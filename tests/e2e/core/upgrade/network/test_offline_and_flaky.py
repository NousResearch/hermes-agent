"""Offline and flaky networks: fail fast, loud and truthfully, or retry within bounds.

Every command runs in a private network namespace. ``edge=None`` is a machine with no network
at all; otherwise the only egress is the test's proxy, whose sites script the failure: a release
host answering 503 with ``Retry-After``, a forge that rate-limits (HTTP 429), an artifact host
serving bytes that don't match the pinned digest.

The contract for a failure (``_seed.assert_nothing_changed``): non-zero exit, a message naming
what failed, no success banner, the checkout / selected PM environment / PM runtime / tool store
exactly as before, and the next ``hermes`` still starts. For a transient fault the contract is
that the operation retries within bounds and then succeeds.

Classes: offline update and offline PM provisioning; a release-host outage that ``main`` (which
reads no release record) must not notice; persistent 429 from the forge (#106026 printed the success
banner after a failed fetch); transient 503 then a digest mismatch on a PM download, for the
tools whose artifacts come from GitHub releases and from nodejs.org (#106027's class: a Node
tarball accepted without verification).
"""

from __future__ import annotations

import shutil

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.network import _netedge as N
from tests.e2e.core.upgrade.network import _seed as S

pytestmark = [
    pytest.mark.platforms("linux"),
    # Every `hermes update` here targets a throwaway sandboxed install, never the real checkout.
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(N.netns_required_reason() is not None, reason=str(N.netns_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

MAIN_RECORD = "/releases/channels/main.json"
FAST = 60  # seconds a no-network failure may take, bounded retries included


@pytest.fixture(scope="module")
def inst(tmp_path_factory):
    return S.seed_install(tmp_path_factory.mktemp("net-flaky"))


def test_offline_update_fails_fast_and_changes_nothing(inst):
    inst.publish("offline")
    before = inst.state()
    r = inst.hermes("update", "--yes", edge=None, timeout=300)
    assert r.secs < FAST, f"an offline update took {r.secs:.0f}s to give up\n" + r.report(inst)
    S.assert_nothing_changed(inst, before, r, "offline update")
    assert "unavailable" in r.out.lower() or "cannot reach" in r.out.lower(), (
        "the offline failure does not say the network was unreachable\n" + r.report(inst))


def test_offline_provisioning_of_a_missing_tool_fails_loudly(inst):
    """The tool store lost an entry (pruned, antivirus, disk cleanup) and the machine is offline:
    the documented remedy `hermes pm install <tool>` must fail within bounds, name the tool and
    the download, and leave every other tool alone."""
    with S.tool_missing(inst, "ripgrep") as tool:
        before = inst.state()
        r = inst.hermes("pm", "install", "ripgrep", edge=None, timeout=300)
        assert r.secs < FAST, f"offline provisioning took {r.secs:.0f}s to give up\n" + r.report(inst)
        S.assert_nothing_changed(inst, before, r, "offline `hermes pm install ripgrep`")
        assert "ripgrep" in r.out and "download failed" in r.out, (
            "the failure does not name the tool and the failed download\n" + r.report(inst))
        assert not (inst.sb.hermes_home / "tools" / tool["entry"]).exists(), (
            "an offline install left a store entry behind\n" + r.report(inst))


def test_main_update_reads_no_release_record_and_survives_a_release_host_outage(inst):
    """The release host answers 503 to everything (a CDN outage, or the regional WAF 403s of
    #128295): ``main`` follows the git branch and never reads a release record, so the update
    lands and the host sees zero record requests."""
    new = inst.publish("release-host-outage")
    assets = N.static_app({}, always={MAIN_RECORD: N.Response(503, b"down\n", {"Retry-After": "1"})})
    edge = inst.edge(assets=assets)
    try:
        r = inst.hermes("update", "--yes", edge=edge)
    finally:
        edge.close()
    assert r.rc == 0 and inst.head() == new, "a release-host outage blocked a main update\n" + r.report(inst)
    reads = [h for h in edge.proxy.requests(S.ASSETS) if h.path.startswith("/releases/channels/")]
    assert not reads, f"main read a release record: {reads}\n" + r.report(inst)


def test_forge_rate_limit_fails_truthfully(inst):
    """Every git request gets HTTP 429 (the repo-scoped throttle users hit): the update must exit
    non-zero, say it was rate limited, and print no success banner (#106026)."""
    inst.publish("forge-429")
    before = inst.state()
    git = N.git_app(inst.gitroot, faults=[N.Response(429, b"rate limited\n", {"Retry-After": "1"})] * 50)
    edge = inst.edge(git=git)
    try:
        r = inst.hermes("update", "--yes", edge=edge, timeout=300)
    finally:
        edge.close()
    assert r.secs < FAST * 2, f"a rate-limited fetch took {r.secs:.0f}s to give up\n" + r.report(inst)
    S.assert_nothing_changed(inst, before, r, "forge rate limit")
    assert "429" in r.out or "rate limit" in r.out.lower(), "the failure does not say the forge rate-limited\n" + r.report(inst)


# Where each tool's pinned artifact is downloaded from, as the PM lock declares it; the edge
# serves those hosts (and the content-addressed hermes-assets mirror) with wrong bytes.
ARTIFACT_HOSTS = {
    "ripgrep": ("github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"),
    "node": ("nodejs.org",),
}


@pytest.mark.parametrize("tool_name", sorted(ARTIFACT_HOSTS))
def test_pm_download_retries_then_refuses_a_digest_mismatch(inst, tool_name):
    """Artifact host: two 503s with Retry-After (retried), then bytes that are not the pinned
    artifact. PM must refuse them, say so, and install nothing."""
    bad = b"not the pinned artifact\n" * 4096
    flaky = {"*": [N.Response(503, b"busy\n", {"Retry-After": "1"})] * 2}
    extra = {host: N.static_app({"*": bad}, faults=flaky) for host in ARTIFACT_HOSTS[tool_name]}
    with S.tool_missing(inst, tool_name) as tool:
        before = inst.state()
        edge = inst.edge(assets=N.static_app({"*": bad}), git=extra.pop("github.com", None), extra=extra)
        try:
            r = inst.hermes("pm", "install", tool_name, edge=edge, timeout=600)
        finally:
            edge.close()
        served = [h for h in edge.proxy.hits if h.kind == "request" and h.host in ARTIFACT_HOSTS[tool_name]]
        assert any(h.status == 503 for h in served) and any(h.status == 200 for h in served), (
            f"the artifact host was not retried past its 503s: {[str(h) for h in served]}\n" + r.report(inst))
        assert "sha256 mismatch" in r.out or "digest" in r.out.lower(), (
            f"{tool_name}: the failure does not say the download failed verification\n" + r.report(inst))
        assert not (inst.sb.hermes_home / "tools" / tool["entry"]).exists(), (
            f"{tool_name}: unverified bytes were published to the tool store\n" + r.report(inst))
        S.assert_nothing_changed(inst, before, r, f"`hermes pm install {tool_name}` with a digest mismatch")
