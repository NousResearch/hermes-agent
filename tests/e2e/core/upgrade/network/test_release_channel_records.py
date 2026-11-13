"""Stable releases: the updater follows the latest published GitHub release exactly, or refuses.

A source install subscribed to ``stable`` resolves its target from GitHub's latest published
release (``api.github.com/repos/<repo>/releases/latest``: non-draft, non-prerelease, strict
``vX.Y.Z``) and verifies the tag on origin points at the commit GitHub reports. No R2 channel
record is read. The edge serves the API (behind the TLS-inspecting proxy that is the
namespace's only egress) and the git server behind the same proxy serves the source. In every
cell origin/main has moved past the release, so "the updater silently followed main" is
visible as a HEAD that landed on the main tip.

Classes: the release is valid (land exactly on its commit), unavailable (404 / 503 / tunnel cut),
unacceptable (draft, prerelease, not JSON), or its commit disagrees with the tag on origin.
In each failure the updater refuses stable with a clear message instead of moving to main.
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

LATEST = f"/repos/{S.REPOSITORY}/releases/latest"


@pytest.fixture(scope="module")
def inst(tmp_path_factory):
    inst = S.seed_install(tmp_path_factory.mktemp("net-channels"))
    r = inst.hermes("update", "--set-channel", "stable", edge=None, timeout=120)
    assert r.rc == 0, "could not subscribe the install to stable\n" + r.report(inst)
    return inst


def _release_and_main_tip(inst: S.Installed, tag: str) -> tuple[str, str]:
    """A release commit on top of the install, then a newer main tip on top of it."""
    release = inst.publish(f"{tag}-release")
    tip = I.publish_commit(inst.origin, inst.root, f"e2e network: {tag} main tip",
                           {f"E2E_NETWORK_{tag}_TIP.txt": "unreleased\n"})
    return release, tip


def _update(inst: S.Installed, api: N.App | None = None, **edge_kw) -> S.Result:
    edge = inst.edge(extra={S.GITHUB_API: api or N.static_app({})}, **edge_kw)
    try:
        return inst.hermes("update", "--yes", edge=edge, timeout=600)
    finally:
        edge.close()


def test_published_release_lands_exactly_on_its_commit_without_reading_r2(inst):
    release, tip = _release_and_main_tip(inst, "valid")
    r = _update(inst, N.static_app(S.stable_release(inst, release)))
    assert r.rc == 0, "update to the published stable release failed\n" + r.report(inst)
    head = inst.head()
    assert head != tip, "stable subscriber was moved to the unreleased main tip\n" + r.report(inst)
    assert head == release, f"stable update landed on {head}, not the released {release}\n" + r.report(inst)
    reads = [h.path for h in r.edge.proxy.requests(S.GITHUB_API)] if r.edge else []
    assert LATEST in reads, f"the latest release was not read through the proxy: {reads}\n" + r.report(inst)
    r2 = [h.path for h in r.edge.proxy.requests(S.ASSETS)] if r.edge else []
    assert not any("/releases/" in path for path in r2), f"stable read R2 release objects: {r2}\n" + r.report(inst)


def _refused(inst: S.Installed, r: S.Result, before: dict, what: str) -> None:
    S.assert_nothing_changed(inst, before, r, what)
    assert "stable" in r.out and "No update was applied" in r.out, (
        f"{what}: the refusal does not name the stable channel and say nothing was applied\n" + r.report(inst))


UNAVAILABLE = {
    "no-release-404": dict(api=N.static_app({})),
    "outage-503": dict(api=N.static_app({}, always={LATEST: N.Response(503, b"down\n")})),
    "tunnel-cut": dict(eof_hosts=[S.GITHUB_API]),
}


@pytest.mark.parametrize("fault", sorted(UNAVAILABLE))
def test_unavailable_release_refuses_and_never_moves_to_main(inst, fault):
    _release_and_main_tip(inst, f"unavailable-{fault}")
    before = inst.state()
    r = _update(inst, **UNAVAILABLE[fault])
    _refused(inst, r, before, f"stable release {fault}")


def _unacceptable(inst: S.Installed, release: str, fault: str) -> dict[str, bytes]:
    if fault == "not-json":
        return {LATEST: b"<html>502 Bad Gateway</html>\n"}
    if fault == "origin-tag-mismatch":
        return S.stable_release(inst, release, api_commit="f" * 40)
    return S.stable_release(inst, release, **{fault: True})


@pytest.mark.parametrize("fault", ["draft", "not-json", "origin-tag-mismatch", "prerelease"])
def test_unacceptable_release_refuses(inst, fault):
    release, _ = _release_and_main_tip(inst, f"unacceptable-{fault}")
    before = inst.state()
    r = _update(inst, N.static_app(_unacceptable(inst, release, fault)))
    _refused(inst, r, before, f"stable release {fault}")
