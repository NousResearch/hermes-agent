"""Proxy-only egress: ``hermes update`` on a network whose ONLY way out is an HTTP(S) proxy.

Every command runs in its own network namespace (no route, no DNS; proven per module by
``_seed.assert_isolated``). The proxy is what a corporate network runs: it tunnels ``CONNECT``,
inspects TLS with the company's own root, may demand credentials, and refuses every host it has no
route for, logging it. The installed checkout was cloned from the official GitHub URL, so the
update's channel read (``hermes-assets.nousresearch.com``), its git fetch and every lazy blob
fetch of the partial clone (``github.com``) have to cross the proxy, and the cell reads the
proxy's log to show they did.

Failure classes: #124022 (updater dies with ``UNEXPECTED_EOF`` when direct TLS is cut and it
ignores the proxy), a proxy that requires auth, and a corporate root supplied via
``SSL_CERT_FILE`` instead of the OS store.
"""

from __future__ import annotations

import shutil

import pytest

from tests.e2e.core._pending_fixes import known_failure
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

GIT_PATH = "/NousResearch/hermes-agent.git/"


@pytest.fixture(scope="module")
def inst(tmp_path_factory):
    return S.seed_install(tmp_path_factory.mktemp("net-proxy"))


def _assert_updated_through_proxy(inst: S.Installed, r: S.Result, new: str) -> None:
    edge = r.edge
    assert edge is not None
    assert r.rc == 0, "update through the proxy failed\n" + r.report(inst)
    assert inst.head() == new, f"update exited 0 but HEAD is {inst.head()}, not the published {new}\n" + r.report(inst)
    channel = [h for h in edge.proxy.requests(S.ASSETS) if h.path.startswith("/releases/channels/")]
    assert channel, "the channel record was never read through the proxy\n" + r.report(inst)
    fetches = [h for h in edge.proxy.requests("github.com") if h.path.startswith(GIT_PATH) and h.status == 200]
    assert any("git-upload-pack" in h.path for h in fetches), "no git fetch crossed the proxy\n" + r.report(inst)
    stray = sorted(edge.proxy.hosts("refused") | edge.proxy.hosts("tls-rejected"))
    assert not stray, f"the update tried to reach hosts the proxy does not route: {stray}\n" + r.report(inst)
    v = inst.version_works()
    assert v.rc == 0 and I.TRACEBACK not in v.out, "`hermes --version` broken after the update\n" + I.describe(v.cp)


def test_update_through_tls_inspecting_proxy_lands_on_the_new_commit(inst):
    """Corporate root installed in the OS trust store, HTTPS_PROXY set, direct egress impossible."""
    new = inst.publish("tls-inspecting")
    edge = inst.edge()
    try:
        r = inst.hermes("update", "--yes", edge=edge)
    finally:
        edge.close()
    _assert_updated_through_proxy(inst, r, new)


def test_update_through_authenticating_proxy_and_credentials_stay_private(inst):
    """The proxy answers 407 without ``Proxy-Authorization``; the user's HTTPS_PROXY carries
    ``user:password@``. The update must authenticate everywhere and never print the password."""
    new = inst.publish("auth-proxy")
    edge = inst.edge(auth=("corp-user", "pr0xy-s3cret-e2e"))
    try:
        r = inst.hermes("update", "--yes", edge=edge)
    finally:
        edge.close()
    _assert_updated_through_proxy(inst, r, new)
    assert "pr0xy-s3cret-e2e" not in r.out, "the proxy password was printed by the updater\n" + r.report(inst)


def test_update_with_corporate_root_only_in_ssl_cert_file(inst):
    """The corporate root is NOT in the OS store; the user points ``SSL_CERT_FILE`` at a bundle
    that has it (the usual workaround on machines they don't administer). Every TLS client the
    updater drives has to trust it: the channel read and the git fetch."""
    new = inst.publish("ssl-cert-file")
    bundle = inst.root / "ssl-cert-file-bundle.pem"
    bundle.write_bytes(N.SYSTEM_CA_BUNDLE.read_bytes() + b"\n" + inst.ca.pem)
    edge = inst.edge()
    try:
        r = inst.hermes("update", "--yes", edge=edge, corporate_root=False, env={"SSL_CERT_FILE": str(bundle)})
    finally:
        edge.close()
    with known_failure(
        r"unable to access 'https://github\.com/NousResearch/hermes-agent\.git/': "
        r"(SSL certificate|server certificate verification failed|SSL certificate problem)",
        "gated on #124654: the updater's git fetch ignores SSL_CERT_FILE, so a corporate root "
        "supplied that way passes the channel read and then fails at `Fetching updates`",
    ):
        _assert_updated_through_proxy(inst, r, new)


def test_tunnel_cut_to_channel_host_fails_fast_and_changes_nothing(inst):
    """The #124022 symptom made deterministic: the proxy accepts the CONNECT to the release
    host and hangs up before TLS (``UNEXPECTED_EOF``). The update must stop, say the channel
    could not be read, and must not silently fall through to the git branch."""
    inst.publish("tunnel-cut")
    before = inst.state()
    edge = inst.edge(eof_hosts=[S.ASSETS])
    try:
        r = inst.hermes("update", "--yes", edge=edge, timeout=300)
    finally:
        edge.close()
    assert r.secs < 60, f"a cut tunnel took {r.secs:.0f}s to fail\n" + r.report(inst)
    S.assert_nothing_changed(inst, before, r, "tunnel cut to the channel host")
    assert "channel" in r.out.lower() and "unavailable" in r.out.lower(), (
        "the failure does not say the release channel was unreachable\n" + r.report(inst))
