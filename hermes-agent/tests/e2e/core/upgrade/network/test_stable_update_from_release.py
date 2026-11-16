"""A stable subscriber on the previous release updates to the next published release.

The install is the newest by-path release (``_install_helpers.upgrade_base("by-path")``), put
there by that release's own installer and subscribed to ``stable``, so the updater that runs is
the one users have today. Upstream, HEAD is published as the latest GitHub release and an
unreleased commit sits on main above it. The update must land exactly on HEAD, not follow main,
and the next ``hermes`` must start.

The edge fakes api.github.com and github.com as in ``test_release_channel_records`` and tunnels
every other host to the real one: from a release, the update downloads the dependencies HEAD adds.
"""

from __future__ import annotations

import shutil

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.network import _netedge as N
from tests.e2e.core.upgrade.network import _seed as S

pytestmark = [
    pytest.mark.allow_network,  # the edge tunnels the hosts it does not fake to the real ones
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
    base = I.upgrade_base("by-path")
    if base is None:
        pytest.skip("no by-path release is reachable from HEAD (a tagless checkout)")
    inst = S.seed_install(tmp_path_factory.mktemp("net-stable-from-release"), base.commit)
    r = inst.hermes("update", "--set-channel", "stable", edge=None, timeout=120)
    assert r.rc == 0, f"{base.ref} could not subscribe to stable\n" + r.report(inst)
    return inst


def test_stable_update_from_the_previous_release_lands_on_the_next_release(inst):
    head = I.head_sha()
    I.git("update-ref", "refs/heads/main", head, cwd=inst.origin)
    tip = I.publish_commit(inst.origin, inst.root, "e2e network: unreleased main tip",
                           {"E2E_NETWORK_STABLE_TIP.txt": "unreleased\n"})
    edge = inst.edge(extra={S.GITHUB_API: N.static_app(S.stable_release(inst, head))}, tunnel_unrouted=True)
    try:
        r = inst.hermes("update", "--yes", edge=edge, timeout=1500)
    finally:
        edge.close()
    assert r.rc == 0, "stable update from the previous release failed\n" + r.report(inst)
    landed = inst.head()
    assert landed != tip, "stable subscriber was moved to the unreleased main tip\n" + r.report(inst)
    assert landed == head, f"stable update landed on {landed}, not the released {head}\n" + r.report(inst)
    failed = [str(h) for h in edge.proxy.hits if h.kind == "tunnel-failed"]
    assert not failed, f"the edge could not reach a real host the update needed: {failed}\n" + r.report(inst)
    reads = [h.path for h in edge.proxy.requests(S.GITHUB_API)]
    assert LATEST in reads, f"the latest release was not read through the proxy: {reads}\n" + r.report(inst)
    v = inst.version_works()
    assert v.rc == 0 and I.TRACEBACK not in v.out, "`hermes --version` broken after the update\n" + I.describe(v.cp)
