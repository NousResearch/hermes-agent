"""A network failure during ``hermes update`` must never take down a healthy gateway (the exit-75
class, #123370: keep it closed).

The upstream is published, but the git transport is refused: the official URL is rewritten to a
loopback port that nobody listens on, the way an offline laptop or a dead proxy looks. The update
fails and says so. The gateway the user started keeps its PID, keeps serving the installed commit,
and keeps answering turns. The checkout stays where it was.
"""

from __future__ import annotations

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.handoff import _handoff as X
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.skipif(not H.BWRAP_OK, reason="bubblewrap sandbox unavailable"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
    pytest.mark.live_system_guard_bypass,  # one bwrap sandbox per cell; killing it reaps everything
]


def _refuse_git_transport(inst: X.Install) -> str:
    dead = f"http://127.0.0.1:{X.free_port()}/NousResearch/hermes-agent.git"
    (inst.home / ".gitconfig").write_text(
        f'[url "{dead}"]\n  insteadOf = {I.OFFICIAL_HTTPS}\n  insteadOf = {I.OFFICIAL_SSH}\n', encoding="utf-8")
    return dead


@pytest.mark.parametrize("column", ["n1", "head"])
def test_refused_fetch_leaves_the_running_gateway_alone(root, column):
    if column == "n1" and not X.refs().base:
        pytest.skip("no release tag reachable (shallow checkout)")
    with FakeLLMServer(default_text="still here") as srv, X.cell(column, root, srv.base_url) as inst:
        before = X.start_gateway(inst)
        installed = inst.sha()
        boots = len(X.gateway_starts(inst))
        X.publish_target(inst)
        _refuse_git_transport(inst)

        up = inst.update(timeout=600)

        diag = inst.diagnostics(up)
        assert up.returncode != 0, f"a refused fetch was reported as success\n{diag}"
        out = (up.stdout + up.stderr).lower()
        assert any(w in out for w in ("fetch", "network", "connect", "unable to access", "could not")), (
            f"the failed update does not say why\n{diag}")
        assert X.TRACEBACK not in up.stdout + up.stderr, f"the failed update crashed\n{diag}"
        assert inst.sha() == installed, f"a failed fetch moved the checkout\n{diag}"
        ident = X.identify(inst.hermes_home)
        assert ident is not None and ident["pid"] == before["pid"], (
            f"the running gateway was stopped or replaced by an update that never fetched: {ident}\n{diag}")
        assert ident["code_sha"] == installed, f"gateway identity changed: {ident}\n{diag}"
        assert len(X.gateway_starts(inst)) == boots, f"the gateway was restarted by a failed update\n{diag}"
        assert [p["pid"] for p in X.gateway_pids(inst)] == [before["pid"]], f"gateway process set changed\n{diag}"
        status, body = X.chat(inst.port, "are you still there?")
        assert status == 200 and X.reply_text(body) == "still here", f"no turn after the failed update: {body}\n{diag}"
