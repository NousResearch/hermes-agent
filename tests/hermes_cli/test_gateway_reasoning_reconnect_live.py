"""A classic CLI keeps showing reasoning after the gateway restarts (#134889).

The owner keeps ``show_reasoning`` on its live session, so a restarted gateway streams no
``reasoning.delta`` to the CLI's redialed socket until the flag is sent again. This drives the CLI's own
``GatewayConnection`` + ``GatewayAgent`` against two successive disposable ``gateway.run`` daemons on one
home, with a loopback provider that reasons before it answers. The connection and agent survive the restart.
"""
from pathlib import Path

import pytest

from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_session_store import GatewaySessionStore
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon
from tests.gateway.test_cli_plugin_command_in_gateway import _dial_running

REASONING = "REASONING-BEFORE-THE-ANSWER"
ANSWER = "THE-ANSWER-AFTER-RESTART"


@pytest.mark.platforms("linux")
def test_reasoning_preference_survives_a_gateway_restart_on_the_same_cli_connection(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))  # the CLI process's own home: where its dial finds the owner
    conn = None
    with FakeLLMServer([Text(ANSWER, reasoning=REASONING), Text(ANSWER, reasoning=REASONING)]) as provider:
        write_hermes_home(home, provider.base_url)
        config = (home / "config.yaml").read_text() + "gateway:\n  multiplex_profiles: false\n"
        (home / "config.yaml").write_text(config + "auxiliary:\n  title_generation:\n    enabled: false\n")
        env = child_env()
        env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
                   OPENAI_API_KEY="loopback-only", OPENAI_BASE_URL=provider.base_url, PYTHONUNBUFFERED="1")
        conn = GatewayConnection(connect=_dial_running(home))
        store = GatewaySessionStore(conn)
        events = []

        def turn(agent, text):
            start = len(events)
            result = agent.run_conversation(text, stream_callback=lambda delta: events.append(("answer", delta)))
            assert not result.get("failed"), result
            return events[start:], result

        try:
            with daemon(root, home, env, barrier=False):
                conn.start(30)
                sid = store.create({})
                agent = GatewayAgent(session_db=store, session_id=sid,
                                     reasoning_callback=lambda text: events.append(("reasoning", text)))
                before, result = turn(agent, "think before restart")
                assert result["final_response"] == ANSWER, result
                assert "".join(t for k, t in before if k == "reasoning") == REASONING, before
                epoch = conn.epoch
            # The daemon is gone; the CLI keeps the same connection and agent objects.
            with daemon(root, home, env, barrier=False):
                after, result = turn(agent, "think after restart")
                assert conn.epoch == epoch + 1, "the restart was not a redial of the same connection"
                assert result["final_response"] == ANSWER, result
                kinds = [kind for kind, _ in after]
                assert "".join(t for k, t in after if k == "reasoning") == REASONING, after
                assert "reasoning" in kinds and "answer" in kinds, after
                assert max(i for i, k in enumerate(kinds) if k == "reasoning") < kinds.index("answer"), after
            assert len(provider.main_requests()) == 2
        finally:
            if conn is not None:
                conn.close()
