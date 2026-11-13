"""A gateway's one-time tips are marked seen in the profile that was shown them (#154977).

Under a multiplexing gateway a served profile's turn reads its own config.yaml, so the "seen" flag
has to land there too. It used to land in the process (host) profile's config instead: the served
profile got the busy-input tip on every busy message, and the host profile lost its own tip.
"""

import queue
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def served(tmp_path, monkeypatch):
    host, profile = tmp_path / "host", tmp_path / "profiles" / "work"
    for home in (host, profile):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("display:\n  tool_progress_command: true\n", encoding="utf-8")
    monkeypatch.setattr("gateway.run._hermes_home", host)
    token = set_hermes_home_override(profile)
    yield host, profile
    reset_hermes_home_override(token)


def _seen(home) -> dict:
    from hermes_cli.config import read_user_config_raw

    cfg = read_user_config_raw(home / "config.yaml")
    return (cfg.get("onboarding") or {}).get("seen") or {}


def _busy_ack() -> str:
    from gateway.config import Platform
    from gateway.run_busy import GatewayBusySessionMixin

    event = SimpleNamespace(source=SimpleNamespace(platform=Platform.TELEGRAM))
    return GatewayBusySessionMixin()._compose_busy_ack_message(
        event, 0.0, None, None, is_steer_mode=False, is_queue_mode=True, is_redirect_mode=False,
        demoted_for_subagents=False, demoted_for_compression=False,
    )


def test_busy_input_tip_is_shown_once_to_a_served_profile(served):
    from agent.onboarding import BUSY_INPUT_FLAG, busy_input_hint_gateway

    host, profile = served
    hint = busy_input_hint_gateway("queue")
    assert hint in _busy_ack()
    assert hint not in _busy_ack()
    assert _seen(profile).get(BUSY_INPUT_FLAG) is True
    assert BUSY_INPUT_FLAG not in _seen(host)


def test_tool_progress_tip_is_marked_seen_in_the_served_profile(served):
    from agent.onboarding import TOOL_PROGRESS_FLAG
    from gateway.run_turn_progress import GatewayTurnProgressMixin

    host, profile = served
    progress = GatewayTurnProgressMixin()
    progress._ctx = SimpleNamespace(_LONG_TOOL_THRESHOLD_S=1, progress_mode="all",
                                    long_tool_hint_fired=[False], progress_queue=queue.Queue())
    progress._progress_onboarding_hint({"duration": 5})
    assert progress._ctx.progress_queue.qsize() == 1
    progress._progress_onboarding_hint({"duration": 5})
    assert progress._ctx.progress_queue.qsize() == 1
    assert _seen(profile).get(TOOL_PROGRESS_FLAG) is True
    assert TOOL_PROGRESS_FLAG not in _seen(host)
