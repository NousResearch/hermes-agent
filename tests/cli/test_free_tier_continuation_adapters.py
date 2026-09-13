"""A continuation gate is not a model answer or an automatic retry."""
from types import SimpleNamespace
from unittest.mock import Mock
import queue

import pytest

from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
from hermes_cli.cli_loops_mixin import CLILoopsMixin
from hermes_cli.free_tier_usage import LIMIT_NOTICE


def gated_result(*, refused=True):
    return {"final_response": LIMIT_NOTICE, "messages": [], "api_calls": 0,
            "completed": not refused, "failed": refused,
            "failure_reason": "free_tier_limit" if refused else None,
            "refusal_reason": "free_tier_limit" if refused else None,
            "failure_retryable": False, "free_tier": {"capped": True}}


@pytest.mark.parametrize("refused", [False, True])
def test_no_goal_or_loop_judge_after_gate(refused):
    cli = SimpleNamespace(_last_turn_result=gated_result(refused=refused),
        _get_goal_manager=Mock(side_effect=AssertionError("must not judge stale history")),
        _get_loop_manager=Mock(side_effect=AssertionError("must not complete or reschedule")))
    CLILoopsMixin._maybe_continue_goal_after_turn(cli)
    CLILoopsMixin._maybe_complete_loop_tick_after_turn(cli)


def test_gate_preserves_every_envelope_beyond_stash_capacity():
    from hermes_cli.prompt_stash import PromptStash, MAX_STASH_ITEMS
    from cli import _VoiceInputMessage
    cli = SimpleNamespace(_pending_input=queue.Queue(), _last_turn_result=gated_result(),
                          _prompt_stash=PromptStash())
    cli._prompt_stash.stash("existing composer draft")
    envelopes = [(f"queued @file:{i}", [f"{i}.png"]) for i in range(MAX_STASH_ITEMS + 5)]
    envelopes.append(_VoiceInputMessage("spoken original"))
    for envelope in envelopes:
        cli._pending_input.put(envelope)
    cli._pending_input.put("/login")
    cli._pending_input.put("/model local")
    assert CLIChatTurnMixin._get_pending_input(cli) == "/login"
    assert CLIChatTurnMixin._get_pending_input(cli) == "/model local"
    with pytest.raises(queue.Empty):
        CLIChatTurnMixin._get_pending_input(cli)
    assert cli._prompt_stash.pop() == ("existing composer draft", [])
    # Admission recovery resumes the actual FIFO, not an inaccessible side list.
    cli._last_turn_result = None
    for envelope in envelopes:
        assert CLIChatTurnMixin._get_pending_input(cli) is envelope
    assert cli._pending_input.empty()


def test_cli_worker_recovers_parked_input_after_model_command():
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    cli._last_turn_result = gated_result()
    cli._should_exit = cli._agent_running = False
    original = ("draft", ["image.png"])
    cli._pending_input.put(original)
    cli._pending_input.put("/model local")
    seen = []
    def dispatch(value):
        seen.append(value)
        if value == "/model local":
            cli._last_turn_result = None
        else:
            cli._should_exit = True
    cli._tui_process_one_input = dispatch
    cli._tui_process_loop()
    assert seen == ["/model local", original]


@pytest.mark.parametrize("stream_kind", ["tts", "tokens", "preview"])
def test_streaming_tts_prints_display_only_gate_notice(monkeypatch, stream_kind):
    import cli as cli_module
    printed = []
    monkeypatch.setattr(cli_module, "_cprint", printed.append)
    cli = SimpleNamespace(_stream_started=stream_kind == "tokens", _stream_box_opened=stream_kind == "tokens",
                          _scrollback_box_width=lambda: 80)
    result = gated_result(refused=False)
    result.update(response_transformed=True, free_tier_notice=LIMIT_NOTICE,
                  final_response="answer\n\n" + LIMIT_NOTICE)
    result["response_previewed"] = stream_kind == "preview"
    turn = SimpleNamespace(result=result, use_streaming_tts=stream_kind == "tts", box_opened=True)
    CLIChatTurnMixin._chat_print_response_panel(cli, turn, result["final_response"])
    assert LIMIT_NOTICE in "\n".join(printed)
    assert "answer" not in "\n".join(printed)


@pytest.mark.parametrize("refused", [False, True])
def test_quiet_gate_never_starts_kanban_judge(monkeypatch, refused):
    import cli as cli_module
    result = gated_result(refused=refused)
    cli = SimpleNamespace(agent=SimpleNamespace(run_conversation=lambda **kw: result),
                          conversation_history=[], session_id="s")
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    judge = Mock()
    monkeypatch.setattr(cli_module, "_run_kanban_goal_loop_q", judge)
    with pytest.raises(SystemExit) as exc:
        cli_module._run_quiet_single_query(cli, "original")
    assert exc.value.code == (1 if refused else 0)
    judge.assert_not_called()


@pytest.mark.parametrize("live_agent", [False, True])
def test_idle_gate_uses_selected_route_until_actual_recovery(tmp_path, monkeypatch, live_agent):
    from cli import HermesCLI
    from hermes_cli import auth, auth_nous, free_tier_usage as usage
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    guest = {"auth_method": "anonymous", "anon_token": "anon_idle_fixture"}
    auth._save_active_provider_state("nous", guest)
    auth_nous._write_shared_nous_state(guest)
    for _ in range(usage.TOOL_CALL_CAP):
        usage.record_completed_tool(usage.current_identity())
    route = {"base_url": "https://welcome-api.nousresearch.com/v1"}
    agent = SimpleNamespace(**route) if live_agent else None
    cli = SimpleNamespace(_last_turn_result=gated_result(), base_url="http://localhost:8080/v1",
        agent=agent, _ensure_runtime_credentials=lambda: True,
        _resolve_turn_agent_config=lambda _: {"runtime": route},
        _drain_interrupt_queue_to_pending_input=Mock(),
        _check_config_mcp_changes=lambda: None, _check_termios_drift=lambda: None,
        _drain_process_notifications=Mock(), _maybe_fire_loop_tick=Mock(), _maybe_resume_parked_goal=Mock())
    HermesCLI._tui_idle_tick(cli)
    cli._drain_process_notifications.assert_not_called()
    cli._maybe_fire_loop_tick.assert_not_called()
    # Reverse the mismatch: /model selected local while CLI.base_url is stale welcome.
    cli.base_url = route["base_url"]
    route["base_url"] = "http://localhost:8080/v1"
    if agent is not None:
        agent.base_url = route["base_url"]
    HermesCLI._tui_idle_tick(cli)
    cli._drain_process_notifications.assert_called_once()
    assert cli._last_turn_result is None
    assert usage.status()["capped"]
    # A completed /login leaves the old agent in place until credentials refresh.
    cli._last_turn_result = gated_result()
    cli.agent = SimpleNamespace(base_url=cli.base_url)
    def refresh_credentials():
        cli.agent = None
        return True
    cli._ensure_runtime_credentials = refresh_credentials
    HermesCLI._tui_idle_tick(cli)
    assert cli._last_turn_result is None
    assert cli._drain_process_notifications.call_count == 2


def test_quiet_preflight_skips_image_auxiliary_when_capped(monkeypatch):
    import cli as mod
    from agent import free_tier
    cli = SimpleNamespace(_claim_active_session=lambda *a, **kw: True,
        _ensure_runtime_credentials=lambda: True, _active_agent_route_signature="route",
        _resolve_turn_agent_config=lambda _: {"signature": "route", "model": "m", "runtime": {}},
        _init_agent=lambda **kw: True, agent=SimpleNamespace())
    monkeypatch.setattr(mod, "_collect_query_images", lambda *a: ("original", ["image.png"]))
    monkeypatch.setattr(mod, "_collect_kanban_task_images", lambda *a: [])
    monkeypatch.setattr(mod, "_finalize_single_query", lambda *a: None)
    monkeypatch.setattr(free_tier, "refusal", lambda *a: gated_result())
    image = Mock(side_effect=AssertionError("no image tool before admission"))
    monkeypatch.setattr(mod, "_route_single_query_images", image)
    def run(cli, query):
        assert query == "original"
        raise SystemExit(1)
    monkeypatch.setattr(mod, "_run_quiet_single_query", run)
    with pytest.raises(SystemExit):
        mod._run_single_query_mode(cli, "original", None, True, True)
    image.assert_not_called()


def test_kanban_continuation_stops_before_judging_gate(monkeypatch):
    import cli as cli_module
    from hermes_cli import goals, kanban_db, kanban_db_connect
    from contextlib import nullcontext
    result = gated_result()
    cli = SimpleNamespace(agent=SimpleNamespace(run_conversation=lambda **kw: result),
                          conversation_history=[], session_id="s")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "fixture")
    monkeypatch.setattr(kanban_db_connect, "connect_closing", lambda: nullcontext(None))
    monkeypatch.setattr(kanban_db, "get_task", lambda *a: SimpleNamespace(title="task", body="", goal_max_turns=3))
    def loop(**kw):
        with pytest.raises(RuntimeError, match="free_tier_limit"):
            kw["run_turn"]("continue")
    monkeypatch.setattr(goals, "run_kanban_goal_loop", loop)
    cli_module._run_kanban_goal_loop_q(cli, "first answer")
    assert cli._last_turn_result is result


def test_oneshot_refusal_is_nonzero_even_with_notice(monkeypatch, capsys):
    from hermes_cli import oneshot
    result = gated_result()
    monkeypatch.setattr(oneshot, "_run_agent", lambda *a, **kw: (LIMIT_NOTICE, result))
    assert oneshot.run_oneshot("original") == 2
    assert LIMIT_NOTICE in capsys.readouterr().out
