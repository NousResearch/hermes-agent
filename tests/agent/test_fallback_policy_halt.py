"""``fallback_policy.halt``: refuse fallback activation when the operator demands it.

A subscription-backed primary that exhausts its usage window can otherwise start spending paid
fallback API budget invisibly (weeks, on long-running unattended work). With ``halt: true`` the
chain is never walked: the refusal is emitted as a visible diagnostic and the primary failure
surfaces through the normal terminal path. Contracts pinned here:

  1. halt → ``try_activate_fallback`` returns False, emits the refusal, touches no chain state.
  2. default (no policy) → activation proceeds exactly as before.
  3. The shared halt reader reads the effective config and fails OPEN on read errors.
  4. halt makes a populated chain unavailable before any fallback-attempt copy is rendered.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.turn_empty_response import _terminal_empty
from agent.turn_response_check import retry_invalid_response
from agent.turn_retry_state import TurnRetryState
from run_agent import AIAgent


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


class TestHaltPolicy:
    def test_halt_makes_chain_empty_before_invalid_response_copy(self):
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        statuses = []
        response = SimpleNamespace(error=None, choices=[], model=getattr(agent, "model", ""))

        with (
            patch(
                "hermes_cli.fallback_config.fallback_halt_active",
                return_value=(True, "fallback halt refusal"),
            ),
            patch.object(agent, "_buffer_diagnostic_status", side_effect=statuses.append),
            patch.object(agent, "_emit_diagnostic_status", side_effect=statuses.append),
            patch.object(agent, "_flush_status_buffer"),
            patch.object(agent, "_buffer_vprint"),
            patch.object(agent, "_invoke_api_request_error_hook"),
            patch.object(agent, "_persist_session"),
            patch("agent.turn_response_check.stop_thinking_spinner", return_value=None),
        ):
            assert agent._has_pending_fallback() is False
            verdict = retry_invalid_response(
                agent,
                response=response,
                error_details=["no choices"],
                _retry=TurnRetryState(),
                thinking_spinner=None,
                messages=[],
                api_messages=[],
                api_kwargs=None,
                active_system_prompt=None,
                conversation_history=None,
                retry_count=0,
                max_retries=1,
                compression_attempts=0,
                api_call_count=1,
                api_request_id="request",
                api_start_time=0.0,
                api_duration=0.1,
                effective_task_id="task",
                turn_id="turn",
            )
            assert agent._fallback_index == 0
            assert agent._fallback_activated is False
            setattr(agent, "_fallback_activated", False)
            setattr(agent, "_fallback_index", 0)
            setattr(agent, "_empty_content_retries", 1)
            setattr(agent, "model", "primary-model")
            setattr(agent, "provider", "primary-provider")
            _terminal_empty(
                agent,
                SimpleNamespace(
                    content=None,
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                    tool_calls=None,
                ),
                "stop",
                [],
            )

        assert verdict.action == "return"
        rendered = "\n".join(statuses).lower()
        for misleading in (
            "trying fallback", "switching to fallback", "activating fallback", "after fallback attempts"
        ):
            assert misleading not in rendered
        fallback_related = [text for text in statuses if "fallback" in text.lower()]
        assert fallback_related and set(fallback_related) == {"fallback halt refusal"}


class TestHaltReader:
    def test_reads_effective_config_with_false_coercion_and_fail_open(self, tmp_path, monkeypatch):
        import hermes_constants
        (tmp_path / "config.yaml").write_text("fallback_policy:\n  halt: true\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(tmp_path))
        from hermes_cli.fallback_config import fallback_halt_active
        active, message = fallback_halt_active()
        assert active is True
        assert "fallback_policy.halt" in message

        (tmp_path / "config.yaml").write_text('fallback_policy:\n  halt: "false"\n')
        assert fallback_halt_active() == (False, "")

        with patch(
            "hermes_cli.config_effective.load_user_config_effective",
            side_effect=RuntimeError("config loader exploded"),
        ):
            assert fallback_halt_active() == (False, "")

    def test_reads_each_bound_profile_home_a_b_a(self, tmp_path):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.fallback_config import fallback_halt_active

        homes = []
        for name, active in (("a", True), ("b", False)):
            home = tmp_path / name
            home.mkdir()
            home.joinpath("config.yaml").write_text(
                f"fallback_policy:\n  halt: {str(active).lower()}\n",
                encoding="utf-8",
            )
            homes.append(home)

        observed = []
        for home in (homes[0], homes[1], homes[0]):
            token = set_hermes_home_override(home)
            try:
                observed.append(fallback_halt_active()[0])
            finally:
                reset_hermes_home_override(token)

        assert observed == [True, False, True]


def test_pending_fallback_guard_emits_halt_refusal_once(monkeypatch):
    from run_agent import AIAgent

    emitted = []
    agent = AIAgent.__new__(AIAgent)
    setattr(agent, "_fallback_chain", [{"provider": "openrouter", "model": "backup"}])
    setattr(agent, "_fallback_index", 0)
    agent._fallback_halt_notified = False
    agent._emit_diagnostic_status = lambda message: emitted.append(message)
    monkeypatch.setattr(
        "hermes_cli.fallback_config.fallback_halt_active",
        lambda: (True, "fallback halt refusal"),
    )

    assert AIAgent._has_pending_fallback(agent) is False
    assert AIAgent._has_pending_fallback(agent) is False
    assert emitted == ["fallback halt refusal"]
