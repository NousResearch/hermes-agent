"""Surface contract for the Claude Agent SDK runtime.

PR4's event projector already emits the canonical Hermes callbacks, so the
interactive surfaces need no per-surface code. The non-interactive ones do, and
these are the three properties they were missing:

* **Reachability.** Every non-interactive surface builds its agent from
  ``resolve_runtime_provider()``. Without a ``claude-code`` branch there, a
  subscription-mode cron job / ACP session / batch run silently resolved to
  ``anthropic`` and resumed the pre-SDK direct-OAuth billing path.
* **Fail fast, not hang.** A missing login must surface as an actionable
  message before work starts, never as a login prompt on a headless process.
* **Teardown.** These surfaces run many agents; each SDK agent owns an OS
  thread and a Claude Code subprocess.

Image input is here too because it is a whole-turn property that every surface
shares: an attachment either reaches the SDK or fails loudly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


@pytest.fixture
def subscription_gate_open(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.claude_code.subscription_enabled", lambda config=None: True
    )


class TestRuntimeProviderResolution:
    def test_resolves_to_the_sdk_runtime_and_not_to_anthropic(
        self, subscription_gate_open
    ):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        with patch(
            "hermes_cli.runtime_provider.resolve_external_process_provider_credentials",
            return_value={
                "provider": "claude-code",
                "api_key": "",
                "base_url": "claude-sdk://subscription",
                "command": "/usr/bin/claude",
                "credentials_owner": "claude-agent-sdk",
                "source": "claude_agent_sdk",
            },
        ):
            runtime = resolve_runtime_provider(requested="claude-code")

        assert runtime["provider"] == "claude-code"
        assert runtime["api_mode"] == "claude_agent_sdk"
        assert runtime["api_key"] == ""
        assert runtime["credentials_owner"] == "claude-agent-sdk"
        # Not a reachable REST endpoint — nothing may treat it as one.
        assert not runtime["base_url"].startswith("http")

    def test_a_missing_claude_cli_propagates_its_actionable_error(
        self, subscription_gate_open
    ):
        from hermes_cli.auth import AuthError
        from hermes_cli.runtime_provider import resolve_runtime_provider

        with patch(
            "hermes_cli.runtime_provider.resolve_external_process_provider_credentials",
            side_effect=AuthError(
                "Could not find the `claude` CLI. Install Claude Code, then run "
                "`claude auth login`."
            ),
        ):
            with pytest.raises(AuthError) as excinfo:
                resolve_runtime_provider(requested="claude-code")
        assert "claude auth login" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ACP
# ---------------------------------------------------------------------------


class TestAcpAuth:
    def test_an_empty_api_key_is_not_an_unconfigured_provider(self):
        from acp_adapter.auth import detect_provider

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": "claude-code",
                "api_key": "",
                "api_mode": "claude_agent_sdk",
                "credentials_owner": "claude-agent-sdk",
            },
        ):
            assert detect_provider() == "claude-code"

    def test_a_genuinely_unconfigured_provider_is_still_rejected(self):
        from acp_adapter.auth import detect_provider

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": "openrouter",
                "api_key": "",
                "api_mode": "chat_completions",
            },
        ):
            assert detect_provider() is None


class TestAcpSessionLifecycle:
    def _manager(self):
        from acp_adapter.session import SessionManager

        return SessionManager(agent_factory=lambda: MagicMock())

    def test_removing_a_session_closes_its_agent(self):
        manager = self._manager()
        state = manager.create_session(cwd=".")
        agent = state.agent
        assert manager.remove_session(state.session_id) is True
        agent.close.assert_called_once()

    def test_cleanup_closes_every_live_agent(self):
        manager = self._manager()
        agents = [manager.create_session(cwd=".").agent for _ in range(3)]
        manager.cleanup()
        for agent in agents:
            agent.close.assert_called_once()

    def test_replacing_an_agent_closes_the_one_it_replaces(self):
        manager = self._manager()
        state = manager.create_session(cwd=".")
        previous = state.agent
        replacement = MagicMock()
        manager.replace_agent(state, replacement)
        previous.close.assert_called_once()
        assert state.agent is replacement

    def test_replacing_an_agent_with_itself_does_not_close_it(self):
        manager = self._manager()
        state = manager.create_session(cwd=".")
        manager.replace_agent(state, state.agent)
        state.agent.close.assert_not_called()

    def test_a_failing_close_never_breaks_teardown(self):
        from acp_adapter.session import SessionManager

        agent = MagicMock()
        agent.close.side_effect = RuntimeError("wedged subprocess")
        SessionManager.close_agent(agent)  # must not raise


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestBatchRunner:
    @pytest.fixture
    def dataset(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text('{"prompt": "hi"}\n', encoding="utf-8")
        return str(path)

    def _runner(self, dataset, **kwargs):
        import batch_runner

        return batch_runner.BatchRunner(
            dataset_file=dataset,
            batch_size=1,
            run_name="preflight-test",
            **kwargs,
        )

    def test_preflight_refuses_to_start_when_not_signed_in(self, dataset):
        runner = self._runner(dataset, api_mode="claude_agent_sdk")
        with patch(
            "agent.claude_runtime.claude_runtime_preflight",
            return_value="Not signed in to Claude — run `claude auth login`.",
        ):
            with pytest.raises(RuntimeError) as excinfo:
                runner._preflight_runtime()
        assert "claude auth login" in str(excinfo.value)

    def test_preflight_is_a_no_op_for_every_other_runtime(self, dataset):
        runner = self._runner(dataset, api_mode="chat_completions")
        with patch(
            "agent.claude_runtime.claude_runtime_preflight",
            side_effect=AssertionError("must not be consulted"),
        ):
            runner._preflight_runtime()

    def test_preflight_passes_when_signed_in(self, dataset):
        runner = self._runner(dataset, api_mode="claude_agent_sdk")
        with patch("agent.claude_runtime.claude_runtime_preflight", return_value=None):
            runner._preflight_runtime()

    def test_provider_and_api_mode_reach_the_agent(self):
        import batch_runner

        config = {
            "distribution": "default",
            "model": "claude-sonnet-5",
            "max_iterations": 3,
            "base_url": "claude-sdk://subscription",
            "api_key": "",
            "provider": "claude-code",
            "api_mode": "claude_agent_sdk",
            "verbose": False,
        }
        with patch("batch_runner.AIAgent") as MockAgent:
            child = MagicMock()
            child.run_conversation.return_value = {
                "messages": [],
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = child
            batch_runner._process_single_prompt(0, {"prompt": "hi"}, 0, config)
            _, kwargs = MockAgent.call_args

        assert kwargs["provider"] == "claude-code"
        assert kwargs["api_mode"] == "claude_agent_sdk"
        # Every prompt closes its agent; the pool has no maxtasksperchild.
        child.close.assert_called_once()

    def test_the_agent_is_closed_even_when_the_turn_raises(self):
        import batch_runner

        config = {
            "distribution": "default",
            "model": "claude-sonnet-5",
            "max_iterations": 3,
            "base_url": "claude-sdk://subscription",
            "api_key": "",
            "provider": "claude-code",
            "api_mode": "claude_agent_sdk",
            "verbose": False,
        }
        with patch("batch_runner.AIAgent") as MockAgent:
            child = MagicMock()
            child.run_conversation.side_effect = RuntimeError("boom")
            MockAgent.return_value = child
            result = batch_runner._process_single_prompt(0, {"prompt": "hi"}, 0, config)

        assert result["success"] is False
        child.close.assert_called_once()


# ---------------------------------------------------------------------------
# Images (a whole-turn property, shared by every surface)
# ---------------------------------------------------------------------------


class TestImageTurns:
    def _agent(self):
        from run_agent import AIAgent

        return AIAgent.__new__(AIAgent)

    def test_a_text_turn_stays_a_plain_string(self):
        agent = self._agent()
        with patch("agent.claude_runtime.run_claude_agent_sdk_turn") as run_turn:
            run_turn.return_value = {"final_response": "ok"}
            agent._run_claude_agent_sdk_turn(
                user_message="hello",
                original_user_message="hello",
                messages=[],
                effective_task_id="t",
            )
        assert run_turn.call_args.kwargs["user_message"] == "hello"

    def test_an_image_turn_becomes_a_stream_json_prompt(self, monkeypatch):
        import agent.claude_sdk_input as sdk_input

        monkeypatch.setattr(sdk_input, "sdk_supports_streaming_input", lambda: True)
        agent = self._agent()
        content = [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        ]
        with patch("agent.claude_runtime.run_claude_agent_sdk_turn") as run_turn:
            run_turn.return_value = {"final_response": "ok"}
            agent._run_claude_agent_sdk_turn(
                user_message=content,
                original_user_message=content,
                messages=[],
                effective_task_id="t",
            )
        prompt = run_turn.call_args.kwargs["user_message"]
        # An async iterable, which is the only structured shape the SDK takes.
        assert hasattr(prompt, "__aiter__")

    def test_an_image_is_never_silently_dropped(self, monkeypatch):
        import agent.claude_sdk_input as sdk_input

        monkeypatch.setattr(sdk_input, "sdk_supports_streaming_input", lambda: False)
        agent = self._agent()
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
        ]
        with patch("agent.claude_runtime.run_claude_agent_sdk_turn") as run_turn:
            result = agent._run_claude_agent_sdk_turn(
                user_message=content,
                original_user_message=content,
                messages=[],
                effective_task_id="t",
            )
        run_turn.assert_not_called()
        assert result["completed"] is False
        assert "auxiliary.vision" in result["final_response"]
        assert "NOT sent" in result["final_response"]
