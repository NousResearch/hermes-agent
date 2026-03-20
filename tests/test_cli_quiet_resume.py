"""Tests for quiet mode with --resume passing conversation history."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch


class _DummyCLI:
    """Minimal stub for HermesCLI used by main()."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.session_id = "20260319_123742_9b9588"
        self.conversation_history = []
        self.tool_progress_mode = "all"
        self._active_agent_route_signature = None
        self.agent = None

    def _ensure_runtime_credentials(self):
        return True

    def _resolve_turn_agent_config(self, query):
        return {
            "signature": self._active_agent_route_signature,
            "model": None,
            "runtime": None,
            "label": None,
        }

    def _init_agent(self, model_override=None, runtime_override=None, route_label=None):
        self.agent = MagicMock()
        self.agent.run_conversation = MagicMock(
            return_value={"final_response": "Your name is Alice"}
        )
        return True


def test_quiet_resume_passes_conversation_history(monkeypatch, capsys):
    """When resuming in quiet mode, conversation_history must be passed to run_conversation()."""
    import cli as cli_mod

    prior_history = [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
    ]

    created = {}

    def fake_cli(**kwargs):
        obj = _DummyCLI(**kwargs)
        # Simulate _init_agent loading session history (as it does with --resume)
        original_init = obj._init_agent

        def patched_init(**kw):
            result = original_init(**kw)
            obj.conversation_history = list(prior_history)
            return result

        obj._init_agent = patched_init
        created["cli"] = obj
        return obj

    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cli)
    monkeypatch.setattr(cli_mod, "build_preloaded_skills_prompt", lambda *a, **kw: ("", [], []))
    # Suppress atexit registration
    monkeypatch.setattr("atexit.register", lambda *a, **kw: None)

    cli_mod.main(query="What is my name?", quiet=True, resume="20260319_123742_9b9588")

    cli_obj = created["cli"]
    # Verify run_conversation was called with conversation_history
    cli_obj.agent.run_conversation.assert_called_once()
    call_kwargs = cli_obj.agent.run_conversation.call_args
    assert call_kwargs.kwargs.get("conversation_history") == prior_history or \
        (len(call_kwargs.args) > 1 and call_kwargs.args[1] == prior_history) or \
        call_kwargs[1].get("conversation_history") == prior_history


def test_quiet_mode_without_resume_passes_empty_history(monkeypatch, capsys):
    """In quiet mode without --resume, conversation_history should be empty or None."""
    import cli as cli_mod

    created = {}

    def fake_cli(**kwargs):
        obj = _DummyCLI(**kwargs)
        created["cli"] = obj
        return obj

    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cli)
    monkeypatch.setattr(cli_mod, "build_preloaded_skills_prompt", lambda *a, **kw: ("", [], []))
    monkeypatch.setattr("atexit.register", lambda *a, **kw: None)

    cli_mod.main(query="Hello", quiet=True)

    cli_obj = created["cli"]
    cli_obj.agent.run_conversation.assert_called_once()
    call_kwargs = cli_obj.agent.run_conversation.call_args
    # conversation_history should be empty list (no prior session)
    history = call_kwargs.kwargs.get("conversation_history") or \
        (call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
    assert history is None or history == []
