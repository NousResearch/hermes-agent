from __future__ import annotations

import sys
import types

import pytest


def test_load_resume_context_hydrates_exact_canonical_session():
    from hermes_cli.oneshot import _load_resume_context

    calls = []

    class SessionDB:
        def resolve_session_id(self, requested):
            calls.append(("resolve", requested))
            return "session_exact"

        def resolve_resume_session_id(self, resolved):
            calls.append(("resume", resolved))
            return "session_leaf"

        def get_messages_as_conversation(self, resolved, repair_alternation=False):
            calls.append(("history", resolved, repair_alternation))
            return [
                {"role": "session_meta", "content": "private metadata"},
                {"role": "user", "content": "thread-specific request"},
                {"role": "assistant", "content": "thread-specific response"},
            ]

        def reopen_session(self, resolved):
            calls.append(("reopen", resolved))

    session_id, history = _load_resume_context(SessionDB(), "session_requested")

    assert session_id == "session_leaf"
    assert history == [
        {"role": "user", "content": "thread-specific request"},
        {"role": "assistant", "content": "thread-specific response"},
    ]
    assert calls == [
        ("resolve", "session_requested"),
        ("resume", "session_exact"),
        ("history", "session_leaf", True),
        ("reopen", "session_leaf"),
    ]


def test_load_resume_context_rejects_unknown_session():
    from hermes_cli.oneshot import _load_resume_context

    class SessionDB:
        def resolve_session_id(self, requested):
            return None

    with pytest.raises(ValueError, match="Session not found: missing"):
        _load_resume_context(SessionDB(), "missing")


def test_load_resume_context_fails_closed_without_session_db():
    from hermes_cli.oneshot import _load_resume_context

    with pytest.raises(RuntimeError, match="cannot resume safely"):
        _load_resume_context(None, "session_exact")


def test_run_agent_binds_resolved_session_and_history(monkeypatch):
    import hermes_cli.oneshot as oneshot_mod

    constructor = {}
    conversation = {}
    closed = []

    class FakeSessionDB:
        def resolve_session_id(self, requested):
            assert requested == "session_requested"
            return "session_exact"

        def resolve_resume_session_id(self, resolved):
            assert resolved == "session_exact"
            return "session_leaf"

        def get_messages_as_conversation(self, resolved, repair_alternation=False):
            assert (resolved, repair_alternation) == ("session_leaf", True)
            return [{"role": "user", "content": "thread-specific request"}]

        def reopen_session(self, resolved):
            assert resolved == "session_leaf"

        def close(self):
            closed.append(True)

    class FakeAgent:
        def __init__(self, **kwargs):
            constructor.update(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt, **kwargs):
            conversation.update({"prompt": prompt, **kwargs})
            return {"final_response": "done"}

        def shutdown_memory_provider(self, *_args):
            pass

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent)
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "gpt-test", "provider": "openai"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "key",
            "base_url": "https://example.invalid",
            "provider": "openai",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(
        oneshot_mod, "_create_session_db_for_oneshot", lambda: FakeSessionDB()
    )

    result = oneshot_mod._run_agent(
        "continue this thread",
        model="gpt-test",
        provider="openai",
        use_config_toolsets=False,
        resume="session_requested",
        pass_session_id=True,
    )

    assert result == ("done", {"final_response": "done"})
    assert constructor["session_id"] == "session_leaf"
    assert constructor["pass_session_id"] is True
    assert conversation == {
        "prompt": "continue this thread",
        "conversation_history": [
            {"role": "user", "content": "thread-specific request"}
        ],
    }
    assert closed == [True]


def test_run_agent_closes_session_db_when_resume_is_unknown(monkeypatch):
    import hermes_cli.oneshot as oneshot_mod

    closed = []

    class FakeSessionDB:
        def resolve_session_id(self, requested):
            assert requested == "missing"
            return None

        def close(self):
            closed.append(True)

    monkeypatch.setitem(
        sys.modules,
        "run_agent",
        types.SimpleNamespace(AIAgent=lambda **_kwargs: None),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "gpt-test", "provider": "openai"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "key",
            "base_url": "https://example.invalid",
            "provider": "openai",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(
        oneshot_mod, "_create_session_db_for_oneshot", lambda: FakeSessionDB()
    )

    with pytest.raises(ValueError, match="Session not found: missing"):
        oneshot_mod._run_agent(
            "continue this thread",
            model="gpt-test",
            provider="openai",
            use_config_toolsets=False,
            resume="missing",
        )

    assert closed == [True]