from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hermes_cli import main_tui_launch


def _args(**overrides):
    base = {
        "continue_last": None,
        "model": None,
        "provider": None,
        "resume": None,
        "toolsets": None,
        "tui": True,
        "tui_dev": False,
    }
    base.update(overrides)
    return Namespace(**base)


def _raise_exit(rc):
    raise SystemExit(rc)


def _mod(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _mock_response(content="ok"):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="test/model",
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        ),
    )


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    # Reset the idempotency guard so each test starts fresh.
    monkeypatch.setattr(mod, "_oneshot_cleanup_done", False)
    return mod
















def test_termux_skips_bundled_skill_sync_when_stamp_fresh(monkeypatch, tmp_path, main_mod):
    calls = []

    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(main_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(main_mod, "_termux_bundled_skills_fingerprint", lambda: "fp1")
    main_mod._mark_termux_bundled_skills_synced()
    monkeypatch.setitem(
        sys.modules,
        "tools.skills_sync",
        types.SimpleNamespace(sync_skills=lambda quiet: calls.append(quiet)),
    )

    assert main_mod._sync_bundled_skills_for_startup() is False
    assert calls == []






def test_exit_after_oneshot_flushes_stdio_and_calls_os_exit(
    monkeypatch, main_mod
):
    flushed = []
    exits = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def flush(self):
            flushed.append(self.name)

    def fake_exit(rc):
        exits.append(rc)
        raise SystemExit(rc)

    monkeypatch.setattr(main_mod.sys, "stdout", FakeStream("stdout"))
    monkeypatch.setattr(main_mod.sys, "stderr", FakeStream("stderr"))
    monkeypatch.setattr(main_mod.os, "_exit", fake_exit)
    monkeypatch.setattr("logging.shutdown", lambda: None)

    with pytest.raises(SystemExit) as exc:
        main_mod._exit_after_oneshot(17)

    assert exc.value.code == 17
    assert exits == [17]
    assert flushed == ["stdout", "stderr"]






def test_oneshot_subprocess_exits_without_teardown_abort():
    program = textwrap.dedent(
        """
        import hermes_cli.oneshot as oneshot
        from hermes_cli.main import _exit_after_oneshot

        oneshot._run_agent = lambda *args, **kwargs: ("ok", {"final_response": "ok"})
        _exit_after_oneshot(oneshot.run_oneshot("hello"))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == b"ok\n"
    # Don't demand byte-empty stderr — an import-time warning from the heavy
    # CLI import chain shouldn't fail this. What matters is no crash traceback.
    assert b"Traceback" not in result.stderr


@pytest.mark.parametrize(
    "terminal_result",
    [
        {"final_response": "busy", "failed": True, "completed": False},
        {"final_response": "stopped", "interrupted": True, "completed": False},
    ],
)
def test_oneshot_returns_nonzero_when_turn_was_not_processed(
    monkeypatch, capsys, terminal_result
):
    import hermes_cli.oneshot as oneshot

    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: (terminal_result["final_response"], terminal_result),
    )

    assert oneshot.run_oneshot("continue") == 2
    assert capsys.readouterr().out == f"{terminal_result['final_response']}\n"


def test_oneshot_does_not_finalize_session_when_turn_admission_failed(monkeypatch):
    import hermes_cli.oneshot as oneshot

    events = []

    class FakeDB:
        def close(self):
            events.append("db_close")

    class FakeAgent:
        session_id = "busy-session"
        _last_turn_admitted = False
        _end_session_on_close = True
        _session_messages = []

        def shutdown_memory_provider(self, *_args):
            events.append("memory_close")

        def close(self):
            if self._end_session_on_close:
                events.append("session_end")
            events.append("agent_close")

    monkeypatch.setattr(oneshot, "_linger_for_background_completions", lambda: None)
    monkeypatch.setattr(
        oneshot, "_flush_oneshot_session_store", lambda _agent: events.append("session_flush")
    )

    agent = FakeAgent()
    oneshot._close_agent(agent, FakeDB())

    assert events == ["memory_close", "agent_close", "db_close"]
    assert agent._end_session_on_close is False








def _stub_plugin_discovery(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )




def test_oneshot_wires_session_db_for_recall(monkeypatch):
    """hermes -z bypasses HermesCLI, but recall still needs SessionDB."""
    from hermes_cli.oneshot import _run_agent

    captured = {}
    sentinel_db = object()

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt, **_kwargs):
            captured["prompt"] = prompt
            return {"final_response": "ok", "failed": False, "partial": False}

    class FakeSessionDB:
        def __new__(cls):
            return sentinel_db

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(sys.modules, "hermes_state", mod("hermes_state", SessionDB=FakeSessionDB))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        mod("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        mod("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: {"session_search"}),
    )

    text, result = _run_agent("recall this")
    assert text == "ok"
    assert not result.get("failed")
    assert captured["session_db"] is sentinel_db
    assert captured["enabled_toolsets"] == ["session_search"]
    assert captured["prompt"] == "recall this"


def test_oneshot_resume_preloads_session_history(monkeypatch):
    """hermes -z --resume must run on the resumed session's transcript."""
    from hermes_cli.oneshot import _run_agent

    captured = {}
    resolve_calls = []
    resumed_history = [
        {"role": "user", "content": "remember the session-only fact: cobalt"},
        {"role": "assistant", "content": "I will remember cobalt."},
    ]

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.session_id = kwargs.get("session_id")
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["run_kwargs"] = kwargs
            loader = kwargs.get("conversation_history_loader")
            if loader is not None:
                captured["loaded_history"] = loader()
            return {"final_response": "cobalt", "failed": False, "partial": False}

        def shutdown_memory_provider(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    class FakeSessionDB:
        def __init__(self):
            self.closed = False

        def assert_resume_safe(self, session_id, **kwargs):
            captured["assert_resume_safe"] = (session_id, kwargs)
            return 2

        def resolve_resume_session_id(self, session_id):
            resolve_calls.append(session_id)
            captured["resolve_resume_session_id"] = session_id
            return "resolved-session"

        def get_session(self, session_id):
            captured["get_session"] = session_id
            return {"id": session_id}

        def get_messages_as_conversation(self, session_id, **kwargs):
            captured["get_messages_as_conversation"] = (session_id, kwargs)
            return resumed_history

        def reopen_session(self, session_id):
            captured["reopen_session"] = session_id

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "run_agent", _mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(sys.modules, "hermes_state", _mod("hermes_state", SessionDB=FakeSessionDB))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _mod("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _mod("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
    )

    text, result = _run_agent("what was the fact?", resume="input-session")

    assert text == "cobalt"
    assert not result.get("failed")
    assert resolve_calls == ["input-session", "resolved-session"]
    assert captured["get_session"] == "resolved-session"
    assert captured["assert_resume_safe"] == ("resolved-session", {"tip_only": True})
    assert captured["get_messages_as_conversation"] == (
        "resolved-session",
        {"repair_alternation": True},
    )
    assert captured["reopen_session"] == "resolved-session"
    assert captured["session_id"] == "resolved-session"
    assert "conversation_history" not in captured["run_kwargs"]
    assert captured["loaded_history"] == resumed_history


def test_oneshot_resume_refuses_oversized_session_before_loading_history(
    monkeypatch,
):
    from hermes_cli.oneshot import _run_agent

    calls = []

    class FakeSessionDB:
        def resolve_resume_session_id(self, session_id):
            calls.append(("resolve", session_id))
            return session_id

        def get_session(self, session_id):
            calls.append(("get_session", session_id))
            return {"id": session_id}

        def reopen_session(self, session_id):
            calls.append(("reopen", session_id))

        def assert_resume_safe(self, session_id, **kwargs):
            calls.append(("assert_resume_safe", session_id, kwargs))
            from hermes_state import SessionResumeTooLargeError

            raise SessionResumeTooLargeError(3, 2, scope="in its tip segment")

        def get_resume_conversations(self, session_id):
            calls.append(("get_resume_conversations", session_id))
            raise AssertionError("must not materialize full lineage after guard rejection")

        def get_messages_as_conversation(self, session_id, **_kwargs):
            calls.append(("get_messages_as_conversation", session_id))
            raise AssertionError("must not load model history after guard rejection")

        def close(self):
            calls.append(("close", None))

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs.get("session_id")
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, _prompt, **kwargs):
            kwargs["conversation_history_loader"]()
            raise AssertionError("loader must reject before the turn runs")

        def shutdown_memory_provider(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "run_agent", _mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(sys.modules, "hermes_state", __import__("hermes_state"))
    monkeypatch.setattr("hermes_cli.oneshot._create_session_db_for_oneshot", FakeSessionDB)
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _mod("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
    )

    with pytest.raises(ValueError, match="safe resume limit is 2"):
        _run_agent("what was the fact?", resume="too-large")

    assert calls == [
        ("resolve", "too-large"),
        ("get_session", "too-large"),
        ("reopen", "too-large"),
        ("resolve", "too-large"),
        ("reopen", "too-large"),
        ("assert_resume_safe", "too-large", {"tip_only": True}),
        ("close", None),
    ]


def test_oneshot_resume_loads_history_from_session_db(monkeypatch, tmp_path):
    """Resumed one-shot turns must use the durable transcript, not a new session."""
    from hermes_cli.oneshot import _run_agent
    from hermes_state import SessionDB

    captured = {}
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session("resumable-session", source="cli")
    session_db.append_message(
        "resumable-session",
        "user",
        "remember the session-only fact: cobalt",
    )
    session_db.append_message(
        "resumable-session",
        "assistant",
        "I will remember cobalt.",
    )
    session_db.end_session("resumable-session", "agent_close")

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.session_id = kwargs.get("session_id")
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["run_kwargs"] = kwargs
            loader = kwargs.get("conversation_history_loader")
            if loader is not None:
                captured["loaded_history"] = loader()
            return {"final_response": "cobalt", "failed": False, "partial": False}

        def shutdown_memory_provider(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "hermes_cli.oneshot._create_session_db_for_oneshot",
        lambda: session_db,
    )
    monkeypatch.setitem(sys.modules, "run_agent", _mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _mod("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _mod("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
    )

    text, result = _run_agent("what was the fact?", resume="resumable-session")

    assert text == "cobalt"
    assert not result.get("failed")
    assert captured["session_id"] == "resumable-session"
    assert captured["prompt"] == "what was the fact?"
    assert [
        (msg["role"], msg["content"])
        for msg in captured["loaded_history"]
    ] == [
        ("user", "remember the session-only fact: cobalt"),
        ("assistant", "I will remember cobalt."),
    ]


def test_oneshot_resume_real_agent_appends_to_resumed_session_and_preserves_identity(
    monkeypatch,
    tmp_path,
    capsys,
):
    import logging

    from hermes_cli.oneshot import run_oneshot
    import hermes_cli.config as config_mod
    import hermes_cli.runtime_provider as runtime_provider_mod
    import hermes_cli.tools_config as tools_config_mod
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    seed_db = SessionDB(db_path=db_path)
    seed_db.create_session("resumable-session", source="cli")
    seed_db.append_message(
        "resumable-session",
        "user",
        "remember the session-only fact: cobalt",
    )
    seed_db.append_message(
        "resumable-session",
        "assistant",
        "I will remember cobalt.",
    )
    seed_db.end_session("resumable-session", "agent_close")
    seed_db.close()

    provider_calls = []

    class DeterministicClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            provider_calls.append(kwargs)
            messages = kwargs["messages"]
            saw_cobalt = any(
                isinstance(message.get("content"), str)
                and "cobalt" in message["content"]
                for message in messages
            )
            content = "resumed fact is cobalt" if saw_cobalt else "missing resumed fact"
            return _mock_response(content)

    monkeypatch.setattr(
        "hermes_cli.oneshot._create_session_db_for_oneshot",
        lambda: SessionDB(db_path=db_path),
    )
    monkeypatch.setattr(config_mod, "load_config", lambda: {"model": {"default": "m"}})
    monkeypatch.setattr(
        runtime_provider_mod,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "k",
            "base_url": "https://example.invalid/v1",
            "provider": "openai",
            "requested_provider": "openai",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )
    with (
        patch("agent.process_bootstrap.OpenAI", DeterministicClient),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        usage_file = tmp_path / "usage.json"
        try:
            rc = run_oneshot(
                "what was the fact?",
                usage_file=str(usage_file),
                resume="resumable-session",
            )
        finally:
            logging.disable(logging.NOTSET)

    assert rc == 0
    assert capsys.readouterr().out == "resumed fact is cobalt\n"
    report = json.loads(usage_file.read_text(encoding="utf-8"))
    assert report["session_id"] == "resumable-session"
    assert provider_calls
    assert provider_calls[-1]["messages"][-1]["content"] == "what was the fact?"

    verify_db = SessionDB(db_path=db_path)
    try:
        rows = verify_db.get_messages_as_conversation("resumable-session")
    finally:
        verify_db.close()
    assert [(row["role"], row["content"]) for row in rows] == [
        ("user", "remember the session-only fact: cobalt"),
        ("assistant", "I will remember cobalt."),
        ("user", "what was the fact?"),
        ("assistant", "resumed fact is cobalt"),
    ]


def test_oneshot_resume_loads_history_after_turn_lease(
    monkeypatch,
    tmp_path,
):
    import logging

    import hermes_cli.config as config_mod
    import hermes_cli.runtime_provider as runtime_provider_mod
    import hermes_cli.tools_config as tools_config_mod
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db_path = tmp_path / "state.db"
    seed_db = SessionDB(db_path=db_path)
    seed_db.create_session("shared-session", source="cli")
    seed_db.append_message("shared-session", "user", "first continuation")
    seed_db.append_message("shared-session", "assistant", "first ack")
    seed_db.end_session("shared-session", "agent_close")
    seed_db.close()

    provider_calls = []
    primary_db = SessionDB(db_path=db_path)
    sibling_appended = False

    class DeterministicClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            provider_calls.append(kwargs)
            messages = kwargs["messages"]
            saw_second = any(
                isinstance(message.get("content"), str)
                and "second continuation committed" in message["content"]
                for message in messages
            )
            return _mock_response(
                "fresh history included" if saw_second else "stale history"
            )

    monkeypatch.setattr(config_mod, "load_config", lambda: {"model": {"default": "m"}})
    monkeypatch.setattr(
        runtime_provider_mod,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "k",
            "base_url": "https://example.invalid/v1",
            "provider": "openai",
            "requested_provider": "openai",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )
    agent = AIAgent(
        api_key="k",
        base_url="https://example.invalid/v1",
        provider="openai",
        requested_provider="openai",
        api_mode="chat_completions",
        model="m",
        quiet_mode=True,
        platform="cli",
        session_id="shared-session",
        session_db=primary_db,
        skip_context_files=True,
        skip_memory=True,
    )

    def fresh_history_loader():
        nonlocal sibling_appended
        assert getattr(agent, "_active_session_turn_lease_holder", None)
        if not sibling_appended:
            sibling_appended = True
            sibling_db = SessionDB(db_path=db_path)
            try:
                sibling_db.reopen_session("shared-session")
                sibling_db.append_message(
                    "shared-session",
                    "user",
                    "second continuation committed",
                )
                sibling_db.append_message("shared-session", "assistant", "second ack")
                sibling_db.end_session("shared-session", "agent_close")
            finally:
                sibling_db.close()
        return primary_db.get_messages_as_conversation(
            "shared-session",
            repair_alternation=True,
        )

    with (
        patch("agent.process_bootstrap.OpenAI", DeterministicClient),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        try:
            result = agent.run_conversation(
                "third continuation",
                conversation_history_loader=fresh_history_loader,
            )
        finally:
            logging.disable(logging.NOTSET)
            agent.close()
            primary_db.close()

    assert result["final_response"] == "fresh history included"
    assert sibling_appended
    assert provider_calls


def test_oneshot_resume_retries_failed_transcript_flush_before_close(
    monkeypatch,
    tmp_path,
    capsys,
):
    import logging

    from hermes_cli.oneshot import run_oneshot
    import hermes_cli.config as config_mod
    import hermes_cli.runtime_provider as runtime_provider_mod
    import hermes_cli.tools_config as tools_config_mod
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    seed_db = SessionDB(db_path=db_path)
    seed_db.create_session("retry-session", source="cli")
    seed_db.append_message("retry-session", "user", "remember cobalt")
    seed_db.append_message("retry-session", "assistant", "stored cobalt")
    seed_db.end_session("retry-session", "agent_close")
    seed_db.close()

    class FlakySessionDB(SessionDB):
        failed_once = False

        def append_messages_batch(self, session_id, messages, **kwargs):
            if (
                session_id == "retry-session"
                and any(msg.get("content") == "what was stored?" for msg in messages)
                and not type(self).failed_once
            ):
                type(self).failed_once = True
                raise OSError("transient write failure")
            return super().append_messages_batch(session_id, messages, **kwargs)

    class DeterministicClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            return _mock_response("resumed fact is cobalt")

    monkeypatch.setattr(
        "hermes_cli.oneshot._create_session_db_for_oneshot",
        lambda: FlakySessionDB(db_path=db_path),
    )
    monkeypatch.setattr(config_mod, "load_config", lambda: {"model": {"default": "m"}})
    monkeypatch.setattr(
        runtime_provider_mod,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "k",
            "base_url": "https://example.invalid/v1",
            "provider": "openai",
            "requested_provider": "openai",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )

    with (
        patch("agent.process_bootstrap.OpenAI", DeterministicClient),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        try:
            rc = run_oneshot("what was stored?", resume="retry-session")
        finally:
            logging.disable(logging.NOTSET)

    assert rc == 0
    assert capsys.readouterr().out == "resumed fact is cobalt\n"
    verify_db = SessionDB(db_path=db_path)
    try:
        rows = verify_db.get_messages_as_conversation("retry-session")
    finally:
        verify_db.close()
    assert [(row["role"], row["content"]) for row in rows][-2:] == [
        ("user", "what was stored?"),
        ("assistant", "resumed fact is cobalt"),
    ]


def test_oneshot_resume_allows_compressed_lineage_when_tip_is_within_limit(
    monkeypatch,
    tmp_path,
    capsys,
):
    import logging

    from hermes_cli.oneshot import run_oneshot
    import hermes_cli.config as config_mod
    import hermes_cli.runtime_provider as runtime_provider_mod
    import hermes_cli.tools_config as tools_config_mod
    import hermes_state
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    seed_db = SessionDB(db_path=db_path)
    seed_db.create_session("compressed-root", source="cli")
    seed_db.append_messages_batch(
        "compressed-root",
        [{"role": "user", "content": f"archived-{idx}"} for idx in range(4)],
    )
    seed_db.end_session("compressed-root", "compression")
    seed_db.create_session("live-tip", source="cli", parent_session_id="compressed-root")
    seed_db.append_message("live-tip", "user", "remember live-tip cobalt")
    seed_db.append_message("live-tip", "assistant", "tip fact stored")
    seed_db.end_session("live-tip", "agent_close")
    seed_db.close()

    class DeterministicClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            messages = kwargs["messages"]
            saw_tip = any(
                isinstance(message.get("content"), str)
                and "live-tip cobalt" in message["content"]
                for message in messages
            )
            saw_archive = any(
                isinstance(message.get("content"), str)
                and "archived-" in message["content"]
                for message in messages
            )
            return _mock_response(
                "tip-only resume worked" if saw_tip and not saw_archive else "wrong scope"
            )

    monkeypatch.setattr(hermes_state, "resolved_max_resume_messages", lambda: 3)
    monkeypatch.setattr(
        "hermes_cli.oneshot._create_session_db_for_oneshot",
        lambda: SessionDB(db_path=db_path),
    )
    monkeypatch.setattr(config_mod, "load_config", lambda: {"model": {"default": "m"}})
    monkeypatch.setattr(
        runtime_provider_mod,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "k",
            "base_url": "https://example.invalid/v1",
            "provider": "openai",
            "requested_provider": "openai",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )

    with (
        patch("agent.process_bootstrap.OpenAI", DeterministicClient),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        try:
            rc = run_oneshot("what does the live tip remember?", resume="live-tip")
        finally:
            logging.disable(logging.NOTSET)

    assert rc == 0
    assert capsys.readouterr().out == "tip-only resume worked\n"
    verify_db = SessionDB(db_path=db_path)
    try:
        assert verify_db.get_resume_message_count("live-tip") > 3
        rows = verify_db.get_messages_as_conversation("live-tip")
    finally:
        verify_db.close()
    assert [(row["role"], row["content"]) for row in rows][-2:] == [
        ("user", "what does the live tip remember?"),
        ("assistant", "tip-only resume worked"),
    ]


def test_oneshot_resume_rejects_compressed_tip_when_tip_exceeds_limit(
    monkeypatch,
    tmp_path,
):
    import logging

    from hermes_cli.oneshot import _run_agent
    import hermes_cli.config as config_mod
    import hermes_cli.runtime_provider as runtime_provider_mod
    import hermes_cli.tools_config as tools_config_mod
    import hermes_state
    from hermes_state import SessionDB, SessionResumeTooLargeError

    db_path = tmp_path / "state.db"
    seed_db = SessionDB(db_path=db_path)
    seed_db.create_session("compressed-root", source="cli")
    seed_db.append_message("compressed-root", "user", "archived")
    seed_db.end_session("compressed-root", "compression")
    seed_db.create_session("large-tip", source="cli", parent_session_id="compressed-root")
    seed_db.append_messages_batch(
        "large-tip",
        [{"role": "user", "content": f"tip-{idx}"} for idx in range(4)],
    )
    seed_db.end_session("large-tip", "agent_close")
    seed_db.close()

    class MustNotCallClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **_kwargs):
            raise AssertionError("oversized tip must fail before provider call")

    monkeypatch.setattr(hermes_state, "resolved_max_resume_messages", lambda: 3)
    monkeypatch.setattr(
        "hermes_cli.oneshot._create_session_db_for_oneshot",
        lambda: SessionDB(db_path=db_path),
    )
    monkeypatch.setattr(config_mod, "load_config", lambda: {"model": {"default": "m"}})
    monkeypatch.setattr(
        runtime_provider_mod,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "k",
            "base_url": "https://example.invalid/v1",
            "provider": "openai",
            "requested_provider": "openai",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )

    with (
        patch("agent.process_bootstrap.OpenAI", MustNotCallClient),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        pytest.raises(SessionResumeTooLargeError, match="in its tip segment"),
    ):
        try:
            _run_agent("should fail", resume="large-tip")
        finally:
            logging.disable(logging.NOTSET)


def test_oneshot_without_resume_real_agent_does_not_see_prior_session(
    monkeypatch,
    tmp_path,
):
    from hermes_cli.oneshot import _run_agent
    import hermes_cli.config as config_mod
    import hermes_cli.runtime_provider as runtime_provider_mod
    import hermes_cli.tools_config as tools_config_mod
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    seed_db = SessionDB(db_path=db_path)
    seed_db.create_session("isolated-session", source="cli")
    seed_db.append_message("isolated-session", "user", "remember cobalt")
    seed_db.append_message("isolated-session", "assistant", "stored cobalt")
    seed_db.end_session("isolated-session", "agent_close")
    seed_db.close()

    class DeterministicClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            messages = kwargs["messages"]
            saw_cobalt = any(
                isinstance(message.get("content"), str)
                and "cobalt" in message["content"]
                for message in messages
            )
            return _mock_response(
                "unexpected prior fact" if saw_cobalt else "no prior fact"
            )

    monkeypatch.setattr(
        "hermes_cli.oneshot._create_session_db_for_oneshot",
        lambda: SessionDB(db_path=db_path),
    )
    monkeypatch.setattr(config_mod, "load_config", lambda: {"model": {"default": "m"}})
    monkeypatch.setattr(
        runtime_provider_mod,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "k",
            "base_url": "https://example.invalid/v1",
            "provider": "openai",
            "requested_provider": "openai",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )
    with (
        patch("agent.process_bootstrap.OpenAI", DeterministicClient),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        text, result = _run_agent("what was the fact?")

    assert text == "no prior fact"
    assert result["session_id"] != "isolated-session"


def test_oneshot_resume_rejects_unknown_session(monkeypatch, tmp_path):
    from hermes_cli.oneshot import _run_agent
    from hermes_state import SessionDB

    session_db = SessionDB(db_path=tmp_path / "state.db")

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setattr(
        "hermes_cli.oneshot._create_session_db_for_oneshot",
        lambda: session_db,
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        mod("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        mod("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
    )
    monkeypatch.setitem(
        sys.modules,
        "run_agent",
        mod("run_agent", AIAgent=lambda **_kwargs: pytest.fail("must not build agent")),
    )

    with pytest.raises(ValueError, match="Session not found: missing-session"):
        _run_agent("what was the fact?", resume="missing-session")


def test_launch_tui_exports_model_provider_and_toolsets(monkeypatch, main_mod):
    captured = {}
    active_path_during_call = None

    monkeypatch.setattr(main_tui_launch, "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")),
    )

    def fake_call(argv, cwd=None, env=None):
        nonlocal active_path_during_call
        captured.update({"argv": argv, "cwd": cwd, "env": env})
        active_path_during_call = Path(env["HERMES_TUI_ACTIVE_SESSION_FILE"])
        assert active_path_during_call.exists()
        return 1

    monkeypatch.setattr(main_mod.subprocess, "call", fake_call)

    with pytest.raises(SystemExit):
        main_mod._launch_tui(
            model="nous/hermes-test", provider="nous", toolsets="web, terminal"
        )

    env = captured["env"]
    assert env["HERMES_MODEL"] == "nous/hermes-test"
    assert env["HERMES_INFERENCE_MODEL"] == "nous/hermes-test"
    assert env["HERMES_TUI_PROVIDER"] == "nous"
    assert env["HERMES_INFERENCE_PROVIDER"] == "nous"
    assert env["HERMES_TUI_TOOLSETS"] == "web,terminal"
    active_path = Path(env["HERMES_TUI_ACTIVE_SESSION_FILE"])
    assert active_path.name.startswith("hermes-tui-active-session-")
    assert active_path.suffix == ".json"
    assert active_path_during_call == active_path
    assert not active_path.exists()
    assert env["NODE_ENV"] == "production"




def test_make_tui_argv_dev_prebuilds_hermes_ink(monkeypatch, main_mod, tmp_path):
    tui_dir = tmp_path / "ui-tui"
    tsx = tui_dir / "node_modules" / ".bin" / "tsx"
    ink_dir = tui_dir / "packages" / "hermes-ink"
    tsx.parent.mkdir(parents=True)
    ink_dir.mkdir(parents=True)
    tsx.write_text("#!/usr/bin/env node\n", encoding="utf-8")

    monkeypatch.setattr(main_tui_launch, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_tui_launch, "_tui_need_npm_install", lambda _tui_dir: False)
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_mod.shutil, "which", lambda bin_name: f"/usr/bin/{bin_name}")

    calls = []

    def fake_run(cmd, cwd=None, **_kwargs):
        calls.append((cmd, cwd))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    argv, cwd = main_tui_launch._make_tui_argv(tui_dir, tui_dev=True)

    assert argv == [str(tsx), "src/entry.tsx"]
    assert cwd == tui_dir
    assert calls == [(["/usr/bin/npm", "run", "build"], str(ink_dir))]
