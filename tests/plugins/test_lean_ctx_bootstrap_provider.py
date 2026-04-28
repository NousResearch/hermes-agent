from pathlib import Path

from agent.context_bootstrap import _lean_ctx_config_present
from plugins.context_bootstrap.lean_ctx import LeanCtxBootstrapProvider, LeanCtxConfig


def test_auto_config_presence_follows_binary_availability(monkeypatch):
    cfg = {"lean_ctx": {"enabled": "auto", "command": "lean-ctx"}}

    monkeypatch.setattr("agent.context_bootstrap.shutil.which", lambda command: None)
    assert _lean_ctx_config_present(cfg) is False

    monkeypatch.setattr("agent.context_bootstrap.shutil.which", lambda command: "/usr/local/bin/lean-ctx")
    assert _lean_ctx_config_present(cfg) is True


def test_provider_builds_first_turn_packet_with_overview_and_symbols(tmp_path):
    calls = []

    def fake_call(tool_name, args, root, timeout_seconds):
        calls.append((tool_name, args, root))
        return f"{tool_name} result"

    provider = LeanCtxBootstrapProvider(
        LeanCtxConfig(command="lean-ctx", max_chars=6000, timeout_seconds=1),
        call_tool=fake_call,
    )

    context = provider.context_for_turn(
        session_id="s1",
        user_message="Review function `dispatchTask` in this repo",
        is_first_turn=True,
        workspace_root=tmp_path,
        conversation_history=[],
    )

    assert "LEAN-CTX BOOTSTRAP CONTEXT" in context
    assert "ctx_overview result" in context
    assert "ctx_symbol:dispatchTask" in context
    assert [call[0] for call in calls] == [
        "ctx_overview",
        "ctx_preload",
        "ctx_handoff",
        "ctx_symbol",
        "ctx_callers",
    ]
    assert all(call[2] == Path(tmp_path).resolve() for call in calls)


def test_provider_builds_delegation_packet_with_separate_budget(tmp_path):
    calls = []

    def fake_call(tool_name, args, root, timeout_seconds):
        calls.append((tool_name, args, root))
        return f"{tool_name} result"

    provider = LeanCtxBootstrapProvider(
        LeanCtxConfig(
            command="lean-ctx",
            delegation_max_chars=800,
            max_chars=6000,
            timeout_seconds=1,
        ),
        call_tool=fake_call,
    )

    context = provider.context_for_delegation(
        goal="Review function `dispatchTask`",
        context="Focus on persona routing.",
        workspace_root=tmp_path,
    )

    assert "LEAN-CTX DELEGATION CONTEXT" in context
    assert "Focus on persona routing" in context
    assert "ctx_symbol:dispatchTask" in context
    assert len(context) <= 800
    assert [call[0] for call in calls] == [
        "ctx_overview",
        "ctx_preload",
        "ctx_handoff",
        "ctx_symbol",
        "ctx_callers",
    ]


def test_provider_bootstraps_session_once(tmp_path):
    provider = LeanCtxBootstrapProvider(
        LeanCtxConfig(command="lean-ctx"),
        call_tool=lambda tool_name, args, root, timeout_seconds: f"{tool_name} result",
    )

    first = provider.context_for_turn(
        session_id="s1",
        user_message="Inspect code",
        is_first_turn=True,
        workspace_root=tmp_path,
    )
    second = provider.context_for_turn(
        session_id="s1",
        user_message="Inspect code again",
        is_first_turn=True,
        workspace_root=tmp_path,
    )

    assert first
    assert second == ""
