"""CLI completion sidecars survive turn staging and real SQLite persistence."""

import types

import pytest

from cli import HermesCLI
from tools.process_registry_notifications import SubagentNotification, async_delegation_display_text


@pytest.mark.parametrize("trusted", [True, False])
def test_cli_async_completion_is_typed_before_crash_persist(
    tmp_path, monkeypatch, trusted
):
    from agent.turn_context import build_turn_context
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.setattr("cli.get_tool_definitions", lambda **kw: [])
    cli = HermesCLI()

    text = "[ASYNC DELEGATION BATCH COMPLETE — deleg_cli]\nMeaningful results"
    event = {
        "delegation_id": "deleg_cli",
        "results": [{"status": "completed"}, {"status": "failed"}],
    }
    metadata = {
        "delegation_id": "deleg_cli", "task_count": 2,
        "completed_count": 1, "failed_count": 1,
        "display_text": async_delegation_display_text(event),
    }
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session(session_id=cli.session_id, source="cli")
        db.set_session_title(cli.session_id, "Completion persistence test")
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
            session_db=db,
            session_id=cli.session_id,
        )
        agent._session_db_created = True
        agent._cached_system_prompt = "SYSTEM"
        agent._skip_mcp_refresh = True
        cli.agent = agent
        captured = {}

        def run_conversation(**kwargs):
            captured["staged"] = dict(agent._pending_cli_user_message)
            captured["kwargs"] = kwargs
            context = build_turn_context(
                agent=agent,
                user_message=kwargs["user_message"],
                system_message=None,
                conversation_history=kwargs["conversation_history"],
                task_id=kwargs["task_id"],
                stream_callback=None,
                persist_user_message=kwargs.get("persist_user_message"),
                persist_user_display_kind=kwargs.get("persist_user_display_kind"),
                persist_user_display_metadata=kwargs.get(
                    "persist_user_display_metadata"
                ),
                restore_or_build_system_prompt=lambda *a, **k: None,
                install_safe_stdio=lambda: None,
                sanitize_surrogates=lambda s: s,
                summarize_user_message_for_log=lambda s: s,
                set_session_context=lambda sid: None,
                set_current_write_origin=lambda o: None,
                ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
            )
            return {
                "final_response": "done",
                "messages": context.messages,
                "api_calls": 1,
                "completed": True,
                "partial": True,
                "response_previewed": True,
            }

        monkeypatch.setattr(agent, "run_conversation", run_conversation)
        monkeypatch.setattr(cli, "_ensure_runtime_credentials", lambda: True)
        monkeypatch.setattr(cli, "_init_agent", lambda **kw: True)
        monkeypatch.setattr(
            cli,
            "_resolve_turn_agent_config",
            lambda message: {
                "signature": cli._active_agent_route_signature,
                "model": None,
                "runtime": None,
            },
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.set_runtime_main", lambda *a, **k: None
        )
        cli.chat(SubagentNotification(text, event) if trusted else text)

        (row,) = db.get_messages_as_conversation(cli.session_id)
        expected_kind = "async_delegation_complete" if trusted else None
        assert captured["staged"].get("display_kind") == expected_kind
        assert captured["kwargs"].get("persist_user_display_kind") == expected_kind
        assert row.get("display_kind") == expected_kind
        assert row.get("display_metadata") == (metadata if trusted else None)
        if trusted:
            assert captured["staged"]["display_metadata"] == metadata
        assert row["content"] == text
        assert captured["kwargs"]["user_message"] == text
