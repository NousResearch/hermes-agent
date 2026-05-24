"""Tests for the lightweight attribution/action ledger."""

import asyncio
from types import SimpleNamespace

from agent.attribution_ledger import (
    classify_side_effect,
    format_attribution_report,
    summarize_tool_call,
)
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner, _is_what_did_you_do_trigger
from gateway.session import SessionSource
from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS, resolve_command
from hermes_state import SessionDB


class TestAttributionLedgerFormatting:
    def test_summarize_tool_call_redacts_secret_arguments(self):
        token = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"

        summary = summarize_tool_call(
            "terminal",
            {"command": f"curl -H 'Authorization: Bearer {token}' https://example.test"},
        )

        assert token not in summary
        assert "terminal" in summary
        assert "curl" in summary

    def test_classify_side_effect_for_core_tools(self):
        assert classify_side_effect("read_file", {}) == "read"
        assert classify_side_effect("patch", {}) == "write"
        assert classify_side_effect("send_message", {}) == "message"
        assert classify_side_effect("terminal", {"command": "git status"}) == "git"
        assert classify_side_effect("terminal", {"command": "python build.py"}) == "process"

    def test_format_report_keeps_empty_buckets_visible(self):
        report = format_attribution_report(
            [], session_id="s1", lineage_id="s1", chat_id="chat-1"
        )

        assert "无可确认项" in report
        assert "未发现可归因证据" in report
        assert "git history" in report

    def test_format_report_groups_completed_failed_and_started(self):
        report = format_attribution_report(
            [
                {
                    "id": 1,
                    "session_id": "s1",
                    "tool_name": "read_file",
                    "status": "completed",
                    "action_summary": "read README.md",
                    "side_effect_class": "read",
                    "started_at": 100.0,
                },
                {
                    "id": 2,
                    "session_id": "s1",
                    "tool_name": "patch",
                    "status": "failed",
                    "action_summary": "patch SKILL.md",
                    "side_effect_class": "write",
                    "started_at": 101.0,
                    "error_preview": "old_string not found",
                },
                {
                    "id": 3,
                    "session_id": "s1",
                    "tool_name": "terminal",
                    "status": "started",
                    "action_summary": "terminal: sleep 999",
                    "side_effect_class": "process",
                    "started_at": 102.0,
                },
            ],
            session_id="s1",
            lineage_id="s1",
            chat_id="chat-1",
        )

        assert "已确认完成" in report
        assert "失败/被阻止" in report
        assert "仅看到开始" in report
        assert "event#1" in report
        assert "event#2" in report
        assert "event#3" in report


class TestGatewayWhatDidYouDoCommand:
    def test_chinese_plain_text_triggers_command(self):
        assert _is_what_did_you_do_trigger("你做了什么")
        assert _is_what_did_you_do_trigger("刚才你干嘛了？")
        assert _is_what_did_you_do_trigger("你到底改了啥")
        assert _is_what_did_you_do_trigger("哪些是你做的")
        assert not _is_what_did_you_do_trigger("你接着做吧")

    def test_command_aliases_resolve_and_bypass_running_agent(self):
        assert resolve_command("whatdidyoudo").name == "what-did-you-do"
        assert resolve_command("wdyd").name == "what-did-you-do"
        assert "what-did-you-do" in ACTIVE_SESSION_BYPASS_COMMANDS

    def test_gateway_handler_reports_current_lineage_events(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session(session_id="parent", source="qqbot")
            db.create_session(session_id="child", source="qqbot", parent_session_id="parent")
            db.append_attribution_event(
                session_id="parent",
                tool_name="read_file",
                status="completed",
                action_summary="read AGENTS.md",
                side_effect_class="read",
                source="qqbot",
                chat_id="chat-1",
                platform_message_id="msg-1",
            )
            db.append_attribution_event(
                session_id="child",
                tool_name="patch",
                status="completed",
                action_summary="patch what-did-you-do skill",
                side_effect_class="write",
                source="qqbot",
                chat_id="chat-1",
                platform_message_id="msg-2",
            )

            source = SessionSource(
                platform=Platform.QQBOT,
                chat_id="chat-1",
                chat_type="dm",
                user_id="user-1",
            )
            event = MessageEvent(text="/what-did-you-do", source=source, message_id="msg-3")

            runner = object.__new__(GatewayRunner)
            runner._session_db = db
            runner.session_store = SimpleNamespace(
                _entries={"qqbot:dm:chat-1": SimpleNamespace(session_id="child")},
                _generate_session_key=lambda src: "qqbot:dm:chat-1",
            )

            response = asyncio.run(runner._handle_what_did_you_do_command(event))

            assert "当前 session: child" in response
            assert "lineage: parent" in response
            assert "read AGENTS.md" in response
            assert "patch what-did-you-do skill" in response
            assert "event#" in response
        finally:
            db.close()
