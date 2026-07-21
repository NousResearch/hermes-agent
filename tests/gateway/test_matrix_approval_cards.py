"""Tests for Matrix approval card formatting and compact lifecycle."""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.matrix.approval_cards import (
    force_redact_command,
    format_pending_expanded,
    format_pending_summarized,
    format_terminal_compact,
    load_matrix_approval_summary_config,
    sanitize_summary,
)


class TestApprovalCardFormatting:
    def test_pending_expanded_includes_full_command_and_reason(self):
        text, html = format_pending_expanded(
            command="rm -rf /tmp/example",
            description="recursive delete",
            allow_permanent=True,
        )
        # Stable producer contract used by Matrix clients to render rich
        # approval controls (including the two-step permanent warning).
        assert "Dangerous command requires approval" in text
        assert "recursive delete" in text
        assert "rm -rf /tmp/example" in text
        assert "✅ once" in text
        assert "🌀 session" in text
        assert "♾️ always" in text
        assert "❌ deny" in text
        assert "❎" not in text
        assert html is not None
        assert "Dangerous command requires approval" in html
        assert "<pre>" in html
        assert "rm -rf /tmp/example" in html

    def test_pending_cards_respect_session_approval_scope(self):
        expanded_text, expanded_html = format_pending_expanded(
            command="rm -rf /tmp/example",
            description="recursive delete",
            allow_permanent=True,
            allow_session=False,
        )
        summarized_text, summarized_html = format_pending_summarized(
            command="rm -rf /tmp/example",
            description="recursive delete",
            summary="Deletes the bounded test directory.",
            allow_permanent=True,
            allow_session=False,
        )

        for text in (expanded_text, expanded_html, summarized_text, summarized_html):
            assert text is not None
            assert "🌀" not in text
            assert "session" not in text.lower()
            assert "♾️" not in text
            assert "always" not in text.lower()

    def test_pending_summarized_keeps_self_contained_plaintext_fallback(self):
        text, html = format_pending_summarized(
            command="git reset --hard HEAD~1",
            description="git reset --hard (destroys uncommitted changes)",
            summary="Discards recent local commits and uncommitted work.",
        )
        assert "Advisory interpretation" in text
        assert "Dangerous command requires approval" in text
        # Plaintext clients cannot rely on formatted HTML or the original event.
        assert "git reset --hard HEAD~1" in text
        assert html is not None
        assert "Dangerous command requires approval" in html
        assert "<details>" in html
        assert html.index("Advisory interpretation") < html.index("<details>")
        assert "<summary>Full command</summary>" in html
        assert "git reset --hard HEAD~1" in html
        assert "Discards recent local commits" in html

    def test_terminal_compact_keeps_advisory_primary_and_command_in_details(self):
        text, html = format_terminal_compact(
            choice="once",
            command="systemctl restart example.service",
            description="stop/restart system service",
            actor="@user:example.org",
            summary="Restarts the example service and briefly interrupts it.",
        )
        assert "Approved once" in text
        assert "Advisory interpretation" in text
        assert "Restarts the example service" in text
        # Plaintext remains audit-complete even though formatted HTML collapses it.
        assert "systemctl restart example.service" in text
        assert html is not None
        assert "<details>" in html
        assert html.index("Advisory interpretation") < html.index("<details>")
        assert "<summary>Full command</summary>" in html
        assert "systemctl restart example.service" in html
        assert "@user:example.org" in html

    def test_sanitize_summary_strips_html_and_bounds_length(self):
        dirty = "<script>alert(1)</script> ```rm -rf /``` does a thing " + ("x" * 600)
        clean = sanitize_summary(dirty, max_chars=80)
        assert "<script>" not in clean
        assert "```" not in clean
        assert len(clean) <= 80

    def test_load_summary_config_defaults_disabled(self):
        cfg = load_matrix_approval_summary_config({})
        assert cfg.enabled is False
        assert cfg.local_timeout_seconds == 90

    def test_load_summary_config_caps_local_timeout_at_90(self):
        cfg = load_matrix_approval_summary_config(
            {
                "matrix": {
                    "approvals": {
                        "llm_summary": {
                            "enabled": True,
                            "provider_policy": "local_only",
                            "local_timeout_seconds": 180,
                        }
                    }
                }
            }
        )
        assert cfg.enabled is True
        assert cfg.local_timeout_seconds == 90

    def test_force_redact_known_prefix(self):
        out = force_redact_command("export OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz")
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz" not in out


class TestMatrixApprovalCardLifecycle:
    @pytest.mark.asyncio
    async def test_approval_html_edit_keeps_matrix_replacement_prefix(self):
        """Approval edits retain Matrix's outer fallback marker."""
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        sent = {}

        async def _send_event(room, event_type, content):
            sent["content"] = content
            return "$replacement"

        adapter._client = types.SimpleNamespace(send_message_event=_send_event)
        text, html = format_pending_summarized(
            command="systemctl restart example.service",
            description="stop/restart system service",
            summary="Restarts the example service.",
        )

        result = await adapter.edit_message(
            "!room:example.org",
            "$approval",
            text,
            metadata={"matrix_formatted_body": html},
        )

        assert result.success is True
        content = sent["content"]
        assert content["body"].startswith("* ")
        assert content["formatted_body"].startswith("* ")
        assert content["m.relates_to"] == {
            "rel_type": "m.replace",
            "event_id": "$approval",
        }
        new_content = content["m.new_content"]
        assert not new_content["body"].startswith("* ")
        assert not new_content["formatted_body"].startswith("* ")
        for required in (
            "Dangerous command requires approval",
            "stop/restart system service",
            "Advisory interpretation",
            "Restarts the example service",
            "systemctl restart example.service",
            "✅ once",
            "❌ deny",
        ):
            assert required in new_content["body"]
        assert new_content["formatted_body"].index(
            "Advisory interpretation"
        ) < new_content["formatted_body"].index("<details>")
        assert "<summary>Full command</summary>" in new_content["formatted_body"]
        assert "systemctl restart example.service" in new_content["formatted_body"]

    @pytest.mark.asyncio
    async def test_send_exec_approval_uses_expanded_card_and_seeds_reactions(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        adapter._client = types.SimpleNamespace()
        adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$evt1"))
        adapter._send_reaction = AsyncMock(return_value="$r")

        with patch(
            "plugins.platforms.matrix.approval_cards.load_matrix_approval_summary_config",
            return_value=types.SimpleNamespace(enabled=False),
        ):
            result = await adapter.send_exec_approval(
                chat_id="!room:example.org",
                command="rm -rf /tmp/test",
                session_key="sess-1",
                description="recursive delete",
            )

        assert result.success is True
        body = adapter.send.await_args.args[1]
        assert "Dangerous command requires approval" in body
        assert "recursive delete" in body
        assert "rm -rf /tmp/test" in body
        assert "❎" not in body
        meta = adapter.send.await_args.kwargs.get("metadata") or {}
        assert "matrix_formatted_body" in meta
        prompt = adapter._approval_prompts_by_event["$evt1"]
        assert prompt.command == "rm -rf /tmp/test"
        assert prompt.allow_session is True
        assert prompt.state == "pending_expanded"
        emojis = [call.args[2] for call in adapter._send_reaction.await_args_list]
        assert emojis == ["✅", "🌀", "♾️", "❌"]

    @pytest.mark.asyncio
    async def test_send_exec_approval_without_session_scope_seeds_once_deny_only(
        self, monkeypatch
    ):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        adapter._client = types.SimpleNamespace()
        adapter.send = AsyncMock(return_value=types.SimpleNamespace(success=True, message_id="$evt2"))
        adapter._send_reaction = AsyncMock(return_value="$r")

        with patch(
            "plugins.platforms.matrix.approval_cards.load_matrix_approval_summary_config",
            return_value=types.SimpleNamespace(enabled=False),
        ):
            result = await adapter.send_exec_approval(
                chat_id="!room:example.org",
                command="rm -rf /tmp/test",
                session_key="sess-2",
                description="recursive delete",
                allow_permanent=True,
                allow_session=False,
            )

        assert result.success is True
        body = adapter.send.await_args.args[1]
        assert "🌀" not in body
        assert "♾️" not in body
        prompt = adapter._approval_prompts_by_event["$evt2"]
        assert prompt.allow_session is False
        emojis = [call.args[2] for call in adapter._send_reaction.await_args_list]
        assert emojis == ["✅", "❌"]

    @pytest.mark.asyncio
    async def test_concurrent_same_session_prompts_keep_distinct_identity(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        adapter._client = types.SimpleNamespace()
        adapter.send = AsyncMock(
            side_effect=[
                types.SimpleNamespace(success=True, message_id="$evt1"),
                types.SimpleNamespace(success=True, message_id="$evt2"),
            ]
        )
        adapter._send_reaction = AsyncMock(return_value="$r")
        adapter._schedule_approval_resolution_watch = MagicMock()

        with patch(
            "plugins.platforms.matrix.approval_cards.load_matrix_approval_summary_config",
            return_value=types.SimpleNamespace(enabled=False),
        ):
            await adapter.send_exec_approval(
                chat_id="!room:example.org",
                command="rm -rf /tmp/first",
                session_key="sess-1",
                metadata={"approval_id": "approval-1"},
            )
            await adapter.send_exec_approval(
                chat_id="!room:example.org",
                command="rm -rf /tmp/second",
                session_key="sess-1",
                metadata={"approval_id": "approval-2"},
            )

        assert set(adapter._approval_prompts_by_event) == {"$evt1", "$evt2"}
        assert adapter._approval_prompt_by_session["sess-1"] == {"$evt1", "$evt2"}
        assert adapter._approval_prompts_by_event["$evt1"].approval_id == "approval-1"
        assert adapter._approval_prompts_by_event["$evt2"].approval_id == "approval-2"

    @pytest.mark.asyncio
    async def test_typed_fifo_resolution_finalizes_only_oldest_real_prompt(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter
        from tools import approval as approval_mod

        session_key = "sess-real-fifo"
        first = approval_mod._ApprovalEntry({"command": "rm -rf /tmp/first"})
        second = approval_mod._ApprovalEntry({"command": "rm -rf /tmp/second"})
        with approval_mod._lock:
            approval_mod._gateway_queues[session_key] = [first, second]

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        adapter._client = types.SimpleNamespace()
        adapter.send = AsyncMock(
            side_effect=[
                types.SimpleNamespace(success=True, message_id="$evt1"),
                types.SimpleNamespace(success=True, message_id="$evt2"),
            ]
        )
        adapter._send_reaction = AsyncMock(return_value="$r")
        adapter._redact_bot_approval_reactions = AsyncMock()
        adapter._finalize_matrix_approval_prompt = AsyncMock()

        try:
            with patch(
                "plugins.platforms.matrix.approval_cards.load_matrix_approval_summary_config",
                return_value=types.SimpleNamespace(enabled=False),
            ):
                await adapter.send_exec_approval(
                    chat_id="!room:example.org",
                    command=first.data["command"],
                    session_key=session_key,
                    metadata={"approval_id": first.approval_id},
                )
                await adapter.send_exec_approval(
                    chat_id="!room:example.org",
                    command=second.data["command"],
                    session_key=session_key,
                    metadata={"approval_id": second.approval_id},
                )

            prompt1 = adapter._approval_prompts_by_event["$evt1"]
            prompt2 = adapter._approval_prompts_by_event["$evt2"]

            assert approval_mod.resolve_gateway_approval(session_key, "once") == 1
            await asyncio.sleep(0.6)

            assert first.event.is_set() is True
            assert second.event.is_set() is False
            assert prompt1.resolved is True
            assert prompt2.resolved is False
            assert "$evt1" not in adapter._approval_prompts_by_event
            assert adapter._approval_prompt_by_session[session_key] == {"$evt2"}
            assert approval_mod.has_blocking_approval(
                session_key,
                approval_id=second.approval_id,
            ) is True
            adapter._finalize_matrix_approval_prompt.assert_awaited_once_with(
                "!room:example.org",
                "$evt1",
                prompt1,
                choice="resolved",
                actor="",
            )
        finally:
            prompt2 = adapter._approval_prompts_by_event.get("$evt2")
            if prompt2 is not None:
                prompt2.resolved = True
                adapter._forget_matrix_approval_prompt("$evt2", prompt2)
            approval_mod.clear_session(session_key)
            await asyncio.sleep(0.6)

    @pytest.mark.asyncio
    async def test_reaction_resolve_edits_terminal_card(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        adapter._user_id = "@bot:example.org"
        prompt = _MatrixApprovalPrompt(
            session_key="sess-1",
            chat_id="!room:example.org",
            message_id="$target",
            approval_id="approval-2",
            command="rm -rf /tmp/x",
            description="recursive delete",
        )
        prompt.summary = "Deletes the bounded directory and its contents."
        prompt.state = "pending_summarized"
        adapter._approval_prompts_by_event["$target"] = prompt
        adapter._approval_prompt_by_session["sess-1"] = {"$target"}
        adapter.edit_message = AsyncMock(return_value=types.SimpleNamespace(success=True))
        adapter._redact_bot_approval_reactions = AsyncMock()

        content = {"m.relates_to": {"event_id": "$target", "key": "✅"}}
        event = types.SimpleNamespace(
            sender="@user:example.org",
            event_id="$react1",
            room_id="!room:example.org",
            content=content,
        )

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
            await adapter._on_reaction(event)

        resolve.assert_called_once_with(
            "sess-1",
            "once",
            approval_id="approval-2",
        )
        assert adapter.edit_message.await_count == 1
        edited = adapter.edit_message.await_args.args[2]
        assert "Approved once" in edited
        assert "Deletes the bounded directory" in edited
        assert "rm -rf /tmp/x" in edited
        metadata = adapter.edit_message.await_args.kwargs.get("metadata") or {}
        edited_html = metadata.get("matrix_formatted_body") or ""
        assert edited_html.index("Advisory interpretation") < edited_html.index("<details>")
        assert "<summary>Full command</summary>" in edited_html
        assert "rm -rf /tmp/x" in edited_html

    @pytest.mark.asyncio
    async def test_resolution_watch_tracks_one_approval_id(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        prompt = _MatrixApprovalPrompt(
            session_key="sess-1",
            chat_id="!room:example.org",
            message_id="$target",
            approval_id="approval-2",
            command="rm -rf /tmp/x",
        )
        adapter._approval_prompts_by_event[prompt.message_id] = prompt
        adapter._approval_prompt_by_session[prompt.session_key] = {prompt.message_id}
        adapter._redact_bot_approval_reactions = AsyncMock()
        adapter._finalize_matrix_approval_prompt = AsyncMock()

        with patch("tools.approval.has_blocking_approval", return_value=False) as pending:
            adapter._schedule_approval_resolution_watch(prompt)
            for _ in range(10):
                if prompt.resolved:
                    break
                await asyncio.sleep(0)

        pending.assert_called_once_with("sess-1", approval_id="approval-2")
        assert prompt.resolved is True
        assert "$target" not in adapter._approval_prompts_by_event
        adapter._finalize_matrix_approval_prompt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_summary_edit_skipped_when_already_resolved(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt
        from plugins.platforms.matrix.approval_cards import MatrixApprovalSummaryConfig

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        prompt = _MatrixApprovalPrompt(
            session_key="sess-1",
            chat_id="!room:example.org",
            message_id="$target",
            command="echo hi",
            description="script execution via -c flag",
        )
        prompt.resolved = True
        adapter.edit_message = AsyncMock()

        with patch(
            "plugins.platforms.matrix.approval_cards.generate_command_summary",
            return_value="Says hello.",
        ):
            cfg = MatrixApprovalSummaryConfig(enabled=True, provider_policy="local_only")
            adapter._schedule_approval_summary(prompt, cfg)
            task = prompt.summary_task
            assert isinstance(task, asyncio.Task)
            await task

        adapter.edit_message.assert_not_awaited()
        assert prompt.state == "pending_expanded"
        assert prompt.generation == 0

    @pytest.mark.asyncio
    async def test_summary_success_commits_after_replacement(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt
        from plugins.platforms.matrix.approval_cards import MatrixApprovalSummaryConfig

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        prompt = _MatrixApprovalPrompt(
            session_key="sess-1",
            chat_id="!room:example.org",
            message_id="$target",
            command="docker restart example",
            description="docker restart/stop/kill (container lifecycle)",
        )
        adapter.edit_message = AsyncMock(return_value=types.SimpleNamespace(success=True))

        with patch(
            "plugins.platforms.matrix.approval_cards.generate_command_summary",
            return_value="Restarts the example container.",
        ):
            cfg = MatrixApprovalSummaryConfig(
                enabled=True,
                provider_policy="local_only",
                local_timeout_seconds=90,
            )
            adapter._schedule_approval_summary(prompt, cfg)
            task = prompt.summary_task
            assert isinstance(task, asyncio.Task)
            await task

        assert prompt.state == "pending_summarized"
        assert prompt.summary == "Restarts the example container."
        assert prompt.generation == 1
        assert adapter.edit_message.await_count == 1
        body = adapter.edit_message.await_args.args[2]
        assert "Dangerous command requires approval" in body
        assert "Advisory interpretation" in body
        assert "Restarts the example container" in body
        assert "docker restart example" in body
        assert "✅ once" in body
        meta = adapter.edit_message.await_args.kwargs.get("metadata") or {}
        html = meta.get("matrix_formatted_body") or ""
        assert html.index("Advisory interpretation") < html.index("<details>")
        assert "<summary>Full command</summary>" in html
        assert "docker restart example" in html

    @pytest.mark.asyncio
    async def test_summary_edit_failure_does_not_advance_presented_state(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt
        from plugins.platforms.matrix.approval_cards import MatrixApprovalSummaryConfig

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        prompt = _MatrixApprovalPrompt(
            session_key="sess-1",
            chat_id="!room:example.org",
            message_id="$target",
            command="docker restart example",
            description="container lifecycle",
        )
        adapter.edit_message = AsyncMock(
            return_value=types.SimpleNamespace(success=False, error="homeserver rejected edit")
        )

        with patch(
            "plugins.platforms.matrix.approval_cards.generate_command_summary",
            return_value="Restarts the example container.",
        ):
            cfg = MatrixApprovalSummaryConfig(enabled=True, provider_policy="local_only")
            adapter._schedule_approval_summary(prompt, cfg)
            task = prompt.summary_task
            assert isinstance(task, asyncio.Task)
            await task

        assert adapter.edit_message.await_count == 1
        assert prompt.state == "pending_expanded"
        assert prompt.summary == ""
        assert prompt.generation == 0

    @pytest.mark.asyncio
    async def test_resolution_race_writes_terminal_replacement_last(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt
        from plugins.platforms.matrix.approval_cards import MatrixApprovalSummaryConfig

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        prompt = _MatrixApprovalPrompt(
            session_key="sess-1",
            chat_id="!room:example.org",
            message_id="$target",
            command="docker restart example",
            description="container lifecycle",
        )
        summary_edit_started = asyncio.Event()
        edits: list[str] = []

        async def _edit_message(room_id, event_id, body, metadata=None):
            if "Dangerous command requires approval" in body:
                edits.append("summary")
                summary_edit_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    # Model a transport that completed the edit at cancellation time.
                    return types.SimpleNamespace(success=True)
            edits.append("terminal")
            return types.SimpleNamespace(success=True)

        adapter.edit_message = AsyncMock(side_effect=_edit_message)

        with patch(
            "plugins.platforms.matrix.approval_cards.generate_command_summary",
            return_value="Restarts the example container.",
        ):
            cfg = MatrixApprovalSummaryConfig(enabled=True, provider_policy="local_only")
            adapter._schedule_approval_summary(prompt, cfg)
            await summary_edit_started.wait()
            prompt.resolved = True
            await adapter._finalize_matrix_approval_prompt(
                prompt.chat_id,
                prompt.message_id,
                prompt,
                choice="once",
                actor="@user:example.org",
            )

        assert edits == ["summary", "terminal"]
        assert prompt.state == "terminal_once"
        assert prompt.summary == ""
        assert prompt.generation == 1

    @pytest.mark.asyncio
    async def test_terminal_edit_failure_does_not_claim_compaction(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        prompt = _MatrixApprovalPrompt(
            session_key="sess-1",
            chat_id="!room:example.org",
            message_id="$target",
            command="docker restart example",
            description="container lifecycle",
            resolved=True,
        )
        adapter.edit_message = AsyncMock(
            return_value=types.SimpleNamespace(success=False, error="edit failed")
        )

        await adapter._finalize_matrix_approval_prompt(
            prompt.chat_id,
            prompt.message_id,
            prompt,
            choice="denied",
            actor="@user:example.org",
        )

        assert prompt.state == "pending_expanded"
        assert prompt.generation == 0
