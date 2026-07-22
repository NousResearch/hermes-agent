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
    generate_command_summary,
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

    def test_pending_summarized_preserves_typed_and_reaction_actions(self):
        text, html = format_pending_summarized(
            command="echo safe",
            description="run a bounded command",
            summary="Prints a bounded value.",
            allow_permanent=True,
            allow_session=True,
        )
        assert html is not None
        for required in (
            "!approve session",
            "!approve always",
            "!approve",
            "!deny",
            "✅ once",
            "🌀 session",
            "♾️ always",
            "❌ deny",
        ):
            assert required in text
            assert required in html

    def test_long_command_is_audit_complete_in_all_card_states(self):
        tail = "; audit-tail-sentinel"
        command = "printf start;" + ("x" * 2100) + tail
        cards = (
            format_pending_expanded(command=command, description="bounded test"),
            format_pending_summarized(
                command=command,
                description="bounded test",
                summary="Prints a bounded value.",
            ),
            format_terminal_compact(
                choice="once",
                command=command,
                description="bounded test",
            ),
        )

        for text, html in cards:
            assert html is not None
            assert tail in text
            assert tail in html

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
        for actionable in (
            "Dangerous command requires approval",
            "!approve",
            "!deny",
            "✅ once",
            "🌀 session",
            "♾️ always",
            "❌ deny",
        ):
            assert actionable not in text
            assert actionable not in html

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

    def test_generate_summary_disabled_policy_never_calls_llm(self):
        with patch("agent.auxiliary_client.call_llm") as call_llm:
            result = generate_command_summary(
                command="echo safe",
                description="bounded test",
                provider_policy="disabled",
            )

        assert result is None
        call_llm.assert_not_called()

    def test_generate_summary_unknown_policy_fails_closed_without_llm(self):
        with patch("agent.auxiliary_client.call_llm") as call_llm:
            result = generate_command_summary(
                command="echo safe",
                description="bounded test",
                provider_policy="local_only_typo",
            )

        assert result is None
        call_llm.assert_not_called()

    def test_generate_summary_local_only_rejects_remote_route(self):
        remote_route = {
            "provider": "openai-codex",
            "model": "gpt-test",
            "base_url": "https://api.example.org/v1",
            "api_key": "test-key",
            "api_mode": "codex_responses",
        }
        with (
            patch(
                "plugins.platforms.matrix.approval_cards._resolve_approval_summary_route",
                return_value=remote_route,
                create=True,
            ),
            patch("agent.auxiliary_client.call_llm") as call_llm,
        ):
            result = generate_command_summary(
                command="echo safe",
                description="bounded test",
                provider_policy="local_only",
            )

        assert result is None
        call_llm.assert_not_called()

    def test_generate_summary_local_only_pins_private_route_without_fallback(self):
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Safe summary."))]
        )
        private_route = {
            "provider": "custom",
            "model": "local-model",
            "base_url": "http://10.0.10.20:8000/v1",
            "api_key": "test-key",
            "api_mode": "chat_completions",
        }
        with (
            patch(
                "plugins.platforms.matrix.approval_cards._resolve_approval_summary_route",
                return_value=private_route,
                create=True,
            ),
            patch("agent.auxiliary_client.call_llm", return_value=response) as call_llm,
        ):
            result = generate_command_summary(
                command="echo safe",
                description="bounded test",
                provider_policy="local_only",
            )

        assert result == "Safe summary."
        kwargs = call_llm.call_args.kwargs
        assert kwargs["provider"] == "custom"
        assert kwargs["model"] == "local-model"
        assert kwargs["base_url"] == "http://10.0.10.20:8000/v1"
        assert kwargs["allow_provider_fallback"] is False

    def test_generate_summary_force_redacts_before_llm_boundary(self):
        token = "sk-proj-" + ("X" * 40)
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Safe summary."))]
        )
        private_route = {
            "provider": "custom",
            "model": "local-model",
            "base_url": "http://10.0.10.20:8000/v1",
            "api_key": "test-key",
            "api_mode": "chat_completions",
        }
        with (
            patch(
                "plugins.platforms.matrix.approval_cards._resolve_approval_summary_route",
                return_value=private_route,
            ),
            patch("agent.auxiliary_client.call_llm", return_value=response) as call_llm,
        ):
            result = generate_command_summary(
                command=f"export OPENAI_API_KEY={token} && echo safe",
                description="bounded test",
                provider_policy="local_only",
            )

        assert result == "Safe summary."
        assert token not in repr(call_llm.call_args.kwargs["messages"])

    def test_force_redact_known_prefix(self):
        raw = "export OPENAI_API_KEY=" + "sk-proj-abcdefghijklmnopqrstuvwxyz"
        out = force_redact_command(raw)
        token = raw.partition("=")[2]
        assert token not in out

    def test_force_redact_failure_is_fail_closed(self):
        token = "sk-" + "proj-abcdefghijklmnopqrstuvwxyz"
        raw = f"export OPENAI_API_KEY={token}"
        with patch("agent.redact.redact_sensitive_text", side_effect=RuntimeError("boom")):
            out = force_redact_command(raw)
        assert raw not in out
        assert token not in out

    def test_matrix_html_sanitizer_preserves_native_disclosure_and_drops_unsafe_markup(self):
        from plugins.platforms.matrix.adapter import _sanitize_matrix_html

        raw = (
            '<details open onclick="alert(1)" style="display:block">'
            '<summary data-kind="command">Full command</summary>'
            '<pre><code class="language-bash" onmouseover="alert(2)">echo safe</code></pre>'
            '</details>'
            '<script>alert(3)</script><style>.hidden{display:none}</style>'
            '<a href="javascript:alert(4)" title="unsafe">bad link</a>'
        )

        expected = (
            '<details><summary>Full command</summary>'
            '<pre><code class="language-bash">echo safe</code></pre></details>'
            '<a>bad link</a>'
        )
        sanitized = _sanitize_matrix_html(raw)

        assert sanitized == expected
        assert _sanitize_matrix_html(sanitized) == expected

    @pytest.mark.parametrize(
        "href",
        (
            "JaVaScRiPt:alert(1)",
            "java&#x0A;script:alert(1)",
            "javascript&#58;alert(1)",
            "data:text/html,unsafe",
        ),
    )
    def test_matrix_html_sanitizer_rejects_obfuscated_unsafe_urls(self, href):
        from plugins.platforms.matrix.adapter import _sanitize_matrix_html

        assert _sanitize_matrix_html(f'<a href="{href}">bad link</a>') == "<a>bad link</a>"


class TestMatrixApprovalCardLifecycle:
    @pytest.mark.asyncio
    async def test_pending_expanded_send_sanitizes_native_formatted_body(self):
        """t0 is a safe Matrix root event with an audit-complete fallback."""
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
            sent["event_type"] = event_type
            sent["content"] = content
            return "$approval"

        adapter._client = types.SimpleNamespace(send_message_event=_send_event)
        text, html = format_pending_expanded(
            command="echo safe",
            description="run a bounded test command",
        )
        assert html is not None
        tainted_html = html.replace(
            "<pre>",
            '<pre onclick="alert(1)" style="display:none">',
        ) + "<script>alert(2)</script>"

        result = await adapter.send(
            "!room:example.org",
            text,
            metadata={"matrix_formatted_body": tainted_html},
        )

        assert result.success is True
        assert str(sent["event_type"]) == "m.room.message"
        content = sent["content"]
        assert set(content) == {"msgtype", "body", "format", "formatted_body"}
        assert content["msgtype"] == "m.text"
        assert content["format"] == "org.matrix.custom.html"
        for required in (
            "Dangerous command requires approval",
            "run a bounded test command",
            "echo safe",
            "✅ once",
            "❌ deny",
        ):
            assert required in content["body"]
            assert required in content["formatted_body"]
        assert "<pre>echo safe</pre>" in content["formatted_body"]
        assert "<script" not in content["formatted_body"]
        assert "alert(" not in content["formatted_body"]
        assert " onclick=" not in content["formatted_body"]
        assert " style=" not in content["formatted_body"]

    @pytest.mark.asyncio
    async def test_pending_summarized_edit_is_sanitized_and_keeps_replacement_prefix(self):
        """t1 retains Matrix's outer marker and complete authoritative content."""
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
            sent["event_type"] = event_type
            sent["content"] = content
            return "$replacement"

        adapter._client = types.SimpleNamespace(send_message_event=_send_event)
        text, html = format_pending_summarized(
            command="systemctl restart example.service",
            description="stop/restart system service",
            summary="Restarts the example service.",
        )
        assert html is not None
        tainted_html = html.replace(
            "<details>",
            '<details open onclick="alert(1)" style="display:block">',
        ) + "<script>alert(2)</script>"

        result = await adapter.edit_message(
            "!room:example.org",
            "$approval",
            text,
            metadata={"matrix_formatted_body": tainted_html},
        )

        assert result.success is True
        assert str(sent["event_type"]) == "m.room.message"
        content = sent["content"]
        assert set(content) == {
            "msgtype",
            "body",
            "format",
            "formatted_body",
            "m.new_content",
            "m.relates_to",
        }
        assert content["msgtype"] == "m.text"
        assert content["format"] == "org.matrix.custom.html"
        assert content["m.relates_to"] == {
            "rel_type": "m.replace",
            "event_id": "$approval",
        }
        new_content = content["m.new_content"]
        assert set(new_content) == {"msgtype", "body", "format", "formatted_body"}
        assert new_content["msgtype"] == "m.text"
        assert new_content["format"] == "org.matrix.custom.html"
        assert content["body"] == "* " + new_content["body"]
        assert content["formatted_body"] == "* " + new_content["formatted_body"]
        for required in (
            "Dangerous command requires approval",
            "stop/restart system service",
            "Advisory interpretation",
            "Restarts the example service",
            "systemctl restart example.service",
            "!approve",
            "!deny",
            "✅ once",
            "❌ deny",
        ):
            assert required in new_content["body"]
            assert required in new_content["formatted_body"]
        assert new_content["formatted_body"].index(
            "Advisory interpretation"
        ) < new_content["formatted_body"].index("<details>")
        assert "<strong>Advisory interpretation:</strong>" in new_content["formatted_body"]
        assert "<summary>Full command</summary>" in new_content["formatted_body"]
        assert "systemctl restart example.service" in new_content["formatted_body"]
        assert "<script" not in new_content["formatted_body"]
        assert "alert(" not in new_content["formatted_body"]
        assert " onclick=" not in new_content["formatted_body"]
        assert " style=" not in new_content["formatted_body"]
        assert " open" not in new_content["formatted_body"]

    @pytest.mark.asyncio
    async def test_terminal_edit_is_sanitized_and_keeps_authoritative_replacement(self):
        """t2 is compact HTML while plaintext and m.new_content remain auditable."""
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
            sent["event_type"] = event_type
            sent["content"] = content
            return "$terminal"

        adapter._client = types.SimpleNamespace(send_message_event=_send_event)
        text, html = format_terminal_compact(
            choice="once",
            command="echo safe",
            description="run a bounded test command",
            actor="operator",
            summary="Prints a bounded test value.",
        )
        assert html is not None
        tainted_html = html.replace(
            "<details>",
            '<details open onfocus="alert(1)" style="display:block">',
        ) + "<style>.hidden{display:none}</style>"

        result = await adapter.edit_message(
            "!room:example.org",
            "$approval",
            text,
            metadata={"matrix_formatted_body": tainted_html},
        )

        assert result.success is True
        assert str(sent["event_type"]) == "m.room.message"
        content = sent["content"]
        assert set(content) == {
            "msgtype",
            "body",
            "format",
            "formatted_body",
            "m.new_content",
            "m.relates_to",
        }
        assert content["msgtype"] == "m.text"
        assert content["format"] == "org.matrix.custom.html"
        assert content["m.relates_to"] == {
            "rel_type": "m.replace",
            "event_id": "$approval",
        }
        new_content = content["m.new_content"]
        assert set(new_content) == {"msgtype", "body", "format", "formatted_body"}
        assert new_content["msgtype"] == "m.text"
        assert new_content["format"] == "org.matrix.custom.html"
        assert content["body"] == "* " + new_content["body"]
        assert content["formatted_body"] == "* " + new_content["formatted_body"]
        for required in (
            "Approved once",
            "operator",
            "run a bounded test command",
            "Advisory interpretation",
            "Prints a bounded test value",
            "echo safe",
        ):
            assert required in new_content["body"]
            assert required in new_content["formatted_body"]
        assert new_content["formatted_body"].index(
            "Advisory interpretation"
        ) < new_content["formatted_body"].index("<details>")
        assert "<summary>Full command</summary>" in new_content["formatted_body"]
        assert "<style" not in new_content["formatted_body"]
        assert "alert(" not in new_content["formatted_body"]
        assert " onfocus=" not in new_content["formatted_body"]
        assert " style=" not in new_content["formatted_body"]
        assert " open" not in new_content["formatted_body"]
        for actionable in (
            "Dangerous command requires approval",
            "!approve",
            "!deny",
            "✅ once",
            "🌀 session",
            "♾️ always",
            "❌ deny",
        ):
            assert actionable not in new_content["body"]
            assert actionable not in new_content["formatted_body"]

    @pytest.mark.asyncio
    async def test_preformatted_whitespace_only_unsafe_html_keeps_generated_fallback(self):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        sent = []

        async def _send_event(room, event_type, content):
            sent.append(content)
            return f"$event-{len(sent)}"

        adapter._client = types.SimpleNamespace(send_message_event=_send_event)
        unsafe_html = " \n<script>alert(1)</script>\t"

        root = await adapter.send(
            "!room:example.org",
            "**visible fallback**",
            metadata={"matrix_formatted_body": unsafe_html},
        )
        edit = await adapter.edit_message(
            "!room:example.org",
            "$approval",
            "**visible fallback**",
            metadata={"matrix_formatted_body": unsafe_html},
        )

        assert root.success is True
        assert edit.success is True
        assert "<strong>visible fallback</strong>" in sent[0]["formatted_body"]
        assert "<strong>visible fallback</strong>" in sent[1]["m.new_content"]["formatted_body"]
        assert sent[1]["formatted_body"] == (
            "* " + sent[1]["m.new_content"]["formatted_body"]
        )
        assert "<script" not in sent[0]["formatted_body"]
        assert "<script" not in sent[1]["formatted_body"]

    @pytest.mark.asyncio
    async def test_preformatted_payload_over_transport_limit_fails_closed(self):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={
                    "homeserver": "https://matrix.example.org",
                    "max_message_length": 500,
                },
            )
        )
        adapter._client = types.SimpleNamespace(send_message_event=AsyncMock())
        oversized_html = "<p>" + ("x" * 600) + "</p>"

        root = await adapter.send(
            "!room:example.org",
            "visible fallback",
            metadata={"matrix_formatted_body": oversized_html},
        )
        edit = await adapter.edit_message(
            "!room:example.org",
            "$approval",
            "visible fallback",
            metadata={"matrix_formatted_body": oversized_html},
        )

        assert root.success is False
        assert edit.success is False
        assert "limit" in (root.error or "").lower()
        assert "limit" in (edit.error or "").lower()
        adapter._client.send_message_event.assert_not_awaited()

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
        sent = {}

        async def _send_event(room, event_type, content):
            sent["event_type"] = event_type
            sent["content"] = content
            return "$evt1"

        adapter._client = types.SimpleNamespace(send_message_event=_send_event)
        adapter._send_reaction = AsyncMock(return_value="$r")
        adapter._schedule_approval_resolution_watch = MagicMock()

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
        assert str(sent["event_type"]) == "m.room.message"
        content = sent["content"]
        assert set(content) == {"msgtype", "body", "format", "formatted_body"}
        assert content["msgtype"] == "m.text"
        assert content["format"] == "org.matrix.custom.html"
        assert "Dangerous command requires approval" in content["body"]
        assert "recursive delete" in content["body"]
        assert "rm -rf /tmp/test" in content["body"]
        assert "❎" not in content["body"]
        assert "<pre>rm -rf /tmp/test</pre>" in content["formatted_body"]
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
                choice="once",
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
    @pytest.mark.parametrize("choice", ["once", "session", "always", "deny"])
    async def test_typed_resolution_watch_preserves_exact_terminal_choice(
        self, monkeypatch, choice
    ):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@user:example.org")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt
        from tools import approval as approval_mod

        session_key = f"sess-typed-{choice}"
        entry = approval_mod._ApprovalEntry({"command": "rm -rf /tmp/x"})
        with approval_mod._lock:
            approval_mod._gateway_queues[session_key] = [entry]

        adapter = MatrixAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"homeserver": "https://matrix.example.org"},
            )
        )
        prompt = _MatrixApprovalPrompt(
            session_key=session_key,
            chat_id="!room:example.org",
            message_id=f"$target-{choice}",
            approval_id=entry.approval_id,
            command=entry.data["command"],
        )
        adapter._approval_prompts_by_event[prompt.message_id] = prompt
        adapter._approval_prompt_by_session[prompt.session_key] = {prompt.message_id}
        adapter._redact_bot_approval_reactions = AsyncMock()
        adapter._finalize_matrix_approval_prompt = AsyncMock()

        try:
            adapter._schedule_approval_resolution_watch(prompt)
            assert approval_mod.resolve_gateway_approval(session_key, choice) == 1
            for _ in range(10):
                if prompt.resolved:
                    break
                await asyncio.sleep(0)

            assert prompt.resolved is True
            adapter._finalize_matrix_approval_prompt.assert_awaited_once_with(
                prompt.chat_id,
                prompt.message_id,
                prompt,
                choice=choice,
                actor="",
            )
        finally:
            prompt.resolved = True
            approval_mod.clear_session(session_key)

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
