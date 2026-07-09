"""Tests for the VK Teams platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_vkteams`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.

Everything routes through the ``platform_registry`` — no core files are
modified by this plugin, so coverage focuses on the adapter itself plus
the plugin-shape hooks (``register()``, ``_env_enablement``,
``_standalone_send``).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_vkteams = load_plugin_adapter("vkteams")

VKTeamsAdapter = _vkteams.VKTeamsAdapter
check_requirements = _vkteams.check_requirements
validate_config = _vkteams.validate_config
is_connected = _vkteams.is_connected
register = _vkteams.register
_env_enablement = _vkteams._env_enablement
_standalone_send = _vkteams._standalone_send
_redact_token = _vkteams._redact_token
_strip_mdv2 = _vkteams._strip_mdv2
_strip_html = _vkteams._strip_html
_escape_html = _vkteams._escape_html
_wrap_markdown_tables = _vkteams._wrap_markdown_tables
DEFAULT_API_BASE = _vkteams.DEFAULT_API_BASE
MAX_MESSAGE_LENGTH = _vkteams.MAX_MESSAGE_LENGTH
GROUP_CHAT_SUFFIX = _vkteams.GROUP_CHAT_SUFFIX


TOKEN = "001.0000000000.0000000000:1000000"


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_adapter(monkeypatch=None, extra=None, **env):
    if monkeypatch is not None:
        for var in (
            "VKTEAMS_BOT_TOKEN", "VKTEAMS_API_BASE", "VKTEAMS_PARSE_MODE",
            "VKTEAMS_POLL_TIME", "VKTEAMS_ALLOWED_USERS", "VKTEAMS_ALLOW_ALL_USERS",
        ):
            monkeypatch.delenv(var, raising=False)
        for var, value in env.items():
            monkeypatch.setenv(var, value)
    config = PlatformConfig(enabled=True, extra=extra or {"token": TOKEN})
    return VKTeamsAdapter(config)


class _FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload)
        self.content = b""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _wire_client(adapter, responses):
    """Attach a mock httpx client returning *responses* for GET calls in order."""
    client = MagicMock()
    client.get = AsyncMock(side_effect=list(responses))
    client.post = AsyncMock()
    adapter._http_client = client
    return client


# ---------------------------------------------------------------------------
# 1. Platform enum (plugin-discovered, not bundled in the core enum)
# ---------------------------------------------------------------------------


def test_platform_enum_resolves_via_plugin_scan():
    from gateway.config import Platform
    p = Platform("vkteams")
    assert p.value == "vkteams"
    assert Platform("vkteams") is p


# ---------------------------------------------------------------------------
# 2. check_requirements / validate_config / is_connected
# ---------------------------------------------------------------------------


class TestRequirements:

    def test_check_requirements_tracks_httpx(self, monkeypatch):
        monkeypatch.setattr(_vkteams, "HTTPX_AVAILABLE", False)
        assert check_requirements() is False
        monkeypatch.setattr(_vkteams, "HTTPX_AVAILABLE", True)
        assert check_requirements() is True

    def test_validate_config_requires_token(self, monkeypatch):
        monkeypatch.delenv("VKTEAMS_BOT_TOKEN", raising=False)
        assert validate_config(PlatformConfig(enabled=True, extra={})) is False
        assert validate_config(
            PlatformConfig(enabled=True, extra={"token": TOKEN})
        ) is True

    def test_token_from_platformconfig_token_field(self, monkeypatch):
        monkeypatch.delenv("VKTEAMS_BOT_TOKEN", raising=False)
        assert validate_config(PlatformConfig(enabled=True, token=TOKEN)) is True

    def test_is_connected_from_env(self, monkeypatch):
        monkeypatch.setenv("VKTEAMS_BOT_TOKEN", TOKEN)
        assert is_connected(PlatformConfig(enabled=True, extra={})) is True
        monkeypatch.delenv("VKTEAMS_BOT_TOKEN", raising=False)
        assert is_connected(PlatformConfig(enabled=True, extra={})) is False


# ---------------------------------------------------------------------------
# 3. Adapter init
# ---------------------------------------------------------------------------


class TestAdapterInit:

    def test_defaults(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter._token == TOKEN
        assert adapter._api_base == DEFAULT_API_BASE
        assert adapter._parse_mode == "MarkdownV2"
        assert adapter._poll_time == 25
        assert adapter._last_event_id == 0

    def test_env_overrides(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            extra={"token": TOKEN},
            VKTEAMS_API_BASE="https://corp.example.ru/bot/v1/",
            VKTEAMS_PARSE_MODE="none",
            VKTEAMS_POLL_TIME="40",
        )
        assert adapter._api_base == "https://corp.example.ru/bot/v1"
        assert adapter._parse_mode == ""  # "none" disables formatting
        assert adapter._poll_time == 40

    def test_extra_wins_over_env_for_api_base(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            extra={"token": TOKEN, "api_base": "https://onprem.corp/bot/v1"},
            VKTEAMS_API_BASE="https://ignored.example/bot/v1",
        )
        assert adapter._api_base == "https://onprem.corp/bot/v1"

    def test_invalid_poll_time_falls_back(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_POLL_TIME="bogus")
        assert adapter._poll_time == 25

    def test_allowlist_parsing(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch, VKTEAMS_ALLOWED_USERS="a@corp.ru, B@corp.ru"
        )
        assert adapter._allowed_users == {"a@corp.ru", "b@corp.ru"}
        # No runner wired → env-only fallback path.
        assert adapter._is_callback_user_authorized("A@corp.ru") is True
        assert adapter._is_callback_user_authorized("evil@corp.ru") is False

    def test_no_allowlist_fails_closed(self, monkeypatch):
        """No allowlist + no runner + no GATEWAY_ALLOW_ALL_USERS -> DENY."""
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        adapter = _make_adapter(monkeypatch)
        assert adapter._is_callback_user_authorized("anyone@corp.ru") is False

    def test_no_allowlist_honors_gateway_allow_all(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        assert adapter._is_callback_user_authorized("anyone@corp.ru") is True

    def test_wildcard_allowlist_allows_all(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_ALLOWED_USERS="*")
        assert adapter._is_callback_user_authorized("anyone@corp.ru") is True

    def test_empty_user_id_denied(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_ALLOW_ALL_USERS="true")
        assert adapter._is_callback_user_authorized("") is False

    def test_callback_auth_delegates_to_runner(self, monkeypatch):
        """When a runner is wired, its _is_user_authorized decides."""
        adapter = _make_adapter(monkeypatch)
        runner = MagicMock()
        runner._is_user_authorized = MagicMock(return_value=False)
        handler = MagicMock()
        handler.__self__ = runner
        adapter._message_handler = handler
        # Even with allow-all env set, the runner's DENY wins.
        adapter._allow_all_users = True
        assert adapter._is_callback_user_authorized(
            "user@corp.ru", chat_id="c", chat_type="group",
        ) is False
        source = runner._is_user_authorized.call_args.args[0]
        assert source.user_id == "user@corp.ru"
        assert source.chat_type == "group"


# ---------------------------------------------------------------------------
# 4. Helpers: redaction, group detection, markdown
# ---------------------------------------------------------------------------


class TestHelpers:

    def test_redact_token(self):
        msg = f"error for url https://x/bot/v1/events/get?token={TOKEN}&pollTime=25"
        redacted = _redact_token(msg, TOKEN)
        assert TOKEN not in redacted
        assert "token=***" in redacted

    def test_redact_unknown_token_via_regex(self):
        msg = "url 'https://x/bot/v1/messages/sendText?token=001.abc&chatId=1'"
        assert "001.abc" not in _redact_token(msg, "")

    def test_group_chat_detection(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter._is_group_chat(f"685000000{GROUP_CHAT_SUFFIX}") is True
        assert adapter._is_group_chat("user@corp.ru") is False

    def test_wrap_markdown_tables(self):
        text = "before\n| a | b |\n|---|---|\n| 1 | 2 |\nafter"
        wrapped = _wrap_markdown_tables(text)
        assert "```\n| a | b |" in wrapped
        assert wrapped.endswith("after")

    def test_strip_mdv2_roundtrip(self):
        assert _strip_mdv2(r"a\.b \(c\)") == "a.b (c)"
        assert _strip_mdv2("*bold* _it_") == "bold it"


class TestFormatMessage:

    def test_bold_and_escape(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        out = adapter.format_message("**bold** and 1.5 (x)")
        assert "*bold*" in out
        # VK Teams does NOT treat '.' as special — it must stay unescaped,
        # or the whole message is rejected with "Format error".
        assert "1.5" in out and r"1\.5" not in out
        # Parens ARE style delimiters (links) → escaped.
        assert r"\(x\)" in out

    def test_telegram_only_punctuation_not_escaped(self, monkeypatch):
        """Regression for the live 'Format error': VK Teams rejects escaping
        of Telegram-only specials (. ! - = + # | { }). They must pass through."""
        adapter = _make_adapter(monkeypatch)
        out = adapter.format_message("Готово! Версия 1.5-rc, cost=3+2. C#/C++ {a|b}")
        for frag in ("Готово!", "1.5-rc", "cost=3+2", "C#/C++", "{a|b}"):
            assert frag in out, f"{frag!r} altered: {out!r}"
        assert "\\" not in out  # nothing in this string is a VK style delimiter

    def test_list_markers_survive(self, monkeypatch):
        """Line-start '- ' and '1.' are VK Teams list syntax — leaving '-'
        and '.' unescaped lets them render as real lists."""
        adapter = _make_adapter(monkeypatch)
        out = adapter.format_message("- пункт\n- пункт 2")
        assert out == "- пункт\n- пункт 2"
        out2 = adapter.format_message("1. первый\n2. второй")
        assert out2 == "1. первый\n2. второй"

    def test_style_delimiters_escaped(self, monkeypatch):
        """Literal style delimiters in plain text must be escaped so they
        don't accidentally start a style."""
        adapter = _make_adapter(monkeypatch)
        out = adapter.format_message(r"a_b c*d e~f g`h [i] (j)")
        for esc in (r"a\_b", r"c\*d", r"e\~f", r"g\`h", r"\[i\]", r"\(j\)"):
            assert esc in out, f"missing {esc!r} in {out!r}"

    def test_header_becomes_bold(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.format_message("## Title").startswith("*Title*")

    def test_code_block_protected(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        src = "```python\nx = a * b  # 1.5\n```"
        out = adapter.format_message(src)
        assert "x = a * b  # 1.5" in out  # body untouched (no escapes)
        assert out.startswith("```python\n")

    def test_link_conversion(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        out = adapter.format_message("see [docs](https://x.y/path_(1))")
        assert "[docs](https://x.y/path_(1\\))" in out

    def test_plain_mode_passthrough(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_PARSE_MODE="none")
        assert adapter.format_message("**raw**") == "**raw**"


class TestFormatMessageHTML:
    """HTML is the recommended mode: VK Teams' MarkdownV2 parser rejects
    valid inline code / lone ``_ * ~`` with 'Format error'; HTML escapes only
    ``& < >`` so that whole class of failure disappears."""

    def _adapter(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_PARSE_MODE="HTML")
        assert adapter._parse_mode == "HTML"
        return adapter

    def test_escape_html_only_three_chars(self):
        assert _escape_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"
        # Ampersand escaped first so entities don't double up.
        assert _escape_html("&<>") == "&amp;&lt;&gt;"

    def test_inline_code_with_lone_tilde_is_valid(self, monkeypatch):
        """The exact live 'Format error' payload — a lone '~' inside inline
        code. In HTML it's a plain literal wrapped in <code>."""
        adapter = self._adapter(monkeypatch)
        out = adapter.format_message("No personalities configured in `~/./config.yaml`")
        assert out == "No personalities configured in <code>~/./config.yaml</code>"

    def test_prose_html_metachars_escaped(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        out = adapter.format_message("latency < 5ms & CPU > 90%")
        assert out == "latency &lt; 5ms &amp; CPU &gt; 90%"

    def test_tag_injection_from_user_text_is_neutralized(self, monkeypatch):
        """A user/model can't smuggle live markup: <b> in prose is escaped."""
        adapter = self._adapter(monkeypatch)
        out = adapter.format_message("oops <b>evil</b> <script>x</script>")
        assert "<b>" not in out and "<script>" not in out
        assert "&lt;b&gt;evil&lt;/b&gt;" in out

    def test_bold_italic_strike(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        assert adapter.format_message("**bold**") == "<b>bold</b>"
        assert adapter.format_message("*it*") == "<i>it</i>"
        assert adapter.format_message("~~old~~ new") == "<s>old</s> new"

    def test_underscore_italic(self, monkeypatch):
        """_Курсив_ must render as italic (regression: it was left literal)."""
        adapter = self._adapter(monkeypatch)
        assert adapter.format_message("_Курсив_ — проверка") == "<i>Курсив</i> — проверка"

    def test_snake_case_not_italicized(self, monkeypatch):
        """Word-boundary guard: identifiers with '_' in prose stay literal."""
        adapter = self._adapter(monkeypatch)
        out = adapter.format_message("var latency_ms and health_check here")
        assert out == "var latency_ms and health_check here"
        assert "<i>" not in out

    def test_double_underscore_not_italic(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        # A dangling '__' (the "not italic" demo) must not become a tag.
        assert adapter.format_message("__ не-italic") == "__ не-italic"

    def test_nested_emphasis(self, monkeypatch):
        """Recursive inline pass: one style nests inside another instead of
        leaving the inner markers literal (regression from the live test)."""
        adapter = self._adapter(monkeypatch)
        assert adapter.format_message("**bold _it_**") == "<b>bold <i>it</i></b>"
        assert adapter.format_message("*it **b** in*") == "<i>it <b>b</b> in</i>"
        assert adapter.format_message("~~**bs**~~") == "<s><b>bs</b></s>"
        assert adapter.format_message("**~~sb~~**") == "<b><s>sb</s></b>"
        # Inline code survives inside bold.
        assert adapter.format_message("**b `c` d**") == "<b>b <code>c</code> d</b>"

    def test_blockquote_inner_emphasis(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        out = adapter.format_message("> Цитата с **жирным**")
        assert out == "<blockquote>Цитата с <b>жирным</b></blockquote>"

    def test_header_becomes_bold(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        assert adapter.format_message("## Сводка") == "<b>Сводка</b>"

    def test_code_block_becomes_pre_and_escapes_body(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        out = adapter.format_message("```bash\nx < y && z\n```")
        assert out.startswith("<pre>") and out.rstrip().endswith("</pre>")
        assert "x &lt; y &amp;&amp; z" in out
        assert "bash" not in out  # language hint dropped

    def test_link_conversion(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        out = adapter.format_message("see [docs](https://x.y/p?a=1&b=2)")
        assert out == 'see <a href="https://x.y/p?a=1&amp;b=2">docs</a>'

    def test_multiple_fences_do_not_cascade(self, monkeypatch):
        """Line-anchored fences: two code blocks with a heading between them
        stay separate, and the heading is NOT swallowed into a <pre>."""
        adapter = self._adapter(monkeypatch)
        src = "```\nblock one\n```\n\n## Между\n\n```\nblock two\n```"
        out = adapter.format_message(src)
        assert out.count("<pre>") == 2 and out.count("</pre>") == 2
        assert "<b>Между</b>" in out
        assert "block one" in out and "block two" in out

    def test_lone_backtick_in_prose_stays_literal(self, monkeypatch):
        """A stray backtick can't open a code span that runs across newlines
        and swallows following paragraphs."""
        adapter = self._adapter(monkeypatch)
        src = "Бэктик ` отдельно.\n\nСледующий абзац с `кодом` тут."
        out = adapter.format_message(src)
        assert "Бэктик ` отдельно." in out       # lone backtick left as-is
        assert "<code>кодом</code>" in out        # real inline code still works
        assert out.count("<code>") == 1

    def test_list_markers_survive(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        assert adapter.format_message("- a\n- b") == "- a\n- b"
        assert adapter.format_message("1. a\n2. b") == "1. a\n2. b"

    def test_strip_html_roundtrip(self):
        assert _strip_html("<b>bold</b> and <code>x&lt;y</code>") == "bold and x<y"
        assert _strip_html('<a href="u">t</a>') == "t"

    def test_strip_formatting_dispatches_to_html(self, monkeypatch):
        adapter = self._adapter(monkeypatch)
        assert adapter._strip_formatting("<b>hi</b>") == "hi"


# ---------------------------------------------------------------------------
# 5. send()
# ---------------------------------------------------------------------------


class TestSend:

    def test_send_success(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = _wire_client(adapter, [_FakeResponse({"ok": True, "msgId": "7001"})])
        result = _run(adapter.send("user@corp.ru", "hello"))
        assert result.success is True
        assert result.message_id == "7001"
        call = client.get.call_args
        assert call.args[0].endswith("/messages/sendText")
        assert call.kwargs["params"]["chatId"] == "user@corp.ru"
        assert call.kwargs["params"]["parseMode"] == "MarkdownV2"

    def test_send_empty_is_noop(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = _wire_client(adapter, [])
        result = _run(adapter.send("user@corp.ru", "   "))
        assert result.success is True
        client.get.assert_not_called()

    def test_send_reply_to_first_chunk_only(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_PARSE_MODE="none")
        client = _wire_client(adapter, [
            _FakeResponse({"ok": True, "msgId": str(7000 + i)}) for i in range(3)
        ])
        long_text = "word " * 2000  # > 4096 chars -> multiple chunks
        result = _run(adapter.send("user@corp.ru", long_text, reply_to="42"))
        assert result.success is True
        calls = client.get.call_args_list
        assert len(calls) > 1
        assert calls[0].kwargs["params"].get("replyMsgId") == "42"
        assert "replyMsgId" not in calls[1].kwargs["params"]
        # Last visible message id + earlier chunks as continuations.
        assert result.message_id == str(7000 + len(calls) - 1)
        assert len(result.continuation_message_ids) == len(calls) - 1

    def test_send_ratelimit_retries_once(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        _wire_client(adapter, [
            _FakeResponse({"ok": False, "description": "ratelimit"},
                          headers={"Retry-After": "0.01"}),
            _FakeResponse({"ok": True, "msgId": "7002"}),
        ])
        result = _run(adapter.send("user@corp.ru", "hi"))
        assert result.success is True
        assert result.message_id == "7002"

    def test_send_bad_format_falls_back_to_plain(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = _wire_client(adapter, [
            _FakeResponse({"ok": False, "description": "invalid message format"}),
            _FakeResponse({"ok": True, "msgId": "7003"}),
        ])
        result = _run(adapter.send("user@corp.ru", "**hi**"))
        assert result.success is True
        second = client.get.call_args_list[1]
        assert "parseMode" not in second.kwargs["params"]

    def test_send_html_carries_parsemode_and_transpiles(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_PARSE_MODE="HTML")
        client = _wire_client(adapter, [_FakeResponse({"ok": True, "msgId": "7100"})])
        result = _run(adapter.send("user@corp.ru", "**bold** `x<y`"))
        assert result.success is True
        params = client.get.call_args.kwargs["params"]
        assert params["parseMode"] == "HTML"
        assert params["text"] == "<b>bold</b> <code>x&lt;y</code>"

    def test_send_html_bad_format_falls_back_to_stripped_plain(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_PARSE_MODE="HTML")
        client = _wire_client(adapter, [
            _FakeResponse({"ok": False, "description": "invalid message format"}),
            _FakeResponse({"ok": True, "msgId": "7101"}),
        ])
        result = _run(adapter.send("user@corp.ru", "**hi** `code`"))
        assert result.success is True
        second = client.get.call_args_list[1]
        assert "parseMode" not in second.kwargs["params"]
        # Tags stripped for the plain retry, not left as literal <b>…</b>.
        assert second.kwargs["params"]["text"] == "hi code"

    def test_send_forbidden_not_retryable(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        _wire_client(adapter, [
            _FakeResponse({"ok": False, "description": "permission denied"}),
        ])
        result = _run(adapter.send("user@corp.ru", "hi"))
        assert result.success is False
        assert result.retryable is False
        assert result.error_kind == "forbidden"


# ---------------------------------------------------------------------------
# 6. edit_message()
# ---------------------------------------------------------------------------


class TestEditMessage:

    def test_midstream_edit_is_plain(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = _wire_client(adapter, [_FakeResponse({"ok": True})])
        result = _run(adapter.edit_message("user@corp.ru", "10", "progress…"))
        assert result.success is True
        params = client.get.call_args.kwargs["params"]
        assert params["msgId"] == "10"
        assert "parseMode" not in params

    def test_finalize_edit_is_formatted(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = _wire_client(adapter, [_FakeResponse({"ok": True})])
        result = _run(
            adapter.edit_message("user@corp.ru", "10", "**done**", finalize=True)
        )
        assert result.success is True
        params = client.get.call_args.kwargs["params"]
        assert params["parseMode"] == "MarkdownV2"
        assert "*done*" in params["text"]

    def test_group_edits_paced_not_skipped(self, monkeypatch):
        """A throttled mid-stream edit PACES (sleeps) then edits — it must
        never report success without actually performing the edit."""
        adapter = _make_adapter(monkeypatch)
        chat = f"685{GROUP_CHAT_SUFFIX}"
        client = _wire_client(adapter, [
            _FakeResponse({"ok": True}), _FakeResponse({"ok": True}),
        ])
        sleeps = []

        async def _fake_sleep(secs):
            sleeps.append(secs)

        monkeypatch.setattr(_vkteams.asyncio, "sleep", _fake_sleep)
        first = _run(adapter.edit_message(chat, "10", "one"))
        second = _run(adapter.edit_message(chat, "10", "two"))
        assert first.success and second.success
        # Both edits reached the API; the second was paced by a sleep.
        assert client.get.call_count == 2
        assert any(s > 0 for s in sleeps)

    def test_finalize_bypasses_throttle(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        chat = f"685{GROUP_CHAT_SUFFIX}"
        client = _wire_client(adapter, [
            _FakeResponse({"ok": True}), _FakeResponse({"ok": True}),
        ])
        _run(adapter.edit_message(chat, "10", "one"))
        result = _run(adapter.edit_message(chat, "10", "final", finalize=True))
        assert result.success is True
        assert client.get.call_count == 2

    def test_midstream_overflow_truncates_and_dedups(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = _wire_client(adapter, [_FakeResponse({"ok": True})])
        adapter._last_edit_at.clear()
        big = "x" * (MAX_MESSAGE_LENGTH + 500)
        first = _run(adapter.edit_message("user@corp.ru", "10", big))
        assert first.success is True
        sent = client.get.call_args.kwargs["params"]["text"]
        assert len(sent) <= MAX_MESSAGE_LENGTH
        # Same saturated preview again resolves to identical truncated text —
        # a genuine no-op (the preview was recorded only after the first
        # successful edit), so no second API call.
        adapter._last_edit_at.clear()
        second = _run(adapter.edit_message("user@corp.ru", "10", big + "y"))
        assert second.success is True
        assert client.get.call_count == 1

    def test_finalize_overflow_splits(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_PARSE_MODE="none")
        responses = [
            _FakeResponse({"ok": True}),                       # editText chunk 1
            _FakeResponse({"ok": True, "msgId": "11"}),        # sendText chunk 2
            _FakeResponse({"ok": True, "msgId": "12"}),        # sendText chunk 3
        ]
        client = _wire_client(adapter, responses)
        big = "word " * 2000
        result = _run(adapter.edit_message("user@corp.ru", "10", big, finalize=True))
        assert result.success is True
        assert result.message_id in ("11", "12")
        assert "10" in result.continuation_message_ids
        assert client.get.call_count == len(
            [c for c in client.get.call_args_list]
        )

    def test_edit_not_modified_is_success(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        _wire_client(adapter, [
            _FakeResponse({"ok": False, "description": "message not modified"}),
        ])
        result = _run(
            adapter.edit_message("user@corp.ru", "10", "same", finalize=True)
        )
        assert result.success is True


# ---------------------------------------------------------------------------
# 7. Inbound: newMessage normalization
# ---------------------------------------------------------------------------


def _new_message_payload(**overrides):
    payload = {
        "msgId": "57883346846815032",
        "chat": {"chatId": "user@corp.ru", "type": "private"},
        "from": {"userId": "user@corp.ru", "firstName": "Ivan", "lastName": "Petrov"},
        "timestamp": 1546290000,
        "text": "привет",
    }
    payload.update(overrides)
    return payload


class TestInbound:

    def test_dispatches_message_event(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        _run(adapter._on_new_message(_new_message_payload()))
        event = adapter.handle_message.call_args.args[0]
        assert event.text == "привет"
        assert event.message_type == MessageType.TEXT
        assert event.source.chat_id == "user@corp.ru"
        assert event.source.chat_type == "dm"
        assert event.source.user_name == "Ivan Petrov"
        assert event.message_id == "57883346846815032"

    def test_group_chat_type(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        payload = _new_message_payload(
            chat={"chatId": f"685{GROUP_CHAT_SUFFIX}", "type": "group", "title": "DevOps"},
        )
        _run(adapter._on_new_message(payload))
        source = adapter.handle_message.call_args.args[0].source
        assert source.chat_type == "group"
        assert source.chat_name == "DevOps"

    def test_own_messages_skipped(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._bot_user_id = "bot@corp.ru"
        adapter.handle_message = AsyncMock()
        payload = _new_message_payload()
        payload["from"] = {"userId": "bot@corp.ru"}
        _run(adapter._on_new_message(payload))
        adapter.handle_message.assert_not_called()

    def test_duplicates_skipped(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        payload = _new_message_payload()
        _run(adapter._on_new_message(payload))
        _run(adapter._on_new_message(payload))
        assert adapter.handle_message.call_count == 1

    def test_reply_part_maps_context(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._bot_user_id = "bot@corp.ru"
        adapter.handle_message = AsyncMock()
        payload = _new_message_payload(parts=[{
            "type": "reply",
            "payload": {"message": {
                "msgId": "111", "text": "original",
                "from": {"userId": "bot@corp.ru", "firstName": "Hermes"},
            }},
        }])
        _run(adapter._on_new_message(payload))
        event = adapter.handle_message.call_args.args[0]
        assert event.reply_to_message_id == "111"
        assert event.reply_to_text == "original"
        assert event.reply_to_is_own_message is True

    def test_mention_rewritten(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        payload = _new_message_payload(
            text="@[user2@corp.ru] глянь",
            parts=[{
                "type": "mention",
                "payload": {"userId": "user2@corp.ru", "firstName": "Petr"},
            }],
        )
        _run(adapter._on_new_message(payload))
        assert adapter.handle_message.call_args.args[0].text == "@Petr глянь"

    def test_empty_message_skipped(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        _run(adapter._on_new_message(_new_message_payload(text="")))
        adapter.handle_message.assert_not_called()

    def test_poll_once_advances_cursor(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        _wire_client(adapter, [_FakeResponse({"ok": True, "events": [
            {"eventId": 41, "type": "newMessage", "payload": _new_message_payload()},
            {"eventId": 42, "type": "editedMessage", "payload": {}},
        ]})])

        async def _drive():
            await adapter._poll_once()
            # Event handling is offloaded to background tasks; drain them.
            pending = list(adapter._background_tasks)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        _run(_drive())
        # Cursor advances synchronously (before dispatch), acks stay ordered.
        assert adapter._last_event_id == 42
        assert adapter.handle_message.call_count == 1

    def test_poll_auth_failure_is_fatal(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        _wire_client(adapter, [
            _FakeResponse({"ok": False, "description": "Invalid token"}),
        ])
        with pytest.raises(_vkteams._FatalPollError):
            _run(adapter._poll_once())
        assert adapter.has_fatal_error is True


class TestConnectionLifecycle:

    def test_connect_is_idempotent(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Simulate a live adapter with a running poll task + open client.
        disconnect = AsyncMock()
        adapter.disconnect = disconnect

        async def _fake_self_get(*a, **k):
            return {"userId": "bot@corp.ru"}, None, None

        async def _drive():
            live = asyncio.create_task(asyncio.sleep(3600))
            adapter._poll_task = live
            with patch.object(adapter, "_api_get", _fake_self_get), \
                 patch.object(_vkteams.httpx, "AsyncClient", return_value=MagicMock()), \
                 patch.object(adapter, "_poll_loop", new=lambda: asyncio.sleep(0)):
                ok = await adapter.connect(is_reconnect=True)
            live.cancel()
            return ok

        ok = _run(_drive())
        assert ok is True
        # The live session was recycled before a new one was started.
        disconnect.assert_awaited_once()

    def test_disconnect_skips_awaiting_current_task(self, monkeypatch):
        """disconnect() invoked from within the poll task must not await it."""
        adapter = _make_adapter(monkeypatch)
        adapter._http_client = None

        async def _drive():
            async def _body():
                adapter._poll_task = asyncio.current_task()
                await adapter.disconnect()  # must return, not deadlock
                return "done"
            return await asyncio.wait_for(_body(), timeout=2.0)

        assert _run(_drive()) == "done"


# ---------------------------------------------------------------------------
# 8. Callback queries (inline buttons)
# ---------------------------------------------------------------------------


def _callback_payload(data, user_id="user@corp.ru"):
    return {
        "queryId": "q-1",
        "callbackData": data,
        "from": {"userId": user_id, "firstName": "Ivan"},
        "message": {
            "msgId": "500",
            "chat": {"chatId": "user@corp.ru", "type": "private"},
            "text": "prompt",
        },
    }


class TestCallbacks:

    def test_exec_approval_resolves(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_ALLOW_ALL_USERS="true")
        _wire_client(adapter, [
            _FakeResponse({"ok": True}),  # answerCallbackQuery
            _FakeResponse({"ok": True}),  # editText (prompt replacement)
        ])
        adapter._register_prompt_state(adapter._approval_state, 3, "session-key-1")
        resolve = MagicMock(return_value=1)
        with patch("tools.approval.resolve_gateway_approval", resolve):
            _run(adapter._on_callback_query(_callback_payload("ea:once:3")))
        resolve.assert_called_once_with("session-key-1", "once")
        assert 3 not in adapter._approval_state

    def test_stale_approval_answers_gracefully(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_ALLOW_ALL_USERS="true")
        client = _wire_client(adapter, [_FakeResponse({"ok": True})])
        _run(adapter._on_callback_query(_callback_payload("ea:once:99")))
        params = client.get.call_args.kwargs["params"]
        assert params["queryId"] == "q-1"
        assert "already been resolved" in params["text"]

    def test_unauthorized_tap_denied(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_ALLOWED_USERS="boss@corp.ru")
        client = _wire_client(adapter, [_FakeResponse({"ok": True})])
        adapter._register_prompt_state(adapter._approval_state, 3, "session-key-1")
        resolve = MagicMock(return_value=1)
        with patch("tools.approval.resolve_gateway_approval", resolve):
            _run(adapter._on_callback_query(
                _callback_payload("ea:once:3", user_id="intruder@corp.ru")
            ))
        resolve.assert_not_called()
        assert 3 in adapter._approval_state  # untouched — still resolvable
        assert "not authorized" in client.get.call_args.kwargs["params"]["text"]

    def test_clarify_choice_resolves(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_ALLOW_ALL_USERS="true")
        _wire_client(adapter, [
            _FakeResponse({"ok": True}),  # answerCallbackQuery
            _FakeResponse({"ok": True}),  # editText
        ])
        adapter._register_prompt_state(adapter._clarify_state, "cid", "session-key-2")
        entry = MagicMock()
        entry.choices = ["red", "green"]
        resolve = MagicMock(return_value=True)
        with patch.dict("tools.clarify_gateway._entries", {"cid": entry}, clear=False), \
             patch("tools.clarify_gateway.resolve_gateway_clarify", resolve):
            _run(adapter._on_callback_query(_callback_payload("cl:cid:1")))
        resolve.assert_called_once_with("cid", "green")
        assert "cid" not in adapter._clarify_state

    def test_clarify_other_flips_to_text_capture(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, VKTEAMS_ALLOW_ALL_USERS="true")
        _wire_client(adapter, [
            _FakeResponse({"ok": True}), _FakeResponse({"ok": True}),
        ])
        adapter._register_prompt_state(adapter._clarify_state, "cid", "session-key-2")
        mark = MagicMock(return_value=True)
        with patch("tools.clarify_gateway.mark_awaiting_text", mark):
            _run(adapter._on_callback_query(_callback_payload("cl:cid:other")))
        mark.assert_called_once_with("cid")
        # State stays until the typed answer arrives.
        assert "cid" in adapter._clarify_state


# ---------------------------------------------------------------------------
# 9. Prompt senders store button state
# ---------------------------------------------------------------------------


class TestSsrfGuard:

    def test_send_image_blocks_unsafe_url(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = _wire_client(adapter, [_FakeResponse({"ok": True, "msgId": "9"})])
        with patch("tools.url_safety.is_safe_url", return_value=False):
            result = _run(adapter.send_image(
                "user@corp.ru", "http://169.254.169.254/latest/meta-data/",
            ))
        # Falls back to base behavior: posts the URL as text, never fetches it.
        assert result.success is True
        params = client.get.call_args.kwargs["params"]
        # URL is posted as text (MarkdownV2 may escape the dots).
        assert params["text"].replace("\\", "").endswith("169.254.169.254/latest/meta-data/")
        # The internal URL was NOT downloaded and NOT uploaded as a file.
        client.post.assert_not_called()

    def test_send_image_safe_url_uploads(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._send_file_multipart = AsyncMock(
            return_value=_vkteams.SendResult(success=True, message_id="9")
        )
        adapter._read_local_file = AsyncMock(return_value=b"\xff\xd8\xff")
        with patch("tools.url_safety.is_safe_url", return_value=True), \
             patch.object(_vkteams, "cache_image_from_url",
                          AsyncMock(return_value="/cache/img_abc.jpg")):
            result = _run(adapter.send_image("user@corp.ru", "https://cdn.example/x.jpg"))
        assert result.success is True
        adapter._send_file_multipart.assert_awaited_once()


class TestPromptStateEviction:

    def test_expired_entries_pruned_on_register(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Insert a stale entry directly, then register a fresh one.
        adapter._approval_state[1] = ("old", 0.0)  # created long ago (t=0)
        adapter._register_prompt_state(adapter._approval_state, 2, "fresh")
        assert 1 not in adapter._approval_state  # pruned by TTL
        assert adapter._approval_state[2][0] == "fresh"

    def test_size_cap_drops_oldest(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        cap = _vkteams.PROMPT_STATE_MAX_SIZE
        # Fill past the cap with recent timestamps so only the size cap applies.
        for i in range(cap + 10):
            adapter._register_prompt_state(adapter._clarify_state, f"c{i}", f"s{i}")
        assert len(adapter._clarify_state) <= cap


class TestPromptSenders:

    def test_send_exec_approval_registers_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = _wire_client(adapter, [_FakeResponse({"ok": True, "msgId": "600"})])
        result = _run(adapter.send_exec_approval(
            "user@corp.ru", "rm -rf /tmp/x", "session-key-9",
        ))
        assert result.success is True
        assert [v[0] for v in adapter._approval_state.values()] == ["session-key-9"]
        keyboard = json.loads(
            client.get.call_args.kwargs["params"]["inlineKeyboardMarkup"]
        )
        datas = [b["callbackData"] for row in keyboard for b in row]
        approval_id = list(adapter._approval_state.keys())[0]
        assert f"ea:once:{approval_id}" in datas
        assert f"ea:deny:{approval_id}" in datas

    def test_send_clarify_with_choices(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = _wire_client(adapter, [_FakeResponse({"ok": True, "msgId": "601"})])
        result = _run(adapter.send_clarify(
            "user@corp.ru", "Which?", ["red", "green"], "cid-1", "session-key-3",
        ))
        assert result.success is True
        assert adapter._clarify_state["cid-1"][0] == "session-key-3"
        params = client.get.call_args.kwargs["params"]
        assert "1. red" in params["text"]
        keyboard = json.loads(params["inlineKeyboardMarkup"])
        datas = [b["callbackData"] for row in keyboard for b in row]
        assert "cl:cid-1:0" in datas and "cl:cid-1:other" in datas

    def test_send_slash_confirm_registers_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        _wire_client(adapter, [_FakeResponse({"ok": True, "msgId": "602"})])
        result = _run(adapter.send_slash_confirm(
            "user@corp.ru", "Reload", "Reload MCP servers?", "session-key-4", "cf-1",
        ))
        assert result.success is True
        assert adapter._slash_confirm_state["cf-1"][0] == "session-key-4"


# ---------------------------------------------------------------------------
# 10. get_chat_info / typing
# ---------------------------------------------------------------------------


class TestChatInfoAndTyping:

    def test_private_chat_maps_to_dm(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        _wire_client(adapter, [_FakeResponse({
            "ok": True, "type": "private", "firstName": "Ivan", "lastName": "Petrov",
        })])
        info = _run(adapter.get_chat_info("user@corp.ru"))
        assert info["type"] == "dm"
        assert info["name"] == "Ivan Petrov"

    def test_getinfo_failure_falls_back_to_suffix(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        _wire_client(adapter, [_FakeResponse({"ok": False, "description": "boom"})])
        info = _run(adapter.get_chat_info(f"685{GROUP_CHAT_SUFFIX}"))
        assert info["type"] == "group"

    def test_typing_throttled_within_window(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = _wire_client(adapter, [
            _FakeResponse({"ok": True}), _FakeResponse({"ok": True}),
        ])
        _run(adapter.send_typing("user@corp.ru"))
        _run(adapter.send_typing("user@corp.ru"))
        assert client.get.call_count == 1
        params = client.get.call_args.kwargs["params"]
        assert params["actions"] == "typing"

    def test_stop_typing_clears_throttle(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = _wire_client(adapter, [
            _FakeResponse({"ok": True}), _FakeResponse({"ok": True}),
            _FakeResponse({"ok": True}),
        ])
        _run(adapter.send_typing("user@corp.ru"))
        _run(adapter.stop_typing("user@corp.ru"))
        _run(adapter.send_typing("user@corp.ru"))
        assert client.get.call_count == 3


# ---------------------------------------------------------------------------
# 11. Plugin shape: register / env enablement / standalone send
# ---------------------------------------------------------------------------


class _FakeCtx:
    def __init__(self):
        self.kwargs = None

    def register_platform(self, **kwargs):
        self.kwargs = kwargs


class TestPluginShape:

    def test_register_contract(self):
        ctx = _FakeCtx()
        register(ctx)
        kw = ctx.kwargs
        assert kw["name"] == "vkteams"
        assert kw["required_env"] == ["VKTEAMS_BOT_TOKEN"]
        assert kw["allowed_users_env"] == "VKTEAMS_ALLOWED_USERS"
        assert kw["allow_all_env"] == "VKTEAMS_ALLOW_ALL_USERS"
        assert kw["cron_deliver_env_var"] == "VKTEAMS_HOME_CHANNEL"
        assert kw["max_message_length"] == MAX_MESSAGE_LENGTH
        assert callable(kw["adapter_factory"])
        assert callable(kw["standalone_sender_fn"])
        assert callable(kw["env_enablement_fn"])

    def test_registry_entry_constructs(self):
        """The kwargs must match PlatformEntry's schema exactly."""
        from gateway.platform_registry import PlatformEntry
        ctx = _FakeCtx()
        register(ctx)
        kw = dict(ctx.kwargs)
        entry = PlatformEntry(source="plugin", **kw)
        assert entry.name == "vkteams"

    def test_env_enablement_requires_token(self, monkeypatch):
        monkeypatch.delenv("VKTEAMS_BOT_TOKEN", raising=False)
        assert _env_enablement() is None

    def test_env_enablement_seeds_extra(self, monkeypatch):
        monkeypatch.setenv("VKTEAMS_BOT_TOKEN", TOKEN)
        monkeypatch.setenv("VKTEAMS_API_BASE", "https://onprem.corp/bot/v1/")
        monkeypatch.setenv("VKTEAMS_HOME_CHANNEL", f"685{GROUP_CHAT_SUFFIX}")
        monkeypatch.delenv("VKTEAMS_HOME_CHANNEL_NAME", raising=False)
        seed = _env_enablement()
        assert seed["token"] == TOKEN
        assert seed["api_base"] == "https://onprem.corp/bot/v1"
        assert seed["home_channel"]["chat_id"] == f"685{GROUP_CHAT_SUFFIX}"

    def test_standalone_send_without_token(self, monkeypatch):
        monkeypatch.delenv("VKTEAMS_BOT_TOKEN", raising=False)
        result = _run(_standalone_send(
            PlatformConfig(enabled=True, extra={}), "user@corp.ru", "hi",
        ))
        assert "error" in result

    def test_standalone_send_success(self, monkeypatch):
        monkeypatch.delenv("VKTEAMS_BOT_TOKEN", raising=False)
        pconfig = PlatformConfig(enabled=True, extra={"token": TOKEN})

        client = MagicMock()
        client.get = AsyncMock(return_value=_FakeResponse({"ok": True, "msgId": "800"}))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch.object(_vkteams.httpx, "AsyncClient", return_value=client):
            result = _run(_standalone_send(pconfig, "user@corp.ru", "cron ping"))
        assert result.get("success") is True
        assert result["message_id"] == "800"
        params = client.get.call_args.kwargs["params"]
        assert params["chatId"] == "user@corp.ru"
        assert params["token"] == TOKEN
