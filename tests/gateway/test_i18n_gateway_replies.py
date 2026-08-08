"""Tests for gateway i18n — behaviour under non-English language catalogs.

Verifies that gateway reply strings translated via t() produce the expected
locale output and fall back to English when a key is missing.

Coverage:
  - _gateway_provider_error_reply (auth, policy, rate-limit, generic)
  - /stop and /goal slash handlers
  - Voice-channel handler t() reachable from a clean runner
  - Topic-mode handler t() reachable from a clean runner
  - Busy-reject, steer, activity, and contract-draft key lookups
"""

from unittest.mock import MagicMock

import pytest

from agent import i18n
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_event(text="/stop", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
        chat_type="dm",
    )
    return MessageEvent(text=text, source=source, message_id="m1")


def _make_runner():
    """Bare GatewayRunner — just enough surface for t()-emitting handlers."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()
    runner._pending_approvals = {}
    runner._pending_messages = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner.config = MagicMock()
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = False
    runner.session_store = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = MagicMock()
    runner.hooks.loaded_hooks = False
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner.onboarding_store = MagicMock()
    runner._paired_users_data = {}
    return runner


def _set_lang(monkeypatch, lang: str):
    monkeypatch.setenv("HERMES_LANGUAGE", lang)
    i18n.reset_language_cache()


# ---------------------------------------------------------------------------
# _gateway_provider_error_reply
# ---------------------------------------------------------------------------


class TestGatewayProviderErrorReply:
    """_gateway_provider_error_reply maps raw error text → t("gateway.provider_*")."""

    def test_auth_error_english(self, monkeypatch):
        _set_lang(monkeypatch, "en")
        from gateway.run import _gateway_provider_error_reply
        result = _gateway_provider_error_reply("Provider authentication failed: bad key")
        assert "authentication failed" in result
        assert "credentials" in result.lower()

    def test_auth_error_spanish(self, monkeypatch):
        _set_lang(monkeypatch, "es")
        from gateway.run import _gateway_provider_error_reply
        result = _gateway_provider_error_reply("Provider authentication failed: bad key")
        assert "autenticación" in result.lower()
        assert "credenciales" in result.lower()

    def test_policy_rejection_english(self, monkeypatch):
        _set_lang(monkeypatch, "en")
        from gateway.run import _gateway_provider_error_reply
        result = _gateway_provider_error_reply("HTTP 403 content policy violation")
        assert "rejected" in result.lower()
        assert "rephrasing" in result.lower()

    def test_policy_rejection_french(self, monkeypatch):
        _set_lang(monkeypatch, "fr")
        from gateway.run import _gateway_provider_error_reply
        result = _gateway_provider_error_reply("HTTP 403 content policy violation")
        assert "rejeté" in result.lower()
        assert "reformulez" in result.lower()

    def test_rate_limit_english(self, monkeypatch):
        _set_lang(monkeypatch, "en")
        from gateway.run import _gateway_provider_error_reply
        result = _gateway_provider_error_reply("Rate limited after 3 retries")
        assert "rate-limiting" in result.lower()

    def test_rate_limit_japanese(self, monkeypatch):
        _set_lang(monkeypatch, "ja")
        from gateway.run import _gateway_provider_error_reply
        result = _gateway_provider_error_reply("Rate limited after 3 retries")
        assert "制限" in result

    def test_generic_failure_english(self, monkeypatch):
        _set_lang(monkeypatch, "en")
        from gateway.run import _gateway_provider_error_reply
        result = _gateway_provider_error_reply("Some unknown API error occurred")
        assert "failed after retries" in result.lower()

    def test_generic_failure_chinese(self, monkeypatch):
        _set_lang(monkeypatch, "zh")
        from gateway.run import _gateway_provider_error_reply
        result = _gateway_provider_error_reply("Some unknown API error occurred")
        assert "失败" in result

    def test_missing_key_falls_back_to_english(self, monkeypatch, tmp_path):
        fake_locales = tmp_path / "locales"
        fake_locales.mkdir()
        (fake_locales / "en.yaml").write_text(
            "gateway:\n  provider_failed: 'EN fallback'\n", encoding="utf-8"
        )
        (fake_locales / "xx.yaml").write_text(
            "gateway:\n  provider_failed: ''\n", encoding="utf-8"
        )
        monkeypatch.setattr(i18n, "_locales_dir", lambda: fake_locales)
        _set_lang(monkeypatch, "xx")
        from gateway.run import _gateway_provider_error_reply
        result = _gateway_provider_error_reply("HTTP 500 server error")
        assert result == "EN fallback"


# ---------------------------------------------------------------------------
# slash command handlers
# ---------------------------------------------------------------------------


class TestSlashCommandI18n:
    """Representative slash handlers that emit t()-translated replies."""

    @pytest.mark.asyncio
    async def test_stop_no_active_spanish(self, monkeypatch):
        _set_lang(monkeypatch, "es")
        runner = _make_runner()
        runner._running_agents = {}
        event = _make_event(text="/stop")
        result = await runner._handle_stop_command(event)
        assert "activa" in result.lower()

    @pytest.mark.asyncio
    async def test_stop_no_active_english(self, monkeypatch):
        _set_lang(monkeypatch, "en")
        runner = _make_runner()
        runner._running_agents = {}
        event = _make_event(text="/stop")
        result = await runner._handle_stop_command(event)
        assert "no active" in result.lower()

    @pytest.mark.asyncio
    async def test_goal_none_set_german(self, monkeypatch):
        _set_lang(monkeypatch, "de")
        runner = _make_runner()
        event = _make_event(text="/goal clear")
        result = await runner._handle_goal_command(event)
        assert "kein" in result.lower()
        assert "ziel" in result.lower()


# ---------------------------------------------------------------------------
# voice-channel handlers (t() reachable through a bare runner)
# ---------------------------------------------------------------------------


class TestVoiceHandlerI18n:
    """Voice-channel gateway replies accessible from a clean runner."""

    @pytest.mark.asyncio
    async def test_voice_unsupported_platform_spanish(self, monkeypatch):
        _set_lang(monkeypatch, "es")
        runner = _make_runner()
        event = _make_event(platform=Platform.DISCORD)
        result = await runner._handle_voice_channel_join(event)
        assert "compatibles" in result.lower()

    @pytest.mark.asyncio
    async def test_voice_unsupported_platform_english(self, monkeypatch):
        _set_lang(monkeypatch, "en")
        runner = _make_runner()
        event = _make_event(platform=Platform.DISCORD)
        result = await runner._handle_voice_channel_join(event)
        assert "not supported" in result.lower()


# ---------------------------------------------------------------------------
# topic-mode handler (t() reachable through a bare runner)
# ---------------------------------------------------------------------------


class TestTopicI18n:
    """Topic-mode gateway replies."""

    @pytest.mark.asyncio
    async def test_topic_not_telegram_french(self, monkeypatch):
        _set_lang(monkeypatch, "fr")
        runner = _make_runner()
        event = _make_event(platform=Platform.DISCORD)
        result = await runner._handle_topic_command(event)
        assert "telegram" in result.lower()


# ---------------------------------------------------------------------------
# busy-reject / steer / activity / contract-draft key lookups
# ---------------------------------------------------------------------------


class TestNewI18nKeys:
    """Direct t() calls for the gateway keys added in this PR.
    These prove each key resolves to a translated non-empty string in
    every tested locale, confirming catalog parity and translation
    correctness without needing the full busy-session dispatch machinery."""

    # -- busy_reject --

    @pytest.mark.parametrize("lang,key,expected_substring", [
        ("en", "gateway.busy_reject.model",    "Agent is running"),
        ("en", "gateway.busy_reject.moa",      "Agent is running"),
        ("en", "gateway.busy_reject.generic",  "/model"),
        ("es", "gateway.busy_reject.model",    "agente"),
        ("es", "gateway.busy_reject.generic",  "/model"),  # placeholder formatted
        ("de", "gateway.busy_reject.generic",  "Agent"),
        ("fr", "gateway.busy_reject.codex_runtime", "exécution"),
        ("pt", "gateway.busy_reject.model",    "execução"),
        ("ru", "gateway.busy_reject.moa",      "занят"),
        ("ja", "gateway.busy_reject.generic",  "実行"),
    ])
    def test_busy_reject_keys(self, monkeypatch, lang, key, expected_substring):
        _set_lang(monkeypatch, lang)
        result = i18n.t(key, name="model")
        assert expected_substring.lower() in result.lower()

    # -- steer --

    @pytest.mark.parametrize("lang,key,expected_substring", [
        ("en", "gateway.steer.usage_no_agent", "Usage: /steer"),
        ("es", "gateway.steer.usage_no_agent", "Uso:"),
        ("de", "gateway.steer.usage_no_agent", "Verwendung"),
    ])
    def test_steer_usage_keys(self, monkeypatch, lang, key, expected_substring):
        _set_lang(monkeypatch, lang)
        result = i18n.t(key)
        assert expected_substring.lower() in result.lower()

    # -- activity --

    @pytest.mark.parametrize("lang,key", [
        ("en", "gateway.activity.still_on_it"),
        ("en", "gateway.activity.one_sec"),
        ("ko", "gateway.activity.one_sec"),
        ("it", "gateway.activity.still_on_it"),
        ("zh", "gateway.activity.one_sec"),
    ])
    def test_activity_keys_non_empty(self, monkeypatch, lang, key):
        _set_lang(monkeypatch, lang)
        result = i18n.t(key)
        assert len(result) > 0
        # English baseline: exact text
        if lang == "en" and "still_on_it" in key:
            assert result == "still on it"
        if lang == "en" and "one_sec" in key:
            assert result == "one sec"

    # -- goal.contract_draft_failed --

    @pytest.mark.parametrize("lang,key,expected_substring", [
        ("en", "gateway.goal.contract_draft_failed", "draft"),
        ("es", "gateway.goal.contract_draft_failed", "contrato"),
        ("de", "gateway.goal.contract_draft_failed", "Vertrag"),
        ("ru", "gateway.goal.contract_draft_failed", "контракт"),
    ])
    def test_contract_draft_failed_keys(self, monkeypatch, lang, key, expected_substring):
        _set_lang(monkeypatch, lang)
        result = i18n.t(key)
        assert expected_substring.lower() in result.lower()
