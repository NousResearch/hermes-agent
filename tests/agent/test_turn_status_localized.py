"""Turn retry/fallback status lines: chat shows the active language, matchers keep reading English.

The producers emit ``tl()`` values whose ``str`` is English; the gateway status filter matches that English
(noise suppression, provider-error rewrite) and only then renders the active language.
"""

import pytest

from agent import i18n
from agent.status_output import StatusOutputMixin
from agent.turn_recovery import activate_codex_app_server_fallback, compute_error_backoff
from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message
from hermes_constants import get_hermes_home

_DE_FALLBACK = "⚠️ Ratenlimit erreicht — Wechsel zum Fallback-Anbieter..."
_DE_RETRY = "⏳ Neuer Versuch in {wait}s (Versuch {attempt}/{max})..."


class _Agent(StatusOutputMixin):
    log_prefix = ""
    suppress_status_output = True
    platform = "telegram"
    thinking_callback = None

    def __init__(self):
        self.statuses = []
        self.status_callback = lambda kind, text: self.statuses.append((kind, text))
        self.activity = []

    def _has_pending_fallback(self):
        return True

    def _try_activate_fallback(self, **_kwargs):
        return False

    def _touch_activity(self, desc, **_kwargs):
        self.activity.append(desc)

    def _client_log_context(self):
        return ""


@pytest.fixture
def german_overlay(monkeypatch):
    overlay = get_hermes_home() / "locales"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "de.yaml").write_text(
        f'"display.status.turn.fallback_rate_limited": "{_DE_FALLBACK}"\n'
        f'"display.status.turn.retry_wait": "{_DE_RETRY}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_LANGUAGE", "de")
    i18n.reset_language_cache()
    yield
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n.reset_language_cache()


def test_turn_status_reaches_chat_localized_while_matchers_read_english(german_overlay):
    agent = _Agent()
    assert activate_codex_app_server_fallback(agent, {"error": "Too many requests, please slow down"}) is False
    compute_error_backoff(
        agent, RuntimeError("upstream 503"), retry_count=1, max_retries=3, is_rate_limited=False,
        is_zai_coding_overload=False, base_url="https://example.invalid/v1", model="m",
    )
    agent._flush_status_buffer()

    [(fallback_kind, fallback), (retry_kind, retry)] = agent.statuses
    # The value other code matches stays English...
    assert fallback == "⚠️ Rate limited — switching to fallback provider..."
    assert retry.startswith("⏳ Retrying in ")
    # ...a deliverable line reaches the chat in the active language...
    assert _prepare_gateway_status_message(Platform.TELEGRAM, fallback_kind, fallback) == _DE_FALLBACK
    # ...and retry chatter is still recognised as noise from its English, whatever the language.
    assert _prepare_gateway_status_message(Platform.TELEGRAM, retry_kind, retry) is None
    assert i18n.render_localized(retry).startswith("⏳ Neuer Versuch in ")
    assert all(not str(a).startswith("⏳ Neuer") for a in agent.activity)  # stored activity stays English
