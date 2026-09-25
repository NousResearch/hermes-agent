"""Operator-only notices never reach a chat whose readers are not the operator (#107899).

A personal-account platform (the WhatsApp linked-device bridge) answers the operator's contacts,
and a customer-facing bot answers customers. ``display.third_party_chat`` marks those platforms:
the agent may legitimately end a turn silently, and machinery written for the operator —
"⚡ Interrupting current task", "The model returned only a silence marker… Try again or rephrase" —
must not be posted into someone else's thread.
"""

from pathlib import Path

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.display_config import chat_readers_are_third_party, resolve_display_setting
from gateway.platforms.event import MessageEvent
from gateway.response_filters import is_machinery_display_kind
from gateway.run_busy import _busy_ack_suppressed
from gateway.run_turn import _silence_allowed_for_turn
from gateway.session import SessionSource

WHATSAPP = Platform.WHATSAPP.value
TELEGRAM = Platform.TELEGRAM.value


def _source(platform: Platform = Platform.WHATSAPP) -> SessionSource:
    return SessionSource(platform=platform, chat_id="905551112233", chat_type="dm", user_id="905551112233")


def _event(platform: Platform = Platform.WHATSAPP) -> MessageEvent:
    return MessageEvent(text="are you free at 12?", source=_source(platform), message_id="msg-1")


def _pin_home(monkeypatch, tmp_path: Path, yaml_text: str | None = None) -> Path:
    """Point the gateway's config reader at a temp home (the real loader, no mocks)."""
    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    if yaml_text is not None:
        (home / "config.yaml").write_text(yaml_text, encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setattr(gateway_run, "get_hermes_home_override", lambda: None)
    return home


# ── The setting itself ──────────────────────────────────────────────────────


def test_a_personal_account_platform_is_third_party_without_configuration():
    """The linked-device bridge is never the operator talking to themselves."""
    assert resolve_display_setting({}, WHATSAPP, "third_party_chat") is True
    assert resolve_display_setting({}, TELEGRAM, "third_party_chat") is False


def test_a_per_platform_override_beats_the_platform_default():
    cfg = {"display": {"platforms": {WHATSAPP: {"third_party_chat": False}}}}
    assert resolve_display_setting(cfg, WHATSAPP, "third_party_chat") is False


def test_the_global_setting_covers_every_platform():
    cfg = {"display": {"third_party_chat": True}}
    assert resolve_display_setting(cfg, TELEGRAM, "third_party_chat") is True
    assert chat_readers_are_third_party(cfg, TELEGRAM) is True


def test_the_setting_reaches_the_turn_path_through_the_gateway_config_loader(monkeypatch, tmp_path):
    """Real loader, real config file: whatever the operator writes is what the turn path sees."""
    _pin_home(monkeypatch, tmp_path, "display:\n  platforms:\n    whatsapp:\n      third_party_chat: false\n")
    assert chat_readers_are_third_party(gateway_run._load_gateway_config(), WHATSAPP) is False

    _pin_home(monkeypatch, tmp_path, "display:\n  third_party_chat: true\n")
    assert chat_readers_are_third_party(gateway_run._load_gateway_config(), TELEGRAM) is True


# ── Silence markers ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("platform", "display_kind", "expected"),
    [
        (Platform.WHATSAPP, None, True),  # a contact's chat: the agent may say nothing
        (Platform.TELEGRAM, "internal_notification", True),  # machinery, unchanged behavior
        (Platform.TELEGRAM, None, False),  # an addressed message to the operator stays visible
    ],
)
def test_silence_markers_are_allowed_for_machinery_and_third_party_chats(platform, display_kind, expected):
    assert _silence_allowed_for_turn(_source(platform), display_kind) is expected


def test_an_operator_can_mark_any_platform_third_party(monkeypatch, tmp_path):
    _pin_home(monkeypatch, tmp_path, "display:\n  platforms:\n    telegram:\n      third_party_chat: true\n")
    assert _silence_allowed_for_turn(_source(Platform.TELEGRAM), None) is True


def test_a_readable_config_does_not_change_machinery_silence(monkeypatch, tmp_path):
    _pin_home(monkeypatch, tmp_path)
    assert _silence_allowed_for_turn(_source(Platform.TELEGRAM), "internal_notification") is True
    assert is_machinery_display_kind("internal_notification") is True


# ── Busy acknowledgements ───────────────────────────────────────────────────


def test_busy_ack_is_suppressed_for_a_third_party_chat(monkeypatch, tmp_path):
    _pin_home(monkeypatch, tmp_path)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raising=False)
    assert _busy_ack_suppressed(_event(Platform.WHATSAPP)) is True
    assert _busy_ack_suppressed(_event(Platform.TELEGRAM)) is False


def test_the_global_kill_switch_still_suppresses_every_platform(monkeypatch, tmp_path):
    _pin_home(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    assert _busy_ack_suppressed(_event(Platform.TELEGRAM)) is True


def test_an_opt_back_in_restores_operator_notices_on_a_third_party_platform(monkeypatch, tmp_path):
    """``third_party_chat: false`` is a real opt-out, not a one-way switch."""
    _pin_home(monkeypatch, tmp_path, "display:\n  platforms:\n    whatsapp:\n      third_party_chat: false\n")
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raising=False)
    assert _busy_ack_suppressed(_event(Platform.WHATSAPP)) is False
    assert _silence_allowed_for_turn(_source(Platform.WHATSAPP), None) is False
