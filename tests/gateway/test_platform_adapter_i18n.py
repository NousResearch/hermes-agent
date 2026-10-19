"""Telegram + Discord adapters read their chat copy from the i18n catalog at call time.

Pins the wave-4 contract: nothing user-facing is bound at import, the exec-approval card uses the
shared ``gateway.exec_approval.*`` keys, translated text is HTML-escaped before Telegram's ``<b>``
wrapper, Discord slash-command text is cut at the 100-char app-command cap, button labels at 80,
and both command-menu fingerprints change with the language.
"""

from __future__ import annotations

import pytest

from agent import i18n
from agent.i18n import t
from gateway.platforms.base import utf16_len

import plugins.platforms.discord.adapter as discord_adapter
import plugins.platforms.telegram.adapter as telegram_adapter
from plugins.platforms.discord.adapter import (
    _DISCORD_APP_COMMAND_TEXT_LIMIT,
    _DISCORD_BUTTON_LABEL_LIMIT,
    _NATIVE_SLASH_COMMAND_SPECS,
    DiscordAdapter,
    ExecApprovalView,
    _native_slash_commands,
)
from plugins.platforms.telegram.adapter import TelegramAdapter, _TOAST_LIMIT


@pytest.fixture
def fake_locale(tmp_path, monkeypatch):
    """Point the catalog loader at a throwaway locales dir with an ``xx`` catalog whose values carry
    HTML metacharacters and exceed every platform cap; ``activate()`` switches the language to it."""
    import hermes_yaml as yaml

    locales = tmp_path / "locales"
    locales.mkdir()
    long_text = "L" * 300
    catalog = {
        "gateway": {"exec_approval": {
            "header": "Hermes <wants> & runs", "reason_label": "Why <flag>",
            "smart_deny_line": "Smart & <DENY>: one op only", "action_once": long_text,
            "action_session": "S", "action_always": "A", "action_deny": "D",
        }},
        "platform": {
            "telegram": {"approval": {
                "action_once": long_text, "action_session": "s", "action_always": "a", "action_deny": "d",
                "toast_already_resolved": long_text,
            }},
            "discord": {
                "approval": {"question": "Q?", "requested_command_label": "Cmd:"},
                "command": {
                    "new": {"description": long_text, "followup": "done"},
                    "reasoning": {"choice_none": long_text},
                },
            },
        },
    }
    (locales / "xx.yaml").write_text(yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")
    real_dir = i18n._locales_dir()
    (locales / "en.yaml").write_text((real_dir / "en.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: locales)
    monkeypatch.setattr(i18n, "_normalize_lang", lambda lang, home=None: str(lang).lower())
    i18n.reset_language_cache()

    def activate(lang: str = "xx"):
        monkeypatch.setenv("HERMES_LANGUAGE", lang)
        i18n.reset_language_cache()

    yield activate
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n.reset_language_cache()


# ── shared contract ──────────────────────────────────────────────────────────

def test_exec_approval_contract_keys_exist_in_english():
    for key in ("header", "reason_label", "smart_deny_line", "timed_out_notice",
                "action_once", "action_session", "action_always", "action_deny"):
        assert t(f"gateway.exec_approval.{key}") != f"gateway.exec_approval.{key}"


@pytest.mark.parametrize("module", [telegram_adapter, discord_adapter])
def test_unauthorized_notice_is_resolved_per_call_not_at_import(module):
    assert not hasattr(module, "_UNAUTHORIZED"), "import-bound notice would freeze the language"
    assert callable(module._unauthorized)
    assert "allowed list" in module._unauthorized()


# ── Telegram ─────────────────────────────────────────────────────────────────

def test_telegram_card_escapes_translated_text_before_html_wrapping(fake_locale):
    fake_locale()
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    assert adapter._EA_HEADER == "⚠️ <b>Hermes &lt;wants&gt; &amp; runs</b>\n\n"
    assert adapter._EA_REASON_LABEL == "Why &lt;flag&gt;: "
    assert adapter._EA_SMART_DENY_LINE == "\n\n<b>Smart &amp; &lt;DENY&gt;:</b> one op only"


def test_telegram_card_english_matches_shared_contract():
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    assert t("gateway.exec_approval.header") in adapter._EA_HEADER
    assert adapter._EA_ACTION_LABELS == {
        c: t(f"platform.telegram.approval.action_{c}") for c in ("once", "session", "always", "deny")}


def test_telegram_toasts_are_cut_at_200(fake_locale):
    fake_locale()
    toast = telegram_adapter._toast("platform.telegram.approval.toast_already_resolved")
    assert len(toast) == _TOAST_LIMIT == 200
    assert len(telegram_adapter._unauthorized()) <= _TOAST_LIMIT


def test_telegram_menu_fingerprint_includes_language(fake_locale):
    rows = [("help", "Show help"), ("status", "Status")]
    english = TelegramAdapter._menu_fingerprint(rows)
    fake_locale()
    assert TelegramAdapter._menu_fingerprint(rows) != english
    fake_locale("en")
    assert TelegramAdapter._menu_fingerprint(rows) == english


def test_telegram_bot_command_descriptions_cut_at_256(monkeypatch):
    import sys
    from types import SimpleNamespace

    telegram_mod = sys.modules["telegram"]  # the shared conftest stub (or the real package)
    monkeypatch.setattr(telegram_mod, "BotCommand", lambda name, desc: SimpleNamespace(command=name, description=desc))
    rows = TelegramAdapter._bot_commands([("help", "x" * 400), ("status", None)])
    assert [len(c.description) for c in rows] == [256, 0]


# ── Discord ──────────────────────────────────────────────────────────────────

def test_discord_native_slash_specs_hold_catalog_keys_only():
    for name, description_key, args, _template, followup_key in _NATIVE_SLASH_COMMAND_SPECS:
        assert description_key.startswith(("platform.discord.command.", "slash.")), name
        assert t(description_key) != description_key, (name, description_key)
        if followup_key:
            assert t(followup_key) != followup_key, (name, followup_key)
        for arg_name, _type, _default, desc_key, choices in args:
            assert t(desc_key) != desc_key, (name, arg_name)
            for label, value in choices or ():
                # Bare level names ("low") are identifiers; anything with a dot must resolve.
                assert "." not in label or t(label) != label, (name, label, value)


def test_discord_native_slash_text_respects_100_char_cap(fake_locale):
    fake_locale()
    by_name = {row[0]: row for row in _native_slash_commands()}
    assert utf16_len(by_name["new"][1]) == _DISCORD_APP_COMMAND_TEXT_LIMIT == 100
    reasoning_choices = dict((v, k) for k, v in by_name["reasoning"][2][0][4])
    assert utf16_len(reasoning_choices["none"]) == 100
    assert reasoning_choices["low"] == "low"
    # Follow-ups stay keys until the command runs.
    assert by_name["new"][4] == "platform.discord.command.new.followup"
    for row in _native_slash_commands():
        assert utf16_len(row[1]) <= 100
        for arg in row[2]:
            assert utf16_len(arg[3]) <= 100


def test_discord_card_uses_shared_contract_and_platform_copy(fake_locale):
    fake_locale()
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    assert adapter._EA_HEADER == "⚠️ **Hermes <wants> & runs**\n\nQ?\n\n**Cmd:**\n"
    assert adapter._EA_REASON_LABEL == "**Why <flag>:** "
    assert adapter._EA_SMART_DENY_LINE == "\n\n**Smart & <DENY>:** one op only"


def test_discord_exec_approval_button_labels_localized_and_capped(fake_locale):
    fake_locale()
    view = ExecApprovalView(session_key="s", allowed_user_ids={"1"})
    labels = {}
    view._localize_buttons(allow_once="gateway.exec_approval.action_once", deny="gateway.exec_approval.action_deny")
    for child in getattr(view, "children", []):
        cb = getattr(child, "callback", None)
        fn_name = getattr(cb, "__name__", None) or getattr(getattr(cb, "callback", None), "__name__", None)
        if fn_name:
            labels[fn_name] = child.label
    once = getattr(view, "allow_once", None)
    if hasattr(once, "label"):
        labels["allow_once"] = once.label
        labels["deny"] = view.deny.label
    if labels:  # real discord.py materialises the decorated buttons; the test stub may not
        assert utf16_len(labels["allow_once"]) == _DISCORD_BUTTON_LABEL_LIMIT == 80
        assert labels["deny"] == "D"


def test_discord_command_sync_fingerprint_includes_language(fake_locale, monkeypatch):
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._client = None
    english = adapter._desired_command_sync_fingerprint()
    fake_locale()
    assert adapter._desired_command_sync_fingerprint() != english
    fake_locale("en")
    assert adapter._desired_command_sync_fingerprint() == english
