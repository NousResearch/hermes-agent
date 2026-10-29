"""The receiving Telegram bot's callbacks must render under that bot's locale."""
from types import MethodType, SimpleNamespace

import pytest

from agent.i18n import reset_language_cache, t
from gateway.run import _profile_runtime_scope
from hermes_constants import get_hermes_home
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
async def test_confirmation_button_uses_own_profile_language(tmp_path):
    launch, family = tmp_path / "launch", tmp_path / "family"
    for home, lang in ((launch, "en"), (family, "ru")):
        home.mkdir()
        (home / "config.yaml").write_text(f"display:\n  language: {lang}\n")
    reset_language_cache()
    adapter = object.__new__(TelegramAdapter)
    adapter._update_receipt_dir = family  # captured at construction time by the real adapter
    adapter._accept_update = lambda: None
    adapter._callback_ctx = lambda query: {}
    outcomes = []

    async def confirmed(self, query, data, ctx):
        outcomes.append((get_hermes_home(), t("platform.telegram.slash_confirm.resolved_once")))

    adapter._handle_slash_confirm_callback = MethodType(confirmed, adapter)
    try:
        with _profile_runtime_scope(launch):
            await adapter._handle_callback_query(
                SimpleNamespace(callback_query=SimpleNamespace(data="sc:once:synthetic")), None
            )
            assert get_hermes_home() == launch
        assert len(outcomes) == 1
        assert outcomes[0][0] == family
        assert "Одобрено" in outcomes[0][1]
    finally:
        reset_language_cache()


@pytest.mark.asyncio
async def test_picker_button_auth_and_response_stay_in_own_profile(tmp_path):
    launch, family = tmp_path / "launch", tmp_path / "family"
    for home, lang in ((launch, "en"), (family, "ru")):
        home.mkdir()
        (home / "config.yaml").write_text(f"display:\n  language: {lang}\n")
    reset_language_cache()
    adapter = object.__new__(TelegramAdapter)
    adapter._update_receipt_dir = family
    adapter._accept_update = lambda: None
    adapter._callback_ctx = lambda query: {}
    observed = []

    async def authorized(self, query, ctx, denial_text):
        observed.append(("auth", get_hermes_home()))
        return True

    async def picker(self, query, data, chat_id):
        observed.append(("picker", get_hermes_home()))

    adapter._callback_authorized = MethodType(authorized, adapter)
    adapter._handle_choice_picker_callback = MethodType(picker, adapter)
    query = SimpleNamespace(data="cp:synthetic", message=SimpleNamespace(chat_id=42))
    try:
        with _profile_runtime_scope(launch):
            await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)
            assert get_hermes_home() == launch
        assert observed == [("auth", family), ("picker", family)]
    finally:
        reset_language_cache()
