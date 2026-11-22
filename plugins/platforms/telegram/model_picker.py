"""Telegram model-picker provider drill-down."""

import asyncio
from contextlib import suppress

from agent.i18n import t
from plugins.platforms.telegram.callback_answer import TOAST_LIMIT, answer_callback


async def show_provider_models(adapter, query, chat_id: str, state: dict, slug: str) -> None:
    provider = next((row for row in state["providers"] if row["slug"] == slug), None)
    if provider is None:
        await answer_callback(query, t("platform.telegram.picker.provider_not_found")[:TOAST_LIMIT])
        return

    # Codex hides account-gated models until discovery is fresh. Wait before publishing
    # model-index callbacks so the buttons and their indices refer to the same catalog.
    reload_provider = state.get("reload_provider")
    if slug == "openai-codex" and reload_provider is not None:
        refreshed = None
        with suppress(Exception):
            refreshed = await asyncio.to_thread(reload_provider, slug)
        if adapter._model_picker_state.get(chat_id) is not state:
            await answer_callback(query, t("platform.telegram.picker.expired_rerun")[:TOAST_LIMIT])
            return
        if refreshed is not None:
            provider.update(refreshed)

    state["selected_provider"] = slug
    state["selected_provider_name"] = provider.get("name", slug)
    state["model_list"] = provider.get("models", [])
    await adapter._picker_show_models(query, state, 0)
