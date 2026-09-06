"""Which title stage is allowed to spend a platform rename.

Titling is two-stage: a derived slice of the user's own words lands inline, and
the model's version replaces it a moment later. A local sidebar wants both. A
Discord thread or a Telegram topic wants only the second — renaming twice lands
on the same name at twice the cost, and Discord allows two channel renames per
ten minutes, so the throwaway can be the one that survives.
"""

from __future__ import annotations

import types

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner


def _attach(lane):
    """Attach the title callback for *lane* and return (callback, renames)."""
    renames: list = []
    source = types.SimpleNamespace(platform=Platform.DISCORD, chat_id="chan-1")

    runner = types.SimpleNamespace(
        _is_telegram_topic_lane=lambda src: lane == "telegram",
        _is_discord_auto_thread_lane=lambda src: lane == "discord",
        _is_relay_discord_channel_lane=lambda src: False,
        _schedule_telegram_topic_title_rename=(
            lambda src, sid, title: renames.append(title)
        ),
        _schedule_discord_semantic_thread_rename=(
            lambda src, sid, title: renames.append(title)
        ),
    )
    holder = types.SimpleNamespace(
        _runner=runner,
        _attach_session_title_callback=TurnRunner._attach_session_title_callback,
    )
    agent = types.SimpleNamespace(session_id="sess-1")
    holder._attach_session_title_callback(
        holder, agent, types.SimpleNamespace(source=source)
    )
    return agent._on_session_title, renames


@pytest.mark.parametrize("lane", ["telegram", "discord"])
def test_the_rename_waits_for_the_model_title(lane):
    callback, renames = _attach(lane)

    callback("fix the flaky auth test in log", "derived")
    assert renames == []

    callback("Fix flaky auth test", "llm")
    assert renames == ["Fix flaky auth test"]


@pytest.mark.asyncio
async def test_native_thread_rename_passes_only_the_initial_name_guard():
    """The shared rename lane must honor the strict native adapter contract."""
    calls: list[tuple[str, str, str | None]] = []

    class StrictNativeAdapter:
        async def rename_thread(
            self,
            thread_id: str,
            name: str,
            *,
            only_if_current_name: str | None = None,
        ) -> bool:
            calls.append((thread_id, name, only_if_current_name))
            return True

    class NativeRenameRunner:
        _is_discord_auto_thread_lane = GatewayRunner._is_discord_auto_thread_lane
        _sanitize_discord_thread_title = GatewayRunner._sanitize_discord_thread_title
        _rename_discord_auto_thread_for_session_title = (
            GatewayRunner._rename_discord_auto_thread_for_session_title
        )

        def __init__(self, adapter):
            self.adapters = {Platform.DISCORD: adapter}

        def _adapter_for_source(self, source):
            return self.adapters[source.platform]

    source = types.SimpleNamespace(
        platform=Platform.DISCORD,
        chat_id="999",
        chat_type="thread",
        thread_id="999",
        auto_thread_created=True,
        auto_thread_initial_name="Initial words",
    )

    runner = NativeRenameRunner(StrictNativeAdapter())
    await runner._rename_discord_auto_thread_for_session_title(
        source,
        "session-1",
        "Semantic Session Title",
    )

    assert calls == [("999", "Semantic Session Title", "Initial words")]
