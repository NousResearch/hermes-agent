"""Retain a committed model pick if a delayed post-run write saw the old route."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_messaging_session_picks import _compressed_chat


@pytest.mark.asyncio
async def test_postrun_sync_cannot_overwrite_a_pick_that_committed_after_its_read(tmp_path, monkeypatch):
    runner, _source, _route, viewer, _root, _fences = await _compressed_chat(tmp_path)
    store = runner.persistence
    sid = "seg-1"
    store.sessions.update_model(sid, "original", provider="openrouter")
    entered, release = threading.Event(), threading.Event()
    replace = store.sessions.replace_session_config

    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5), "post-run write was not released"
        return replace(*args, **kwargs)

    monkeypatch.setattr(store.sessions, "replace_session_config", paused)
    agent = SimpleNamespace(model="turn-fallback", provider="openrouter", requested_provider="openrouter",
                            base_url="https://openrouter.ai/api/v1", api_mode="chat_completions")
    task = asyncio.create_task(asyncio.to_thread(runner._sync_session_model_from_agent, sid, agent))
    try:
        assert await asyncio.to_thread(entered.wait, 5), "post-run sync did not read the route"
        await asyncio.to_thread(store.sessions.update_model, sid, "explicit-pick", provider="openrouter")
        release.set()
        await asyncio.wait_for(task, 5)
        row = store.sessions.get(sid)
        assert row.model == "explicit-pick"
        assert row.model_config["gateway_runtime"]["model"] == "explicit-pick"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("intervening_pick", [False, True])
async def test_postrun_sync_respects_the_turn_start_config(tmp_path, intervening_pick):
    runner, _source, _route, viewer, _root, _ = await _compressed_chat(tmp_path)
    store = runner.persistence
    sid = "seg-1"
    try:
        store.sessions.update_model(sid, "original", provider="openrouter")
        fences = runner._capture_model_sync_fences(sid)
        assert [fence.session_id for fence in fences] == [_root, sid]
        if intervening_pick:
            store.sessions.update_model(sid, "explicit-pick", provider="openrouter")
        agent = SimpleNamespace(model="turn-fallback", provider="openrouter", requested_provider="openrouter",
                                base_url="https://openrouter.ai/api/v1", api_mode="chat_completions")
        await asyncio.to_thread(runner._sync_session_model_from_agent, sid, agent, turn_fences=fences)
        expected = "explicit-pick" if intervening_pick else "turn-fallback"
        row = store.sessions.get(sid)
        assert row.model == expected
        assert row.model_config["gateway_runtime"]["model"] == expected
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_postrun_sync_preserves_a_different_saved_route_on_the_new_tip(tmp_path):
    from persistence.conversation.lineage import CompressionChild
    from persistence.conversation.messages import NewMessage

    runner, _source, _route, viewer, _root, _ = await _compressed_chat(tmp_path)
    store = runner.persistence
    try:
        store.sessions.update_model("seg-1", "original", provider="openrouter")
        fences = runner._capture_model_sync_fences("seg-1")
        store.lineage.publish_compression_child("seg-1", CompressionChild("seg-2", "telegram"),
            [NewMessage.from_live({"role": "user", "content": "carried"})], require_compression_lease=False)
        store.sessions.update_model("seg-2", "explicit-pick", provider="openrouter")
        agent = SimpleNamespace(model="turn-fallback", provider="openrouter", requested_provider="openrouter",
                                base_url="https://openrouter.ai/api/v1", api_mode="chat_completions")
        await asyncio.to_thread(runner._sync_session_model_from_agent, "seg-2", agent, turn_fences=fences)
        assert store.sessions.get("seg-2").model == "explicit-pick"
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()
