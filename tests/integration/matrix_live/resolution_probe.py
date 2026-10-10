"""Observe the receiving adapter's real fetch and crypto boundaries in Linux."""

from __future__ import annotations

import asyncio
import gc
import json
from contextvars import ContextVar
from urllib.parse import unquote

from gateway.platforms.base import BasePlatformAdapter
from hermes_constants import get_hermes_home
from plugins.platforms.matrix.reply_context import MatrixEventContext


def _write_json(path, value):
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value), encoding="utf-8")
    partial.replace(path)


def _observe(adapter, home):
    active = ContextVar("matrix_resolution_probe", default=False)
    cache = adapter._event_context_cache
    client = adapter._client
    request = client.api.request
    decrypt = client.crypto.decrypt_megolm_event
    get_session = client.crypto.crypto_store.get_group_session
    replacement_decrypted = False
    reaction_page = []

    def config():
        path = home / "resolution-config.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    async def pause(event_id):
        if (home / "resolution-started.json").exists():
            return
        store = client.crypto.crypto_store
        _write_json(home / "resolution-started.json", {
            "event_id": event_id,
            "reaction_page": reaction_page,
            "adapter_module": type(adapter).__module__,
            "user_id": adapter._user_id,
            "home": str(home),
            "store_type": f"{type(store).__module__}.{type(store).__qualname__}",
        })
        async with asyncio.timeout(8):
            while not (home / "resolution-release").exists():
                await asyncio.sleep(0.01)

    async def observed_request(method, path, *args, **kwargs):
        nonlocal reaction_page
        response = await request(method, path, *args, **kwargs)
        settings = config() if active.get() else None
        if settings and settings["barrier"] == "reaction" and str(path).endswith("/m.annotation"):
            reaction_page = [event["event_id"] for event in response["chunk"]]
        if (
            settings
            and settings["barrier"] == "fetch"
            and unquote(str(path)).endswith(f"/event/{settings['target']}")
        ):
            assert response["event_id"] == settings["target"], response
            await pause(settings["target"])
        return response

    async def observed_decrypt(event):
        nonlocal replacement_decrypted
        settings = config() if active.get() else None
        if settings:
            paused_id = (
                settings["target"]
                if settings["barrier"] == "original"
                else settings["replacement"]
            )
            if (
                settings["barrier"] in {"original", "replacement", "reaction"}
                and event.event_id == paused_id
            ):
                await pause(paused_id)
        result = await decrypt(event)
        if settings and event.event_id == settings["replacement"]:
            replacement_decrypted = True
            if settings["barrier"] == "relation":
                from plugins.platforms.matrix.effective_event import event_content

                store = client.crypto.crypto_store
                _write_json(home / "resolution-decrypted.json", {
                    "event_id": str(event.event_id),
                    "relation": event_content(result).get("m.relates_to"),
                    "user_id": adapter._user_id,
                    "home": str(home),
                    "store_type": f"{type(store).__module__}.{type(store).__qualname__}",
                })
        return result

    async def observed_session(*args, **kwargs):
        settings = config() if active.get() else None
        if settings and settings["barrier"] == "store" and replacement_decrypted:
            await pause(settings["replacement"])
        return await get_session(*args, **kwargs)

    def scoped(operation, *, is_read, clear=True):
        async def observed(*args, **kwargs):
            if not is_read and args[0] is not adapter:
                return await operation(*args, **kwargs)
            settings = config()
            if not settings or (settings["scope"] == "event") != is_read:
                return await operation(*args, **kwargs)
            if clear and not (home / "resolution-started.json").exists():
                cache._entries.clear()
                cache.max_entries = 2
                gc.collect()
            token = active.set(True)
            try:
                return await operation(*args, **kwargs)
            finally:
                active.reset(token)

        return observed

    client.api.request = observed_request
    client.crypto.decrypt_megolm_event = observed_decrypt
    client.crypto.crypto_store.get_group_session = observed_session
    adapter.read_matrix_context = scoped(adapter.read_matrix_context, is_read=True)
    type(adapter).fetch_inbound_context = scoped(type(adapter).fetch_inbound_context, is_read=False)
    type(adapter).prepare_turn_context = scoped(type(adapter).prepare_turn_context, is_read=False, clear=False)


def register(ctx):
    original_init = BasePlatformAdapter.__init__

    def observed_init(adapter, *args, **kwargs):
        original_init(adapter, *args, **kwargs)
        if adapter.platform.value != "matrix":
            return
        home = get_hermes_home()
        connect = adapter.connect
        redact = adapter._on_redaction
        message = adapter._on_room_message

        async def observed_message(event):
            config = home / "resolution-config.json"
            if config.exists() and json.loads(config.read_text(encoding="utf-8"))["scope"] != "event":
                # Catch-up and thread history run only where the gate requires a mention.
                adapter._free_rooms.discard(str(event.room_id))
            await message(event)
            path = home / "resolution-observed-events.json"
            seen = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            seen.append(str(event.event_id))
            _write_json(path, seen)

        async def observed_redact(event):
            await redact(event)
            path = home / "resolution-config.json"
            if not path.exists():
                return
            settings = json.loads(path.read_text(encoding="utf-8"))
            if str(event.redacts) != settings["withdraw"]:
                return
            cache = adapter._event_context_cache
            if settings.get("eviction", True):
                for index in range(cache.max_entries):
                    cache.store(
                        str(event.room_id),
                        f"$resolution-pressure{index}",
                        MatrixEventContext("", "unrelated"),
                    )
                gc.collect()
            _write_json(home / "resolution-redaction.json", {
                "event_id": str(event.redacts),
                "room_id": str(event.room_id),
                "entries": len(cache._entries),
                "limit": cache.max_entries,
            })

        async def observed_connect(*connect_args, **connect_kwargs):
            result = await connect(*connect_args, **connect_kwargs)
            assert adapter._client.crypto is not None
            _observe(adapter, home)
            return result

        adapter.connect = observed_connect
        adapter._on_redaction = observed_redact
        adapter._on_room_message = observed_message

    BasePlatformAdapter.__init__ = observed_init
