"""Bounded image-pack discovery and native sticker selections for Matrix sessions."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Awaitable

from gateway.config import Platform
from gateway.session_context import get_session_env, get_session_transport
from hermes_constants import hermes_home_key
from plugins.platforms.matrix.read_context import _raw_event, _read_access
from plugins.platforms.matrix.room_inspection import _content, room_permissions

STANDARD = "m.room.image_pack"
LEGACY = "im.ponies.room_emotes"
PRIVATE = "im.ponies.user_emotes"
REFERENCES = {"m.image_pack.rooms": STANDARD, "im.ponies.emote_rooms": LEGACY}
MAX_PACKS = 20
MAX_ITEMS = 100
MAX_REFERENCES = 20
MAX_STATE_EVENTS = 1000
MAX_SELECTIONS = 256
SELECTION_TTL = 300.0
_MXC = re.compile(
    r"mxc://(?:[a-zA-Z0-9.-]+|\[[0-9a-fA-F:.]+\])(?::[0-9]{1,5})?/[^/?#\s]+\Z"
)


class PackError(ValueError):
    pass


def _safe_json(value: Any, *, depth: int = 0, budget: list[int] | None = None) -> None:
    if budget is None:
        budget = [256]
    budget[0] -= 1
    if budget[0] < 0 or depth > 4:
        raise PackError("image metadata exceeds the catalog budget")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, str) and len(value) <= 1200:
        return
    if isinstance(value, list):
        for item in value:
            _safe_json(item, depth=depth + 1, budget=budget)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 256:
                raise PackError("image metadata has an invalid key")
            _safe_json(item, depth=depth + 1, budget=budget)
        return
    raise PackError("image metadata is not bounded JSON")


def _usage(value: Any) -> tuple[str, ...]:
    if value is None or value == []:
        return ("emoticon", "sticker")
    if (
        not isinstance(value, list)
        or len(value) > 2
        or any(item not in ("emoticon", "sticker") for item in value)
    ):
        raise PackError("pack usage is malformed")
    return tuple(value)


def _info(value: Any, max_bytes: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PackError("image info must be an object")
    _safe_json(value)
    for key in ("w", "h", "size"):
        number = value.get(key)
        if key in value and (
            not isinstance(number, int) or isinstance(number, bool) or number < 0
        ):
            raise PackError(f"image {key} must be a non-negative integer")
    if value.get("size", 0) > max_bytes:
        raise PackError("image exceeds the Matrix media limit")
    mime = value.get("mimetype")
    if mime is not None and (
        not isinstance(mime, str) or not re.fullmatch(r"image/[a-zA-Z0-9.+-]+", mime)
    ):
        raise PackError("image MIME type is invalid")
    if "thumbnail_url" in value and not _valid_mxc(value["thumbnail_url"]):
        raise PackError("image thumbnail URL is not MXC")
    if "thumbnail_info" in value:
        _info(value["thumbnail_info"], max_bytes)
    return value


def _valid_mxc(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= 1200
        and _MXC.fullmatch(value) is not None
    )


def _compression_tip(
    session_store: Any, session_key: str, session_id: str
) -> str | None:
    db = session_store._db_for_key(session_key)
    return db.get_compression_tip(session_id) if db is not None else None


@dataclass(frozen=True)
class PackSource:
    source: str
    event_type: str
    room_id: str | None = None
    state_key: str | None = None
    reference_type: str | None = None

    def identity(self, bot: str) -> dict[str, Any]:
        return {
            "source": self.source,
            "event_type": self.event_type,
            "room_id": self.room_id,
            "state_key": self.state_key,
            "account_user_id": bot if self.source != "room" else None,
        }


@dataclass(frozen=True)
class PackImage:
    shortcode: str
    content_json: str
    fingerprint: str

    @classmethod
    def parse(
        cls, shortcode: str, raw: Any, pack: dict, event_type: str, max_bytes: int
    ) -> PackImage | None:
        if not isinstance(shortcode, str) or not shortcode or len(shortcode) > 256:
            raise PackError("image shortcode is invalid")
        if not isinstance(raw, dict) or not _valid_mxc(raw.get("url")):
            raise PackError("image URL is not MXC")
        usage = (
            _usage(raw.get("usage"))
            if event_type != STANDARD and raw.get("usage")
            else _usage(pack.get("usage"))
        )
        if "sticker" not in usage:
            return None
        body = raw.get("body", shortcode)
        if not isinstance(body, str) or not body or len(body) > 1200:
            raise PackError("image body exceeds the sticker budget or is invalid")
        info = _info(raw.get("info", {}), max_bytes)
        content = {"url": raw["url"], "body": body, "info": info}
        content_json = json.dumps(content, ensure_ascii=False, sort_keys=True)
        if len(content_json.encode("utf-8")) > 8192:
            raise PackError("image metadata exceeds the sticker budget")
        descriptor = {
            key: pack.get(key)
            for key in ("display_name", "description", "attribution", "usage")
        }
        _safe_json(descriptor)
        fingerprint = json.dumps(
            [content, descriptor, usage], ensure_ascii=False, sort_keys=True
        )
        return cls(shortcode, content_json, fingerprint)

    def listed(self, selection_id: str) -> dict[str, Any]:
        return {
            "selection_id": selection_id,
            "shortcode": self.shortcode,
            **json.loads(self.content_json),
        }


@dataclass(frozen=True)
class Selection:
    source: PackSource
    image: PackImage
    home: str
    client: Any
    bot: str
    device: str
    store: Any
    owner_profile: str | None
    room_id: str
    requester: str
    session_key: str
    session_id: str
    expires_at: float
    crypto: Any
    state_store: Any
    owner_key: tuple[Any, ...]


@dataclass(frozen=True)
class PackRequest:
    adapter: Any
    client: Any
    bot: str
    device: str
    store: Any
    owner_profile: str | None
    home: str
    room_id: str
    requester: str
    session_key: str
    session_id: str
    crypto: Any
    state_store: Any
    api: Any
    homeserver: str
    access_token: str | None
    client_device: str | None
    crypto_store: Any
    session_store: Any
    stored_session_id: str | None
    admissions: dict[str, str] = field(default_factory=dict)

    @classmethod
    async def capture(cls, adapter: Any, room_id: str, requester: str) -> PackRequest:
        session_key = get_session_env("HERMES_SESSION_KEY")
        session_id = get_session_env("HERMES_SESSION_ID")
        session_store = getattr(adapter, "_session_store", None)
        stored_session_id = (
            session_store.peek_session_id(session_key)
            if session_store is not None
            else None
        )
        if (
            session_store is not None
            and stored_session_id != session_id
            and (
                not stored_session_id
                or not session_id
                or await asyncio.to_thread(
                    _compression_tip, session_store, session_key, stored_session_id
                )
                != session_id
            )
        ):
            raise PackError("Matrix image-pack conversation changed; list packs again")
        return cls(
            adapter,
            adapter._client,
            adapter._user_id,
            adapter._device_id,
            adapter._store_dir,
            adapter._owner_profile,
            hermes_home_key(),
            room_id,
            requester,
            session_key,
            session_id,
            getattr(adapter._client, "crypto", None),
            getattr(adapter._client, "state_store", None),
            getattr(adapter._client, "api", None),
            str(getattr(getattr(adapter._client, "api", None), "base_url", "")),
            getattr(getattr(adapter._client, "api", None), "token", None),
            getattr(adapter._client, "device_id", None),
            getattr(getattr(adapter._client, "crypto", None), "crypto_store", None),
            session_store,
            stored_session_id,
        )

    def owner_key(self) -> tuple[Any, ...]:
        return (
            id(self.api),
            self.homeserver,
            self.access_token,
            self.client_device,
            id(self.crypto_store),
        )

    def check(self) -> None:
        adapter = self.adapter
        if (
            get_session_transport()[0] is not adapter
            or adapter._closing
            or adapter._client is not self.client
            or not self.bot
            or self.client.mxid != self.bot
            or adapter._user_id != self.bot
            or adapter._device_id != self.device
            or adapter._store_dir != self.store
            or adapter._owner_profile != self.owner_profile
            or hermes_home_key() != self.home
            or getattr(self.client, "crypto", None) is not self.crypto
            or getattr(self.client, "state_store", None) is not self.state_store
            or getattr(self.client, "api", None) is not self.api
            or str(getattr(self.api, "base_url", "")) != self.homeserver
            or getattr(self.api, "token", None) != self.access_token
            or getattr(self.client, "device_id", None) != self.client_device
            or getattr(self.crypto, "crypto_store", None) is not self.crypto_store
            or get_session_env("HERMES_SESSION_PLATFORM") != "matrix"
            or get_session_env("HERMES_SESSION_CHAT_ID") != self.room_id
            or get_session_env("HERMES_SESSION_USER_ID") != self.requester
            or get_session_env("HERMES_SESSION_KEY") != self.session_key
            or not self.session_id
            or get_session_env("HERMES_SESSION_ID") != self.session_id
            or getattr(adapter, "_session_store", None) is not self.session_store
        ):
            raise PackError("Matrix image-pack owner or profile changed")
        if (
            self.session_store is not None
            and self.session_store.peek_session_id(self.session_key)
            != self.stored_session_id
        ):
            raise PackError("Matrix image-pack conversation changed; list packs again")
        for room_id, chat_type in self.admissions.items():
            if (
                room_id not in adapter._joined_rooms
                or adapter._is_sender_authorized(
                    self.requester, chat_type=chat_type, chat_id=room_id
                )
                is not True
                or (
                    adapter._allowed_room_ids
                    and room_id not in adapter._allowed_room_ids
                    and chat_type != "dm"
                )
            ):
                raise PackError("Matrix room admission changed")
        runner = getattr(adapter, "gateway_runner", None)
        if (
            runner is not None
            and runner._adapters_for_profile(self.owner_profile).get(Platform.MATRIX)
            is not adapter
        ):
            raise PackError("Matrix image-pack owner is no longer live")

    async def access(self, room_id: str) -> None:
        self.check()
        client, chat_type, error = await _read_access(
            self.adapter, room_id, self.requester
        )
        self.check()
        if error is not None:
            raise PackError(error["error"])
        if (
            client is not self.client
            or room_id not in self.adapter._joined_rooms
            or self.adapter._is_sender_authorized(
                self.requester, chat_type=chat_type, chat_id=room_id
            )
            is not True
        ):
            raise PackError("Matrix room admission changed")
        assert chat_type is not None
        self.admissions[room_id] = chat_type

    async def read(self, operation: Awaitable[Any]) -> Any:
        try:
            self.check()
        except PackError:
            if asyncio.iscoroutine(operation):
                operation.close()
            raise
        result = await asyncio.wait_for(operation, timeout=10.0)
        self.check()
        await self.access(self.room_id)
        return result

    async def account(self, event_type: str) -> dict:
        try:
            raw = await self.read(self.client.get_account_data(event_type))
        except Exception as exc:
            if (
                getattr(exc, "errcode", None) == "M_NOT_FOUND"
                or type(exc).__name__ == "MNotFound"
            ):
                return {}
            raise
        if not isinstance(raw, dict):
            raise PackError("account pack data must be an object")
        return raw

    async def state(
        self,
        room_id: str,
        event_type: str,
        state_key: str = "",
        *,
        full_event: bool = False,
    ) -> Any:
        try:
            return await self.read(
                self.client.get_state_event(
                    room_id,
                    event_type,
                    state_key,
                    **({"format": "event"} if full_event else {}),
                )
            )
        except Exception as exc:
            self.check()
            if (
                getattr(exc, "errcode", None) == "M_NOT_FOUND"
                or type(exc).__name__ == "MNotFound"
            ):
                await self.access(self.room_id)
                return {}
            raise

    async def pack(self, source: PackSource) -> dict:
        if source.room_id is None:
            return await self.account(source.event_type)
        await self.access(source.room_id)
        if source.reference_type:
            references = await self.account(source.reference_type)
            keys = (
                references.get("rooms", {}).get(source.room_id)
                if isinstance(references.get("rooms"), dict)
                else None
            )
            if not isinstance(keys, dict) or source.state_key not in keys:
                raise PackError(
                    "Matrix account pack reference changed; list packs again"
                )
        if source.state_key is None:
            raise PackError("Matrix room pack has no state key")
        value = await self.state(source.room_id, source.event_type, source.state_key)
        await self.access(source.room_id)
        if source.reference_type:
            references = await self.account(source.reference_type)
            rooms = references.get("rooms")
            keys = rooms.get(source.room_id) if isinstance(rooms, dict) else None
            if not isinstance(keys, dict) or source.state_key not in keys:
                raise PackError(
                    "Matrix account pack reference changed; list packs again"
                )
        return _content(value)


def _selections(adapter: Any) -> OrderedDict[str, Selection]:
    if not hasattr(adapter, "_image_pack_selections"):
        adapter._image_pack_selections = OrderedDict()
    records = adapter._image_pack_selections
    now = time.monotonic()
    for key, selection in list(records.items()):
        if selection.expires_at <= now:
            del records[key]
    return records


@dataclass
class Catalog:
    request: PackRequest
    packs: list[dict]
    errors: list[dict]
    truncated: bool = False
    item_count: int = 0
    inspected_images: int = 0
    inspected_packs: int = 0

    def error(self, error: dict) -> None:
        if len(self.errors) < MAX_ITEMS:
            self.errors.append(error)
            return
        self.truncated = True

    def add(self, source: PackSource, content: dict) -> None:
        if self.inspected_packs >= MAX_PACKS or self.inspected_images >= MAX_ITEMS:
            self.truncated = True
            return
        self.inspected_packs += 1
        pack = content.get("pack", {})
        images = content.get("images")
        if not isinstance(pack, dict) or not isinstance(images, dict):
            self.error({
                **source.identity(self.request.bot),
                "error": "pack or images is malformed",
            })
            return
        try:
            _usage(pack.get("usage"))
        except PackError as exc:
            self.error({**source.identity(self.request.bot), "error": str(exc)})
            return
        metadata = {}
        for key, limit in (
            ("display_name", 256),
            ("description", 1200),
            ("attribution", 1200),
        ):
            value = pack.get(key)
            if value is not None and (not isinstance(value, str) or len(value) > limit):
                self.error({
                    **source.identity(self.request.bot),
                    "error": f"pack {key} exceeds the catalog budget or is invalid",
                })
                return
            metadata[key] = value
        items = []
        budget = MAX_ITEMS - self.inspected_images
        for shortcode, raw in islice(images.items(), budget):
            self.inspected_images += 1
            try:
                image = PackImage.parse(
                    shortcode,
                    raw,
                    pack,
                    source.event_type,
                    self.request.adapter._max_media_bytes,
                )
            except PackError as exc:
                self.error({**source.identity(self.request.bot), "error": str(exc)})
                continue
            if image is None:
                continue
            key = uuid.uuid4().hex
            records = _selections(self.request.adapter)
            records[key] = Selection(
                source,
                image,
                self.request.home,
                self.request.client,
                self.request.bot,
                self.request.device,
                self.request.store,
                self.request.owner_profile,
                self.request.room_id,
                self.request.requester,
                self.request.session_key,
                self.request.session_id,
                time.monotonic() + SELECTION_TTL,
                self.request.crypto,
                self.request.state_store,
                self.request.owner_key(),
            )
            while len(records) > MAX_SELECTIONS:
                records.popitem(last=False)
            items.append(image.listed(key))
            self.item_count += 1
        self.truncated |= len(images) > budget
        self.packs.append({
            **source.identity(self.request.bot),
            **metadata,
            "items": items,
        })

    async def room_state(self) -> list:
        request = self.request
        try:
            state = await request.read(request.client.get_state(request.room_id))
        except PackError:
            raise
        except Exception as exc:
            request.check()
            await request.access(request.room_id)
            self.error({
                "source": "room",
                "error": f"room state read failed: {type(exc).__name__}",
            })
            return []
        if not isinstance(state, list) or len(state) > MAX_STATE_EVENTS:
            self.truncated = True
            self.error({
                "source": "room",
                "error": "Matrix room state exceeds the image-pack discovery budget",
            })
            return []
        return state

    async def discover(self) -> dict:
        request = self.request
        await request.access(request.room_id)
        for event in await self.room_state():
            raw = _raw_event(event)
            event_type, key = raw.get("type"), raw.get("state_key")
            if event_type in (STANDARD, LEGACY):
                if (
                    not isinstance(key, str)
                    or len(key) > 256
                    or raw.get("room_id", request.room_id) != request.room_id
                ):
                    self.error({
                        "source": "room",
                        "error": "pack state identity is malformed",
                    })
                    continue
                self.add(
                    PackSource("room", event_type, request.room_id, key), _content(raw)
                )
        reference_count = 0
        for reference_type, event_type in REFERENCES.items():
            try:
                data = await request.account(reference_type)
                rooms = data.get("rooms", {})
                if not isinstance(rooms, dict):
                    raise PackError("account pack rooms must be an object")
                for room_id, keys in islice(rooms.items(), MAX_REFERENCES):
                    if reference_count >= MAX_REFERENCES:
                        self.truncated = True
                        break
                    if (
                        not isinstance(room_id, str)
                        or len(room_id) > 256
                        or not room_id.startswith("!")
                        or not isinstance(keys, dict)
                    ):
                        self.error({
                            "source": "account_reference",
                            "error": "pack reference is malformed",
                        })
                        reference_count += 1
                        continue
                    remaining = MAX_REFERENCES - reference_count
                    for key in islice(keys, remaining):
                        reference_count += 1
                        if (
                            not isinstance(key, str)
                            or len(key) > 256
                            or not isinstance(keys[key], dict)
                        ):
                            self.error({
                                "source": "account_reference",
                                "error": "pack reference is malformed",
                            })
                            continue
                        source = PackSource(
                            "account_reference",
                            event_type,
                            room_id,
                            key,
                            reference_type,
                        )
                        try:
                            content = await request.pack(source)
                        except PackError as exc:
                            request.check()
                            self.error({
                                "source": "account_reference",
                                "error": str(exc),
                            })
                            continue
                        except Exception as exc:
                            request.check()
                            self.error({
                                "source": "account_reference",
                                "error": f"pack read failed: {type(exc).__name__}",
                            })
                            continue
                        self.add(source, content)
                    self.truncated |= len(keys) > remaining
                self.truncated |= len(rooms) > MAX_REFERENCES
            except PackError as exc:
                request.check()
                self.error({"source": "account_reference", "error": str(exc)})
            except Exception as exc:
                request.check()
                self.error({
                    "source": "account_reference",
                    "error": f"account read failed: {type(exc).__name__}",
                })
        try:
            private = await request.account(PRIVATE)
            if private:
                self.add(PackSource("bot_account", PRIVATE), private)
        except Exception as exc:
            request.check()
            self.error({
                "source": "bot_account",
                "error": str(exc)
                if isinstance(exc, PackError)
                else f"account read failed: {type(exc).__name__}",
            })
        await request.access(request.room_id)
        return {
            "packs": self.packs,
            "errors": self.errors,
            "truncated": self.truncated,
            "untrusted_data": True,
            "account_user_id": request.bot,
        }


async def _send(
    request: PackRequest, selection_id: str, reply_to: str | None, thread_id: str | None
) -> dict:
    selection = _selections(request.adapter).get(selection_id)
    if selection is None or (
        selection.home,
        selection.client,
        selection.bot,
        selection.device,
        selection.store,
        selection.owner_profile,
        selection.room_id,
        selection.requester,
        selection.session_key,
        selection.session_id,
        selection.crypto,
        selection.state_store,
        selection.owner_key,
    ) != (
        request.home,
        request.client,
        request.bot,
        request.device,
        request.store,
        request.owner_profile,
        request.room_id,
        request.requester,
        request.session_key,
        request.session_id,
        request.crypto,
        request.state_store,
        request.owner_key(),
    ):
        raise PackError("Matrix image selection is unavailable; list packs again")
    await request.access(request.room_id)
    content = await request.pack(selection.source)
    pack, images = content.get("pack", {}), content.get("images", {})
    if not isinstance(pack, dict) or not isinstance(images, dict):
        raise PackError("Matrix image pack changed; list packs again")
    fresh = PackImage.parse(
        selection.image.shortcode,
        images.get(selection.image.shortcode),
        pack,
        selection.source.event_type,
        request.adapter._max_media_bytes,
    )
    if fresh is None or fresh.fingerprint != selection.image.fingerprint:
        raise PackError("Matrix image selection changed; list packs again")

    async def permissions() -> bool:
        power = _content(await request.state(request.room_id, "m.room.power_levels"))
        encryption = _content(await request.state(request.room_id, "m.room.encryption"))
        algorithm = encryption.get("algorithm")
        if algorithm is not None and algorithm != "m.megolm.v1.aes-sha2":
            raise PackError("Matrix room encryption algorithm is unsupported")
        create = await request.state(request.room_id, "m.room.create", full_event=True)
        levels = room_permissions(
            power,
            encryption,
            create,
            request.requester,
            request.bot,
            event_type="m.sticker",
        )
        wire_type = levels["required"]["send_event_type"]
        if wire_type == "m.room.encrypted" and request.crypto is None:
            raise PackError("Matrix encryption keys are unavailable")
        if (
            not levels["bot"]["creator_override"]
            and levels["bot"]["level"] < levels["required"]["send_message"]
        ):
            raise PackError(f"Matrix bot cannot send {wire_type} in this room")
        return wire_type == "m.room.encrypted"

    encrypted = await permissions()
    try:
        from mautrix.client.api import ClientAPI
        from mautrix.types import EventType, RoomID
    except ImportError as exc:
        raise PackError("Matrix SDK is unavailable") from exc

    payload = json.loads(fresh.content_json)
    request.adapter._apply_relation_metadata(
        request.room_id, payload, reply_to=reply_to, metadata={"thread_id": thread_id}
    )
    wire_type = EventType.STICKER
    wire_content = payload
    if encrypted:
        wire_content = await request.read(
            request.client.encrypt(RoomID(request.room_id), EventType.STICKER, payload)
        )
        if not await permissions():
            raise PackError("Matrix room encryption changed")
        wire_type = EventType.ROOM_ENCRYPTED
    for _attempt in range(2):
        await request.access(request.room_id)
        if selection.source.room_id and selection.source.room_id != request.room_id:
            await request.access(selection.source.room_id)
        final_pack = await request.pack(selection.source)
        final_images, final_metadata = (
            final_pack.get("images"),
            final_pack.get("pack", {}),
        )
        if not isinstance(final_images, dict) or not isinstance(final_metadata, dict):
            raise PackError("Matrix image pack changed; list packs again")
        final_image = PackImage.parse(
            selection.image.shortcode,
            final_images.get(selection.image.shortcode),
            final_metadata,
            selection.source.event_type,
            request.adapter._max_media_bytes,
        )
        if final_image is None or final_image.fingerprint != fresh.fingerprint:
            raise PackError("Matrix image selection changed; list packs again")
        await request.access(request.room_id)
        if request.state_store is None:
            break
        current_encryption = await asyncio.wait_for(
            request.state_store.is_encrypted(request.room_id), timeout=10.0
        )
        request.check()
        if not current_encryption or wire_type == EventType.ROOM_ENCRYPTED:
            break
        if request.crypto is None:
            raise PackError("Matrix encryption keys are unavailable")
        wire_content = await request.read(
            request.client.encrypt(RoomID(request.room_id), EventType.STICKER, payload)
        )
        if not await permissions():
            raise PackError("Matrix room encryption changed")
        wire_type = EventType.ROOM_ENCRYPTED
    request.check()
    if _selections(request.adapter).get(selection_id) is not selection:
        raise PackError("Matrix image selection expired; list packs again")
    try:
        event_id = await ClientAPI.send_message_event(
            request.client, RoomID(request.room_id), wire_type, wire_content
        )
    except Exception as exc:
        return {"error": str(exc)}
    try:
        request.check()
    except PackError as exc:
        return {
            "success": True,
            "event_id": str(event_id),
            "warning": f"{exc} after the server accepted the sticker",
        }
    request.adapter._thread_fallbacks.remember_sent(
        request.room_id, payload, str(event_id)
    )
    return {"success": True, "event_id": str(event_id)}


async def matrix_image_packs(
    adapter: Any,
    action: str,
    room_id: str,
    *,
    requester: str,
    selection_id: str | None = None,
    reply_to: str | None = None,
    thread_id: str | None = None,
) -> dict:
    try:
        request = await PackRequest.capture(adapter, room_id, requester)
        if action == "list":
            return await Catalog(request, [], []).discover()
        if action == "send" and isinstance(selection_id, str):
            return await _send(request, selection_id, reply_to, thread_id)
        return {"error": "action must be list or send with a selection_id"}
    except PackError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Matrix image-pack request failed: {type(exc).__name__}"}
