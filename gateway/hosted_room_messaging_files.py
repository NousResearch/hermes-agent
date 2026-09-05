"""Files and Full reply consumer for authorized Group Chat messaging owners.

Only the command/menu entry imports this owner. File-source imports stay lazy so
the independent text-only Messaging source retains its existing startup path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone

from agent.i18n import t
from gateway.choice_picker import ChoicePage, ChoiceProgress, PAGE_TIMEOUT_SECONDS, choice_label

_STATUS_KEYS = frozenset({
    "unsupported", "unavailable", "denied", "expired", "empty", "no_match",
    "partial", "getting", "busy", "delivered", "unknown", "failed", "rate",
    "error", "ambiguous", "large",
})
MAX_MENUS = 128
MAX_HISTORY = 8
_RATE_LIMITS = {"read": 30, "send": 6}


def text(key, **values):
    return t("gateway.group_files." + key, **values)


def _status_text(outcome, fallback):
    return text(outcome if outcome in _STATUS_KEYS else fallback)


def label(value, limit=80):
    text = " ".join(
        "".join(
            char
            for char in str(value)
            if unicodedata.category(char) not in {"Cc", "Cf"}
        ).split()
    )
    for char in "\\`*_{}[]<>|":
        text = text.replace(char, "")
    text = text.replace("@", "\uff20")
    if text.startswith("/"):
        text = "\uff0f" + text[1:]
    return text[:limit] or t("gateway.group_files.file")


def size_label(size):
    return (
        f"{size / 1_000_000:.1f} MB"
        if size >= 1_000_000
        else f"{max(1, round(size / 1000))} KB"
    )


def _clip_caption_part(value, limit):
    if len(value) <= limit:
        return value
    if limit <= 1:
        return "\u2026"
    tail = limit // 2
    return value[:limit - tail - 1] + "\u2026" + value[-tail:]


def _error(exc):
    if isinstance(exc, ImportError):
        return text("unavailable")
    code = getattr(exc, "code", str(exc))
    if code == "file_code_ambiguous":
        return text("ambiguous")
    if code in {"file_too_large", "too_large"}:
        return text("large")
    if code == "file_integrity_failed":
        return text("integrity")
    if code == "file_unavailable":
        return text("removed")
    if code == "file_invalid_request":
        return text("invalid")
    if code == "classic_files_on_desktop":
        return text("classic")
    if code in {"file_access_denied", "denied"}:
        return text("denied")
    if code in {"unsupported", "file_access_unsupported"}:
        return text("unsupported")
    if code == "attachment_cursor_reset_required":
        return text("cursor_reset")
    if code == "file_lookup_limit":
        return text("lookup_limit")
    return text("error")


def _source_key(runner, event):
    from gateway.hosted_room_messaging import messaging_transport_profile

    source = event.source
    values = [
        str(getattr(source.platform, "value", source.platform)),
        runner._group_chat_profile(event),
        messaging_transport_profile(event),
    ]
    values.extend(
        str(getattr(source, field, None) or "")
        for field in (
            "chat_id",
            "thread_id",
            "scope_id",
            "guild_id",
            "user_id",
            "user_id_alt",
        )
    )
    home = runner.config.get_home_channel(source.platform)
    values.append(None if home is None else [
        home.platform.value,
        str(home.chat_id),
        str(home.thread_id or ""),
        str(home.user_id or ""),
        str(home.scope_id or ""),
        str(getattr(home, "selection_id", None) or ""),
    ])
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode()
    ).hexdigest()


def _room_key(room):
    return tuple(
        room.get(field)
        for field in (
            "room_id",
            "authority_gateway_id",
            "authority_epoch",
            "_room_mode",
            "_remote_member_id",
        )
    )


def _authorized(runner, event, expected=None, adapter=None):
    from gateway.hosted_room_messaging import (
        is_machine_authored,
        is_message_edit,
        relay_provenance_is_unknown,
    )

    authorize = getattr(runner, "_is_user_authorized_for_source", None)
    if (
        is_machine_authored(event)
        or is_message_edit(event)
        or relay_provenance_is_unknown(event)
        or not callable(authorize)
        or authorize(event.source) is not True
        or not runner._can_control_group_chats(event)
    ):
        raise PermissionError("denied")
    if expected is not None and _source_key(runner, event) != expected:
        raise PermissionError("denied")
    if adapter is not None and runner._adapter_for_source(event.source) is not adapter:
        raise PermissionError("denied")


def _rate(runner, source_key, kind):
    now = time.monotonic()
    buckets = getattr(runner, "_group_file_rates", None)
    if not isinstance(buckets, OrderedDict):
        buckets = runner._group_file_rates = OrderedDict()
    key = (source_key, kind)
    recent = [stamp for stamp in buckets.pop(key, []) if now - stamp < 60]
    if len(recent) >= _RATE_LIMITS[kind]:
        buckets[key] = recent
        return False
    buckets[key] = recent + [now]
    while len(buckets) > 2048:
        buckets.popitem(last=False)
    return True


class FilesMenu:
    def __init__(self, runner, event, backend, command):
        _authorized(runner, event)
        self.runner, self.event, self.backend, self.command = (
            runner,
            event,
            backend,
            command,
        )
        self.source_key = _source_key(runner, event)
        self.adapter = runner._adapter_for_source(event.source)
        self.profile = runner._group_chat_profile(event)
        self.deadline = time.monotonic() + PAGE_TIMEOUT_SECONDS
        self.room = None
        self.reference = ""
        self.query = ""
        self.pages = []
        self.position = -1
        self.actions = {}
        self.revision = 0
        self.long_codes = False
        self.handle = secrets.token_hex(8)

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("menu expired")
        _authorized(self.runner, self.event, self.source_key, self.adapter)

    async def fresh_room(self):
        from gateway.hosted_room_messaging import list_messaging_rooms, resolve_room

        self.check()
        rooms = await asyncio.to_thread(
            list_messaging_rooms, self.backend, profile=self.profile
        )
        self.check()
        if self.room is None:
            return rooms
        current = resolve_room(rooms, self.reference)
        if _room_key(current) != _room_key(self.room):
            raise PermissionError("denied")
        return current

    async def bind(self, reference):
        from gateway.hosted_room_messaging import resolve_room, room_reference

        rooms = await self.fresh_room()
        self.room = resolve_room(rooms, reference)
        self.reference = room_reference(self.room)

    def page(self, title, actions, *, full_width=False):
        self.actions = {}
        self.revision += 1
        choices = []
        for caption, action in actions:
            if self.event.source.platform.value == "telegram":
                icon = {"files": "📎", "bots": "🤖"}.get(action[0])
                if action[0] == "room":
                    icon = "🕘" if caption == text("activity") else "‹"
                if action[0] == "groups":
                    icon, caption = "‹", text("groups")
                if icon:
                    caption = f"{icon} {caption}"
            token = f"{self.handle}:{self.revision}:{len(choices)}"
            self.actions[token] = action
            choices.append({"label": caption, "value": token, "full_width": full_width})
        return ChoicePage(title[:2048], choices)

    def failure(self, exc, action):
        message = _error(exc)
        code = getattr(exc, "code", str(exc))
        try:
            self.check()
        except (PermissionError, TimeoutError):
            return message
        if code in {"denied", "file_access_denied"}:
            return message
        actions = []
        if code not in {
            "unsupported",
            "file_access_unsupported",
            "classic_files_on_desktop",
            "file_too_large",
            "too_large",
            "file_unavailable",
            "file_lookup_limit",
            "attachment_cursor_reset_required",
        }:
            actions.append((text("retry"), action))
        actions.extend([
            (text("show_latest"), ("files", None)),
            (text("back"), ("room", None)),
        ])
        return self.page(message, actions)

    async def room_page(self, detail=None, *, view="room"):
        from gateway.hosted_room_messaging import format_room_detail

        current = await self.fresh_room()
        if detail is None:
            detail = await asyncio.to_thread(
                format_room_detail,
                self.backend,
                current,
                room_command=self.command,
                show_approvals=self.runner._can_approve_group_chats(self.event),
            )
        actions = await self.room_content_actions(current)
        if view != "room":
            actions.append((text("activity"), ("room", None)))
        if view != "bots" and current.get("members"):
            actions.append((text("bots"), ("bots", None)))
        actions.append((text("back_groups"), ("groups", None)))
        await self.fresh_room()
        return self.page(detail, actions)

    async def room_content_actions(self, current):
        """Only offer content known to exist; navigation never waits on delivery."""
        from gateway.hosted_room_file_delivery import native_document_limit
        from gateway.hosted_room_file_lookup import latest_reply

        try:
            native_document_limit(self.adapter, self.event.source)
        except Exception:
            return []

        async def available(function, **kwargs):
            try:
                return await asyncio.wait_for(asyncio.to_thread(function, **kwargs), 2)
            except Exception:
                return None

        catalog, reply = await asyncio.gather(
            available(getattr(self.backend, "list_files", None), room=current, profile=self.profile, limit=1),
            available(lambda **kwargs: latest_reply(self.backend, **kwargs),
                      room=current, profile=self.profile),
        )
        actions = []
        if catalog and (catalog.get("items") or catalog.get("has_more")):
            actions.append((text("files"), ("files", None)))
        if reply:
            actions.append((text("full_reply"), ("reply", None)))
        return actions

    async def files_page(self, cursor=None, *, direction="latest"):
        current = await self.fresh_room()
        if not callable(getattr(self.backend, "list_files", None)):
            raise ImportError("Files backend is not installed")
        page = await asyncio.wait_for(
            asyncio.to_thread(
                self.backend.list_files,
                room=current,
                profile=self.profile,
                cursor=cursor,
                query=self.query,
                limit=8,
            ),
            20,
        )
        await self.fresh_room()
        if direction == "latest":
            self.pages, self.position = [page], 0
        else:
            self.pages = self.pages[: self.position + 1] + [page]
            self.pages = self.pages[-MAX_HISTORY:]
            self.position = len(self.pages) - 1
        return self.render_files()

    def render_files(self, notice=""):
        page = self.pages[self.position]
        actions = [
            (caption, ("file", (dict(item), False, "")))
            for item, caption in zip(page["items"], self._file_labels(page["items"]))
        ]
        if self.position > 0:
            actions.append((text("newer"), ("newer", None)))
        if page["has_more"]:
            actions.append((text("older"), ("older", None)))
        actions.extend([
            (text("search"), ("search", None)), (text("back"), ("room", None))
        ])
        if len(actions) < 12:
            actions.append((text("show_latest"), ("files", None)))
        empty = (
            text("partial")
            if page["has_more"]
            else text("no_match")
            if self.query
            else text("empty")
        )
        title = text("title", name=label(self.room.get("name")))
        if notice or not page["items"]:
            title += "\n" + (notice or empty)
        rendered = self.page(title, actions)
        return ChoicePage(rendered.title, [
            {**choice, "full_width": index < len(page["items"])}
            for index, choice in enumerate(rendered.choices)
        ])

    def file_label(self, item):
        return self._file_labels([item])[0]

    def _file_labels(self, items):
        """Fit once per loaded row, then disambiguate final renderer captions."""
        from hermes_time import get_timezone

        # Files rows are not current-choice markers: Telegram gives them 64 chars.
        limit = 64 if self.event.source.platform.value == "telegram" else 100
        zone = get_timezone()
        date_format = text("date_format")

        def identity(item):
            return item["event_id"], item["attachment_id"]

        loaded = {
            identity(item): item
            for page in self.pages
            for item in page["items"]
        }
        loaded.update((identity(item), item) for item in items)
        records, minutes, seconds = {}, Counter(), Counter()
        for key, item in loaded.items():
            name = _clip_caption_part(label(item["name"], len(item["name"])), 42)
            producer = label(item["producer"]["label"], 20)
            instant = datetime.fromtimestamp(item["shared_at"], timezone.utc).astimezone(zone)
            minute = instant.strftime(date_format)
            group = (name, producer, minute)
            minutes[group] += 1
            seconds[(*group, instant.second)] += 1
            records[key] = (name, producer, instant, group, size_label(item["size"]))

        def fit(record, code=""):
            name, producer, instant, group, size = record
            precision = date_format
            if minutes[group] > 1:
                precision += ":%S"
                if seconds[(*group, instant.second)] > 1:
                    precision += ".%f %z"
            date = instant.strftime(precision)
            # A sanitized filename cannot start with this reserved code prefix.
            prefix = f"[{code}] " if code else ""
            fixed = text("file_label", name="", producer="", date=date, size=size)
            available = limit - len(prefix) - len(fixed)
            if available < 2:
                return code  # A rare long collision prefix still fits as a full code.
            producer_width = min(len(producer), 20, max(1, available // 3))
            name_width = min(len(name), 42, available - producer_width)
            producer_width = min(20, available - name_width)
            caption = prefix + text(
                "file_label", name=_clip_caption_part(name, name_width),
                producer=_clip_caption_part(producer, producer_width), date=date, size=size,
            )
            return choice_label({"label": caption}, limit)

        captions = {key: fit(record) for key, record in records.items()}
        collisions = defaultdict(list)
        for key, caption in captions.items():
            collisions[caption].append(key)
        ambiguous = [key for group in collisions.values() if len(group) > 1 for key in group]
        if ambiguous:
            from gateway.hosted_room_file_lookup import selection_digest

            codes = {key: selection_digest(self.room, loaded[key]) for key in ambiguous}
            for key, code in codes.items():
                length = 8
                while length < len(code) and any(
                    other.startswith(code[:length])
                    for other_key, other in codes.items() if other_key != key
                ):
                    length += 4
                captions[key] = fit(records[key], code[:length])
        return [captions[identity(item)] for item in items]

    def plain_files(self):
        from gateway.hosted_room_file_lookup import selection_digest

        page = self.pages[self.position]
        lines = ["**" + text("title", name=label(self.room.get("name"))) + "**"]
        for item, caption in zip(page["items"], self._file_labels(page["items"])):
            code = selection_digest(self.room, item)[: 64 if self.long_codes else 8]
            lines.extend([
                caption,
                f"`{self.command} {self.reference} file {code}`",
            ])
        if not page["items"]:
            lines.append(
                text("partial")
                if page["has_more"]
                else text("no_match")
                if self.query
                else text("empty")
            )
        if page["has_more"]:
            lines.append(
                text("command_hint", caption=text("older"),
                     command=f"`{self.command} {self.reference} files --older {self.handle}`")
            )
        if self.position > 0:
            lines.append(
                text("command_hint", caption=text("newer"),
                     command=f"`{self.command} {self.reference} files --newer {self.handle}`")
            )
        lines.append(text("command_hint", caption=text("search"),
                          command=f"`{self.command} {self.reference} files <text>`"))
        lines.append(text("command_hint", caption=text("full_reply"),
                          command=f"`{self.command} {self.reference} reply`"))
        return "\n".join(lines)

    async def previous(self):
        return await self.refresh_page(max(0, self.position - 1))

    async def refresh_page(self, index):
        from gateway.hosted_room_file_lookup import resolve_file, selection_digest

        current = await self.fresh_room()
        probe = await asyncio.to_thread(
            self.backend.list_files, room=current, profile=self.profile, limit=1
        )

        async def refresh(item):
            try:
                return await asyncio.to_thread(
                    resolve_file,
                    self.backend,
                    room=current,
                    profile=self.profile,
                    code=selection_digest(current, item),
                )
            except Exception as exc:
                if getattr(exc, "code", "") == "file_unavailable":
                    return None
                raise

        items = await asyncio.wait_for(
            asyncio.gather(*(refresh(item) for item in self.pages[index]["items"])), 20
        )
        await self.fresh_room()
        self.pages[index] = {
            **self.pages[index],
            "items": [item for item in items if item is not None],
            **({"latest_seq": probe["latest_seq"]} if "latest_seq" in probe else {}),
        }
        self.check()
        self.position = index
        return self.render_files()

    async def latest_reply(self):
        from gateway.hosted_room_file_lookup import latest_reply

        current = await self.fresh_room()
        reply = await asyncio.wait_for(
            asyncio.to_thread(
                latest_reply, self.backend, room=current, profile=self.profile
            ),
            20,
        )
        await self.fresh_room()
        return {
            **reply,
            "name": "reply.md",
            "size": len(reply["text"].encode("utf-8")),
            "reply": True,
        }

    async def prepare_file(self, item, confirmed=False, retry=""):
        from gateway.hosted_room_file_delivery import (
            FileDeliveryError,
            native_document_limit,
        )

        await self.fresh_room()
        maximum = native_document_limit(self.adapter, self.event.source)
        if item["size"] > maximum:
            raise FileDeliveryError("too_large")
        if item["size"] > min(10_000_000, maximum) and not confirmed:
            return self.page(
                text("confirm_send", name=label(item["name"]),
                     size=size_label(item["size"])),
                [(text("send"), ("file", (item, True, retry))),
                 (text("cancel"), ("files", None))],
            )
        return ChoiceProgress(text("getting"), lambda: self.deliver(item, retry))

    async def deliver(self, item, retry=""):
        from gateway.hosted_room_file_delivery import (
            Document,
            deliver_document,
            delivery_identity,
        )
        from gateway.hosted_room_file_lookup import resolve_file, selection_digest
        from gateway.hosted_room_messaging import messaging_event_id

        current = await self.fresh_room()
        immutable = dict(item)
        if item.get("reply"):
            selection_key = hashlib.sha256(
                json.dumps([_room_key(current), item["event_id"], "reply"]).encode()
            ).hexdigest()
        else:
            selection_key = selection_digest(current, item)
        active = getattr(self.runner, "_group_file_inflight", None)
        if not isinstance(active, set):
            active = self.runner._group_file_inflight = set()
        flight = (self.source_key, selection_key)
        if flight in active:
            return text("busy")
        if not _rate(self.runner, self.source_key, "send"):
            return text("rate")
        key, scope = delivery_identity(
            self.source_key, messaging_event_id(self.event) + retry, selection_key
        )
        active.add(flight)

        def load(maximum):
            if immutable.get("reply"):
                result = self.backend.read_shared_message(
                    room=current, profile=self.profile, event_id=immutable["event_id"]
                )
                if result["text"] != immutable["text"]:
                    raise PermissionError("denied")
                return Document("reply.md", result["text"].encode("utf-8"))
            stored = self.backend.read_file(
                room=current,
                profile=self.profile,
                event_id=immutable["event_id"],
                attachment_id=immutable["attachment_id"],
                max_bytes=maximum,
            )
            if any(
                stored.attachment[field] != immutable[field]
                for field in (
                    "attachment_id",
                    "event_id",
                    "kind",
                    "name",
                    "mime",
                    "size",
                )
            ):
                raise PermissionError("denied")
            return Document(stored.attachment["name"], stored.data)

        async def recheck():
            room = await self.fresh_room()
            if immutable.get("reply"):
                selected = await asyncio.to_thread(
                    self.backend.read_shared_message,
                    room=room,
                    profile=self.profile,
                    event_id=immutable["event_id"],
                )
                if any(selected[field] != immutable[field] for field in selected):
                    raise PermissionError("denied")
            else:
                selected = await asyncio.to_thread(
                    resolve_file,
                    self.backend,
                    room=room,
                    profile=self.profile,
                    code=selection_key,
                )
                if any(
                    selected[field] != immutable[field]
                    for field in (
                        "attachment_id",
                        "event_id",
                        "kind",
                        "name",
                        "mime",
                        "size",
                    )
                ):
                    raise PermissionError("denied")
            self.check()

        try:
            anchor = self.runner._reply_anchor_for_event(self.event)
            metadata = dict(
                self.runner._thread_metadata_for_source(self.event.source, anchor) or {}
            )
            outcome = await deliver_document(
                db_path=self.backend.db_path,
                key=key,
                scope=scope,
                adapter=self.adapter,
                source=self.event.source,
                load=load,
                recheck=recheck,
                metadata=metadata,
                reply_to=anchor,
            )
            try:
                self.check()
            except (PermissionError, TimeoutError):
                return _status_text(outcome, "unknown")
            if outcome == "unknown":
                return self.page(
                    text("unknown"),
                    [
                        (
                            text("send_again"),
                            (
                                "file",
                                (immutable, True, ":again:" + secrets.token_hex(8)),
                            ),
                        ),
                        (text("back_files"), ("files", None)),
                    ],
                )
            notice = _status_text(outcome, "failed")
            return (
                self.page(
                    notice,
                    [
                        (text("back_files"), ("backfiles", None)),
                        (text("group_chat"), ("room", None)),
                    ],
                )
                if self.pages
                else notice
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self.failure(exc, ("file", (immutable, True, retry)))
        finally:
            active.discard(flight)

    async def choose(self, chat_id, value):
        action = ("files", None)
        try:
            self.check()
            if str(chat_id) != str(self.event.source.chat_id):
                raise PermissionError("denied")
            action = self.actions.pop(value, None)
            if action is None:
                return text("expired")
            if not _rate(self.runner, self.source_key, "read"):
                return text("rate")
            kind, data = action
            if kind == "file":
                return await self.prepare_file(*data)
            if kind == "reply":
                from gateway.hosted_room_file_delivery import native_document_limit

                await self.fresh_room()
                native_document_limit(self.adapter, self.event.source)
                return ChoiceProgress(text("getting"), self.reply_action)
            if kind == "files":
                return await self.files_page()
            if kind == "older":
                return await self.files_page(
                    self.pages[self.position]["next_cursor"], direction="older"
                )
            if kind == "newer":
                return await self.previous()
            if kind == "backfiles":
                return (
                    await self.refresh_page(self.position)
                    if self.pages
                    else await self.files_page()
                )
            if kind == "search":
                await self.fresh_room()
                return self.page(
                    text("search_with",
                         command=f"`{self.command} {self.reference} files <text>`"),
                    [(text("show_latest"), ("files", None)),
                     (text("back"), ("files", None))],
                )
            if kind == "bots":
                from gateway.hosted_room_messaging import format_room_bot_list

                current = await self.fresh_room()
                detail = await asyncio.to_thread(
                    format_room_bot_list,
                    self.backend,
                    current,
                    room_command=self.command,
                )
                return await self.room_page(detail, view="bots")
            if kind == "groups":
                from gateway.hosted_room_messaging import format_room_list, room_picker_choices

                self.room = None
                rooms = await self.fresh_room()
                choices = await asyncio.to_thread(room_picker_choices, self.backend, rooms)
                self.check()
                if not choices:
                    self.actions = {}
                    return await asyncio.to_thread(
                        format_room_list, self.backend, rooms=rooms, rooms_command=self.command,
                    )
                return self.page(
                    text("groups") + "\n" + t("gateway.group_home.chooser") + "\n"
                    + text("command_hint", caption=text("groups"),
                           command=f"`{self.command} list`"),
                    [(choice["label"], ("bind", choice["value"])) for choice in choices],
                    full_width=True,
                )
            if kind == "bind":
                from gateway.hosted_room_messaging import resolve_room_picker_choice, room_reference

                self.room = None
                rooms = await self.fresh_room()
                self.room = resolve_room_picker_choice(rooms, data)
                self.reference = room_reference(self.room)
                self.pages, self.position, self.query = [], -1, ""
            return await self.room_page()
        except TimeoutError:
            return text("expired")
        except Exception as exc:
            return self.failure(exc, action)

    async def reply_action(self):
        try:
            item = await self.latest_reply()
            return await self.deliver(item)
        except Exception as exc:
            if getattr(exc, "code", "") == "file_unavailable":
                return text("no_reply")
            return _error(exc)

    async def send_page(self, page):
        self.check()
        source = await asyncio.to_thread(
            self.runner._normalize_source_for_session_key, self.event.source
        )
        self.check()
        sent = await self.runner._try_send_choice_picker(
            self.event,
            self.runner._session_key_for_source(source),
            title=page.title,
            choices=list(page.choices),
            on_choice_selected=self.choose,
            reusable=True,
        )
        menus = getattr(self.runner, "_group_file_menus", None)
        if not isinstance(menus, OrderedDict):
            menus = self.runner._group_file_menus = OrderedDict()
        for handle in list(menus):
            if menus[handle].deadline <= time.monotonic():
                menus.pop(handle, None)
        for _attempt in range(4):
            if self.handle not in menus or menus[self.handle] is self:
                break
            self.handle = secrets.token_hex(8)
        else:
            raise RuntimeError("menu identity is unavailable")
        menus[self.handle] = self
        while len(menus) > MAX_MENUS:
            menus.popitem(last=False)
        return sent


async def handle_command(runner, event, backend, query):
    command = f"{runner._typed_command_prefix_for(event.source)}group"
    menu = FilesMenu(runner, event, backend, command)
    try:
        parts = query.split(maxsplit=2)
        await menu.bind(parts[0])
        if menu.room.get("_room_mode") == "desktop":
            return text("classic")
        verb = parts[1].casefold()
        tail = parts[2] if len(parts) == 3 else ""
        if not _rate(runner, menu.source_key, "read"):
            return text("rate")
        if verb == "files":
            if tail.startswith(("--older ", "--newer ")):
                direction, handle = tail.split(maxsplit=1)
                old = getattr(runner, "_group_file_menus", {}).get(handle)
                if (
                    old is None
                    or old.source_key != menu.source_key
                    or _room_key(old.room) != _room_key(menu.room)
                ):
                    return text("expired")
                old.check()
                if direction == "--older" and not old.pages[old.position]["has_more"]:
                    return text("no_older")
                menu.query, menu.pages, menu.position = (
                    old.query,
                    list(old.pages),
                    old.position,
                )
                page = (
                    await menu.previous()
                    if direction == "--newer"
                    else await menu.files_page(
                        menu.pages[menu.position]["next_cursor"], direction="older"
                    )
                )
            else:
                menu.query = tail
                page = await menu.files_page()
            return None if await menu.send_page(page) else menu.plain_files()
        if verb == "reply" and not tail:
            from gateway.hosted_room_file_delivery import native_document_limit

            native_document_limit(menu.adapter, menu.event.source)
            await _ack(menu)
            result = await menu.reply_action()
        elif verb == "file":
            from gateway.hosted_room_file_lookup import resolve_file, selection_digest

            values = tail.split()
            if not 1 <= len(values) <= 2 or (
                len(values) == 2 and values[1].casefold() != "confirm"
            ):
                return text("use_file",
                            command=f"`{command} {menu.reference} file <file-id>`")
            from gateway.hosted_room_file_delivery import native_document_limit

            native_document_limit(menu.adapter, menu.event.source)
            await _ack(menu)
            try:
                item = await asyncio.wait_for(
                    asyncio.to_thread(
                        resolve_file,
                        backend,
                        room=await menu.fresh_room(),
                        profile=menu.profile,
                        code=values[0],
                    ),
                    20,
                )
            except Exception as exc:
                if getattr(exc, "code", "") == "file_code_ambiguous":
                    menu.long_codes = True
                    await menu.fresh_room()
                    matches = list(getattr(exc, "matches", ()))
                    if not matches:
                        return text("ambiguous")
                    menu.pages = [
                        {
                            "items": matches,
                            "has_more": False,
                            "next_cursor": None,
                            "snapshot_seq": max(item["seq"] for item in matches),
                        }
                    ]
                    menu.position = 0
                    if await menu.send_page(menu.render_files(text("ambiguous"))):
                        return None
                    return text("ambiguous") + "\n\n" + menu.plain_files()
                raise
            result = await menu.prepare_file(item, confirmed=len(values) == 2)
            if isinstance(result, ChoiceProgress):
                result = await result.complete()
            elif isinstance(result, ChoicePage):
                if await menu.send_page(result):
                    return None
                code = selection_digest(menu.room, item)
                return (
                    result.title
                    + "\n" + text("confirm_command",
                                  command=f"`{command} {menu.reference} file {code} confirm`")
                )
        else:
            return text("usage", files=f"`{command} {menu.reference} files`",
                        file="`file <file-id>`", reply="`reply`")
        if isinstance(result, ChoicePage):
            if await menu.send_page(result):
                return None
            return (
                result.title
                + "\n" + text("send_again_hint")
            )
        return result
    except ImportError:
        return text("unavailable")
    except TimeoutError:
        return text("expired")
    except Exception as exc:
        return _error(exc)


async def _ack(menu):
    menu.check()
    try:
        anchor = menu.runner._reply_anchor_for_event(menu.event)
        await menu.adapter.send(
            chat_id=menu.event.source.chat_id,
            content=text("getting"),
            reply_to=anchor,
            metadata=menu.runner._thread_metadata_for_source(menu.event.source, anchor),
        )
    except Exception:
        pass


def room_picker_callback(runner, event, backend, command, fallback):
    adapter = runner._adapter_for_source(event.source)
    if getattr(type(adapter), "supports_choice_pages", False) is not True:
        return fallback, False
    source_key = _source_key(runner, event)
    menu = None

    async def selected(chat_id, value):
        nonlocal menu
        try:
            _authorized(runner, event, source_key, adapter)
            if menu is not None:
                return await menu.choose(chat_id, value)
            from gateway.hosted_room_messaging import (
                resolve_room_picker_choice,
                room_reference,
            )

            candidate = FilesMenu(runner, event, backend, command)
            if str(chat_id) != str(event.source.chat_id):
                raise PermissionError("denied")
            if not _rate(runner, candidate.source_key, "read"):
                return text("rate")
            rooms = await candidate.fresh_room()
            room = resolve_room_picker_choice(rooms, value)
            await candidate.bind(room_reference(room))
            menu = candidate
            return await menu.room_page()
        except Exception as exc:
            return _error(exc)

    return selected, True


async def try_room_menu(runner, event, backend, room, command):
    adapter = runner._adapter_for_source(event.source)
    if getattr(type(adapter), "supports_choice_pages", False) is not True:
        return False
    from gateway.hosted_room_messaging import room_reference

    menu = FilesMenu(runner, event, backend, command)
    await menu.bind(room_reference(room))
    return await menu.send_page(await menu.room_page())
