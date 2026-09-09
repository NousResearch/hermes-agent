"""Bounded cross-room browsing over the existing authorized file catalogues."""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict

from gateway.hosted_room_messaging_files import (
    FilesMenu,
    _clip_caption_part,
    _error,
    _rate,
    _room_key,
    label,
    text,
)

PAGE_SIZE = 8
MAX_GLOBAL_MENUS = 8
BATCH_TIMEOUT = 20


class PageReadUnavailable(RuntimeError):
    """Retryable catalogue failure at an already established snapshot cursor."""


class AllFilesMenu(FilesMenu):
    def __init__(self, *args):
        super().__init__(*args)
        self.streams = {}
        self.initialized = False
        self.incomplete = False
        self.has_classic = False
        self.first_page = 0
        self.child = None
        self.lock = asyncio.Lock()
        self.read_deadline = None
        self.failed_page = None

    async def _read(self, function, **kwargs):
        self.check()
        semaphore = getattr(self.runner, "_all_group_files_read_slots", None)
        if semaphore is None:
            semaphore = self.runner._all_group_files_read_slots = asyncio.Semaphore(4)
        await semaphore.acquire()
        job = asyncio.create_task(asyncio.to_thread(function, **kwargs))

        def finished(task):
            semaphore.release()
            if not task.cancelled():
                task.exception()

        # A timed-out remote call keeps its slot until the actual I/O finishes.
        job.add_done_callback(finished)
        value = await asyncio.shield(job)
        self.check()
        return value

    async def _batch(self, operations):
        tasks = [asyncio.create_task(operation) for operation in operations]
        if not tasks:
            return []
        try:
            budget = (
                BATCH_TIMEOUT
                if self.read_deadline is None
                else max(0, self.read_deadline - time.monotonic())
            )
            _done, pending = await asyncio.wait(tasks, timeout=budget)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for task in pending:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.check()
        return results

    async def _inventory(self):
        rooms = await self.fresh_room()
        self.has_classic = any(room.get("_room_mode") == "desktop" for room in rooms)
        current = {
            _room_key(room): room
            for room in rooms
            if room.get("_room_mode") != "desktop"
        }
        if not self.initialized:
            self.streams = {
                key: {"room": room, "items": [], "cursor": None, "loaded": False}
                for key, room in current.items()
            }
            self.initialized = True
        for key in list(self.streams):
            if key not in current:
                del self.streams[key]
                self.incomplete = True
            else:
                self.streams[key]["room"] = current[key]
        # New rooms/files enter on a fresh command, not halfway through paging.
        return current

    async def _fill(self):
        selected = [
            stream
            for stream in self.streams.values()
            if not stream["items"] and (not stream["loaded"] or stream["cursor"])
        ]
        results = await self._batch([
            self._read(
                self.backend.list_files,
                room=stream["room"],
                profile=self.profile,
                cursor=stream["cursor"],
                query=self.query,
                limit=PAGE_SIZE,
            )
            for stream in selected
        ])
        for stream, result in zip(selected, results):
            if isinstance(result, BaseException):
                if (
                    stream["loaded"]
                    and stream["cursor"]
                    and getattr(result, "code", "")
                    not in {
                        "file_access_denied",
                        "file_unavailable",
                        "attachment_cursor_reset_required",
                    }
                ):
                    raise PageReadUnavailable()
                stream["loaded"] = True
                stream["cursor"] = None
                self.incomplete = True
            else:
                stream["loaded"] = True
                stream["items"] = list(result["items"])
                stream["cursor"] = result["next_cursor"] if result["has_more"] else None

    async def _verify_rows(self, rows, current):
        from gateway.hosted_room_file_lookup import resolve_file, selection_digest

        eligible = [
            (current[_room_key(room)], item)
            for room, item in rows
            if _room_key(room) in current
        ]
        results = await self._batch([
            self._read(
                resolve_file,
                backend=self.backend,
                room=room,
                profile=self.profile,
                code=selection_digest(room, item),
            )
            for room, item in eligible
        ])
        verified, retained = [], []
        for (room, item), result in zip(eligible, results):
            if isinstance(result, BaseException):
                self.incomplete = True
                if getattr(result, "code", "") not in {
                    "file_unavailable",
                    "file_access_denied",
                }:
                    retained.append((room, item))
            else:
                verified.append((room, result))
                retained.append((room, item))
        latest = {_room_key(room) for room in await self.fresh_room()}
        return (
            [(room, item) for room, item in verified if _room_key(room) in latest],
            [(room, item) for room, item in retained if _room_key(room) in latest],
        )

    async def open_page(self, index=0):
        async with self.lock:
            previous = (
                self.streams,
                self.pages,
                self.initialized,
                self.first_page,
                self.position,
                self.incomplete,
                self.has_classic,
            )
            self.streams = {
                key: {**stream, "items": list(stream["items"])}
                for key, stream in self.streams.items()
            }
            self.pages = [dict(page) for page in self.pages]
            self.read_deadline = time.monotonic() + BATCH_TIMEOUT
            try:
                result = await self._construct_page(index)
                self.failed_page = None
                return result
            except BaseException as exc:
                (
                    self.streams,
                    self.pages,
                    self.initialized,
                    self.first_page,
                    self.position,
                    self.incomplete,
                    self.has_classic,
                ) = previous
                if self.deadline <= time.monotonic():
                    self.streams.clear()
                    self.pages.clear()
                if isinstance(exc, PageReadUnavailable):
                    self.check()
                    self.failed_page = index
                    return self.page(
                        text("error"), [(text("reload_page"), ("page", index))]
                    )
                raise
            finally:
                self.read_deadline = None

    async def _construct_page(self, index):
        current = await self._inventory()
        offset = index - self.first_page
        if not 0 <= offset <= len(self.pages):
            raise TimeoutError("page expired")
        if offset == len(self.pages):
            if index and not self.pages[-1]["has_more"]:
                raise TimeoutError("page expired")
            rows = []
            for _ in range(PAGE_SIZE):
                # Refill a drained room before choosing the next global row,
                # including when its cursor runs out midway through a page.
                await self._fill()
                current = await self._inventory()
                if any(
                    not stream["items"] and stream["cursor"]
                    for stream in self.streams.values()
                ):
                    self.incomplete = True
                row = min(
                    (
                        (stream["room"], item)
                        for stream in self.streams.values()
                        for item in stream["items"]
                    ),
                    key=lambda row: (
                        -row[1]["shared_at"],
                        str(row[0]["room_id"]),
                        -row[1]["seq"],
                        row[1]["attachment_id"],
                    ),
                    default=None,
                )
                if row is None:
                    break
                room, item = row
                rows.append(row)
                self.streams[_room_key(room)]["items"].remove(item)
            self.pages.append({
                "candidates": rows,
                "rows": [],
                "has_more": any(
                    stream["items"] or stream["cursor"]
                    for stream in self.streams.values()
                ),
            })
            if len(self.pages) > 8:
                self.pages.pop(0)
                self.first_page += 1
                offset -= 1
        (
            self.pages[offset]["rows"],
            self.pages[offset]["candidates"],
        ) = await self._verify_rows(self.pages[offset]["candidates"], current)
        self.position = index
        return self.render()

    def _caption_menu(self, room):
        menu = FilesMenu(self.runner, self.event, self.backend, self.command)
        menu.room = room
        menu.pages = [
            {
                "items": [
                    item
                    for owner, item in self.pages[self.position - self.first_page][
                        "rows"
                    ]
                    if _room_key(owner) == _room_key(room)
                ]
            }
        ]
        return menu

    def render(self):
        from gateway.hosted_room_messaging import room_reference

        page = self.pages[self.position - self.first_page]
        actions = []
        seen = set()
        for index, (room, item) in enumerate(page["rows"]):
            prefix = f"{room_reference(room)}. {label(room.get('name'), 14)} · "
            limit = 64 if self.event.source.platform.value == "telegram" else 100
            caption = self._caption_menu(room)._file_labels(
                [item], max_caption_chars=limit - len(prefix)
            )[0]
            if len(prefix) + len(caption) > limit:
                prefix = f"{room_reference(room)}. "
                caption = self._caption_menu(room)._file_labels(
                    [item], max_caption_chars=limit - len(prefix)
                )[0]
            name = label(item["name"], len(item["name"]))
            if re.fullmatch(r"[0-9a-f]{8,64}", caption) or (
                (len(name) <= 16 and name not in caption)
                or (
                    len(name) > 16
                    and (name[:5] not in caption or name[-4:] not in caption)
                )
            ):
                from gateway.hosted_room_file_lookup import selection_digest

                code = selection_digest(room, item)[:8]
                suffix = f" [{code}]"
                caption = (
                    _clip_caption_part(name, limit - len(prefix) - len(suffix)) + suffix
                )
            caption = (
                prefix + caption if len(prefix) + len(caption) <= limit else caption
            )
            if caption in seen:
                numbered = f"{text('download')} {index + 1} · {room_reference(room)} · "
                caption = (
                    numbered + _clip_caption_part(name, limit - len(numbered))
                )
            seen.add(caption)
            actions.append((caption, ("file", (room, item))))
        if self.position > self.first_page:
            actions.append((
                text("page_action", current=self.position),
                ("page", self.position - 1),
            ))
        if page["has_more"]:
            actions.append((
                text("page_action", current=self.position + 2),
                ("page", self.position + 1),
            ))
        if len(page["rows"]) < len(page["candidates"]):
            actions.append((text("reload_page"), ("page", self.position)))
        if page["rows"] or self.query or page["has_more"] or self.incomplete:
            actions.append((text("search"), ("search", None)))
        actions.append((text("back_groups"), ("groups", None)))
        title = text("page_title", name=text("all_title"), current=self.position + 1)
        if not page["rows"]:
            title += "\n\n" + text("no_match" if self.query else "all_empty")
        if self.incomplete:
            title += "\n\n" + text("some_unavailable")
        if self.has_classic:
            title += "\n\n" + text("classic_all")
        return self.page(title, actions, full_width=True)

    def plain_files(self):
        from gateway.hosted_room_file_lookup import selection_digest
        from gateway.hosted_room_messaging import room_reference

        if self.failed_page is not None:
            return (
                text("error")
                + "\n\n"
                + text(
                    "command_hint",
                    caption=text("reload_page"),
                    command=f"`{self.command} files --page {self.handle} {self.failed_page + 1}`",
                )
            )
        page = self.pages[self.position - self.first_page]
        lines = [
            "**"
            + text("page_title", name=text("all_title"), current=self.position + 1)
            + "**"
        ]
        for room, item in page["rows"]:
            caption = self._caption_menu(room)._file_labels([item], multiline=True)[0]
            lines.extend([
                "",
                caption,
                text("group_from", name=label(room.get("name"), 48)),
                text(
                    "command_hint",
                    caption=text("download"),
                    command=f"`{self.command} {room_reference(room)} file {selection_digest(room, item)[:8]}`",
                ),
            ])
        if not page["rows"]:
            lines.extend(["", text("no_match" if self.query else "all_empty")])
        if self.incomplete:
            lines.extend(["", text("some_unavailable")])
        if self.has_classic:
            lines.extend(["", text("classic_all")])
        lines.append("")
        for index in (self.position - 1, self.position + 1):
            if index < self.first_page or (
                index > self.position and not page["has_more"]
            ):
                continue
            lines.append(
                text(
                    "command_hint",
                    caption=text("page_action", current=index + 1),
                    command=f"`{self.command} files --page {self.handle} {index + 1}`",
                )
            )
        if len(page["rows"]) < len(page["candidates"]):
            lines.append(
                text(
                    "command_hint",
                    caption=text("reload_page"),
                    command=f"`{self.command} files --page {self.handle} {self.position + 1}`",
                )
            )
        if page["rows"] or self.query or page["has_more"] or self.incomplete:
            lines.append(text("all_search", command=f"`{self.command} files <text>`"))
        lines.append(
            text(
                "command_hint", caption=text("view_group"), command=f"`{self.command}`"
            )
        )
        return "\n".join(lines)

    async def choose(self, chat_id, value):
        try:
            self.check()
            if str(chat_id) != str(self.event.source.chat_id):
                raise PermissionError("denied")
            if self.child is not None and value.startswith(self.child.handle + ":"):
                return await self.child.choose(chat_id, value)
            action = self.actions.pop(value, None)
            if action is None:
                return text("expired")
            if not _rate(self.runner, self.source_key, "read"):
                return text("rate")
            kind, data = action
            if kind == "page":
                return await self.open_page(data)
            if kind == "file":
                from gateway.hosted_room_messaging import room_reference

                room, item = data
                child = FilesMenu(self.runner, self.event, self.backend, self.command)
                await child.bind(room_reference(room))
                if _room_key(child.room) != _room_key(room):
                    raise PermissionError("denied")
                self.child = child
                return await child.prepare_file(item)
            if kind == "search":
                return self.page(
                    text("all_search", command=f"`{self.command} files <text>`"),
                    [(text("back_files"), ("page", self.position))],
                )
            self.child = FilesMenu(self.runner, self.event, self.backend, self.command)
            page = self.child.page(text("groups"), [(text("groups"), ("groups", None))])
            return await self.child.choose(chat_id, page.choices[0]["value"])
        except TimeoutError:
            return text("expired")
        except Exception as exc:
            return _error(exc)

    async def send_page(self, page):
        self.check()
        source = await asyncio.to_thread(
            self.runner._normalize_source_for_session_key, self.event.source
        )
        self.check()
        # Global snapshots have their own smaller cache, not the room-menu cache.
        return await self.runner._try_send_choice_picker(
            self.event,
            self.runner._session_key_for_source(source),
            title=page.title,
            choices=list(page.choices),
            on_choice_selected=self.choose,
            reusable=True,
        )


async def handle_all_files(runner, event, backend, query):
    command = f"{runner._typed_command_prefix_for(event.source)}group"
    try:
        menu = AllFilesMenu(runner, event, backend, command)
        if not _rate(runner, menu.source_key, "read"):
            return text("rate")
        if not callable(getattr(backend, "list_files", None)):
            return text("unavailable")
        menus = getattr(runner, "_all_group_file_menus", None)
        if not isinstance(menus, OrderedDict):
            menus = runner._all_group_file_menus = OrderedDict()
        parts = query.split(maxsplit=2)
        index = 0
        if parts and parts[0] == "--page":
            if len(parts) != 3 or not parts[2].isascii() or not parts[2].isdecimal():
                return text("expired")
            old = menus.get(parts[1])
            if (
                old is None
                or old.source_key != menu.source_key
                or old.adapter is not menu.adapter
            ):
                return text("expired")
            old.check()
            menu, index = old, int(parts[2]) - 1
        else:
            from gateway.hosted_room_file_contract import catalog_options

            menu.query = catalog_options({"query": query}).get("query", "")
        page = await menu.open_page(index)
        for handle in list(menus):
            try:
                menus[handle].check()
            except (PermissionError, TimeoutError):
                menus.pop(handle, None)
        menus[menu.handle] = menu
        menus.move_to_end(menu.handle)
        while len(menus) > MAX_GLOBAL_MENUS:
            _handle, retired = menus.popitem(last=False)
            retired.deadline = 0
            retired.streams.clear()
            retired.pages.clear()
        plain = menu.plain_files()
        return None if await menu.send_page(page) else plain
    except TimeoutError:
        return text("expired")
    except Exception as exc:
        return _error(exc)
