"""Shared helpers for gateway platform adapters: message dedup, markdown
stripping, thread participation tracking, GFM table → bullets, mention-pattern
compilation, and fence-aware markdown chunking."""

import asyncio
import contextlib
import contextvars
import hashlib
import hmac
import itertools
import json
import logging
import os
import re
import secrets
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterator, MutableMapping, Optional
from urllib.parse import urljoin, urlsplit
from gateway.platforms.event import MessageEvent
from utils import atomic_json_write

logger = logging.getLogger(__name__)


class MessageDeduplicator:
    """TTL-based message deduplication cache (``if dedup.is_duplicate(msg_id): return``)."""

    def __init__(self, max_size: int = 2000, ttl_seconds: float = 300):
        self._seen: dict[str, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def is_duplicate(self, msg_id: str) -> bool:
        """Return True if *msg_id* was already seen within the TTL window."""
        if not msg_id:
            return False
        now = time.time()
        if msg_id in self._seen:
            if now - self._seen[msg_id] < self._ttl:
                return True
            del self._seen[msg_id]  # expired: treat as new
        self._seen[msg_id] = now
        if len(self._seen) > self._max_size:
            cutoff = now - self._ttl
            self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
            if len(self._seen) > self._max_size:
                # All entries still fresh: keep the newest so max_size holds under load.
                self._seen = dict(sorted(self._seen.items(), key=lambda item: item[1])[-self._max_size:])
        return False

    def contains(self, msg_id: str) -> bool:
        """Return whether *msg_id* is live in the cache without inserting it."""
        seen_at = self._seen.get(msg_id) if msg_id else None
        if seen_at is None:
            return False
        if time.time() - seen_at < self._ttl:
            return True
        del self._seen[msg_id]
        return False

    def discard(self, msg_id: str) -> None:
        """Release a claimed message ID after cancelled/failed handoff."""
        self._seen.pop(msg_id, None)

    def clear(self):
        self._seen.clear()

    def absorb(self, other: "MessageDeduplicator") -> None:
        """Adopt *other*'s still-live IDs (at their original seen times) into this cache."""
        cutoff = time.time() - self._ttl
        self._seen.update({k: v for k, v in other._seen.items() if v > cutoff and k not in self._seen})


def carry_inbound_dedup(predecessor: Any, adapter: Any) -> None:
    """Seed a rebuilt adapter's dedup caches from the instance it replaces (call before connect).

    The runner's reconnect path builds a NEW adapter; without this a platform replaying a recent
    inbound ID after the reconnect (websocket resume, webhook retry, unacked poll batch) is
    admitted and answered a second time."""
    if predecessor is None:
        return
    for name, previous in vars(predecessor).items():
        current = getattr(adapter, name, None)
        if isinstance(previous, MessageDeduplicator) and isinstance(current, MessageDeduplicator) and current is not previous:
            current.absorb(previous)


def hand_over_held_inbound(source: Any, target: Any) -> None:
    """Move inbound ``source`` is holding to ``target`` (#132829, #133399).

    The runner calls it at publish time (retired instance -> replacement) and after disposing a
    failed reconnect candidate (candidate -> retained predecessor): a candidate that fails connect
    can hold acked updates and salvage pending batches in disconnect(), which must not die with it."""
    adopt = getattr(target, "adopt_held_inbound", None)
    if source is not None and callable(adopt):
        adopt(source)


# Worker-thread handoff used by the off-loop persist paths.  A module attribute
# so tests can replace THIS seam instead of patching ``asyncio.to_thread``
# globally.
_to_thread = asyncio.to_thread


async def cancel_task(task: Optional[asyncio.Task]) -> None:
    """Cancel *task* and wait for it to unwind. ``None``/finished tasks are no-ops; awaiting the
    current task would deadlock, so a self-cancel only requests cancellation. Exceptions the task
    dies with are swallowed: at teardown nobody is left to handle them."""
    if task is None or task.done():
        return
    task.cancel()
    if task is not asyncio.current_task():
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


def tag_safe_display_name(value: Any) -> str:
    """A sender display name made safe to sit inside a gateway-built ``[name]`` / ``[name|id]`` tag.

    Display names are set by the sender. On top of the one-line neutralization, the tag's own
    delimiters (``[``, ``]`` and the ``|`` that precedes a trusted id) are dropped, so a name can
    neither end the tag early nor forge a second id or mention span inside it."""
    from gateway.session import neutralize_untrusted_inline_text
    text = neutralize_untrusted_inline_text(value)
    return " ".join("".join(ch for ch in text if ch not in "[]|").split())


def display_width(text: str) -> int:
    """Terminal-style display columns: East-Asian wide/full-width characters count as two.
    Chat clients clip button labels by rendered width, not by codepoints, so a 25-character
    Korean label is as wide as 50 Latin ones."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def numbered_clarify_choices(choices, *, max_label_cols: int, escape=None,
                             max_chars: Optional[int] = None, len_fn=len) -> Optional[str]:
    """``"1. <choice>\n2. <choice>"`` when any choice is wider than ``max_label_cols`` display
    columns, else ``None``. Button-capable adapters put this in the message body and switch to
    positional (``1``, ``2``, …) button labels: a clipped label with no tooltip is unanswerable
    (#78115), and the body is the only place a long option renders in full. Over ``max_chars``
    every line stays (a button whose line was cut is unanswerable): the long options lose their
    middle (start and end kept, so options sharing a long opening still differ) before escaping,
    so a cut never splits an escape. An option whose first difference from its nearest sibling
    falls in that cut middle keeps a window around it too, so options sharing both a long
    opening and a long ending still read differently."""
    texts = [str(c).strip() for c in choices]
    if all(display_width(t) <= max_label_cols for t in texts):
        return None
    esc = escape or (lambda s: s)

    def _render(items) -> str:
        return "\n".join(f"{i}. {esc(t)}" for i, t in enumerate(items, start=1))
    out, each = _render(texts), max(map(len, texts))
    prefix = [[len(os.path.commonprefix([a, b])) for b in texts] for a in texts]
    while max_chars is not None and len_fn(out) > max_chars and each > 1:
        each = min(each - 1, each * 4 // 5) or 1
        out = _render(_distinct_shortenings(texts, each, prefix))
    return out


def _distinct_shortenings(texts: list, each: int, prefix: list) -> list:
    """Every text shortened to about *each* characters, distinct options kept distinct where the
    room allows: each first keeps a window around where it differs from its nearest sibling, and
    options whose windows still read the same (repeated wording shifts them) are re-cut with one
    shared window around the group's earliest difference until they part."""
    out = [_shorten_choice(t, each, max((p for j, p in enumerate(prefix[i]) if j != i), default=0))
           for i, t in enumerate(texts)]
    for _ in texts:
        groups: dict = {}
        for i, line in enumerate(out):
            groups.setdefault(line, []).append(i)
        clashes = [g for g in groups.values() if len(g) > 1 and len({texts[i] for i in g}) > 1]
        if not clashes:
            break
        for group in clashes:
            shared = min(prefix[a][b] for a in group for b in group if texts[a] != texts[b])
            for i in group:
                out[i] = _shorten_choice(texts[i], each, shared)
    return out


def _shorten_choice(text: str, each: int, anchor: int) -> str:
    """*text* in about *each* characters: start and end kept around a ``…``, or, when *anchor*
    (where it first differs from its nearest sibling) would fall in the cut, start, a window
    around *anchor* and end."""
    if len(text) <= each:
        return text
    head, tail = (each + 1) // 2, each // 2
    if anchor < head or anchor >= len(text) - tail:
        return f"{text[:head].rstrip()}…{text[len(text) - tail:].lstrip()}"
    head = tail = each // 4
    width = each - head - tail
    start = min(max(anchor - width // 2, head + 1), len(text) - tail - width - 1)
    if start <= head:
        return f"{text[:(each + 1) // 2].rstrip()}…{text[len(text) - each // 2:].lstrip()}"
    return (f"{text[:head].rstrip()}…{text[start:start + width].strip()}…"
            f"{text[len(text) - tail:].lstrip()}")


def _clip_text(text: str, limit: int, len_fn=len) -> str:
    """*text* within *limit* units of *len_fn*, a cut marked with ``…``."""
    if len_fn(text) <= limit:
        return text
    cut = text[: max(0, limit - 1)]
    while cut and len_fn(cut) > limit - 1:
        cut = cut[:-1]
    return cut + "…"


def clarify_text_with_choices(head: str, choices, *, limit: int, max_label_cols: int, escape=None,
                              len_fn=len) -> tuple[str, bool]:
    """``(text, numbered)`` for a button clarify within *limit*: *head* alone when every option fits
    its button, else *head* plus the numbered list (see :func:`numbered_clarify_choices`). The list
    is reserved first (at least half of *limit*) and the head is cut to what is left, so every
    positional button keeps its line however long the question is."""
    if numbered_clarify_choices(choices, max_label_cols=max_label_cols) is None:
        return _clip_text(head, limit, len_fn), False
    room = max(limit - 2 - len_fn(head), limit // 2)
    listed = numbered_clarify_choices(choices, max_label_cols=0, escape=escape, max_chars=room, len_fn=len_fn)
    return f"{_clip_text(head, max(0, limit - 2 - len_fn(listed)), len_fn)}\n\n{listed}", True


def prompt_id_counter() -> Iterator[int]:
    """Ids for buttons a chat keeps after a restart (approval cards, confirmations, pickers).

    Starts at a random point instead of 1: an old card still visible after a restart would
    otherwise carry the same id as the first new card and answer it."""
    return itertools.count(secrets.randbelow(10**12) + 1)

def caption_is_command(caption: Optional[str]) -> bool:
    """A file's caption that is a slash command (``/goal summarize this``). Adapters skip inlining
    such a file's text: prepended, it hides the command; appended, it becomes the command's
    arguments. The agent still gets the cached path with a note to read it. Parsed by
    :meth:`MessageEvent.get_command`, so attachment-ref prefixes and paths follow the one rule."""
    return MessageEvent(text=caption or "").get_command() is not None


def inline_document_text(caption: str, injection: Optional[str], inlined: list) -> tuple[str, list]:
    """Prepend a document's inlined text to its caption: ``(text, inlined flags)``. Under a command
    caption the text is left out and the flags cleared, so the agent is told to read the file."""
    if not injection or caption_is_command(caption):
        return caption, [False] * len(inlined)
    return (f"{injection}\n\n{caption}" if caption else injection), inlined


def bounded_put(store: MutableMapping[str, Any], key: str, value: Any, cap: int) -> None:
    """Insert into an insertion-ordered mapping with a hard size bound, evicting the oldest keys. A
    re-put moves the key to the newest position so live entries outlast stale ones."""
    store.pop(key, None)
    store[key] = value
    while len(store) > cap:
        del store[next(iter(store))]


def typing_metadata_for_turn(adapter: Any, chat_id: Any, anchor: Any, metadata: Optional[dict]) -> Any:
    """A turn's typing/status metadata, fixed when the turn starts and reused for every refresh and
    clear. ``anchor`` is the turn's own message id: an adapter whose status thread depends on the turn
    (``status_metadata_for_turn``: a top-level Slack DM's status sits on its triggering message) takes
    it from here, not from per-chat state that a later inbound in the chat overwrites. Looked up on
    the class, so a test double's auto-attributes never stand in for the hook."""
    hook = getattr(type(adapter), "status_metadata_for_turn", None)
    if anchor is None or not callable(hook):
        return metadata
    return hook(adapter, str(chat_id), str(anchor), metadata)


def typing_pause_key(chat_id: Any, metadata: Optional[dict]) -> tuple:
    """``_typing_paused`` entry that pauses ONE turn's keep-alive: the chat plus the thread its
    typing metadata (``typing_metadata_for_turn``) targets. A bare ``chat_id`` entry still pauses
    every loop in the chat (approval waits); this one leaves a concurrent topic's typing alone."""
    return chat_id, (metadata or {}).get("thread_id")


# Inline markdown-stripping rules, applied in order: bold, italic, bold/italic
# underscore, code fence markers, inline code, headings. Links and the newline
# squeeze run after these, in that order (see ``strip_markdown``).
_STRIP_RULES = (
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"),
    (re.compile(r"\*(.+?)\*", re.DOTALL), r"\1"),
    (re.compile(r"\b__(?![\s_])(.+?)(?<![\s_])__\b", re.DOTALL), r"\1"),
    (re.compile(r"\b_(?![\s_])(.+?)(?<![\s_])_\b", re.DOTALL), r"\1"),
    (re.compile(r"```[a-zA-Z0-9_+-]*\n?"), ""),
    (re.compile(r"`(.+?)`"), r"\1"),
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")
_HTTP_TARGET_RE = re.compile(r"https?://", re.IGNORECASE)
_NEWLINE_SQUEEZE_RE = re.compile(r"\n{3,}")


def _keep_link_target(match: re.Match[str]) -> str:
    r"""``[label](https://url)`` -> ``label\nurl``.

    The bare URL is the only thing a platform with its own data detection
    (iMessage) can turn back into a tap target, so dropping it makes the link
    unreachable rather than merely unformatted. Non-http targets (``mailto:``,
    relative paths) are not auto-linked, so they keep the label-only behaviour.
    """
    label, target = match.group(1).strip(), match.group(2).strip()
    if not _HTTP_TARGET_RE.match(target):
        return match.group(1)
    return target if label == target else f"{label}\n{target}"


def strip_markdown(text: str, *, keep_link_targets: bool = False) -> str:
    r"""Strip markdown formatting for plain-text platforms (SMS, iMessage, etc.).

    ``keep_link_targets`` rewrites ``[label](https://url)`` as ``label\nurl``
    instead of discarding the URL; pass it on platforms that auto-link bare URLs.
    """
    for pattern, repl in _STRIP_RULES:
        text = pattern.sub(repl, text)
    text = _MD_LINK_RE.sub(_keep_link_target if keep_link_targets else r"\1", text)
    return _NEWLINE_SQUEEZE_RE.sub("\n\n", text).strip()


class ThreadParticipationTracker:
    """Persistent set of threads the bot has participated in (``<platform>_threads.json``);
    ``thread_id in tracker`` checks membership, ``tracker.mark(thread_id)`` persists."""

    _MAX_TRACKED = 500

    def __init__(self, platform_name: str, max_tracked: int = 500):
        self._platform = platform_name
        self._max_tracked = max_tracked
        # ``mark_async`` runs the persist on a worker thread, which removes the
        # accidental serialization the event loop used to provide.  Two locks,
        # never nested the other way round:
        #   ``_lock``    guards ONLY the in-memory set (``_remember``,
        #                ``__contains__``, ``clear`` and the snapshot in
        #                ``_save``); held for microseconds, safe on the loop.
        #   ``_io_lock`` worker-only; serializes snapshot+write in ``_save`` so
        #                two persists cannot interleave and lose an entry.
        # ``_save`` must NOT hold ``_lock`` across ``os.replace``: the adapters'
        # ``thread_id in tracker`` gate runs on the loop thread and would stall
        # for the whole rename.
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._threads: dict[str, None] = dict.fromkeys(str(t) for t in self._load())

    def _state_path(self) -> Path:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / f"{self._platform}_threads.json"

    def _load(self) -> list[str]:
        try:
            data = json.loads(self._state_path().read_text(encoding="utf-8-sig"))
        except Exception:
            return []
        return [str(thread_id) for thread_id in data] if isinstance(data, list) else []

    def _save(self) -> None:
        with self._io_lock:
            with self._lock:
                thread_list = list(self._threads)
                if len(thread_list) > self._max_tracked:
                    thread_list = thread_list[-self._max_tracked:]
                    self._threads = dict.fromkeys(thread_list)
            atomic_json_write(self._state_path(), thread_list, indent=None)

    def _remember(self, thread_id: str) -> bool:
        """Record *thread_id* in memory; ``True`` when a persist is still owed.

        The in-memory half stays synchronous even for :meth:`mark_async` so that
        a ``thread_id in tracker`` check immediately after marking is correct --
        the mention-gating logic in the adapters depends on that.
        """
        with self._lock:
            if thread_id in self._threads:
                return False
            self._threads[thread_id] = None
            return True

    def mark(self, thread_id: str) -> None:
        """Mark *thread_id* as participated and persist.

        Blocking: ends in ``atomic_json_write`` -> ``os.replace``.  Coroutines
        must use :meth:`mark_async` instead.
        """
        if self._remember(thread_id):
            self._save()

    async def mark_async(self, thread_id: str) -> None:
        """Off-loop form of :meth:`mark`.

        ``_save`` ends in ``atomic_json_write`` -> ``os.replace``, whose
        duration is unbounded under filesystem pressure.  Every caller of this
        tracker sits on an inbound-message coroutine in the Discord and Matrix
        adapters, so the rename must not be paid inline on the event loop -- it
        stalls every other adapter's polling and every in-flight turn for as
        long as it runs.
        """
        if self._remember(thread_id):
            await _to_thread(self._save)

    def __contains__(self, thread_id: str) -> bool:
        with self._lock:
            return thread_id in self._threads

    def clear(self) -> None:
        with self._lock:
            self._threads.clear()


async def send_chunks(chunks: list, send_one) -> Any:
    """Send ``chunks`` in order through ``send_one(chunk) -> SendResult``, stopping at the first failure.

    A failure after earlier chunks landed carries the ``partial_overflow`` contract that
    ``BasePlatformAdapter._is_partial_delivery`` reads, so no caller (send retry, plain-text
    fallback, cron standalone fallback) re-sends the head the recipient already has.
    """
    from gateway.platforms.base import SendResult
    result = SendResult(success=False, error="nothing to send")
    for delivered, chunk in enumerate(chunks):
        result = await send_one(chunk)
        if not result.success:
            if delivered:
                raw = dict(result.raw_response) if isinstance(result.raw_response, dict) else {}
                raw.update(partial_overflow=True, delivered_chunks=delivered, total_chunks=len(chunks))
                result.raw_response = raw
            break
    return result


_LOG_HASH_SALT_FILE = "log_hash_salt"
_LOG_HASH_SALTS: dict[str, bytes] = {}
_PROCESS_LOG_HASH_SALT = secrets.token_bytes(32)


def _read_or_create_log_hash_salt(home: Path) -> bytes:
    path = home / _LOG_HASH_SALT_FILE
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        # Atomic create: write a private temp file, hard-link it into place; the loser of a race
        # reads the winner's salt. A crash can never leave a short salt file behind.
        data = secrets.token_bytes(32)
        temporary = home / f".{_LOG_HASH_SALT_FILE}.{os.getpid()}.{secrets.token_hex(8)}"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            try:
                os.link(temporary, path)
            except FileExistsError:
                data = path.read_bytes()
            except OSError:  # no hard links (Android SELinux, exFAT): exclusive create, first wins
                data = _create_log_hash_salt_exclusive(path, data)
        finally:
            temporary.unlink(missing_ok=True)
    if len(data) != 32:
        raise ValueError("log hash salt is invalid")
    return data


def _create_log_hash_salt_exclusive(path: Path, data: bytes) -> bytes:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_bytes()
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return data


def _log_hash_salt() -> bytes:
    from hermes_constants import get_hermes_home, hermes_home_key
    try:
        home = get_hermes_home()
        key = hermes_home_key(home)
    except (OSError, ValueError):
        return _PROCESS_LOG_HASH_SALT
    if (cached := _LOG_HASH_SALTS.get(key)) is None:
        try:
            cached = _read_or_create_log_hash_salt(home)
        except (OSError, ValueError):
            # Must never raise into a log call; a process-random salt still correlates within this
            # run. Cached so an unwritable home doesn't retry file I/O on every log line.
            cached = _PROCESS_LOG_HASH_SALT
        _LOG_HASH_SALTS[key] = cached
    return cached


def hash_for_log(value: object) -> str:
    """A short stable digest of an identifier (chat id, address) for INFO logs: lines about the
    same chat still correlate, but the log never carries the identifier itself. Keyed with a
    per-install salt so low-entropy ids (phone numbers, guessable emails) can't be recovered from
    a shared log by hashing candidates."""
    if value is None or value == "":
        return "<none>"
    data = str(value).encode("utf-8", "surrogatepass")
    return "h:" + hmac.new(_log_hash_salt(), data, hashlib.sha256).hexdigest()[:10]


def redact_phone(phone: str) -> str:
    """Redact a phone number for logging, preserving country code and last 4."""
    if not phone:
        return "<none>"
    if len(phone) <= 8:
        return phone[:2] + "****" + phone[-2:] if len(phone) > 4 else "****"
    return phone[:4] + "****" + phone[-4:]


# ─── GFM table → bullets. Discord calls convert_table_to_bullets(); Telegram imports the
# primitives but keeps its own MarkdownV2-aware renderer.
# Delimiter row: optional outer pipes, dash cells (optional alignment colons). Requires at
# least one internal '|' so a lone '---' rule is NOT matched.
TABLE_SEPARATOR_RE = re.compile(r'^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*){1,}\|?\s*$')


def is_table_row(line: str) -> bool:
    """Return True if *line* could plausibly be a table data row."""
    return '|' in line.strip()


def split_markdown_table_row(line: str) -> list[str]:
    """Split a GFM table row into stripped cells (delegates to agent.markdown_tables)."""
    from agent.markdown_tables import split_table_row
    return split_table_row(line)


def _render_table_block(table_block: list[str]) -> str:
    """Render a GFM table as bold-heading + bullet groups (same alignment logic as Telegram's
    renderer: without a row-label column the full row is data and the heading bullet is skipped)."""
    headers = split_markdown_table_row(table_block[0]) if len(table_block) >= 3 else []
    if len(headers) < 2:
        return "\n".join(table_block)
    has_row_label_col = len(split_markdown_table_row(table_block[2])) == len(headers) + 1
    rendered_groups: list[str] = []
    for index, row in enumerate(table_block[2:], start=1):
        cells = split_markdown_table_row(row)
        if has_row_label_col:
            heading = cells[0] if cells and cells[0] else f"Row {index}"
            data_cells = cells[1:]
        else:
            heading = next((cell for cell in cells if cell), f"Row {index}")
            data_cells = cells
        data_cells = (data_cells + [""] * len(headers))[: len(headers)]
        bullets = [f"• {header}: {value}" for header, value in zip(headers, data_cells)
                   if has_row_label_col or value != heading]
        rendered_groups.append("\n".join([f"**{heading}**", *bullets]))
    return "\n\n".join(rendered_groups)


def convert_table_to_bullets(text: str) -> str:
    """Rewrite GFM pipe tables into bold-heading + bullet groups; fenced code is left alone."""
    if '|' not in text or '-' not in text:
        return text
    lines = text.split('\n')
    out: list[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        is_fence_line = line.lstrip().startswith('```')
        in_fence ^= is_fence_line
        if not (in_fence or is_fence_line) and '|' in line and i + 1 < len(lines) \
                and TABLE_SEPARATOR_RE.match(lines[i + 1]):
            j = i + 2
            while j < len(lines) and is_table_row(lines[j]):
                j += 1
            out.append(_render_table_block(lines[i:j]))
            i = j
        else:
            out.append(line)
            i += 1
    return '\n'.join(out)


def voice_mention_gate_enabled(extra: Optional[dict]) -> bool:
    """``voice_mention_patterns``: opt-in, because it sends unaddressed members' voice notes to STT."""
    from utils import is_truthy_value
    return is_truthy_value((extra or {}).get("voice_mention_patterns"), default=False)


def _gate_stt_scope(adapter: Any, source: Any) -> Any:
    """The routed profile's runtime scope for *source*, as the turn will enter it, so the gate's STT
    reads that profile's ``stt`` config and secrets; a no-op outside a runner."""
    runner = getattr(adapter, "gateway_runner", None)
    enter = getattr(runner, "_async_profile_scope_for_source", None)
    if source is None or not callable(enter):
        return contextlib.nullcontext()
    adapter._canonicalize(source)
    return enter(source)


def _sender_may_use_bot(adapter: Any, source: Any) -> bool:
    """The runner's own allowlist verdict for *source*, checked BEFORE gate STT: a sender the turn
    would refuse must not spend the operator's STT provider. True outside a runner."""
    authorize = getattr(getattr(adapter, "gateway_runner", None), "_is_user_authorized_for_source", None)
    if source is None or not callable(authorize):
        return True
    adapter._canonicalize(source)
    return bool(authorize(source))


def bypasses_chat_order(text: Any) -> bool:
    """A running-agent escape hatch (``/stop``, ``/approve``, ``/deny``): it acts on work already in
    flight, so it skips :class:`ChatOrderedWork`'s per-chat wait and never waits on an earlier voice
    note's gate STT. Every other command (``/new``, ``/undo``, ``/status``) keeps its place behind it."""
    from gateway.session_commands import _ESCAPE_HATCHES
    from hermes_cli.commands import resolve_command
    head = str(text or "").lstrip().split(maxsplit=1)
    if not head or not head[0].startswith("/"):
        return False
    cmd = resolve_command(head[0][1:].split("@", 1)[0])
    return cmd is not None and cmd.name in _ESCAPE_HATCHES


async def spoken_mention_clips(paths: list[str], patterns, *, adapter: Any = None, source: Any = None,
                               keep: frozenset = frozenset()) -> list[str]:
    """Transcribe a group voice note that only a spoken wake word could admit and match ``patterns``
    against the transcript (the text gate runs before STT, so it never sees one: #102330). STT runs
    in the profile *source* routes to (``adapter.gateway_runner``) and never when that profile
    disables it. On a match, the clips to attach; the turn reuses their transcripts, one STT call
    per clip. Otherwise ``[]``, and the clips fetched only for this check are deleted (never those
    in *keep*, which another owner still references)."""
    from gateway.run_inbound_transcription import (remember_gate_transcript, stt_enabled_for_turn,
                                                   transcribe_one_clip)
    results: dict = {}
    try:
        async with _gate_stt_scope(adapter, source):
            from tools.transcription_tools import is_stt_enabled
            enabled = stt_enabled_for_turn(getattr(adapter, "gateway_runner", None)) \
                and _sender_may_use_bot(adapter, source) and await asyncio.to_thread(is_stt_enabled)
            for path in paths if enabled else ():
                try:
                    results[path] = await transcribe_one_clip(path)
                except Exception as exc:  # health: allow BLE001 -- an STT failure is a non-match, never a crash
                    logger.warning("Spoken mention check could not transcribe %s: %s", path, exc)
            if any(text and any(p.search(text) for p in patterns) for text, _note in results.values()):
                for path, (text, _note) in results.items():
                    if text is not None:
                        remember_gate_transcript(path, text)
                return list(paths)
    except Exception as exc:  # health: allow BLE001 -- an unroutable source is dropped by ingress anyway
        logger.warning("Spoken mention check could not enter the routed profile: %s", exc)
    for path in paths:
        if path not in keep:
            with contextlib.suppress(OSError):
                os.remove(path)
    return []


class ChatOrderedWork:
    """Inbound work that may wait on gate-time STT runs off the platform's single reader, FIFO per
    chat: a slow voice note delays only its own chat, and a message sent after it in that chat
    still arrives after it. At most ``limit`` gate transcriptions run at once."""

    def __init__(self, adapter: Any, limit: int = 4):
        self._adapter, self._tails, self._slots = adapter, {}, asyncio.Semaphore(limit)

    def busy(self, chat: str) -> bool:
        tail = self._tails.get(chat)
        return tail is not None and not tail.done()

    def submit(self, chat: str, work, *, transcribes: bool) -> asyncio.Task:
        prev = self._tails.get(chat)

        async def run():
            if prev is not None and not prev.done():
                await asyncio.wait([prev])
            try:
                if transcribes:
                    async with self._slots:
                        await work()
                else:
                    await work()
            except Exception:  # health: allow BLE001 -- one message's failure must not stall its chat
                logger.exception("[%s] inbound work for chat %s failed", getattr(self._adapter, "name", "?"), chat)

        task = asyncio.create_task(run())
        self._tails[chat] = task
        task.add_done_callback(lambda t: self._tails.pop(chat, None) if self._tails.get(chat) is t else None)
        tasks = getattr(self._adapter, "_background_tasks", None)
        if isinstance(tasks, set):
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        return task


def chat_ordered_work(adapter: Any) -> ChatOrderedWork:
    """The adapter's :class:`ChatOrderedWork`, created on first use."""
    return adapter.__dict__.setdefault("_chat_ordered_work", ChatOrderedWork(adapter))


def compile_mention_patterns(raw, *, log_prefix: str, platform_label: str | None = None,
                             display_label: str | None = None, defaults: list[str] | None = None,
                             logger_: logging.Logger | None = None) -> list[re.Pattern]:
    """Compile regex wake-word/mention patterns from config or env values.

    * **Config-style** (dingtalk, telegram): pass ``platform_label``. ``raw`` must be a
      list or string (else warn + ``[]``); non-string entries skipped; info log on load.
    * **Wakeword-style** (photon, bluebubbles): pass ``defaults``. ``raw`` may be None
      (defaults), a string (JSON list or comma/newline separated), a list, or a scalar.
    ``log_prefix`` is interpolated into every log line so per-adapter output stays
    byte-identical to the historical inline implementations.
    """
    log = logger_ or logger

    def _compile(patterns, warn_fmt, *warn_args):
        compiled: list[re.Pattern] = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                log.warning(warn_fmt, log_prefix, *warn_args, pattern, exc)
        return compiled

    if platform_label is not None:
        display = display_label or platform_label
        if raw is None:
            return []
        patterns = [raw] if isinstance(raw, str) else raw
        if not isinstance(patterns, list):
            log.warning("[%s] %s mention_patterns must be a list or string; got %s",
                        log_prefix, platform_label, type(patterns).__name__)
            return []
        compiled = _compile([p for p in patterns if isinstance(p, str) and p.strip()],
                            "[%s] Invalid %s mention pattern %r: %s", display)
        if compiled:
            log.info("[%s] Loaded %d %s mention pattern(s)", log_prefix, len(compiled), display)
        return compiled
    if raw is None:
        patterns = list(defaults or [])
    elif isinstance(raw, str):
        text = raw.strip()
        try:
            loaded = json.loads(text) if text else []
        except Exception:
            loaded = None
        patterns = loaded if isinstance(loaded, list) else [
            part.strip() for line in text.splitlines() for part in line.split(",")]
    else:
        patterns = raw if isinstance(raw, list) else [raw]
    texts = [t for t in (str(p).strip() for p in patterns) if t]
    return _compile(texts, "[%s] Invalid mention pattern %r: %s")


# ─── Fence-Aware Markdown Chunking ───────────────────────────────────────────
# Shared core for gateway/stream_consumer.py (``prefer_paragraphs=False, balance_fences=True``),
# paragraph mode (``prefer_paragraphs=True, balance_fences=False``, derived from the Yuanbao adapter,
# now a plugin) and weixin (``greedy_pack_blocks``).


def text_has_unclosed_fence(text: str) -> bool:
    """Return True when *text* ends inside an unclosed ``` code fence."""
    return sum(line.startswith('```') for line in text.split('\n')) % 2 == 1


def text_ends_with_table_row(text: str) -> bool:
    """True when the last non-empty line starts and ends with ``|``."""
    return _is_pipe_row(text.rstrip().split('\n')[-1])


def is_fence_atom(text: str) -> bool:
    """True when an atomic block is a code block (starts with ```)."""
    return text.lstrip().startswith('```')


def _is_pipe_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith('|') and stripped.endswith('|')


def is_table_atom(text: str) -> bool:
    """True when an atomic block is a table (first line is ``|...|``)."""
    return _is_pipe_row(text.split('\n')[0])


_SENTENCE_END_NEWLINE_RE = re.compile(r'[。！？.!?]\n')


def _cp_budget(text, budget, len_fn):
    """Code-point count of the longest prefix of *text* within *budget* ``len_fn`` units
    (callers guarantee ``len_fn(text) > budget``, so plain ``len`` needs no search)."""
    if len_fn is len:
        return budget
    from gateway.platforms.base import _custom_unit_to_cp  # heavyweight; lazy
    return _custom_unit_to_cp(text, budget, len_fn)


def split_at_paragraph_boundary(text, max_chars, len_fn=None):
    """Split at the nearest paragraph boundary within *max_chars*; return (head, tail).

    Priority: blank line → newline after sentence-ending punctuation (CJK/ASCII)
    → last newline → forced split at the window boundary. ``head + tail == text``
    always holds. *len_fn* measures in custom units (e.g. UTF-16 code units).
    """
    _len = len_fn or len
    if _len(text) <= max_chars:
        return text, ''
    window = text[:_cp_budget(text, max_chars, _len)]
    pos = window.rfind('\n\n')
    sentence_ends = [m.end() for m in _SENTENCE_END_NEWLINE_RE.finditer(window)]
    cut = pos + 2 if pos > 0 else (sentence_ends[-1] if sentence_ends else 0)
    if not cut:
        pos = window.rfind('\n')
        cut = pos + 1 if pos > 0 else len(window)
    return text[:cut], text[cut:]


def split_markdown_atoms(text: str) -> list[str]:
    """Split markdown into indivisible atoms: fenced code blocks, tables
    (consecutive ``|...|`` lines) and paragraphs. Blank lines belong to no atom."""
    atoms: list[str] = []
    current_lines: list[str] = []
    in_fence = False

    def _flush_current() -> None:
        if current_lines:
            atom = '\n'.join(current_lines)
            if atom.strip():
                atoms.append(atom)
            current_lines.clear()

    for line in text.split('\n'):
        if in_fence:
            current_lines.append(line)
            if line.startswith('```'):
                in_fence = False
                _flush_current()
        elif line.startswith('```'):
            _flush_current()
            in_fence = True
            current_lines.append(line)
        elif line.strip() == '':
            _flush_current()
        else:
            # A table line and a non-table line never share an atom.
            if current_lines and _is_pipe_row(current_lines[-1]) != _is_pipe_row(line):
                _flush_current()
            current_lines.append(line)
    _flush_current()
    return atoms


def infer_block_separator(prev_chunk: str, next_chunk: str) -> str:
    """``'\\n'`` when the boundary sits at a code fence or a continued table, else ``'\\n\\n'``."""
    prev_trimmed = prev_chunk.rstrip()
    next_trimmed = next_chunk.lstrip()
    if prev_trimmed.endswith('```') or next_trimmed.startswith('```'):
        return '\n'
    if text_ends_with_table_row(prev_chunk) and next_trimmed and _is_pipe_row(next_trimmed.split('\n')[0]):
        return '\n'
    return '\n\n'


def merge_streaming_fences(chunks: list[str]) -> list[str]:
    """Rejoin chunks truncated mid-fence: while chunk *i* has an unclosed fence
    and a successor exists, merge the successor in via :func:`infer_block_separator`."""
    result: list[str] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        while text_has_unclosed_fence(current) and i + 1 < len(chunks):
            current = current + infer_block_separator(current, chunks[i + 1]) + chunks[i + 1]
            i += 1
        result.append(current)
        i += 1
    return result


def balance_fences_across_chunks(chunks: list[str]) -> list[str]:
    """Close orphaned ``` fences at each chunk boundary and reopen (with the
    original language tag) on the next, so every chunk is fence-balanced alone."""
    if len(chunks) <= 1:
        return chunks
    out: list[str] = []
    carry_lang = None
    for chunk in chunks:
        body = f"```{carry_lang}\n{chunk}" if carry_lang is not None else chunk
        in_code, lang = fence_state_after(chunk, carry_lang is not None, carry_lang or "")
        carry_lang = lang if in_code else None
        out.append(body + "\n```" if in_code else body)
    return out


def _table_header_above(lines: list[str], i: int) -> int:
    """Index of the header row of the pipe table containing ``lines[i]``, else -1. Same rules as
    :func:`convert_table_to_bullets`: a header with a ``|`` over a separator row, then rows with a
    ``|`` (outer pipes optional)."""
    while i >= 1 and is_table_row(lines[i]):
        if TABLE_SEPARATOR_RE.match(lines[i]) and is_table_row(lines[i - 1]):
            return i - 1
        i -= 1
    return -1


def _split_inside_table(head: str, tail: str) -> bool:
    return (is_table_row(head.rstrip('\n').rsplit('\n', 1)[-1])
            and is_table_row(tail.lstrip().split('\n', 1)[0]))


def table_row_start(text: str, split_at: int) -> int:
    """The newline before the table row a mid-line *split_at* falls in, so a row that fits a fresh
    chunk is never cut inside a cell; *split_at* unchanged when the line is not a table row (or is
    the first line, which only an oversized row reaches)."""
    newline = text.rfind('\n', 0, split_at)
    if newline < 1 or text[split_at:split_at + 1] in ('\n', ''):
        return split_at
    line_end = text.find('\n', split_at)
    lines = text[:line_end if line_end >= 0 else len(text)].split('\n')
    return newline if _table_header_above(lines, len(lines) - 1) >= 0 else split_at


def table_rollback_point(text: str, split_at: int, fits) -> int:
    """When ``text[:split_at]`` ends inside a pipe table that *fits* whole in a fresh chunk,
    return the offset of its header line so the table moves intact; else *split_at*."""
    if not _split_inside_table(text[:split_at], text[split_at:]):
        return split_at
    head_lines = text[:split_at].split('\n')
    lines = head_lines + [text[split_at:].lstrip().split('\n', 1)[0]]  # sees a separator just past the cut
    header = _table_header_above(lines, len(lines) - 1)
    if header <= 0:
        return split_at
    start = len('\n'.join(lines[:header])) + 1
    if header >= len(head_lines) - 2:  # header (+ separator) with no rows never ends a chunk
        return start
    end = start
    for line in text[start:].split('\n'):
        if not is_table_row(line):
            break
        end += len(line) + 1
    return start if fits(text[start:end].rstrip('\n')) else split_at


def table_header_to_repeat(chunk: str, rest: str, max_units: int, len_fn=len):
    """Header + separator lines to prepend to *rest* when *chunk* ended mid-table, else None
    (also None when the header alone exceeds *max_units*, so a huge header can't stall)."""
    if not _split_inside_table(chunk, rest):
        return None
    lines = chunk.rstrip('\n').split('\n')
    header = _table_header_above(lines, len(lines) - 1)
    if header < 0:
        return None
    block = f"{lines[header]}\n{lines[header + 1]}"
    return block if len_fn(block) <= max_units else None


def fence_state_after(text: str, in_code: bool = False, lang: str = "") -> tuple[bool, str]:
    """Walk ``text`` line by line toggling on ``` lines; return the final (in_code, lang)."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            tag = stripped[3:].split()
            in_code, lang = (False, "") if in_code else (True, tag[0] if tag else "")
    return in_code, lang


def greedy_pack_blocks(blocks, max_length, len_fn=None, sep="\n\n", overflow=None):
    """Greedily pack *blocks* (joined with *sep*) into chunks of at most *max_length*; an
    oversized block goes through *overflow(block)* (-> list of chunks) if given, else as-is."""
    _len = len_fn or len
    packed: list[str] = []
    current = ""
    for block in blocks:
        candidate = block if not current else f"{current}{sep}{block}"
        if _len(candidate) <= max_length:
            current = candidate
            continue
        if current:
            packed.append(current)
            current = ""
        if _len(block) <= max_length:
            current = block
        elif overflow is not None:
            packed.extend(overflow(block))
        else:
            packed.append(block)
    if current:
        packed.append(current)
    return packed


def split_text_fence_aware(text, limit, len_fn=None, *, prefer_paragraphs=True,
                           balance_fences=False):
    """Split markdown into chunks of at most *limit*, respecting fences.

    ``prefer_paragraphs=True`` (yuanbao-derived): atoms (fences, tables, paragraphs) are
    greedily merged, oversized non-atomic chunks split at paragraph boundaries, small
    neighbours re-merged; a single atom larger than *limit* is emitted oversize, not broken.
    ``prefer_paragraphs=False`` (stream_consumer-derived): newline-preferred hard splitting
    with headroom reserved for fence markers. ``balance_fences=True`` closes/reopens fences
    at chunk boundaries so each chunk renders standalone.
    """
    if not text:
        return []
    if prefer_paragraphs:
        chunks = _chunk_markdown_paragraphs(text, limit, len_fn)
    else:
        chunks = _chunk_newline_preferred(text, limit, len_fn or len)
    return balance_fences_across_chunks(chunks) if balance_fences else chunks


def _chunk_markdown_paragraphs(text, max_chars, len_fn=None):
    """Yuanbao-derived paragraph/atom chunking pipeline (see module docs)."""
    _len = len_fn or len
    if _len(text) <= max_chars:
        return [text]
    # Phase 2: greedy merge; oversized fence/table atoms stay indivisible.
    chunks: list[str] = []
    indivisible_set: set[int] = set()
    current_parts: list[str] = []
    current_len = 0
    for atom in split_markdown_atoms(text):
        atom_len = _len(atom)
        sep_len = 2 if current_parts else 0
        if current_len + sep_len + atom_len > max_chars and current_parts:
            chunks.append('\n\n'.join(current_parts))
            current_parts, current_len, sep_len = [], 0, 0
        if (not current_parts and atom_len > max_chars
                and (is_fence_atom(atom) or is_table_atom(atom))):
            indivisible_set.add(len(chunks))
            chunks.append(atom)
            continue
        current_parts.append(atom)
        current_len += sep_len + atom_len
    if current_parts:
        chunks.append('\n\n'.join(current_parts))
    # Phase 3: split still-oversized divisible chunks at paragraph boundaries.
    result: list[str] = []
    for idx, chunk in enumerate(chunks):
        if _len(chunk) <= max_chars or idx in indivisible_set or text_has_unclosed_fence(chunk):
            result.append(chunk)
            continue
        remaining = chunk
        while _len(remaining) > max_chars:
            head, remaining = split_at_paragraph_boundary(remaining, max_chars, len_fn=len_fn)
            if not head:
                head, remaining = remaining[:max_chars], remaining[max_chars:]
            if head:
                result.append(head)
        if remaining:
            result.append(remaining)
    # Phase 4: merge small chunks with neighbours.
    merged: list[str] = result[:1]
    for chunk in result[1:]:
        combined = merged[-1] + '\n\n' + chunk
        if _len(combined) <= max_chars:
            merged[-1] = combined
        else:
            merged.append(chunk)
    return [c for c in merged if c]


def _chunk_newline_preferred(text, limit, len_fn):
    """Stream-consumer-derived newline-preferred splitting (no balancing)."""
    if len_fn(text) <= limit:
        return [text]
    # Reserve headroom for fence markers a balancing pass may add.
    split_limit = max(limit - 16, limit // 2, 1) if "```" in text else limit
    chunks: list[str] = []
    remaining = text
    while len_fn(remaining) > split_limit:
        budget = _cp_budget(remaining, split_limit, len_fn)
        split_at = remaining.rfind("\n", 0, budget)
        if split_at < budget // 2:
            split_at = budget
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


# ─── Discord channel obfuscation (Bot API change, mandatory Nov 16 2026) ───────
# Channels a bot lacks VIEW_CHANNEL on are still dispatched over the Gateway with the
# name replaced by "___hidden___", sensitive fields nulled and flag 1 << 17 set; over
# HTTP they are omitted. https://discord.com/developers/docs/change-log (Aug 12, 2026)
DISCORD_CHANNEL_OBFUSCATED_FLAG = 1 << 17
DISCORD_OBFUSCATED_CHANNEL_NAME = "___hidden___"


def is_discord_channel_obfuscated(channel) -> bool:
    """True when a discord.py channel object is an obfuscated placeholder.

    Skip these wherever guild channels are enumerated (channel directory, message
    backfill): history reads and sends always fail with a permission error. Flag
    first; the sentinel name is the fallback for discord.py builds that don't
    expose the flag (a visible channel literally named ``___hidden___`` is then
    also skipped — deliberate bias toward hiding). #90154
    """
    try:
        flag_value = channel.flags.value
    except AttributeError:
        flag_value = None
    if isinstance(flag_value, int) and flag_value & DISCORD_CHANNEL_OBFUSCATED_FLAG:
        return True
    return getattr(channel, "name", None) == DISCORD_OBFUSCATED_CHANNEL_NAME


def platform_http_client(**kwargs: Any) -> Any:
    """``httpx.AsyncClient`` for a platform adapter: connect-time SSRF validation
    (``tools.url_safety.create_ssrf_safe_async_client``), every redirect hop re-checked
    (``gateway.platforms.base.ssrf_redirect_guard``, run before any caller ``response`` hooks) and the
    shared keepalive pool limits (``platform_httpx_limits()``; pass ``limits=None`` for httpx's
    defaults). Defaults ``timeout=30.0`` and ``follow_redirects=True``; other kwargs go to httpx."""
    from gateway.platforms._http_client_limits import platform_httpx_limits
    from gateway.platforms.base import ssrf_redirect_guard
    from tools.url_safety import create_ssrf_safe_async_client
    hooks = dict(kwargs.pop("event_hooks", None) or {})
    hooks["response"] = [ssrf_redirect_guard, *hooks.get("response", ())]
    kwargs.setdefault("timeout", 30.0)
    kwargs.setdefault("follow_redirects", True)
    if "limits" not in kwargs:
        kwargs["limits"] = platform_httpx_limits()
    elif kwargs["limits"] is None:
        del kwargs["limits"]
    return create_ssrf_safe_async_client(event_hooks=hooks, **kwargs)


_AIOHTTP_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_AIOHTTP_MAX_REDIRECTS = 10


def _origin(url: str) -> tuple:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    return scheme, parts.hostname, parts.port or (443 if scheme == "https" else 80)


def _sends_credentials(session: Any, kwargs: dict) -> bool:
    """Whether a ``session.get(**kwargs)`` would carry an ``Authorization`` header or basic auth."""
    headers = [*getattr(session, "headers", {}), *(kwargs.get("headers") or {})]
    return (getattr(session, "_default_auth", None) is not None or kwargs.get("auth") is not None
            or any(str(name).lower() == "authorization" for name in headers))


@contextlib.asynccontextmanager
async def ssrf_checked_aiohttp_get(session: Any, url: str, *, first_hop_vetted: bool = False, **kwargs: Any):
    """``session.get(url)`` on an aiohttp session with every redirect hop checked by
    ``tools.url_safety.async_is_safe_url``. aiohttp follows redirects by default, so a pre-flight
    check alone lets a public URL 302 to loopback or cloud metadata. Raises ``ValueError`` on a
    blocked hop, a credential-bearing cross-origin hop, or too many redirects. ``first_hop_vetted``
    skips the check on ``url`` itself for a caller that vetted it another way (a host allowlist);
    redirect targets are always checked."""
    from tools.url_safety import async_is_safe_url
    # aiohttp spends one ``total`` across its own redirect loop and the body read; each hop here
    # is a fresh request, so the whole chain shares one deadline or ``total`` multiplies per hop.
    total = getattr(kwargs.get("timeout") or getattr(session, "timeout", None), "total", None)
    async with asyncio.timeout(total if isinstance(total, (int, float)) else None):
        source_origin, current_url = _origin(url), url
        for hop in range(_AIOHTTP_MAX_REDIRECTS + 1):
            if (hop or not first_hop_vetted) and not await async_is_safe_url(current_url):
                raise ValueError("Blocked URL targeting a private or internal address")
            # aiohttp strips Authorization on a cross-origin redirect; a fresh session.get() re-applies
            # session defaults, so refuse the hop rather than hand the first origin's credentials on.
            if _origin(current_url) != source_origin and _sends_credentials(session, kwargs):
                raise ValueError("Blocked credential-bearing cross-origin redirect")
            async with session.get(current_url, allow_redirects=False, **kwargs) as response:
                location = response.headers.get("Location")
                if response.status in _AIOHTTP_REDIRECT_STATUSES and location:
                    current_url = urljoin(current_url, location)
                    continue
                yield response
                return
        raise ValueError("Too many redirects")

# The slowest uplink a media upload's budgets still cover, and the wall-clock ceiling they grow to.
MIN_MEDIA_UPLOAD_RATE = 128 * 1024  # B/s
MEDIA_UPLOAD_DEADLINE_MAX = 3600.0


def _on_disk_size(value: Any) -> int:
    if isinstance(value, (str, os.PathLike)):
        try:
            return os.stat(value).st_size
        except (OSError, ValueError):
            return 0  # a URL or platform file id
    try:  # an open file handle
        return os.fstat(value.fileno()).st_size
    except (AttributeError, OSError, ValueError):
        return 0


def media_upload_seconds(sources: Any) -> float:
    """Seconds the on-disk payloads in ``sources`` (paths, open files; URLs and ids count 0) take
    at ``MIN_MEDIA_UPLOAD_RATE``."""
    return sum(_on_disk_size(source) for source in sources) / MIN_MEDIA_UPLOAD_RATE


def scaled_upload_deadline(deadline: float, upload_seconds: float) -> float:
    """A media send's wall-clock cap widened by 1.5x its upload time (at most
    ``MEDIA_UPLOAD_DEADLINE_MAX``, never below ``deadline``): a platform answers an upload only once
    the file has arrived, so a fixed cap reports a slow-but-delivered file as failed (#133093)."""
    if upload_seconds <= 0:
        return deadline
    return max(deadline, min(MEDIA_UPLOAD_DEADLINE_MAX, deadline + 1.5 * upload_seconds))


# ── Long reply handoff (#152012) ─────────────────────────────────────────────────────────────────
# A reply that needs more than an adapter's ``MAX_SPLIT_MESSAGES`` goes out whole as a document:
# a runaway reply sent as hundreds of messages gets the bot flood-banned, and cutting it loses text.


def reply_file_parts(text: str, limit_bytes: int) -> list[bytes]:
    """``text`` as UTF-8 parts of at most ``limit_bytes`` each, cut at a line break when one is in the
    back half of a part and never inside a character; one part when it fits."""
    data = text.encode("utf-8")
    parts: list[bytes] = []
    while len(data) > limit_bytes:
        cut = data.rfind(b"\n", 0, limit_bytes) + 1
        if cut < limit_bytes // 2:
            cut = limit_bytes
            while cut and (data[cut] & 0xC0) == 0x80:  # a UTF-8 continuation byte: back up to its lead byte
                cut -= 1
        parts.append(data[:cut])
        data = data[cut:]
    return [*parts, data] if data or not parts else parts


def reply_document_sender(adapter: Any) -> Any:
    """The adapter's upload for a reply file: ``send_reply_document`` (an upload whose failure is a
    failed result, never a text notice reported as sent), else a native ``send_document``; None when
    it has neither (the base ``send_document`` only posts a failure notice)."""
    from gateway.platforms.base import BasePlatformAdapter
    if callable(getattr(type(adapter), "send_reply_document", None)):
        return adapter.send_reply_document
    native = getattr(type(adapter), "send_document", None) not in (None, BasePlatformAdapter.send_document)
    return adapter.send_document if native else None


async def send_reply_as_file(adapter: Any, chat_id: str, text: str, *, reply_to: Optional[str] = None,
                             metadata: Optional[dict] = None) -> Optional[list[str]]:
    """Upload ``text`` whole as ``reply-<id>.md`` (numbered files when it is over the adapter's
    ``REPLY_FILE_MAX_BYTES``), the localized caption on the first. Returns the files' message ids, or
    ``None`` when the adapter cannot upload or an upload was not confirmed: the caller then delivers
    the text as messages and deletes nothing, so no text is lost."""
    import shutil
    import tempfile
    from agent.i18n import t

    send = reply_document_sender(adapter)
    if send is None:
        return None
    parts = reply_file_parts(text, int(getattr(adapter, "REPLY_FILE_MAX_BYTES", 8 * 1024 * 1024)))
    stem, ids = f"reply-{secrets.token_hex(4)}", []
    folder = await asyncio.to_thread(tempfile.mkdtemp, prefix="hermes-reply-")
    try:
        for index, part in enumerate(parts, 1):
            name = f"{stem}.md" if len(parts) == 1 else f"{stem}-{index}.md"
            path = os.path.join(folder, name)
            await asyncio.to_thread(Path(path).write_bytes, part)
            result = await send(chat_id, path, caption=t("platform.shared.response_as_file") if index == 1 else None,
                                file_name=name, reply_to=reply_to if index == 1 else None, metadata=metadata)
            raw = getattr(result, "raw_response", None)
            if not getattr(result, "success", False) or (isinstance(raw, dict) and raw.get("ambiguous")):
                logger.warning("Long reply file %s was not confirmed delivered: %s", name, getattr(result, "error", None))
                return None
            if getattr(result, "message_id", None):
                ids.append(str(result.message_id))
    finally:
        await asyncio.to_thread(shutil.rmtree, folder, True)
    return ids


def reply_message_cap(adapter: Any) -> Optional[int]:
    """How many messages one reply may take on ``adapter`` before it goes out as a file
    (``MAX_SPLIT_MESSAGES``); None when it has no cap or cannot upload one."""
    cap = getattr(adapter, "MAX_SPLIT_MESSAGES", None)
    if type(cap) is not int or cap <= 1 or reply_document_sender(adapter) is None:
        return None
    return cap


def _over_reply_cap(adapter: Any, content: str, metadata: Optional[dict]) -> bool:
    """``content`` needs more messages than the adapter's per-reply cap (``reply_message_count``)."""
    cap, count = reply_message_cap(adapter), getattr(type(adapter), "reply_message_count", None)
    if cap is None or not callable(count) or not content:
        return False
    # Fast path for ordinary messages: escaping, UTF-16 units and line-break splits grow a text by
    # at most 8x, so one this short always fits the cap and is never formatted twice.
    if len(content) * 8 <= cap * int(getattr(adapter, "MAX_MESSAGE_LENGTH", 0) or 0):
        return False
    return adapter.reply_message_count(content, metadata) > cap


async def send_long_reply_as_file(adapter: Any, chat_id: str, content: str, *, reply_to: Optional[str] = None,
                                  metadata: Optional[dict] = None) -> Any:
    """A whole reply that needs more than the adapter's ``MAX_SPLIT_MESSAGES`` messages goes out as
    one file instead: the ``SendResult`` of that upload, or ``None`` to send it as messages as usual
    (under the cap, no upload support, or the upload failed: nothing is lost either way)."""
    return (await try_reply_file(adapter, chat_id, content, reply_to=reply_to, metadata=metadata))[1]


async def try_reply_file(adapter: Any, chat_id: str, content: str, *, reply_to: Optional[str] = None,
                         metadata: Optional[dict] = None) -> tuple[bool, Any]:
    """``(tried, result)`` for :func:`send_long_reply_as_file`: ``tried`` says an upload was attempted,
    so a caller that recounts later (``hand_off_split``) never uploads the same reply twice. A piece
    of a streamed reply (``reply_file_off``) is never filed on its own."""
    if _REPLY_FILE_OFF.get() or not _over_reply_cap(adapter, content, metadata):
        return False, None
    return True, await _file_result(adapter, chat_id, content, reply_to, metadata)


async def _file_result(adapter: Any, chat_id: str, content: str, reply_to: Optional[str],
                       metadata: Optional[dict]) -> Any:
    ids = await send_reply_as_file(adapter, chat_id, content, reply_to=reply_to, metadata=metadata)
    if ids is None:
        logger.warning("Long reply could not be sent as a file; sending it as messages")
        return None
    from gateway.platforms.base import SendResult
    return SendResult(success=True, message_id=ids[0] if ids else None,
                      raw_response={"message_ids": ids, "sent_as_file": True})


# Sends that must not become a reply file: the stream consumer sends one reply as many pieces
# (``adapter.send`` per preview, seal and tail) and counts them itself, so no piece is filed on its own;
# and once a reply's upload failed, its message fallback must not try the upload again. A complete
# message the consumer sends (commentary) opts back in with ``reply_file_off(False)``.
_REPLY_FILE_OFF: contextvars.ContextVar[bool] = contextvars.ContextVar("hermes_reply_file_off", default=False)


@contextlib.contextmanager
def reply_file_off(on: bool = True) -> Iterator[None]:
    token = _REPLY_FILE_OFF.set(on)
    try:
        yield
    finally:
        _REPLY_FILE_OFF.reset(token)


async def hand_off_split(adapter: Any, chat_id: str, content: str, messages: Any, landed: tuple = (), *,
                         reply_to: Optional[str] = None, metadata: Optional[dict] = None) -> Any:
    """An adapter's send is about to post ``messages`` (a count, or a callable computing it) for
    ``content`` after the ``landed`` message ids it already posted for it, a count its preflight did
    not predict (a refused rich send going out as MarkdownV2 chunks). Past ``MAX_SPLIT_MESSAGES`` the
    whole reply goes out as one file and ``landed`` is deleted once the upload is confirmed: the
    file's ``SendResult``. ``None`` sends the messages as planned (under the cap, a stream piece, or
    the upload failed: nothing deleted)."""
    cap = reply_message_cap(adapter)
    if cap is None or _REPLY_FILE_OFF.get() or not content:
        return None
    if len(content) * 8 <= (cap - len(landed)) * int(getattr(adapter, "MAX_MESSAGE_LENGTH", 0) or 0):
        return None  # same fast path as ``_over_reply_cap``: this short a text always fits
    if len(landed) + (messages() if callable(messages) else messages) <= cap:
        return None
    result = await _file_result(adapter, chat_id, content, reply_to, metadata)
    if result is not None:
        for message_id in landed:
            if not await adapter.delete_message(chat_id, str(message_id)):
                logger.warning("Long reply: could not delete earlier part %s", message_id)
    return result
