"""Discord reconnect recovery: the missed-message backfill's config, and the planned-stop replay.

A planned stop (restart, SIGUSR1, SIGTERM through the clean path) withdraws every adapter's message
handler first (``gateway.run_runtime.drain_gateway_runtime``) and tears the adapter down after. Inbound
that reaches the bot in between got no turn: a text batch still in its debounce was flushed into an
adapter with no handler and dropped, a message the drain refused got only a notice, and Discord does
not redeliver what was sent while the process was down. The stop replay records that window in one
small file under the profile's Hermes home and, after the next connect, sends each message once
through the live ingress gates (``_dispatch_recovered_message``), the same path as the opt-in
``missed_message_backfill``. The window is the stop's own: it opens when admission is withdrawn,
messages that already reached a session before it are excluded by id, and a window older than
``_MAX_AGE_SECONDS`` (a long outage, not a restart) is not replayed. Crashes write no window.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections import deque
from contextlib import suppress
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_STATE_FILENAME = "discord_stop_replay.json"
_STATE_VERSION = 1
_MAX_AGE_SECONDS = 1800.0
_MAX_REPLAYS = 20
_HISTORY_LIMIT = 100  # per channel, inside a window of seconds to minutes
_RECENT_SECONDS = 1800.0  # channels active this recently are scanned for the window
_CLOCK_MARGIN_SECONDS = 1.0  # local clock vs. Discord snowflake time
_FALSE = {"false", "0", "no", "off"}


def _ref(message: Any) -> Optional[tuple[str, str]]:
    message_id = str(getattr(message, "id", "") or "")
    channel_id = str(getattr(getattr(message, "channel", None), "id", "") or "")
    return (message_id, channel_id) if message_id and channel_id else None


def message_refs(event: Any) -> list[tuple[str, str]]:
    """``(message_id, channel_id)`` of every Discord message one event carries (a text batch merges
    several; the batch's ``raw_message`` is only the first)."""
    merged = getattr(event, "_discord_message_refs", None)
    if merged:
        return list(merged)
    ref = _ref(getattr(event, "raw_message", None))
    return [ref] if ref else []


def recovery_extra(discord_cfg: dict) -> dict:
    """``config.extra`` seeds for the ``discord.missed_message_backfill`` and ``discord.stop_replay``
    YAML keys (extra only: neither has an env bridge)."""
    seeded: dict[str, Any] = {}
    if isinstance(discord_cfg.get("missed_message_backfill"), dict):
        seeded["missed_message_backfill"] = dict(discord_cfg["missed_message_backfill"])
    if "stop_replay" in discord_cfg:
        seeded["stop_replay"] = discord_cfg["stop_replay"]
    return seeded


class StopReplayWindow:
    """The stop window of one adapter process and its state file (atomic, mode 600)."""

    def __init__(self, hermes_home: Path) -> None:
        self.path = Path(hermes_home) / "gateway" / _STATE_FILENAME
        self.withdrawn_at: Optional[float] = None
        self.opened_at: Optional[float] = None
        self.batch_window = 0.0
        self.deferred: dict[str, str] = {}
        self._dispatched: deque[tuple[float, str, str]] = deque(maxlen=512)

    @property
    def open(self) -> bool:
        return self.opened_at is not None

    def note_dispatched(self, refs: list[tuple[str, str]]) -> None:
        now = time.time()
        self._dispatched.extend((now, message_id, channel_id) for message_id, channel_id in refs)

    def withdraw(self, batch_window: float) -> None:
        """Admission was withdrawn: open the window and record what already reached a session."""
        now = time.time()
        self.withdrawn_at = now
        self.opened_at = self.opened_at or now
        self.batch_window = batch_window
        self.save()

    def defer(self, refs: list[tuple[str, str]], batch_window: float) -> None:
        self.opened_at = self.opened_at or time.time()
        self.batch_window = batch_window
        for message_id, channel_id in refs:
            self.deferred[message_id] = channel_id
        self.save()

    def save(self) -> None:
        from utils import atomic_json_write
        floor = (self.withdrawn_at or self.opened_at) - self.batch_window - _CLOCK_MARGIN_SECONDS
        recent = time.time() - _RECENT_SECONDS
        payload = {
            "version": _STATE_VERSION,
            "withdrawn_at": self.withdrawn_at, "opened_at": self.opened_at, "floor": floor,
            "deferred": self.deferred,
            # Reached a session before the stop (steered, queued, answered): never replayed.
            "skip": sorted({mid for ts, mid, _ in self._dispatched if ts >= floor} - set(self.deferred)),
            "channels": sorted({cid for ts, _, cid in self._dispatched if ts >= recent}
                               | set(self.deferred.values())),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            atomic_json_write(self.path, payload, mode=0o600)
        except OSError as exc:
            logger.warning("[Discord] Cannot write the stop-replay window to %s: %s", self.path, exc)

    def load(self) -> Optional[dict]:
        """The previous process's window, or None (missing; unreadable is logged and removed)."""
        import json
        try:
            raw = self.path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("[Discord] Ignoring unreadable stop-replay window %s: %s", self.path, exc)
            return None
        try:
            state = json.loads(raw)
            if state.get("version") != _STATE_VERSION:
                raise ValueError(f"version {state.get('version')!r}")
            state["opened_at"] = float(state["opened_at"])
            state["floor"] = float(state["floor"])
            state["deferred"] = {str(k): str(v) for k, v in dict(state.get("deferred") or {}).items()}
            state["skip"] = {str(v) for v in state.get("skip") or ()}
            state["channels"] = [str(v) for v in state.get("channels") or ()]
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("[Discord] Ignoring corrupt stop-replay window %s: %s", self.path, exc)
            self.clear()
            return None
        return state

    def clear(self) -> None:
        with suppress(FileNotFoundError):
            self.path.unlink()


class DiscordRecoveryMixin:
    """Reconnect recovery for ``DiscordAdapter``: listed before ``BasePlatformAdapter`` so its
    ``set_message_handler`` / ``handle_message`` / ``_enqueue_text_event`` wrap the base ones."""

    # ── missed-message backfill config ────────────────────────────────────

    def _missed_message_backfill_enabled(self) -> bool:
        """Whether to reconcile Discord messages missed while the gateway was down."""
        from plugins.platforms.discord.adapter import _scoped_gate_env
        configured = self.config.extra.get("missed_message_backfill")
        if isinstance(configured, dict) and "enabled" in configured:
            value = configured["enabled"]
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "on")
            return bool(value)
        raw = _scoped_gate_env("DISCORD_MISSED_MESSAGE_BACKFILL", "false")
        return str(raw).strip().lower() in ("true", "1", "yes", "on")

    def _missed_message_backfill_channels(self) -> set[str]:
        """Channels to scan for missed messages after reconnect: union of allowed and
        free-response channels by default; ``channels: "*"`` scans every text channel."""
        from plugins.platforms.discord.adapter import _decode_json_list_literal
        configured = self.config.extra.get("missed_message_backfill")
        if isinstance(configured, dict) and "channels" in configured:
            raw = configured.get("channels")
            channels = self._gate_csv_set(raw)
            # An explicit list (YAML list or JSON-list string, even empty) is authoritative — the
            # operator disabled the scan; only the default "" string falls through to the env/default.
            if channels or isinstance(_decode_json_list_literal(raw), list):
                return channels
        raw = self._gate_env("DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS")
        if not raw.strip():
            allowed = self._get_allowed_channels()
            return allowed | self._discord_free_response_channels()
        return self._gate_csv_set(raw)

    def _missed_message_backfill_number(self, key: str, env_key: str, default, cast, lo, hi=None):
        """Numeric ``missed_message_backfill.<key>`` (dict extra wins over env), clamped to [lo, hi]."""
        from plugins.platforms.discord.adapter import _scoped_gate_env
        configured = self.config.extra.get("missed_message_backfill")
        raw = configured.get(key, default) if isinstance(configured, dict) else _scoped_gate_env(env_key, str(default))
        try:
            value = cast(raw)
        except (TypeError, ValueError):
            value = cast(default)
        return max(lo, value) if hi is None else max(lo, min(value, hi))

    def _missed_message_backfill_window_seconds(self) -> float:
        return self._missed_message_backfill_number(
            "window_seconds", "DISCORD_MISSED_MESSAGE_BACKFILL_WINDOW_SECONDS", 21600, float, 60.0)

    def _missed_message_backfill_limit(self) -> int:
        return self._missed_message_backfill_number("limit", "DISCORD_MISSED_MESSAGE_BACKFILL_LIMIT", 100, int, 1, 500)

    def _missed_message_backfill_max_dispatches(self) -> int:
        return self._missed_message_backfill_number(
            "max_dispatches", "DISCORD_MISSED_MESSAGE_BACKFILL_MAX_DISPATCHES", 10, int, 1, 100)

    def _missed_message_backfill_max_attempts(self) -> int:
        """Lifetime re-dispatch ceiling for ONE message, independent of completion state:
        ``max_dispatches`` caps a scan, not a row, so without this any message whose completion
        can never be recorded is re-run on every reconnect (#113631)."""
        return self._missed_message_backfill_number(
            "max_attempts", "DISCORD_MISSED_MESSAGE_BACKFILL_MAX_ATTEMPTS", 3, int, 1, 100)

    # ── planned-stop replay: recording ────────────────────────────────────

    def _stop_replay_enabled(self) -> bool:
        """``discord.stop_replay`` (default on)."""
        value = self.config.extra.get("stop_replay", True)
        return value if isinstance(value, bool) else str(value).strip().lower() not in _FALSE

    @cached_property
    def _stop_replay(self) -> StopReplayWindow:
        return StopReplayWindow(self._discord_recovery_store.hermes_home)

    def _stop_replay_batch_window(self) -> float:
        return max(self._text_batch_delay_seconds, self._text_batch_split_delay_seconds)

    def set_message_handler(self, handler) -> None:
        if handler is None and self._message_handler is not None and self._stop_replay_enabled():
            self._stop_replay.withdraw(self._stop_replay_batch_window())
        super().set_message_handler(handler)

    def _enqueue_text_event(self, event) -> None:
        refs = message_refs(event)
        super()._enqueue_text_event(event)
        batch = self._pending_text_batches.get(self._text_batch_key(event))
        if batch is not None and batch is not event:
            batch._discord_message_refs = [*message_refs(batch), *refs]

    async def handle_message(self, event) -> None:
        if self._message_handler is None and self._stop_replay.open:
            self.defer_refused_inbound(event)
            return
        if self._message_handler is not None:
            self._stop_replay.note_dispatched(message_refs(event))
        await super().handle_message(event)

    def defer_refused_inbound(self, event) -> None:
        """Record *event* for the replay after the restart (admission withdrawn or drain refusal)."""
        refs = message_refs(event)
        if not refs or not self._stop_replay_enabled():
            return
        self._stop_replay.defer(refs, self._stop_replay_batch_window())
        for message_id, channel_id in refs:
            logger.info("[%s] Inbound message %s in chat %s deferred to the replay after the stop",
                        self.name, message_id, channel_id)

    # ── planned-stop replay: after the next connect ───────────────────────

    async def _run_reconnect_recovery(self) -> None:
        await self._replay_stop_window()
        if self._missed_message_backfill_enabled():
            await self._run_missed_message_backfill()

    async def _replay_stop_window(self) -> None:
        window = self._stop_replay
        if window.open or not self._client:
            return  # this process is the one stopping
        state = window.load()
        if state is None:
            return
        if not self._stop_replay_enabled():
            window.clear()
            return
        age = time.time() - state["opened_at"]
        if not 0 <= age <= _MAX_AGE_SECONDS:
            logger.info("[%s] Stop-replay window skipped: it opened %.0fs ago (bound %.0fs)",
                        self.name, age, _MAX_AGE_SECONDS)
            window.clear()
            return
        candidates = await self._stop_replay_candidates(state)
        dispatched = 0
        for message in candidates[:_MAX_REPLAYS]:
            if await self._replay_one(message):
                dispatched += 1
        logger.info("[%s] Stop-replay complete: deferred=%d candidates=%d dispatched=%d capped=%d",
                    self.name, len(state["deferred"]), len(candidates), dispatched,
                    max(0, len(candidates) - _MAX_REPLAYS))
        if not window.open:
            window.clear()

    async def _replay_one(self, message: Any) -> bool:
        message_id = str(message.id)
        if self._dedup.contains(message_id):
            return False
        self._record_recovery_attempt(message, status="queued")
        try:
            admitted = await self._dispatch_recovered_message(message, claim=True)
        except Exception as exc:
            self._dedup.discard(message_id)
            self._record_recovery_attempt(message, status="failed", error=str(exc))
            logger.warning("[%s] Stop-replay of message %s failed: %s", self.name, message_id, exc,
                           exc_info=True)
            return False
        if admitted:
            logger.info("[%s] Replayed message %s in chat %s after the planned stop", self.name,
                        message_id, getattr(message.channel, "id", "?"))
        return admitted

    async def _stop_replay_candidates(self, state: dict) -> list:
        """Deferred messages, plus channel history inside the window that no session saw."""
        found: dict[str, Any] = {}
        for message_id, channel_id in state["deferred"].items():
            channel = await self._recovery_channel(channel_id)
            if channel is None:
                continue
            try:
                found[message_id] = await channel.fetch_message(int(message_id))
            except Exception as exc:  # health: allow BLE001 -- deleted, no access, REST error: that one is not replayed
                logger.warning("[%s] Cannot fetch deferred message %s: %s", self.name, message_id, exc)
        floor = dt.datetime.fromtimestamp(state["floor"], dt.UTC)
        channel_ids = set(state["channels"]) | self._missed_message_backfill_channels()
        channel_ids.discard("*")
        seen: set[str] = set()
        for channel_id in sorted(channel_ids):
            channel = await self._recovery_channel(channel_id)
            if channel is None:
                continue
            async for message in self._iter_channel_and_thread_messages(
                    channel, limit=_HISTORY_LIMIT, after=floor, seen_channels=seen):
                message_id = str(message.id)
                created = getattr(message, "created_at", None)
                if (message_id in found or message_id in state["skip"]
                        or not isinstance(created, dt.datetime) or created < floor):
                    continue
                if await self._should_backfill_discord_message(message):
                    found[message_id] = message
        return [found[key] for key in sorted(found, key=int)]

    async def _recovery_channel(self, channel_id: str) -> Any:
        from gateway.platforms.helpers import is_discord_channel_obfuscated
        try:
            channel = self._client.get_channel(int(channel_id))
            if channel is None:
                channel = await self._client.fetch_channel(int(channel_id))
        except Exception as exc:  # health: allow BLE001 -- same contract as the backfill's channel fetch: skip it
            logger.debug("[%s] Cannot fetch recovery channel %s: %s", self.name, channel_id, exc)
            return None
        return None if is_discord_channel_obfuscated(channel) else channel
