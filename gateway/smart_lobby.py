"""Optional smart lobby routing for multiplexed messaging profiles.

A configured lobby message is classified against profile descriptions, then
handed to the selected profile's own Discord adapter in a dedicated thread.
The source lobby remains a control surface; the durable conversation begins in
the target profile/thread namespace.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Mapping, Optional, cast

from gateway.config import Platform

logger = logging.getLogger(__name__)
_SNOWFLAKE_RE = re.compile(r"^[0-9]{5,30}$")
_EXPLICIT_PROFILE_RE = re.compile(
    r"^\s*(?:\[(?P<bracket>[a-z0-9_-]+)\]|@(?P<mention>[a-z0-9_-]+)|(?P<colon>[a-z0-9_-]+):)\s*",
    re.IGNORECASE,
)


@contextmanager
def _target_profile_runtime_scope(profile: str):
    """Enter the same config/secret scope used by a target profile turn."""
    from gateway.run import _profile_runtime_scope
    from hermes_cli.profiles import get_profile_dir

    with _profile_runtime_scope(get_profile_dir(profile)):
        yield


@dataclass(frozen=True)
class SmartLobbyCandidate:
    profile: str
    channel_id: str
    description: str


@dataclass(frozen=True)
class SmartLobbyConfig:
    platform: str
    chat_id: str
    default_profile: str
    candidates: dict[str, SmartLobbyCandidate]
    min_confidence: float = 0.65
    timeout_seconds: float = 20.0


@dataclass(frozen=True)
class SmartLobbyDecision:
    profile: str
    confidence: float
    title: str = ""


@dataclass(frozen=True)
class SmartLobbyRoute:
    source_key: str
    profile: str
    channel_id: str
    title: str
    status: str
    thread_id: Optional[str] = None
    error_kind: Optional[str] = None


class SmartLobbyStore:
    """Small durable idempotency ledger for external thread creation."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._init_lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        self._initialize(conn)
        return conn

    def _initialize(self, conn: sqlite3.Connection) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS smart_lobby_routes (
                    source_key TEXT PRIMARY KEY,
                    profile TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    thread_id TEXT,
                    error_kind TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
            try:
                self.path.chmod(0o600)
            except OSError:
                pass
            self._initialized = True

    @staticmethod
    def _decode(row: sqlite3.Row) -> SmartLobbyRoute:
        return SmartLobbyRoute(
            source_key=str(row["source_key"]),
            profile=str(row["profile"]),
            channel_id=str(row["channel_id"]),
            title=str(row["title"]),
            status=str(row["status"]),
            thread_id=str(row["thread_id"]) if row["thread_id"] else None,
            error_kind=str(row["error_kind"]) if row["error_kind"] else None,
        )

    def reserve(
        self, *, source_key: str, profile: str, channel_id: str, title: str
    ) -> tuple[bool, SmartLobbyRoute]:
        """Atomically reserve one source message; first writer wins."""
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO smart_lobby_routes
                    (source_key, profile, channel_id, title, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'reserved', ?, ?)
                """,
                (source_key, profile, channel_id, title, now, now),
            )
            row = conn.execute(
                "SELECT * FROM smart_lobby_routes WHERE source_key=?", (source_key,)
            ).fetchone()
            conn.commit()
        if row is None:  # pragma: no cover - SQLite invariant
            raise RuntimeError("smart lobby reservation disappeared")
        return cursor.rowcount == 1, self._decode(row)

    def update(
        self,
        source_key: str,
        *,
        status: str,
        thread_id: Optional[str] = None,
        error_kind: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE smart_lobby_routes
                SET status=?, thread_id=COALESCE(?, thread_id), error_kind=?, updated_at=?
                WHERE source_key=?
                """,
                (status, thread_id, error_kind, time.time(), source_key),
            )
            conn.commit()


def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def parse_smart_lobby_config(raw: Any) -> Optional[SmartLobbyConfig]:
    """Validate the optional raw ``gateway.smart_lobby`` mapping."""
    if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
        return None
    platform = str(raw.get("platform") or "discord").strip().lower()
    chat_id = str(raw.get("chat_id") or "").strip()
    default_profile = str(raw.get("default_profile") or "default").strip() or "default"
    if platform != "discord" or not _SNOWFLAKE_RE.fullmatch(chat_id):
        return None

    candidates: dict[str, SmartLobbyCandidate] = {}
    raw_candidates = raw.get("candidates")
    if isinstance(raw_candidates, Mapping):
        for raw_profile, value in raw_candidates.items():
            if not isinstance(raw_profile, str) or not isinstance(value, Mapping):
                continue
            profile = raw_profile.strip().lower()
            channel_id = str(value.get("channel_id") or "").strip()
            description = str(value.get("description") or "").strip()
            if (
                not profile
                or profile == default_profile
                or not _SNOWFLAKE_RE.fullmatch(channel_id)
                or not description
            ):
                continue
            try:
                from hermes_cli.profiles import validate_profile_name

                validate_profile_name(profile)
            except (ImportError, ValueError):
                continue
            candidates[profile] = SmartLobbyCandidate(
                profile=profile,
                channel_id=channel_id,
                description=description[:1_000],
            )
    if not candidates:
        return None
    return SmartLobbyConfig(
        platform=platform,
        chat_id=chat_id,
        default_profile=default_profile,
        candidates=candidates,
        min_confidence=_bounded_float(raw.get("min_confidence"), 0.65, 0.0, 1.0),
        timeout_seconds=_bounded_float(raw.get("timeout_seconds"), 20.0, 1.0, 120.0),
    )


def _strip_code_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        first_newline = value.find("\n")
        if first_newline >= 0:
            value = value[first_newline + 1 :]
        if value.rstrip().endswith("```"):
            value = value.rstrip()[:-3]
    return value.strip()


def _clean_title(value: Any, fallback: str = "Hermes task") -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"^[#>*_`\-]+\s*", "", text).strip()
    return (text[:80] or fallback[:80] or "Hermes task")


def _routed_prompt_notice(text: str, user_name: Any) -> str:
    """Render the original lobby prompt visibly in the destination thread."""
    cleaned = str(text or "").strip()
    if len(cleaned) > 1_400:
        cleaned = cleaned[:1_399] + "…"
    quoted = "\n".join(f"> {line}" if line else ">" for line in cleaned.splitlines())
    author = _clean_title(user_name, fallback="the lobby")
    return f"**Routed from `#hermes` for {author}**\n{quoted}"


def parse_classifier_decision(
    text: str, candidates: set[str], min_confidence: float
) -> Optional[SmartLobbyDecision]:
    """Parse and validate the classifier's strict JSON decision."""
    try:
        payload = json.loads(_strip_code_fence(text))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    profile = str(payload.get("profile") or "").strip().lower()
    confidence = _bounded_float(payload.get("confidence"), -1.0, -1.0, 1.0)
    if profile not in candidates or confidence < min_confidence:
        return None
    return SmartLobbyDecision(
        profile=profile,
        confidence=confidence,
        title=_clean_title(payload.get("title"), fallback="Hermes task"),
    )


class GatewaySmartLobbyMixin:
    """Gateway mixin implementing the optional Discord smart-lobby edge."""

    async def _classify_smart_lobby(
        self, text: str, config: SmartLobbyConfig
    ) -> Optional[SmartLobbyDecision]:
        # Explicit profile prefixes are deterministic and avoid an auxiliary call.
        explicit = _EXPLICIT_PROFILE_RE.match(text or "")
        if explicit:
            profile = next((value for value in explicit.groupdict().values() if value), "")
            profile = profile.lower()
            if profile in config.candidates:
                remainder = (text or "")[explicit.end() :].strip()
                return SmartLobbyDecision(
                    profile=profile,
                    confidence=1.0,
                    title=_clean_title(remainder or text),
                )

        candidate_lines = "\n".join(
            f"- {name}: {candidate.description}"
            for name, candidate in sorted(config.candidates.items())
        )
        prompt = (
            "Choose exactly one specialist profile for this user request. "
            "Return JSON only with keys profile, confidence (0-1), and a concise "
            "Discord thread title (max 80 characters). If no specialist clearly "
            "owns it, return profile=default with low confidence.\n\n"
            f"Profiles:\n{candidate_lines}\n\nUser request:\n{text[:8_000]}"
        )
        try:
            from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning

            response = await asyncio.wait_for(
                async_call_llm(
                    task="smart_lobby_router",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=160,
                    timeout=config.timeout_seconds,
                ),
                timeout=config.timeout_seconds + 2.0,
            )
            content = extract_content_or_reasoning(response)
        except Exception as exc:
            logger.warning("Smart lobby classifier unavailable: %s", exc)
            return None
        decision = parse_classifier_decision(
            content,
            set(config.candidates),
            config.min_confidence,
        )
        if decision is not None and decision.title == "Hermes task":
            decision = dataclasses.replace(decision, title=_clean_title(text))
        return decision

    def _smart_lobby_target_adapter(self, profile: str):
        profile_maps = getattr(self, "_profile_adapters", None) or {}
        mapping = profile_maps.get(profile) or {}
        return mapping.get(Platform.DISCORD)

    def _get_smart_lobby_store(self) -> SmartLobbyStore:
        store = getattr(self, "_smart_lobby_store", None)
        if isinstance(store, SmartLobbyStore):
            return store
        from hermes_constants import get_hermes_home

        store = SmartLobbyStore(get_hermes_home() / "gateway" / "smart_lobby.db")
        self._smart_lobby_store = store
        return store

    @staticmethod
    def _smart_lobby_source_key(source: Any) -> Optional[str]:
        message_id = str(getattr(source, "message_id", "") or "").strip()
        if not message_id:
            return None
        return ":".join(
            (
                "discord",
                str(getattr(source, "guild_id", "") or "dm"),
                str(getattr(source, "chat_id", "") or "unknown"),
                message_id,
            )
        )

    async def _maybe_route_smart_lobby(self, event) -> bool:
        """Route one eligible lobby event; return True when it was consumed."""
        config = parse_smart_lobby_config(
            getattr(getattr(self, "config", None), "smart_lobby", None)
        )
        if config is None or getattr(event, "internal", False):
            return False
        source = getattr(event, "source", None)
        if source is None or source.platform is not Platform.DISCORD:
            return False
        if str(source.chat_id or "") != config.chat_id:
            return False
        source_key = self._smart_lobby_source_key(source)
        if source_key is None:
            # External thread creation needs a stable idempotency key. Events
            # without one stay on the ordinary lobby path.
            return False
        try:
            if event.get_command():
                return False
        except Exception:
            if str(getattr(event, "text", "") or "").lstrip().startswith("/"):
                return False

        decision = await self._classify_smart_lobby(str(event.text or ""), config)
        if decision is None or decision.profile == config.default_profile:
            return False
        candidate = config.candidates.get(decision.profile)
        if candidate is None:
            return False
        source_adapter = getattr(self, "_adapter_for_source")(source)
        target_adapter = self._smart_lobby_target_adapter(decision.profile)
        if target_adapter is None:
            logger.warning(
                "Smart lobby target adapter unavailable: profile=%s platform=discord",
                decision.profile,
            )
            if source_adapter is not None:
                try:
                    await source_adapter.send(
                        source.chat_id,
                        f"`{decision.profile}` owns this request, but its Discord bot is "
                        "unavailable. The request was not executed in the lobby.",
                    )
                except Exception:
                    logger.debug("Smart lobby unavailable-target notice failed", exc_info=True)
            return True
        create_thread = getattr(target_adapter, "create_handoff_thread", None)
        build_source = getattr(target_adapter, "build_source", None)
        handle_message = getattr(target_adapter, "handle_message", None)
        if not callable(create_thread) or not callable(build_source) or not callable(handle_message):
            if source_adapter is not None:
                try:
                    await source_adapter.send(
                        source.chat_id,
                        f"`{decision.profile}` owns this request, but its Discord adapter "
                        "cannot create a routed thread. The request was not executed in the lobby.",
                    )
                except Exception:
                    logger.debug("Smart lobby incapable-target notice failed", exc_info=True)
            return True

        target_auth = getattr(target_adapter, "_is_sender_authorized", None)
        with _target_profile_runtime_scope(decision.profile):
            authorized = (
                target_auth(source.user_id, "thread", candidate.channel_id)
                if callable(target_auth)
                else None
            )
        if authorized is not True:
            logger.warning(
                "Smart lobby target profile rejected sender: profile=%s user=%s",
                decision.profile,
                source.user_id,
            )
            if source_adapter is not None:
                try:
                    await source_adapter.send(
                        source.chat_id,
                        f"`{decision.profile}` owns this request, but you are not authorized "
                        "for that profile. The request was not executed.",
                    )
                except Exception:
                    logger.debug("Smart lobby authorization notice failed", exc_info=True)
            return True

        store = self._get_smart_lobby_store()
        created, reservation = await asyncio.to_thread(
            store.reserve,
            source_key=source_key,
            profile=decision.profile,
            channel_id=candidate.channel_id,
            title=_clean_title(decision.title, event.text),
        )
        if not created:
            # A previous process/turn already crossed the external-side-effect
            # boundary. Never create or dispatch again. If the thread id was
            # persisted, give the user a stable continuation link; otherwise
            # report the fail-closed reservation for operator recovery.
            if source_adapter is not None:
                if reservation.thread_id:
                    notice = (
                        f"Already routed to `{reservation.profile}` → "
                        f"<#{reservation.thread_id}>"
                    )
                else:
                    notice = (
                        f"Routing for this message is already reserved for "
                        f"`{reservation.profile}` and will not be repeated."
                    )
                try:
                    await source_adapter.send(source.chat_id, notice)
                except Exception:
                    logger.debug("Smart lobby duplicate acknowledgment failed", exc_info=True)
            return True

        thread_id = await cast(
            Awaitable[Any],
            create_thread(candidate.channel_id, _clean_title(decision.title, event.text)),
        )
        if not thread_id:
            await asyncio.to_thread(
                store.update,
                source_key,
                status="failed",
                error_kind="thread_create",
            )
            logger.warning(
                "Smart lobby could not create target thread: profile=%s channel=%s",
                decision.profile,
                candidate.channel_id,
            )
            if source_adapter is not None:
                try:
                    await source_adapter.send(
                        source.chat_id,
                        f"I selected `{decision.profile}` but could not create its thread. "
                        "The request was not executed; please retry with an explicit profile prefix.",
                    )
                except Exception:
                    logger.debug("Smart lobby thread-create error notice failed", exc_info=True)
            return True
        await asyncio.to_thread(
            store.update,
            source_key,
            status="thread_created",
            thread_id=str(thread_id),
        )

        routed_source = build_source(
            chat_id=str(thread_id),
            chat_name=f"#{decision.profile} / {_clean_title(decision.title, event.text)}",
            chat_type="thread",
            user_id=getattr(source, "user_id", None),
            user_name=getattr(source, "user_name", None),
            thread_id=str(thread_id),
            chat_topic=None,
            user_id_alt=getattr(source, "user_id_alt", None),
            chat_id_alt=None,
            is_bot=False,
            scope_id=getattr(source, "scope_id", None),
            guild_id=getattr(source, "guild_id", None),
            parent_chat_id=candidate.channel_id,
            message_id=getattr(source, "message_id", None),
            # Role authorization is profile-local. Never carry a role decision
            # made by the lobby/default bot into the target bot's source.
            role_authorized=False,
        )
        routed_event = dataclasses.replace(event, source=routed_source)
        # The synthetic event preserves the real user turn in Hermes state, but
        # it did not originate as a Discord message in the new thread. Mirror a
        # bounded visible copy so the thread is understandable to the user.
        try:
            await cast(
                Awaitable[Any],
                target_adapter.send(
                    str(thread_id),
                    _routed_prompt_notice(str(event.text or ""), source.user_name),
                ),
            )
        except Exception:
            logger.debug("Smart lobby prompt mirror failed", exc_info=True)
        session_key_fn = getattr(target_adapter, "_text_batch_key", None)
        target_session_key = (
            session_key_fn(routed_event) if callable(session_key_fn) else None
        )
        try:
            await cast(Awaitable[Any], handle_message(routed_event))
        except Exception:
            await asyncio.to_thread(
                store.update,
                source_key,
                status="failed",
                thread_id=str(thread_id),
                error_kind="dispatch",
            )
            logger.exception(
                "Smart lobby target dispatch failed after thread creation: profile=%s thread=%s",
                decision.profile,
                thread_id,
            )
            if source_adapter is not None:
                await source_adapter.send(
                    source.chat_id,
                    f"I created <#{thread_id}> for `{decision.profile}`, but dispatch failed. "
                    "Please continue in that thread or retry here.",
                )
            return True

        if target_session_key:
            processing_task = (
                getattr(target_adapter, "_session_tasks", {}) or {}
            ).get(target_session_key)
            if not isinstance(processing_task, asyncio.Task):
                await asyncio.to_thread(
                    store.update,
                    source_key,
                    status="failed",
                    thread_id=str(thread_id),
                    error_kind="dispatch_start",
                )
                if source_adapter is not None:
                    try:
                        await source_adapter.send(
                            source.chat_id,
                            f"I created <#{thread_id}> for `{decision.profile}`, but its "
                            "processing task did not start. Please continue in that thread.",
                        )
                    except Exception:
                        logger.debug("Smart lobby dispatch-start notice failed", exc_info=True)
                return True

        await asyncio.to_thread(
            store.update,
            source_key,
            status="dispatched",
            thread_id=str(thread_id),
        )
        if source_adapter is not None:
            try:
                await source_adapter.send(
                    source.chat_id,
                    f"Routed to `{decision.profile}` → <#{thread_id}>",
                )
            except Exception:
                logger.debug("Smart lobby route acknowledgment failed", exc_info=True)
        logger.info(
            "Smart lobby routed request: profile=%s channel=%s thread=%s confidence=%.3f",
            decision.profile,
            candidate.channel_id,
            thread_id,
            decision.confidence,
        )
        return True
