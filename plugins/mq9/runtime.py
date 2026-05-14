"""Runtime manager for Hermes mq9 plugin.

Implements:
- background passive inbox server (Phase 2 minimal)
- register/discover/call operations used by plugin tools
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from .mq9_client import Mq9Client, Mq9Error

logger = logging.getLogger(__name__)

DEFAULT_NATS_URL = "nats://127.0.0.1:4222"


@dataclass
class RuntimeConfig:
    nats_url: str
    agent_name: str
    mailbox: str
    mailbox_ttl: int
    auto_register: bool
    passive_serve: bool
    poll_interval_s: float
    startup_timeout_s: float
    default_discover_limit: int
    default_call_timeout_s: float
    passive_execute_mode: str
    oneshot_timeout_s: float
    oneshot_provider: str
    oneshot_model: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "nats_url": self.nats_url,
            "agent_name": self.agent_name,
            "mailbox": self.mailbox,
            "mailbox_ttl": self.mailbox_ttl,
            "auto_register": self.auto_register,
            "passive_serve": self.passive_serve,
            "poll_interval_s": self.poll_interval_s,
            "startup_timeout_s": self.startup_timeout_s,
            "default_discover_limit": self.default_discover_limit,
            "default_call_timeout_s": self.default_call_timeout_s,
            "passive_execute_mode": self.passive_execute_mode,
            "oneshot_timeout_s": self.oneshot_timeout_s,
            "oneshot_provider": self.oneshot_provider,
            "oneshot_model": self.oneshot_model,
        }


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _as_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _sanitize_name(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"[^a-z0-9._-]", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-._") or "hermes-agent"


def _default_agent_name() -> str:
    host = socket.gethostname().split(".")[0]
    host = _sanitize_name(host)
    return f"hermes-{host}-{os.getpid()}"


def _strip_mq9_uri(mailbox: str) -> str:
    value = mailbox.strip()
    for prefix in ("mq9://broker/", "mq9s://broker/"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _extract_mailbox(agent: dict[str, Any]) -> str | None:
    raw = agent.get("mailbox")
    if isinstance(raw, str) and raw.strip():
        return _strip_mq9_uri(raw)

    metadata = agent.get("metadata")
    if isinstance(metadata, dict):
        mq9_info = metadata.get("mq9")
        if isinstance(mq9_info, dict):
            for key in ("mailbox", "mail_address"):
                value = mq9_info.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _load_runtime_config(plugin_id: str = "mq9") -> RuntimeConfig:
    merged: dict[str, Any] = {}

    try:
        from hermes_cli.config import cfg_get, load_config

        config = load_config() or {}
        legacy_top = config.get(plugin_id)
        legacy_plugins = cfg_get(config, "plugins", plugin_id, default={})
        entries = cfg_get(config, "plugins", "entries", plugin_id, default={})

        for source in (legacy_top, legacy_plugins, entries):
            if isinstance(source, dict):
                merged.update(source)
    except Exception as exc:  # pragma: no cover - hermes runtime only
        logger.debug("mq9 plugin: failed reading hermes config, fallback defaults: %s", exc)

    env_map = {
        "nats_url": os.getenv("HERMES_MQ9_NATS_URL"),
        "agent_name": os.getenv("HERMES_MQ9_AGENT_NAME"),
        "mailbox": os.getenv("HERMES_MQ9_MAILBOX"),
        "mailbox_ttl": os.getenv("HERMES_MQ9_MAILBOX_TTL"),
        "auto_register": os.getenv("HERMES_MQ9_AUTO_REGISTER"),
        "passive_serve": os.getenv("HERMES_MQ9_PASSIVE_SERVE"),
        "poll_interval_s": os.getenv("HERMES_MQ9_POLL_INTERVAL"),
        "startup_timeout_s": os.getenv("HERMES_MQ9_STARTUP_TIMEOUT"),
        "default_discover_limit": os.getenv("HERMES_MQ9_DISCOVER_LIMIT"),
        "default_call_timeout_s": os.getenv("HERMES_MQ9_CALL_TIMEOUT"),
        "passive_execute_mode": os.getenv("HERMES_MQ9_PASSIVE_EXECUTE_MODE"),
        "oneshot_timeout_s": os.getenv("HERMES_MQ9_ONESHOT_TIMEOUT"),
        "oneshot_provider": os.getenv("HERMES_MQ9_ONESHOT_PROVIDER"),
        "oneshot_model": os.getenv("HERMES_MQ9_ONESHOT_MODEL"),
    }
    for key, value in env_map.items():
        if value not in (None, ""):
            merged[key] = value

    agent_name = str(merged.get("agent_name") or _default_agent_name()).strip()
    agent_name = _sanitize_name(agent_name)

    mailbox = str(merged.get("mailbox") or f"hermes.mq9.{agent_name}.inbox").strip()
    mailbox = _strip_mq9_uri(mailbox)

    execute_mode = str(merged.get("passive_execute_mode") or "minimal").strip().lower()
    if execute_mode not in {"minimal", "oneshot"}:
        execute_mode = "minimal"

    return RuntimeConfig(
        nats_url=str(merged.get("nats_url") or DEFAULT_NATS_URL).strip(),
        agent_name=agent_name,
        mailbox=mailbox,
        mailbox_ttl=_as_int(merged.get("mailbox_ttl"), 24 * 3600, 60, 30 * 24 * 3600),
        auto_register=_as_bool(merged.get("auto_register"), True),
        passive_serve=_as_bool(merged.get("passive_serve"), True),
        poll_interval_s=_as_float(merged.get("poll_interval_s"), 0.25, 0.05, 5.0),
        startup_timeout_s=_as_float(merged.get("startup_timeout_s"), 8.0, 1.0, 60.0),
        default_discover_limit=_as_int(merged.get("default_discover_limit"), 10, 1, 100),
        default_call_timeout_s=_as_float(merged.get("default_call_timeout_s"), 25.0, 1.0, 300.0),
        passive_execute_mode=execute_mode,
        oneshot_timeout_s=_as_float(merged.get("oneshot_timeout_s"), 90.0, 5.0, 600.0),
        oneshot_provider=str(merged.get("oneshot_provider") or "").strip(),
        oneshot_model=str(merged.get("oneshot_model") or "").strip(),
    )


class MQ9HermesRuntime:
    """Shared runtime used by mq9 plugin hooks and tools."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ctx: Any = None

        self._worker_thread: threading.Thread | None = None
        self._worker_stop: threading.Event | None = None

        self._mailbox_address: str | None = None
        self._registered_card: dict[str, Any] | None = None
        self._registered_names: set[str] = set()
        self._started_at: float | None = None
        self._last_error: str | None = None

        self._receive_count: int = 0
        self._reply_count: int = 0
        self._last_receive_ts: int | None = None
        self._last_reply_ts: int | None = None

    def attach_context(self, ctx: Any) -> None:
        with self._lock:
            self._ctx = ctx

    def current_config(self) -> RuntimeConfig:
        return _load_runtime_config(plugin_id="mq9")

    def status(self) -> dict[str, Any]:
        cfg = self.current_config()
        with self._lock:
            thread = self._worker_thread
            running = bool(thread and thread.is_alive())
            return {
                "ok": self._last_error is None,
                "running": running,
                "mailbox_address": self._mailbox_address,
                "agent_name": cfg.agent_name,
                "registered": self._registered_card is not None,
                "registered_card": self._registered_card,
                "registered_names": sorted(self._registered_names),
                "started_at": int(self._started_at) if self._started_at else None,
                "last_error": self._last_error,
                "receive_count": self._receive_count,
                "reply_count": self._reply_count,
                "last_receive_ts": self._last_receive_ts,
                "last_reply_ts": self._last_reply_ts,
                "config": cfg.as_dict(),
            }

    def start_background(self, *, reason: str = "manual") -> dict[str, Any]:
        cfg = self.current_config()

        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return self.status()

            stop_event = threading.Event()
            ready_event = threading.Event()
            self._worker_stop = stop_event
            self._last_error = None

            thread = threading.Thread(
                target=self._thread_main,
                name="mq9-plugin-runtime",
                args=(cfg, stop_event, ready_event, reason),
                daemon=True,
            )
            self._worker_thread = thread
            thread.start()

        ready_event.wait(timeout=cfg.startup_timeout_s)
        return self.status()

    def stop_background(
        self,
        *,
        reason: str = "manual",
        unregister: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            stop_event = self._worker_stop
            thread = self._worker_thread

        if stop_event:
            stop_event.set()
        if thread:
            thread.join(timeout=3.0)

        with self._lock:
            if self._worker_thread and not self._worker_thread.is_alive():
                self._worker_thread = None
                self._worker_stop = None

        if unregister:
            try:
                self._unregister_registered_agents_best_effort()
            except Exception as exc:
                with self._lock:
                    self._last_error = f"mq9 stop cleanup failed: {exc}"
                logger.warning("mq9 runtime stop cleanup failed: %s", exc)

        logger.info("mq9 plugin runtime stopped: reason=%s", reason)
        return self.status()

    def register_self(self, args: dict[str, Any]) -> dict[str, Any]:
        ensure_runtime = _as_bool(args.get("ensure_runtime"), True)
        if ensure_runtime:
            self.start_background(reason="tool-register-self")

        cfg = self.current_config()
        agent_name = _sanitize_name(str(args.get("agent_name") or cfg.agent_name))
        user_mailbox = str(args.get("mailbox") or "").strip()
        description = str(args.get("description") or "").strip()
        tags = args.get("tags") if isinstance(args.get("tags"), list) else None

        with self._lock:
            known_mailbox = self._mailbox_address

        async def _register() -> dict[str, Any]:
            async with Mq9Client(cfg.nats_url) as client:
                mailbox = known_mailbox
                if user_mailbox:
                    mailbox = await self._ensure_mailbox(
                        client,
                        _strip_mq9_uri(user_mailbox),
                        cfg.mailbox_ttl,
                    )
                if not mailbox:
                    mailbox = await self._ensure_mailbox(client, cfg.mailbox, cfg.mailbox_ttl)

                card = self._build_agent_card(
                    agent_name=agent_name,
                    mailbox=mailbox,
                    description=description,
                    tags=tags,
                )
                await client.register_agent(agent_card=card)
                return {"mailbox": mailbox, "card": card}

        result = self._run_async(_register())
        with self._lock:
            self._mailbox_address = result["mailbox"]
            self._registered_card = result["card"]
            self._registered_names.add(agent_name)
            self._last_error = None

        return {
            "ok": True,
            "mailbox": result["mailbox"],
            "agent_name": agent_name,
            "card": result["card"],
        }

    def unregister_self(self, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        explicit_name = str(args.get("agent_name") or "").strip()

        with self._lock:
            names = set(self._registered_names)
            card = self._registered_card

        if explicit_name:
            names = {explicit_name}
        elif card and isinstance(card.get("name"), str) and card.get("name"):
            names.add(str(card["name"]))
        if not names:
            cfg = self.current_config()
            names.add(cfg.agent_name)

        cfg = self.current_config()
        errors: list[dict[str, str]] = []
        removed: list[str] = []

        async def _unregister() -> None:
            async with Mq9Client(cfg.nats_url) as client:
                for name in sorted(names):
                    try:
                        await client.unregister_agent(name=name)
                        removed.append(name)
                    except Mq9Error as exc:
                        if "not found" in str(exc).lower():
                            removed.append(name)
                            continue
                        errors.append({"name": name, "error": str(exc)})

        self._run_async(_unregister())

        with self._lock:
            for name in removed:
                self._registered_names.discard(name)
            if self._registered_card and self._registered_card.get("name") in removed:
                self._registered_card = None
            if errors:
                self._last_error = (
                    f"mq9 unregister had {len(errors)} failure(s): {errors[0]['error']}"
                )
            elif removed:
                self._last_error = None

        return {
            "ok": len(errors) == 0,
            "removed": removed,
            "errors": errors,
            "requested_names": sorted(names),
        }

    def discover(self, args: dict[str, Any]) -> dict[str, Any]:
        cfg = self.current_config()
        query = args.get("query")
        query = str(query).strip() if query is not None else None
        if query == "":
            query = None

        limit = _as_int(args.get("limit"), cfg.default_discover_limit, 1, 100)
        prefer_name = str(args.get("prefer_name") or "").strip() or None

        async def _discover() -> dict[str, Any]:
            async with Mq9Client(cfg.nats_url) as client:
                agents = await client.discover_agents(query=query, limit=limit)
                candidates = self._normalize_candidates(agents)

                if prefer_name and not any(
                    item.get("name") == prefer_name for item in candidates
                ):
                    # Fallback discovery by exact name improves robustness when
                    # semantic search is polluted by stale historical agents.
                    fallback_agents = await client.discover_agents(
                        query=prefer_name,
                        limit=limit,
                    )
                    fallback_candidates = self._normalize_candidates(fallback_agents)
                    exact_fallback = [
                        item
                        for item in fallback_candidates
                        if item.get("name") == prefer_name
                    ]
                    candidates = self._dedupe_candidates(exact_fallback + candidates)

            if prefer_name:
                candidates.sort(key=lambda it: 0 if it.get("name") == prefer_name else 1)

            return {
                "ok": True,
                "query": query,
                "limit": limit,
                "count": len(candidates),
                "agents": candidates,
            }

        return self._run_async(_discover())

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        cfg = self.current_config()

        timeout_s = _as_float(
            args.get("timeout_s"),
            cfg.default_call_timeout_s,
            1.0,
            300.0,
        )
        from_agent = _sanitize_name(str(args.get("from_agent") or cfg.agent_name))

        raw_message = args.get("message")
        if raw_message is None:
            raise ValueError("mq9_call requires 'message'")
        if isinstance(raw_message, str):
            message: dict[str, Any] = {"instruction": raw_message}
        elif isinstance(raw_message, dict):
            message = raw_message
        else:
            message = {"payload": raw_message}

        target_mailbox = str(args.get("target_mailbox") or "").strip()
        target_mailbox = _strip_mq9_uri(target_mailbox) if target_mailbox else ""

        query = str(args.get("query") or "").strip() or None
        prefer_name = str(args.get("prefer_name") or "").strip() or None

        async def _call() -> dict[str, Any]:
            selected_agent: dict[str, Any] | None = None
            mailbox = target_mailbox
            attempt_errors: list[dict[str, str]] = []
            attempted_targets: list[str] = []

            async with Mq9Client(cfg.nats_url) as client:
                if not mailbox:
                    if not query:
                        raise ValueError(
                            "mq9_call requires target_mailbox, or query for discover"
                        )
                    agents = await client.discover_agents(query=query, limit=20)
                    candidates = self._normalize_candidates(agents)

                    if prefer_name and not any(
                        item.get("name") == prefer_name for item in candidates
                    ):
                        # Secondary lookup by name avoids getting stuck on
                        # stale semantic results.
                        fallback_agents = await client.discover_agents(
                            query=prefer_name,
                            limit=20,
                        )
                        fallback_candidates = self._normalize_candidates(fallback_agents)
                        exact_fallback = [
                            item
                            for item in fallback_candidates
                            if item.get("name") == prefer_name
                        ]
                        candidates = self._dedupe_candidates(exact_fallback + candidates)

                    if prefer_name:
                        candidates.sort(
                            key=lambda it: 0 if it.get("name") == prefer_name else 1,
                        )
                    if not candidates:
                        raise RuntimeError(
                            f"mq9_call discover found no callable mailbox: query={query!r}"
                        )
                    for index, candidate in enumerate(candidates):
                        selected_agent = candidate
                        mailbox = str(candidate["_mailbox"])
                        attempted_targets.append(mailbox)

                        # Keep retries bounded when several stale candidates
                        # exist; give the first one full timeout.
                        attempt_timeout = timeout_s
                        if index > 0:
                            attempt_timeout = min(timeout_s, 8.0)

                        try:
                            call_result = await client.mq9_call(
                                from_agent=from_agent,
                                target_mailbox=mailbox,
                                message=message,
                                timeout_s=attempt_timeout,
                            )
                            break
                        except (TimeoutError, Mq9Error) as exc:
                            attempt_errors.append(
                                {
                                    "name": str(candidate.get("name", "")),
                                    "target_mailbox": mailbox,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                }
                            )
                    else:
                        raise RuntimeError(
                            "mq9_call timed out for all discovered candidates: "
                            f"{attempt_errors}"
                        )
                else:
                    attempted_targets.append(mailbox)
                    call_result = await client.mq9_call(
                        from_agent=from_agent,
                        target_mailbox=mailbox,
                        message=message,
                        timeout_s=timeout_s,
                    )

            return {
                "ok": True,
                "from_agent": from_agent,
                "target_mailbox": mailbox,
                "selected_agent": selected_agent,
                "attempted_targets": attempted_targets,
                "correlation_id": call_result.correlation_id,
                "callback_mailbox": call_result.callback_mailbox,
                "response": call_result.response,
            }

        return self._run_async(_call())

    def _thread_main(
        self,
        cfg: RuntimeConfig,
        stop_event: threading.Event,
        ready_event: threading.Event,
        reason: str,
    ) -> None:
        try:
            asyncio.run(self._run_worker(cfg, stop_event, ready_event, reason))
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                self._last_error = str(exc)
            ready_event.set()
            logger.warning("mq9 runtime thread crashed: %s", exc)

    async def _run_worker(
        self,
        cfg: RuntimeConfig,
        stop_event: threading.Event,
        ready_event: threading.Event,
        reason: str,
    ) -> None:
        logger.info(
            "mq9 runtime start: reason=%s nats=%s passive=%s auto_register=%s",
            reason,
            cfg.nats_url,
            cfg.passive_serve,
            cfg.auto_register,
        )

        if not cfg.passive_serve and not cfg.auto_register:
            ready_event.set()
            while not stop_event.is_set():
                await asyncio.sleep(0.2)
            return

        try:
            async with Mq9Client(cfg.nats_url) as client:
                mailbox = await self._ensure_mailbox(client, cfg.mailbox, cfg.mailbox_ttl)
                with self._lock:
                    self._mailbox_address = mailbox
                    self._started_at = time.time()

                if cfg.auto_register:
                    card = self._build_agent_card(
                        agent_name=cfg.agent_name,
                        mailbox=mailbox,
                        description="",
                        tags=None,
                    )
                    await client.register_agent(agent_card=card)
                    with self._lock:
                        self._registered_card = card
                        self._registered_names.add(cfg.agent_name)

                ready_event.set()

                if not cfg.passive_serve:
                    while not stop_event.is_set():
                        await asyncio.sleep(0.2)
                    return

                async_stop = asyncio.Event()

                async def watch_stop() -> None:
                    while not stop_event.is_set():
                        await asyncio.sleep(0.2)
                    async_stop.set()

                async def on_message(message: Any) -> None:
                    with self._lock:
                        self._receive_count += 1
                        self._last_receive_ts = int(time.time())

                    body = message.parse_json()
                    if not isinstance(body, dict):
                        return
                    if body.get("type") != "mq9_call":
                        return

                    callback_mailbox = body.get("reply_to")
                    correlation_id = body.get("correlation_id")
                    if not callback_mailbox or not correlation_id:
                        return

                    result_payload, call_ok = await self._handle_inbound_call(
                        cfg=cfg,
                        correlation_id=str(correlation_id),
                        payload=body.get("payload"),
                    )
                    response = {
                        "type": "mq9_call_reply",
                        "ok": call_ok,
                        "from": cfg.agent_name,
                        "correlation_id": correlation_id,
                        "result": result_payload,
                        "ts": int(time.time()),
                    }

                    await client.send_message(str(callback_mailbox), response)
                    with self._lock:
                        self._reply_count += 1
                        self._last_reply_ts = int(time.time())

                await asyncio.gather(
                    client.subscribe_loop(
                        mailbox,
                        on_message,
                        group_name=f"{cfg.agent_name}-passive",
                        deliver="earliest",
                        poll_interval=cfg.poll_interval_s,
                        stop_event=async_stop,
                    ),
                    watch_stop(),
                )

        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            logger.warning("mq9 runtime worker error: %s", exc)
            ready_event.set()

    async def _ensure_mailbox(self, client: Mq9Client, name: str, ttl: int) -> str:
        mailbox_name = _strip_mq9_uri(name)
        mailbox = await client.create_mailbox(
            ttl=ttl,
            name=mailbox_name,
            idempotent=True,
        )
        return mailbox.mail_address

    def _unregister_registered_agents_best_effort(self) -> None:
        cfg = self.current_config()
        with self._lock:
            names = set(self._registered_names)
            card = self._registered_card
        if card and isinstance(card.get("name"), str) and card.get("name"):
            names.add(str(card["name"]))
        if not names:
            return

        async def _unregister() -> dict[str, Any]:
            removed: list[str] = []
            errors: list[dict[str, str]] = []
            async with Mq9Client(cfg.nats_url) as client:
                for name in sorted(names):
                    try:
                        await client.unregister_agent(name=name)
                        removed.append(name)
                    except Mq9Error as exc:
                        if "not found" in str(exc).lower():
                            removed.append(name)
                            continue
                        errors.append({"name": name, "error": str(exc)})
            return {"removed": removed, "errors": errors}

        result = self._run_async(_unregister())
        removed = set(result.get("removed", []))
        errors = result.get("errors", [])
        with self._lock:
            for name in removed:
                self._registered_names.discard(name)
            if (
                self._registered_card
                and self._registered_card.get("name") in removed
            ):
                self._registered_card = None
            if not errors:
                return
            self._last_error = (
                f"mq9 runtime cleanup unregister failed: {errors[0]['name']}: "
                f"{errors[0]['error']}"
            )

    @staticmethod
    def _normalize_candidates(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for item in agents:
            mailbox = _extract_mailbox(item)
            if not mailbox:
                continue
            normalized = dict(item)
            normalized["_mailbox"] = mailbox
            candidates.append(normalized)
        return candidates

    @staticmethod
    def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in candidates:
            key = (str(item.get("name", "")), str(item.get("_mailbox", "")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _build_agent_card(
        self,
        *,
        agent_name: str,
        mailbox: str,
        description: str,
        tags: list[str] | None,
    ) -> dict[str, Any]:
        desc = description.strip() or (
            "Hermes agent reachable via mq9 plugin. "
            "Can handle coding tasks including Python and HTTP service work."
        )
        tag_list = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        if not tag_list:
            tag_list = ["hermes", "assistant", "coding", "python", "backend"]

        skills: list[dict[str, Any]] = []
        ctx = self._ctx
        toolsets = getattr(ctx, "toolsets", None) if ctx is not None else None
        if isinstance(toolsets, dict):
            for toolset_name, toolset in toolsets.items():
                tool_items = getattr(toolset, "tools", None)
                if not isinstance(tool_items, list):
                    continue
                for tool in tool_items:
                    tool_name = getattr(tool, "name", None)
                    if not tool_name:
                        continue
                    skills.append(
                        {
                            "id": f"{toolset_name}-{tool_name}",
                            "name": str(tool_name),
                            "tags": [str(toolset_name)],
                        }
                    )
        if not skills:
            skills.append(
                {
                    "id": "hermes-general-coding",
                    "name": "General coding assistant",
                    "tags": ["coding", "python", "backend"],
                    "examples": [
                        "write a Python HTTP server",
                        "implement a backend API",
                    ],
                }
            )

        return {
            "name": agent_name,
            "mailbox": f"mq9://broker/{mailbox}",
            "description": desc,
            "tags": tag_list,
            "skills": skills,
            "metadata": {
                "mq9": {
                    "mailbox": mailbox,
                    "transport": "nats",
                }
            },
        }

    async def _handle_inbound_call(
        self,
        *,
        cfg: RuntimeConfig,
        correlation_id: str,
        payload: Any,
    ) -> tuple[dict[str, Any], bool]:
        if cfg.passive_execute_mode != "oneshot":
            return (
                {
                    "mode": "minimal",
                    "summary": (
                        "mq9 passive serve handled this task in minimal mode. "
                        "Enable passive_execute_mode=oneshot to execute with Hermes."
                    ),
                    "request": payload,
                },
                True,
            )

        try:
            answer = await asyncio.to_thread(
                self._run_oneshot_executor,
                cfg,
                correlation_id,
                payload,
            )
            return (
                {
                    "mode": "oneshot",
                    "summary": "Hermes oneshot executor finished remote task.",
                    "answer": answer,
                    "request": payload,
                },
                True,
            )
        except Exception as exc:
            logger.warning("mq9 oneshot executor failed: %s", exc)
            return (
                {
                    "mode": "oneshot",
                    "summary": "Hermes oneshot executor failed; returning error payload.",
                    "error": str(exc),
                    "request": payload,
                },
                False,
            )

    def _run_oneshot_executor(
        self,
        cfg: RuntimeConfig,
        correlation_id: str,
        payload: Any,
    ) -> str:
        prompt = self._build_oneshot_prompt(cfg.agent_name, correlation_id, payload)
        hermes_bin = self._resolve_hermes_binary()

        env = dict(os.environ)
        env["HERMES_ACCEPT_HOOKS"] = "1"
        env["HERMES_YOLO_MODE"] = "1"
        env["HERMES_MQ9_AUTO_REGISTER"] = "0"
        env["HERMES_MQ9_PASSIVE_SERVE"] = "0"
        env["HERMES_MQ9_PASSIVE_EXECUTE_MODE"] = "minimal"

        if cfg.oneshot_provider:
            env["HERMES_INFERENCE_PROVIDER"] = cfg.oneshot_provider
        if cfg.oneshot_model:
            env["HERMES_INFERENCE_MODEL"] = cfg.oneshot_model

        command = [
            hermes_bin,
            "-z",
            prompt,
            "--ignore-user-config",
            "--ignore-rules",
        ]

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=cfg.oneshot_timeout_s,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"exit={completed.returncode}"
            raise RuntimeError(f"hermes oneshot failed: {detail}")

        output = (completed.stdout or "").strip()
        if not output:
            raise RuntimeError("hermes oneshot returned empty output")
        return output

    @staticmethod
    def _resolve_hermes_binary() -> str:
        found = shutil.which("hermes")
        if found:
            return found

        candidate = os.path.join(os.path.dirname(sys.executable), "hermes")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

        raise FileNotFoundError(
            "Cannot find 'hermes' executable in PATH or current Python environment."
        )

    @staticmethod
    def _extract_task_text(payload: Any) -> str:
        if isinstance(payload, str):
            text = payload.strip()
            return text or "(empty task)"

        if isinstance(payload, dict):
            for key in ("instruction", "task", "prompt", "query"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return json.dumps(payload, ensure_ascii=False)

        if payload is None:
            return "(no payload)"
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def _build_oneshot_prompt(
        cls,
        agent_name: str,
        correlation_id: str,
        payload: Any,
    ) -> str:
        task_text = cls._extract_task_text(payload)
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return (
            f"You are remote Hermes agent '{agent_name}'. "
            "Complete the incoming task and return only the final answer. "
            "Do not call external tools.\n\n"
            f"Correlation ID: {correlation_id}\n"
            f"Task:\n{task_text}\n\n"
            f"Payload JSON:\n{payload_text}\n"
        )

    @staticmethod
    def _run_async(coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:  # pragma: no cover - defensive
                error["exc"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if error:
            raise error["exc"]
        return result.get("value")
