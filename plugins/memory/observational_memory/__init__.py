"""Observational Memory provider — shared local markdown memory for Hermes.

This bridges Hermes into the Observational Memory (`om`) ecosystem so Hermes
can read the same profile/active context that Claude Code and Codex use, and
optionally write Hermes sessions back into that memory store.

Package dependency: observational-memory
Config file: $HERMES_HOME/observational_memory.json
Optional secret: OM_HERMES_API_KEY in the active profile's .env
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY_DIR = str(Path.home() / ".local" / "share" / "observational-memory")
_DEFAULT_ENV_FILE = str(Path.home() / ".config" / "observational-memory" / "env")
_MAX_NOTE_LEN = 600
_MAX_STARTUP_SECTION_CHARS = 4000

_PRIORITY_MAP = {
    "high": "🔴",
    "medium": "🟡",
    "low": "🟢",
}


CONTEXT_SCHEMA = {
    "name": "om_context",
    "description": (
        "Load compact Observational Memory context shared across Hermes, Claude Code, "
        "and Codex. Returns startup profile/active context plus optionally relevant "
        "memory search results for the current task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional task/query to retrieve the most relevant memories.",
            },
            "limit": {
                "type": "integer",
                "description": "Max relevant memories to include (default: 4, max: 10).",
            },
        },
        "required": [],
    },
}

SEARCH_SCHEMA = {
    "name": "om_search",
    "description": (
        "Search Observational Memory for relevant preferences, project history, "
        "prior decisions, or recent cross-agent context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {
                "type": "integer",
                "description": "Max results to return (default: 5, max: 10).",
            },
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "om_remember",
    "description": (
        "Store an explicit local observation in Observational Memory right away. "
        "Use for durable preferences, corrections, decisions, or project facts that "
        "should survive future sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact or observation to store.",
            },
            "importance": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Priority level to assign (default: medium).",
            },
        },
        "required": ["content"],
    },
}


def _default_settings() -> dict:
    return {
        "llm_provider": "inherit-existing",
        "llm_model": "",
        "memory_dir": _DEFAULT_MEMORY_DIR,
        "env_file": _DEFAULT_ENV_FILE,
        "search_backend": "bm25",
        "writeback_mode": "incremental",
    }


def _config_path(hermes_home: str | None = None) -> Path:
    if hermes_home:
        return Path(hermes_home) / "observational_memory.json"
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "observational_memory.json"


def _load_settings(hermes_home: str | None = None) -> dict:
    settings = _default_settings()
    path = _config_path(hermes_home)
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            if isinstance(raw, dict):
                settings.update(raw)
        except Exception:
            logger.debug("Failed to read Observational Memory config from %s", path)
    return settings


def _package_available() -> bool:
    return importlib.util.find_spec("observational_memory") is not None


def _sanitize_note(text: str) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= _MAX_NOTE_LEN:
        return compact
    return compact[: _MAX_NOTE_LEN - 3].rstrip() + "..."


class ObservationalMemoryProvider(MemoryProvider):
    """Shared local markdown memory via Observational Memory."""

    def __init__(self):
        self._settings = _default_settings()
        self._config = None
        self._session_id = ""
        self._writeback_mode = "incremental"
        self._writer_enabled = False
        self._pending_messages = []
        self._pending_lock = threading.Lock()
        self._prefetch_lock = threading.Lock()
        self._prefetch_query = ""
        self._prefetch_result = ""
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "observational_memory"

    def is_available(self) -> bool:
        return _package_available()

    def get_config_schema(self):
        return [
            {
                "key": "llm_provider",
                "description": "Observational Memory writer provider",
                "default": "inherit-existing",
                "choices": ["inherit-existing", "anthropic", "openai"],
            },
            {
                "key": "llm_model",
                "description": "Observer/reflector model override (optional)",
                "default": "",
            },
            {
                "key": "memory_dir",
                "description": "Observational Memory data directory",
                "default": _DEFAULT_MEMORY_DIR,
            },
            {
                "key": "env_file",
                "description": "Observational Memory env file path",
                "default": _DEFAULT_ENV_FILE,
            },
            {
                "key": "search_backend",
                "description": "Search backend",
                "default": "bm25",
                "choices": ["bm25", "qmd", "qmd-hybrid", "none"],
            },
            {
                "key": "writeback_mode",
                "description": "How Hermes writes back into Observational Memory",
                "default": "incremental",
                "choices": ["incremental", "session_end", "off"],
            },
            {
                "key": "api_key",
                "description": "API key for the selected writer provider (optional if OM is already configured)",
                "secret": True,
                "env_var": "OM_HERMES_API_KEY",
            },
        ]

    def save_config(self, values, hermes_home):
        path = _config_path(hermes_home)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = _load_settings(hermes_home)
        existing.update(values)
        path.write_text(json.dumps(existing, indent=2) + "\n")

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        hermes_home = kwargs.get("hermes_home")
        self._settings = _load_settings(hermes_home)
        self._writeback_mode = (
            str(self._settings.get("writeback_mode", "incremental")).strip()
            or "incremental"
        )

        self._apply_env_bridge()
        self._config = self._build_config()
        self._config.ensure_memory_dir()
        self._ensure_startup_memory()
        self._ensure_search_ready()
        self._writer_enabled = (
            self._writeback_mode != "off" and self._writeback_is_configured()
        )
        if self._writeback_mode != "off" and not self._writer_enabled:
            logger.warning(
                "Observational Memory writeback is configured as '%s' but "
                "the LLM provider is not available — writeback will be "
                "disabled for this session. Check that an API key is set "
                "(OM_HERMES_API_KEY in .env, or an existing OM env file).",
                self._writeback_mode,
            )

    def system_prompt_block(self) -> str:
        if not self._config:
            return ""

        parts = [
            "# Observational Memory",
            "Shared local markdown memory across Hermes, Claude Code, and Codex.",
        ]
        if self._writer_enabled:
            parts.append("Hermes session writeback is active.")
        else:
            parts.append(
                "Hermes session writeback is inactive; om_remember still stores explicit notes locally."
            )

        profile = self._truncate_prompt_section(
            self._read_text(self._config.profile_path),
            label="Startup Profile",
        )
        active = self._truncate_prompt_section(
            self._read_text(self._config.active_path),
            label="Active Context",
        )
        if profile:
            parts.append(profile)
        if active:
            parts.append(active)

        return "\n\n".join(p for p in parts if p.strip())

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        query = (query or "").strip()
        if not query:
            return ""

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)

        with self._prefetch_lock:
            if self._prefetch_query == query and self._prefetch_result:
                return self._prefetch_result

        results = self._search(query, limit=4)
        formatted = self._format_prefetch(results)
        if formatted:
            with self._prefetch_lock:
                self._prefetch_query = query
                self._prefetch_result = formatted
        return formatted

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        query = (query or "").strip()
        if not query:
            return

        def _run():
            formatted = self._format_prefetch(self._search(query, limit=4))
            with self._prefetch_lock:
                self._prefetch_query = query
                self._prefetch_result = formatted

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="om-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        if not self._config or self._writeback_mode == "off":
            return

        messages = []
        if user_content and user_content.strip():
            messages.append(self._make_message("user", user_content))
        if assistant_content and assistant_content.strip():
            messages.append(self._make_message("assistant", assistant_content))
        if not messages:
            return

        with self._pending_lock:
            self._pending_messages.extend(messages)

        if self._writeback_mode == "incremental":
            self._flush_pending(force=False)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        active_thread = self._sync_thread
        if active_thread and active_thread.is_alive():
            active_thread.join(timeout=10.0)
            if active_thread.is_alive():
                logger.warning(
                    "Observational Memory sync is still running after 10s; "
                    "deferring final session flush until the current sync finishes."
                )
                self._defer_final_flush(active_thread)
                return
        self._flush_pending(force=True)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not content or not content.strip():
            return
        note = f"Built-in {target} memory {action}: {_sanitize_note(content)}"
        try:
            self._append_manual_observation(note, priority="medium")
        except Exception as e:
            logger.debug("Observational Memory mirror write failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [CONTEXT_SCHEMA, SEARCH_SCHEMA, REMEMBER_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "om_context":
            query = str(args.get("query", "") or "").strip()
            limit = self._coerce_limit(args.get("limit"), default=4)
            text = self._build_context(
                query=query, limit=limit, include_search=bool(query)
            )
            return json.dumps({"provider": self.name, "text": text})

        if tool_name == "om_search":
            query = str(args.get("query", "") or "").strip()
            if not query:
                return json.dumps({"error": "query is required"})
            limit = self._coerce_limit(args.get("limit"), default=5)
            results = self._search(query, limit=limit)
            payload = [
                {
                    "rank": item.rank,
                    "score": round(float(item.score), 4),
                    "source": item.document.source.value,
                    "heading": item.document.heading,
                    "excerpt": self._excerpt(item.document.content),
                }
                for item in results
            ]
            return json.dumps({"provider": self.name, "results": payload})

        if tool_name == "om_remember":
            content = str(args.get("content", "") or "").strip()
            if not content:
                return json.dumps({"error": "content is required"})
            importance = (
                str(args.get("importance", "medium") or "medium").strip().lower()
            )
            if importance not in _PRIORITY_MAP:
                importance = "medium"
            stored = self._append_manual_observation(content, priority=importance)
            return json.dumps(stored)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

    # -- Internal helpers -------------------------------------------------

    def _apply_env_bridge(self) -> None:
        provider = (
            str(self._settings.get("llm_provider", "inherit-existing")).strip().lower()
        )
        model = str(self._settings.get("llm_model", "") or "").strip()
        if (
            provider
            and provider != "inherit-existing"
            and not os.environ.get("OM_LLM_PROVIDER")
        ):
            os.environ["OM_LLM_PROVIDER"] = provider
        if model and not os.environ.get("OM_LLM_MODEL"):
            os.environ["OM_LLM_MODEL"] = model

        api_key = os.environ.get("OM_HERMES_API_KEY", "").strip()
        if not api_key:
            return
        if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key

    def _build_config(self):
        from observational_memory.config import Config as OMConfig

        memory_dir = Path(
            str(self._settings.get("memory_dir") or _DEFAULT_MEMORY_DIR)
        ).expanduser()
        env_file = Path(
            str(self._settings.get("env_file") or _DEFAULT_ENV_FILE)
        ).expanduser()

        # OM loads provider/env settings from the env file into process state,
        # so bootstrap a minimal config for that side effect before creating
        # the final config with Hermes-specific overrides.
        bootstrap_cfg = OMConfig(memory_dir=memory_dir, env_file=env_file)
        bootstrap_cfg.load_env_file()

        kwargs: Dict[str, Any] = {
            "memory_dir": memory_dir,
            "env_file": env_file,
        }
        search_backend = (
            str(self._settings.get("search_backend", "bm25")).strip() or "bm25"
        )
        kwargs["search_backend"] = search_backend

        provider = (
            str(self._settings.get("llm_provider", "inherit-existing")).strip().lower()
        )
        if provider and provider != "inherit-existing":
            kwargs["llm_provider"] = provider
        model = str(self._settings.get("llm_model", "") or "").strip()
        if model:
            kwargs["llm_model"] = model

        return OMConfig(**kwargs)

    def _writeback_is_configured(self) -> bool:
        if not self._config:
            return False
        try:
            self._config.validate_provider_config()
            return True
        except Exception:
            return False

    def _ensure_startup_memory(self) -> None:
        if not self._config:
            return
        try:
            from observational_memory.startup_memory import ensure_startup_memory

            ensure_startup_memory(self._config)
        except Exception as e:
            logger.debug("Observational Memory startup refresh skipped: %s", e)

    def _refresh_startup_memory(self) -> None:
        if not self._config:
            return
        try:
            from observational_memory.startup_memory import refresh_startup_memory

            refresh_startup_memory(self._config)
        except Exception as e:
            logger.debug("Observational Memory startup refresh failed: %s", e)

    def _ensure_search_ready(self) -> None:
        if not self._config or self._config.search_backend == "none":
            return
        try:
            from observational_memory.search import get_backend, reindex

            backend = get_backend(self._config.search_backend, self._config)
            if not backend.is_ready():
                reindex(self._config)
        except Exception as e:
            logger.debug("Observational Memory reindex skipped: %s", e)

    def _reindex_search(self) -> None:
        if not self._config or self._config.search_backend == "none":
            return
        try:
            from observational_memory.search import reindex

            reindex(self._config)
        except Exception as e:
            logger.debug("Observational Memory reindex failed: %s", e)

    def _search(self, query: str, limit: int = 5):
        if not self._config:
            return []
        try:
            from observational_memory.search import get_backend, reindex

            backend = get_backend(self._config.search_backend, self._config)
            if not backend.is_ready() and self._config.search_backend != "none":
                reindex(self._config)
                backend = get_backend(self._config.search_backend, self._config)
            return backend.search(query, limit=limit)
        except Exception as e:
            logger.debug("Observational Memory search failed: %s", e)
            return []

    def _format_prefetch(self, results) -> str:
        if not results:
            return ""
        lines = ["## Observational Memory Recall"]
        for item in results:
            lines.append(f"- {item.document.source.value}: {item.document.heading}")
            excerpt = self._excerpt(item.document.content)
            if excerpt:
                lines.append(f"  {excerpt}")
        return "\n".join(lines)

    def _build_context(self, query: str, limit: int, *, include_search: bool) -> str:
        if not self._config:
            return ""

        self._ensure_startup_memory()
        parts = []
        profile = self._read_text(self._config.profile_path)
        active = self._read_text(self._config.active_path)
        if profile:
            parts.append(profile)
        if active:
            parts.append(active)

        if include_search:
            results = self._search(query, limit=limit)
            if results:
                formatted = ["## Relevant Memory"]
                for item in results:
                    formatted.append(
                        f"- {item.document.source.value}: {item.document.heading}"
                    )
                    excerpt = self._excerpt(item.document.content)
                    if excerpt:
                        formatted.append(f"  {excerpt}")
                parts.append("\n".join(formatted))

        if not parts:
            fallback = self._read_text(self._config.reflections_path)
            if fallback:
                parts.append(fallback)

        return "\n\n".join(p for p in parts if p.strip())

    def _append_manual_observation(self, content: str, *, priority: str) -> dict:
        if not self._config:
            raise RuntimeError("Observational Memory is not initialized")

        marker = _PRIORITY_MAP.get(priority, _PRIORITY_MAP["medium"])
        note = _sanitize_note(content)
        now = datetime.now(timezone.utc)
        date_label = now.strftime("%Y-%m-%d")
        time_label = now.strftime("%H:%M")
        entry = f"- {marker} {time_label} {note}"

        self._config.ensure_memory_dir()
        obs_path = self._config.observations_path
        if obs_path.exists():
            text = obs_path.read_text()
        else:
            text = "# Observations\n\n<!-- Auto-maintained by the Observer. -->\n"

        today_pattern = re.compile(
            rf"(?ms)^## {re.escape(date_label)}\n.*?(?=^## \d{{4}}-\d{{2}}-\d{{2}}|\Z)"
        )
        match = today_pattern.search(text)
        if not match:
            addition = f"## {date_label}\n\n### Observations\n{entry}\n"
            updated = text.rstrip() + "\n\n" + addition
        else:
            section = match.group(0).rstrip()
            obs_match = re.search(r"(?ms)^### Observations\n.*?(?=^### |\Z)", section)
            if obs_match:
                obs_block = obs_match.group(0).rstrip() + "\n" + entry
                section = (
                    section[: obs_match.start()]
                    + obs_block
                    + section[obs_match.end() :]
                )
            else:
                section = section.rstrip() + "\n\n### Observations\n" + entry
            updated = text[: match.start()] + section + text[match.end() :]

        obs_path.write_text(updated.rstrip() + "\n")
        self._refresh_startup_memory()
        self._reindex_search()
        return {"stored": True, "priority": priority, "content": note}

    def _flush_pending(self, *, force: bool) -> None:
        if not self._writer_enabled or not self._config:
            return
        active_thread = self._sync_thread
        if (
            active_thread
            and active_thread.is_alive()
            and active_thread is not threading.current_thread()
        ):
            return

        pending = self._take_pending_messages(force=force)
        if not pending:
            return

        if force:
            self._run_observer_batch(pending, force=True)
            return

        def _run():
            try:
                self._run_observer_batch(pending, force=False)
            finally:
                if self._sync_thread is threading.current_thread():
                    self._sync_thread = None

        self._sync_thread = threading.Thread(
            target=_run, daemon=True, name="om-observe"
        )
        self._sync_thread.start()

    def _defer_final_flush(self, active_thread: threading.Thread) -> None:
        def _run():
            try:
                active_thread.join()
                self._flush_pending(force=True)
            finally:
                if self._sync_thread is threading.current_thread():
                    self._sync_thread = None

        self._sync_thread = threading.Thread(
            target=_run, daemon=True, name="om-observe-finalize"
        )
        self._sync_thread.start()

    def _take_pending_messages(self, *, force: bool) -> list:
        with self._pending_lock:
            pending = list(self._pending_messages)
            threshold = 1 if force else getattr(self._config, "min_messages", 5)
            if len(pending) < threshold:
                return []
            self._pending_messages.clear()
            return pending

    def _restore_pending_messages(self, pending: list) -> None:
        with self._pending_lock:
            self._pending_messages[:0] = pending

    def _run_observer_batch(self, pending: list, *, force: bool) -> None:
        try:
            from observational_memory.observe import run_observer

            cfg = replace(self._config, min_messages=1) if force else self._config
            run_observer(pending, cfg, dry_run=False)
        except Exception as e:
            logger.warning("Observational Memory writeback failed: %s", e)
            self._restore_pending_messages(pending)
            return

        # After writing observations, check if the reflector needs to catch up.
        # This ensures long-term memory builds even without a background scheduler.
        self._maybe_run_reflector()

    def _maybe_run_reflector(self) -> None:
        """Run the reflector if daily reflections have fallen behind observations."""
        if not self._config:
            return
        try:
            from observational_memory.reflect import (
                reflector_catchup_needed,
                run_reflector,
            )

            if reflector_catchup_needed(self._config):
                logger.info(
                    "Observational Memory: reflector catch-up needed, running..."
                )
                run_reflector(self._config)
                self._refresh_startup_memory()
                self._reindex_search()
                logger.info("Observational Memory: reflector catch-up complete")
        except Exception as e:
            logger.debug("Observational Memory reflector catch-up skipped: %s", e)

    def _make_message(self, role: str, content: str):
        from observational_memory.transcripts import Message

        return Message(
            role=role,
            content=content.strip(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="hermes",
        )

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            if path.exists() and path.stat().st_size > 0:
                return path.read_text().strip()
        except Exception:
            return ""
        return ""

    @staticmethod
    def _truncate_prompt_section(text: str, *, label: str) -> str:
        text = (text or "").strip()
        if len(text) <= _MAX_STARTUP_SECTION_CHARS:
            return text

        notice = (
            f"\n\n[{label} truncated to {_MAX_STARTUP_SECTION_CHARS} chars for prompt safety. "
            "Use om_context for the full text.]"
        )
        max_body = max(_MAX_STARTUP_SECTION_CHARS - len(notice) - 3, 0)
        trimmed = text[:max_body].rstrip()
        if not trimmed:
            return notice.strip()
        return trimmed + "..." + notice

    @staticmethod
    def _excerpt(content: str) -> str:
        lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
        if lines and lines[0].startswith("## "):
            lines = lines[1:]
        text = " ".join(lines[:4]).strip()
        if len(text) > 260:
            return text[:257].rstrip() + "..."
        return text

    @staticmethod
    def _coerce_limit(value: Any, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(1, min(parsed, 10))


def register(ctx) -> None:
    """Register Observational Memory as a memory provider plugin.

    Called by both the memory provider system (which passes a collector
    with register_memory_provider) and the general plugin system (which
    passes a PluginContext without that method). Guard against the latter
    to avoid noisy startup warnings.
    """
    if hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(ObservationalMemoryProvider())
    else:
        # General plugin loader — memory registration is handled separately
        # by the memory provider system. No-op here is expected.
        pass
