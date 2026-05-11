"""Cognee client configuration and sync bridge.

Cognee exposes async ``remember`` / ``recall`` APIs. Hermes memory providers are
synchronous, so this module owns one background asyncio loop and exposes a small
``run_async`` helper for safe sync calls.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict

from hermes_cli.config import cfg_get, load_config
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER = "openai"
_DEFAULT_DATASET = "hermes_memory"
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


@dataclass
class CogneeClientConfig:
    """Runtime config for the Cognee SDK."""

    api_key: str = ""
    provider: str = _DEFAULT_PROVIDER
    base_url: str = ""
    dataset_name: str = _DEFAULT_DATASET
    enabled: bool = True

    @classmethod
    def from_global_config(cls) -> "CogneeClientConfig":
        """Read Cognee config from Hermes config.yaml and environment.

        Resolution order is: environment variables override config.yaml, and
        config.yaml overrides safe defaults. Secrets live in ``.env`` via the
        env vars below; non-secrets may be stored under ``memory.cognee``.
        """

        try:
            cfg = load_config()
        except Exception:
            logger.debug("Failed to load Hermes config for Cognee", exc_info=True)
            cfg = {}

        mem_cfg = cfg_get(cfg, "memory", "cognee", default={}) or {}
        model_cfg = cfg_get(cfg, "model", default={}) or {}
        try:
            file_cfg_path = get_hermes_home() / "cognee.json"
            if file_cfg_path.exists():
                import json
                file_cfg = json.loads(file_cfg_path.read_text(encoding="utf-8"))
                if isinstance(file_cfg, dict):
                    mem_cfg = {**mem_cfg, **{k: v for k, v in file_cfg.items() if v not in (None, "")}}
        except Exception:
            logger.debug("Failed to load Cognee provider config file", exc_info=True)

        provider = (
            os.environ.get("LLM_PROVIDER")
            or os.environ.get("COGNEE_LLM_PROVIDER")
            or str(mem_cfg.get("provider") or mem_cfg.get("llm_provider") or "").strip()
            or str(model_cfg.get("provider") or "").strip()
            or _DEFAULT_PROVIDER
        )
        api_key = (
            os.environ.get("LLM_API_KEY")
            or os.environ.get("COGNEE_LLM_API_KEY")
            or str(mem_cfg.get("api_key") or "").strip()
            or str(model_cfg.get("api_key") or "").strip()
        )
        base_url = (
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("COGNEE_LLM_BASE_URL")
            or str(mem_cfg.get("base_url") or "").strip()
            or str(model_cfg.get("base_url") or "").strip()
        )
        dataset_name = (
            os.environ.get("COGNEE_DATASET_NAME")
            or str(mem_cfg.get("dataset_name") or "").strip()
            or _DEFAULT_DATASET
        )
        enabled = mem_cfg.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}

        return cls(
            api_key=api_key,
            provider=provider,
            base_url=base_url,
            dataset_name=dataset_name,
            enabled=bool(enabled),
        )

    def apply_to_environment(self) -> None:
        """Expose config in the env names Cognee v1.0 understands.

        Uses direct assignment (not ``setdefault``) for the provider's own
        env vars so that values stay correct even when another component
        (e.g. an inference provider) has already set the same var before
        the memory provider was initialised.
        """

        if self.api_key:
            os.environ["LLM_API_KEY"] = self.api_key
        if self.provider:
            os.environ["LLM_PROVIDER"] = self.provider
        if self.base_url:
            os.environ["LLM_BASE_URL"] = self.base_url
            os.environ["LLM_ENDPOINT"] = self.base_url
        if self.provider == "deepseek":
            os.environ["LLM_MODEL"] = "deepseek/deepseek-chat"
            gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
            if gemini_key:
                os.environ["EMBEDDING_PROVIDER"] = "gemini"
                os.environ["EMBEDDING_MODEL"] = "gemini/gemini-embedding-001"
                os.environ["EMBEDDING_DIMENSIONS"] = "768"
                os.environ["EMBEDDING_API_KEY"] = gemini_key

        # SQLite multi-process safety — force WAL mode + non-zero busy timeout
        # so the gateway and agent can both write to Cognee's databases.
        os.environ.setdefault("DATABASE_CONNECT_ARGS", json.dumps({"timeout": 60}))
        ensure_wal_mode(timeout_ms=5000)


def _loop_worker(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Return the process-wide background event loop for Cognee calls."""

    global _loop, _loop_thread
    with _loop_lock:
        if _loop and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(target=_loop_worker, args=(_loop,), daemon=True, name="cognee-loop")
        _loop_thread.start()
        return _loop


def run_async(awaitable: Awaitable[Any], timeout: float | None = None) -> Any:
    """Run an awaitable on the Cognee loop and block for the result."""

    future = asyncio.run_coroutine_threadsafe(awaitable, get_event_loop())
    return future.result(timeout=timeout)


def stop_event_loop(timeout: float = 2.0) -> None:
    """Stop the global Cognee loop if it is running."""

    global _loop, _loop_thread
    with _loop_lock:
        loop = _loop
        thread = _loop_thread
        _loop = None
        _loop_thread = None
    if loop and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread and thread.is_alive():
        thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# WAL mode enforcement for multi-process SQLite safety
# ---------------------------------------------------------------------------

_COGNEE_DB_GLOBS = (
    ".cognee_system/databases/**/*.lance.db",
    ".data_storage/**/cache.db",
    ".cognee_system/**/*.sqlite",
)


def _find_cognee_databases() -> list[str]:
    """Yield paths to known Cognee SQLite databases on disk."""
    import cognee

    if hasattr(cognee, "__file__") and cognee.__file__:
        root = Path(cognee.__file__).resolve().parent
        found: list[str] = []
        for pattern in _COGNEE_DB_GLOBS:
            found.extend(str(p) for p in root.glob(pattern))
        return found
    return []


def ensure_wal_mode(timeout_ms: int = 5000) -> None:
    """Set WAL journal mode + busy timeout on all Cognee SQLite databases.

    SQLite's default ``delete`` journal mode serialises all writers, causing
    ``database is locked`` when two processes (e.g. gateway + agent) access
    Cognee concurrently.  WAL mode allows one writer + multiple readers,
    and a non-zero busy timeout makes writers retry instead of failing fast.

    This is idempotent — WAL mode is stored in the database file header and
    persists across restarts.  Safe to call on every ``initialize()``.
    """
    try:
        import cognee
    except Exception:
        logger.debug("Cognee not importable — skipping WAL setup")
        return

    dbs = _find_cognee_databases(cognee)
    if not dbs:
        logger.debug("No Cognee SQLite databases found for WAL setup")
        return

    for path in dbs:
        try:
            conn = sqlite3.connect(str(path), timeout=timeout_ms / 1000)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={timeout_ms}")
            conn.close()
            logger.debug("WAL mode set on %s", path)
        except Exception as exc:
            logger.warning("Could not set WAL on %s: %s", path, exc)


def _find_cognee_databases(cognee_module) -> list[str]:
    """Yield paths to known Cognee SQLite databases on disk."""
    if hasattr(cognee_module, "__file__") and cognee_module.__file__:
        root = Path(cognee_module.__file__).resolve().parent
        found: list[str] = []
        for pattern in _COGNEE_DB_GLOBS:
            found.extend(str(p) for p in root.glob(pattern))
        return found
    return []


def _call_accepting_kwargs(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call SDK functions while tolerating minor Cognee signature drift."""

    try:
        sig = inspect.signature(func)
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts_var_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except (TypeError, ValueError):
        pass
    return func(*args, **kwargs)


async def cognee_remember(data: Any, **kwargs: Any) -> Any:
    import cognee

    result = _call_accepting_kwargs(cognee.remember, data, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def cognee_recall(query_text: str, **kwargs: Any) -> Any:
    import cognee

    result = _call_accepting_kwargs(cognee.recall, query_text, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def cognee_forget(**kwargs: Any) -> Any:
    import cognee

    forget = getattr(cognee, "forget", None)
    if forget is None:
        prune = getattr(cognee, "prune", None)
        if prune is None:
            raise RuntimeError("Installed cognee package has no forget() or prune() API")
        result = _call_accepting_kwargs(prune, **kwargs)
    else:
        result = _call_accepting_kwargs(forget, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def serialize_result(value: Any) -> Any:
    """Best-effort JSON-safe serialization for Cognee SDK result objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): serialize_result(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_result(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return serialize_result(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return serialize_result(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        data: Dict[str, Any] = {
            k: v for k, v in vars(value).items() if not k.startswith("_")
        }
        if data:
            return serialize_result(data)
    return str(value)
