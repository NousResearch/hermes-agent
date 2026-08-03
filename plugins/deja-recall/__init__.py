"""deja-recall: local deterministic recall for persisted Hermes sessions."""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import index

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    source_paths: tuple[Path, ...] = ()
    index_path: Path = Path("~/.cache/hermes/recall/index.sqlite3").expanduser()
    result_count: int = 3
    recency_days: float = 30.0
    max_injected_chars: int = 2400


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(int(value), high))
    except (TypeError, ValueError):
        return default


def _load_settings() -> Settings:
    try:
        from hermes_cli.config import load_config
        from hermes_constants import get_hermes_home
        root = Path(get_hermes_home())
        cfg = load_config()
        entries = cfg.get("plugins", {}).get("entries", {}) if isinstance(cfg, dict) else {}
        raw = entries.get("deja-recall", {}) if isinstance(entries, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
    except Exception as exc:
        logger.debug("deja-recall: config unavailable: %s", exc)
        root, raw = Path("~/.hermes").expanduser(), {}
    configured = raw.get("session_paths", raw.get("session_path", []))
    if isinstance(configured, str):
        configured = [configured]
    paths = tuple(Path(str(p)).expanduser() for p in configured if p) if isinstance(configured, list) else ()
    if not paths:
        paths = (root / "state.db",)
    index_path = Path(str(raw.get("index_path", root / "cache" / "deja-recall.sqlite3"))).expanduser()
    try:
        recency = max(1.0, min(float(raw.get("recency_days", 30.0)), 3650.0))
    except (TypeError, ValueError):
        recency = 30.0
    return Settings(
        enabled=bool(raw.get("enabled", True)),
        source_paths=paths,
        index_path=index_path,
        result_count=_bounded_int(raw.get("result_count", 3), 3, 1, 10),
        recency_days=recency,
        max_injected_chars=_bounded_int(raw.get("max_injected_chars", 2400), 2400, 128, 8000),
    )


_settings = Settings(enabled=False)
_jobs: queue.Queue[None] = queue.Queue(maxsize=1)
_worker_started = False
_lock = threading.Lock()


def _refresh() -> None:
    try:
        report = index.refresh_index(_settings.source_paths, _settings.index_path)
        if report.sources_failed:
            logger.warning("deja-recall: %d transcript source(s) could not be indexed", report.sources_failed)
    except Exception as exc:
        logger.warning("deja-recall: index refresh failed: %s", exc)


def _worker() -> None:
    while True:
        _jobs.get()
        try:
            _refresh()
        finally:
            _jobs.task_done()


def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if not _worker_started:
            threading.Thread(target=_worker, name="deja-recall-index", daemon=True).start()
            _worker_started = True


def _enqueue_refresh(**_: Any) -> None:
    if not _settings.enabled:
        return
    _ensure_worker()
    try:
        _jobs.put_nowait(None)
    except queue.Full:
        logger.debug("deja-recall: refresh already queued")


def _pre_llm_call(user_message: Any = "", session_id: str = "", is_first_turn: bool = False, **_: Any):
    if not _settings.enabled or not is_first_turn or not isinstance(user_message, str) or not user_message.strip():
        return None
    try:
        # Cold-cache construction is queued so first-turn latency never depends on indexing.
        if not _settings.index_path.is_file():
            _enqueue_refresh()
            return None
        hits = index.recall(
            _settings.index_path,
            user_message,
            active_session_id=session_id,
            limit=_settings.result_count,
            recency_days=_settings.recency_days,
        )
        if not hits:
            return None
        text = index.render(user_message, hits, max_chars=_settings.max_injected_chars)
        return {"context": text} if text else None
    except Exception as exc:
        logger.warning("deja-recall: retrieval failed: %s", exc)
        return None


def register(ctx) -> None:
    global _settings
    _settings = _load_settings()
    if not _settings.enabled:
        logger.debug("deja-recall: disabled by configuration")
        return
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("on_session_finalize", _enqueue_refresh)
    _enqueue_refresh()
