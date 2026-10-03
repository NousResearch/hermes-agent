"""Profile-scoped extension point for optional content-search backends.

Backends are advisory: returning ``None`` declines a request, and exceptions or
invalid responses fail safely to Hermes' native search implementation.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchBackendRequest:
    pattern: str
    path: str
    file_glob: Optional[str]
    limit: int
    offset: int
    output_mode: str
    context: int
    environment_kind: str
    is_local: bool
    cwd: str


@dataclass(frozen=True)
class SearchBackendMatch:
    path: str
    line_number: int
    content: str


@dataclass
class SearchBackendResult:
    backend: str
    route_reason: str
    matches: List[SearchBackendMatch] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    truncated: bool = False
    limit_reason: Optional[str] = None
    warning: Optional[str] = None


@dataclass(frozen=True)
class SearchBackendDecline:
    route_reason: str


@dataclass(frozen=True)
class SearchBackendRun:
    result: Optional[SearchBackendResult]
    route_reason: Optional[str] = None


@dataclass(frozen=True)
class _BackendEntry:
    name: str
    callback: Callable[[SearchBackendRequest], object]
    token: object


class SearchBackendRegistration:
    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release
        self._disposed = False

    @property
    def active(self) -> bool:
        return not self._disposed

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._release()


_LOCK = threading.RLock()
_BACKENDS: dict[str, dict[str, _BackendEntry]] = {}


def _clean_label(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128 and "\n" not in value


def _within_root(value: str, root: Path) -> bool:
    try:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve(strict=True)
        return candidate == root or root in candidate.parents
    except (OSError, TypeError, ValueError):
        return False


def _valid_result(result: SearchBackendResult, request: SearchBackendRequest) -> bool:
    """Core-owned containment/type checks before plugin output reaches file_tools."""
    try:
        requested = Path(request.path)
        if not requested.is_absolute():
            requested = Path(request.cwd) / requested
        root = requested.resolve(strict=True)
    except (OSError, TypeError, ValueError):
        return False
    if not root.is_dir() or not _clean_label(result.backend) or not _clean_label(result.route_reason):
        return False
    if not isinstance(result.total_count, int) or isinstance(result.total_count, bool) or result.total_count < 0:
        return False
    if not isinstance(result.matches, (list, tuple)) or not isinstance(result.files, (list, tuple)):
        return False
    if not isinstance(result.counts, dict):
        return False
    if result.total_count < max(len(result.matches), len(result.files), len(result.counts)):
        return False
    if result.limit_reason is not None and not _clean_label(result.limit_reason):
        return False
    if result.warning is not None and (not isinstance(result.warning, str) or len(result.warning) > 2000):
        return False
    if any(
        not isinstance(match, SearchBackendMatch)
        or not isinstance(match.line_number, int)
        or isinstance(match.line_number, bool)
        or match.line_number < 1
        or not isinstance(match.content, str)
        or not _within_root(match.path, root)
        for match in result.matches
    ):
        return False
    if any(not isinstance(path, str) or not _within_root(path, root) for path in result.files):
        return False
    if any(
        not isinstance(path, str) or not _within_root(path, root)
        or not isinstance(count, int) or isinstance(count, bool) or count < 0
        for path, count in result.counts.items()
    ):
        return False
    return True


def register_search_backend(
    name: str,
    callback: Callable[[SearchBackendRequest], object],
    *,
    scope: str | None = None,
) -> SearchBackendRegistration:
    """Register one optional backend for a Hermes profile.

    Re-registering the same name replaces that generation; disposing an older
    handle never removes its replacement.
    """
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("Search backend name must not be empty")
    if not callable(callback):
        raise TypeError("Search backend callback must be callable")
    key = hermes_home_key(scope)
    token = object()
    entry = _BackendEntry(clean_name, callback, token)
    with _LOCK:
        _BACKENDS.setdefault(key, {})[clean_name] = entry

    def release() -> None:
        with _LOCK:
            scoped = _BACKENDS.get(key)
            if scoped is None or scoped.get(clean_name) is not entry:
                return
            scoped.pop(clean_name, None)
            if not scoped:
                _BACKENDS.pop(key, None)

    return SearchBackendRegistration(release)


def run_search_backends(
    request: SearchBackendRequest, *, scope: str | None = None,
) -> Optional[SearchBackendRun]:
    """Run profile backends in deterministic name order, or return ``None`` when absent."""
    key = hermes_home_key(scope)
    with _LOCK:
        entries = sorted(_BACKENDS.get(key, {}).values(), key=lambda entry: entry.name)
    if not entries:
        return None
    last_decline: Optional[str] = None
    for entry in entries:
        try:
            response = entry.callback(request)
        except Exception as exc:
            logger.warning("Search backend %s failed: %s", entry.name, exc)
            return SearchBackendRun(None, f"backend_error:{entry.name}")
        if response is None:
            continue
        if isinstance(response, SearchBackendDecline):
            if not _clean_label(response.route_reason):
                logger.warning("Search backend %s returned an invalid decline reason", entry.name)
                return SearchBackendRun(None, f"backend_invalid:{entry.name}")
            last_decline = response.route_reason
            continue
        if isinstance(response, SearchBackendResult):
            try:
                valid = _valid_result(response, request)
            except (AttributeError, TypeError, ValueError):
                valid = False
            if valid:
                return SearchBackendRun(response)
            logger.warning("Search backend %s returned an unsafe or malformed result", entry.name)
            return SearchBackendRun(None, f"backend_invalid:{entry.name}")
        logger.warning("Search backend %s returned unsupported result %s", entry.name, type(response).__name__)
        return SearchBackendRun(None, f"backend_invalid:{entry.name}")
    return SearchBackendRun(None, last_decline) if last_decline else None
