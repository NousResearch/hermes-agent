"""Config-driven per-chat context for inbound gateway messages.

The public config accepts either an inline mapping or a path to a JSON
mapping.  File-backed maps are designed for external orchestrators that update
chat bindings while the gateway is running, so unchanged files stay parsed in
an in-process cache while every observed file state is validated independently.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.session import SessionSource


logger = logging.getLogger(__name__)

CONFIGURED_CHANNEL_CONTEXT_HEADER = "[Configured chat context]"
MAX_CHANNEL_CONTEXT_FILE_BYTES = 1024 * 1024
MAX_CHANNEL_CONTEXT_ENTRIES = 4096
MAX_CHANNEL_CONTEXT_KEY_BYTES = 512
MAX_CHANNEL_CONTEXT_VALUE_BYTES = 2 * 1024
_MAX_CACHED_FILES = 32
_MAX_WARNING_KEYS = 128


@dataclass(frozen=True)
class _FileSignature:
    """Metadata that changes for rewrites and atomic file replacements."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, result: os.stat_result) -> "_FileSignature":
        return cls(
            device=int(result.st_dev),
            inode=int(result.st_ino),
            size=int(result.st_size),
            mtime_ns=int(
                getattr(result, "st_mtime_ns", int(result.st_mtime * 1_000_000_000))
            ),
            ctime_ns=int(
                getattr(result, "st_ctime_ns", int(result.st_ctime * 1_000_000_000))
            ),
        )


@dataclass(frozen=True)
class _CachedFileMap:
    signature: _FileSignature
    contexts: dict[str, str]


def canonical_channel_context_key(source: "SessionSource") -> str:
    """Return the platform-qualified key used by ``channel_context_map``.

    ``SessionSource.chat_id`` intentionally remains the adapter's raw ID.  The
    platform namespace prevents the same raw ID on two platforms from sharing
    context.  Colons inside a native chat ID are preserved verbatim.
    """

    platform = str(getattr(source.platform, "value", source.platform) or "").strip()
    chat_id = str(source.chat_id or "")
    if not platform or not chat_id:
        return ""
    return f"{platform}:{chat_id}"


def merge_channel_context(
    adapter_context: Optional[str],
    configured_context: Optional[str],
) -> str:
    """Combine adapter and configured context without mutating the event."""

    blocks: list[str] = []
    if adapter_context:
        blocks.append(adapter_context)
    if configured_context:
        blocks.append(
            f"{CONFIGURED_CHANNEL_CONTEXT_HEADER}\n{configured_context}"
        )
    return "\n\n".join(blocks)


class ChannelContextResolver:
    """Resolve bounded per-chat context from inline or file-backed config."""

    def __init__(self, *, max_cached_files: int = _MAX_CACHED_FILES) -> None:
        self._max_cached_files = max(1, int(max_cached_files))
        self._file_cache: OrderedDict[Path, _CachedFileMap] = OrderedDict()
        self._warning_keys: OrderedDict[tuple[Any, ...], None] = OrderedDict()
        self._lock = threading.RLock()

    def clear(self) -> None:
        """Clear process-local caches (primarily useful for isolated tests)."""

        with self._lock:
            self._file_cache.clear()
            self._warning_keys.clear()

    def context_for(
        self,
        source: "SessionSource",
        config_value: Any,
        *,
        config_home: Path,
    ) -> Optional[str]:
        """Return configured context for ``source``, or ``None`` when absent."""

        lookup_key = canonical_channel_context_key(source)
        if not lookup_key:
            return None

        if isinstance(config_value, Mapping):
            return self._inline_context(config_value, lookup_key)
        elif isinstance(config_value, str):
            configured_path = config_value.strip()
            if not configured_path:
                return None
            try:
                path = Path(configured_path).expanduser()
                if not path.is_absolute():
                    path = Path(config_home) / path
                path = Path(os.path.abspath(path))
            except (OSError, RuntimeError, ValueError):
                self._warn_once(
                    ("invalid-path", type(configured_path).__name__),
                    "Ignoring invalid gateway.channel_context_map path",
                )
                return None
            contexts = self._load_file(path)
        elif config_value in (None, ""):
            return None
        else:
            self._warn_once(
                ("invalid-config-type", type(config_value).__name__),
                "Ignoring gateway.channel_context_map with unsupported type %s",
                type(config_value).__name__,
            )
            return None

        return contexts.get(lookup_key)

    def _inline_context(
        self,
        raw: Mapping[Any, Any],
        lookup_key: str,
    ) -> Optional[str]:
        """Validate only the requested inline entry to keep lookup O(1)."""

        if len(raw) > MAX_CHANNEL_CONTEXT_ENTRIES:
            self._warn_once(
                ("too-many-entries", "inline config", len(raw)),
                "Ignoring inline config for gateway.channel_context_map: "
                "%d entries exceeds limit %d",
                len(raw),
                MAX_CHANNEL_CONTEXT_ENTRIES,
            )
            return None
        if lookup_key not in raw:
            return None

        value = raw[lookup_key]
        if not isinstance(value, str):
            self._warn_once(
                ("inline-entry-type", type(value).__name__),
                "Ignoring non-string gateway.channel_context_map context from inline config",
            )
            return None
        if not value.strip():
            self._warn_once(
                ("inline-entry-blank",),
                "Ignoring blank gateway.channel_context_map context from inline config",
            )
            return None
        try:
            key_bytes = len(lookup_key.encode("utf-8"))
            value_bytes = len(value.encode("utf-8"))
        except UnicodeEncodeError:
            self._warn_once(
                ("inline-entry-encoding",),
                "Ignoring non-UTF-8 gateway.channel_context_map entry from inline config",
            )
            return None
        if (
            key_bytes > MAX_CHANNEL_CONTEXT_KEY_BYTES
            or value_bytes > MAX_CHANNEL_CONTEXT_VALUE_BYTES
        ):
            self._warn_once(
                ("inline-entry-oversized", key_bytes, value_bytes),
                "Ignoring oversized gateway.channel_context_map entry from inline config",
            )
            return None
        return value

    def _normalize_mapping(
        self,
        raw: Mapping[Any, Any],
        *,
        label: str,
    ) -> dict[str, str]:
        if len(raw) > MAX_CHANNEL_CONTEXT_ENTRIES:
            self._warn_once(
                ("too-many-entries", label, len(raw)),
                "Ignoring %s for gateway.channel_context_map: %d entries exceeds limit %d",
                label,
                len(raw),
                MAX_CHANNEL_CONTEXT_ENTRIES,
            )
            return {}

        contexts: dict[str, str] = {}
        invalid_type = 0
        invalid_key = 0
        oversized = 0
        blank = 0

        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                invalid_type += 1
                continue
            platform, separator, chat_id = key.partition(":")
            if (
                not separator
                or not platform
                or not chat_id
                or key != key.strip()
                or any(character.isspace() for character in platform)
            ):
                invalid_key += 1
                continue
            if not value.strip():
                blank += 1
                continue
            try:
                key_bytes = len(key.encode("utf-8"))
                value_bytes = len(value.encode("utf-8"))
            except UnicodeEncodeError:
                invalid_type += 1
                continue
            if (
                key_bytes > MAX_CHANNEL_CONTEXT_KEY_BYTES
                or value_bytes > MAX_CHANNEL_CONTEXT_VALUE_BYTES
            ):
                oversized += 1
                continue
            contexts[key] = value

        rejected = invalid_type + invalid_key + oversized + blank
        if rejected:
            # Do not log keys or values: both may contain sensitive identifiers
            # or operator-provided instructions.
            self._warn_once(
                (
                    "invalid-entries",
                    label,
                    len(raw),
                    invalid_type,
                    invalid_key,
                    oversized,
                    blank,
                ),
                "Ignored %d invalid gateway.channel_context_map entries from %s "
                "(type=%d, key=%d, oversized=%d, blank=%d)",
                rejected,
                label,
                invalid_type,
                invalid_key,
                oversized,
                blank,
            )
        return contexts

    def _load_file(self, path: Path) -> dict[str, str]:
        # Retry once if an external writer changes the file between stat/open.
        # Writers should still use a sibling temp file + os.replace() so readers
        # never observe partially-written JSON.
        for _attempt in range(2):
            try:
                path_stat = path.stat()
            except (OSError, ValueError) as exc:
                self._drop_cached_path(path)
                self._warn_once(
                    ("file-stat", str(path), type(exc).__name__),
                    "Could not read gateway.channel_context_map file %s (%s)",
                    path,
                    type(exc).__name__,
                )
                return {}

            signature = _FileSignature.from_stat(path_stat)
            cached = self._get_cached(path, signature)
            if cached is not None:
                return cached

            if not stat.S_ISREG(path_stat.st_mode):
                self._cache(path, signature, {})
                self._warn_once(
                    ("not-regular", str(path), signature),
                    "Ignoring non-regular gateway.channel_context_map file %s",
                    path,
                )
                return {}
            if signature.size > MAX_CHANNEL_CONTEXT_FILE_BYTES:
                self._cache(path, signature, {})
                self._warn_once(
                    ("file-too-large", str(path), signature),
                    "Ignoring gateway.channel_context_map file %s: %d bytes exceeds limit %d",
                    path,
                    signature.size,
                    MAX_CHANNEL_CONTEXT_FILE_BYTES,
                )
                return {}

            try:
                with path.open("rb") as handle:
                    opened_stat = os.fstat(handle.fileno())
                    opened_signature = _FileSignature.from_stat(opened_stat)
                    if opened_signature != signature:
                        continue
                    payload = handle.read(MAX_CHANNEL_CONTEXT_FILE_BYTES + 1)
            except (OSError, ValueError) as exc:
                self._drop_cached_path(path)
                self._warn_once(
                    ("file-open", str(path), type(exc).__name__),
                    "Could not read gateway.channel_context_map file %s (%s)",
                    path,
                    type(exc).__name__,
                )
                return {}

            if len(payload) > MAX_CHANNEL_CONTEXT_FILE_BYTES:
                self._cache(path, opened_signature, {})
                self._warn_once(
                    ("file-read-too-large", str(path), opened_signature),
                    "Ignoring gateway.channel_context_map file %s: content exceeds limit %d",
                    path,
                    MAX_CHANNEL_CONTEXT_FILE_BYTES,
                )
                return {}

            try:
                after_signature = _FileSignature.from_stat(path.stat())
            except (OSError, ValueError):
                self._drop_cached_path(path)
                return {}
            if after_signature != opened_signature:
                continue

            try:
                data = json.loads(payload)
            except (UnicodeError, ValueError, RecursionError) as exc:
                self._cache(path, opened_signature, {})
                self._warn_once(
                    ("invalid-json", str(path), opened_signature),
                    "Ignoring invalid gateway.channel_context_map JSON in %s (%s)",
                    path,
                    type(exc).__name__,
                )
                return {}
            if not isinstance(data, dict):
                self._cache(path, opened_signature, {})
                self._warn_once(
                    ("non-mapping-json", str(path), opened_signature),
                    "Ignoring gateway.channel_context_map file %s: JSON root must be an object",
                    path,
                )
                return {}

            contexts = self._normalize_mapping(data, label=f"file {path}")
            self._cache(path, opened_signature, contexts)
            return contexts

        self._drop_cached_path(path)
        self._warn_once(
            ("file-changing", str(path)),
            "Ignoring gateway.channel_context_map file %s while it is changing",
            path,
        )
        return {}

    def _get_cached(
        self,
        path: Path,
        signature: _FileSignature,
    ) -> Optional[dict[str, str]]:
        with self._lock:
            cached = self._file_cache.get(path)
            if cached is None or cached.signature != signature:
                return None
            self._file_cache.move_to_end(path)
            return cached.contexts

    def _cache(
        self,
        path: Path,
        signature: _FileSignature,
        contexts: dict[str, str],
    ) -> None:
        with self._lock:
            self._file_cache[path] = _CachedFileMap(signature, dict(contexts))
            self._file_cache.move_to_end(path)
            while len(self._file_cache) > self._max_cached_files:
                self._file_cache.popitem(last=False)

    def _drop_cached_path(self, path: Path) -> None:
        with self._lock:
            self._file_cache.pop(path, None)

    def _warn_once(
        self,
        key: tuple[Any, ...],
        message: str,
        *args: Any,
    ) -> None:
        with self._lock:
            if key in self._warning_keys:
                self._warning_keys.move_to_end(key)
                return
            self._warning_keys[key] = None
            while len(self._warning_keys) > _MAX_WARNING_KEYS:
                self._warning_keys.popitem(last=False)
        logger.warning(message, *args)
