"""Persistent cooldown tracking for provider failover."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class _CooldownState:
    count: int = 0
    until: float = 0.0
    reason: str = "rate_limit"


def build_cooldown_key(provider: str, api_key: Any, reason: str) -> str:
    """Build a provider- or provider/key-scoped cooldown key without exposing keys.

    Credential sources may be callbacks or opaque objects.  They are not stable
    credentials, so never invoke, stringify, or hash them; use provider scope.
    """
    provider = (provider or "").strip().lower()
    if reason == "billing" or not isinstance(api_key, str) or not api_key:
        return provider
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"{provider}:{fingerprint}"


_PERSISTABLE_PROVIDER_SCOPES = frozenset({
    "ai-gateway", "alibaba", "anthropic", "arcee", "azure-foundry", "bedrock",
    "copilot", "copilot-acp", "deepseek", "fireworks", "gemini", "gmi",
    "huggingface", "kilocode", "kimi-coding", "kimi-coding-cn", "lmstudio",
    "minimax", "minimax-cn", "minimax-oauth", "moa", "nous", "novita",
    "nvidia", "ollama-cloud", "opencode-go", "opencode-zen", "openai",
    "openai-api", "openai-codex", "openrouter", "qwen-oauth", "stepfun",
    "tencent-tokenhub", "vertex", "xai", "xai-oauth", "xiaomi", "zai",
})


def _is_persistable_provider_scope(provider: str) -> bool:
    """Allow known provider identifiers and configured custom-provider scopes."""
    return provider in _PERSISTABLE_PROVIDER_SCOPES or provider == "custom" or provider.startswith("custom:")


def _is_safe_persisted_key(key: str) -> bool:
    if not isinstance(key, str):
        return False
    provider, separator, fingerprint = key.rpartition(":")
    if not separator:
        return _is_persistable_provider_scope(key)
    return _is_persistable_provider_scope(provider) and len(fingerprint) == 16 and all(
        char in "0123456789abcdef" for char in fingerprint
    )


class CooldownManager:
    """Thread-safe cooldown tracker backed by an atomic JSON file."""

    def __init__(
        self,
        base_seconds: float = 60.0,
        multiplier: float = 5.0,
        max_seconds: float = 3600.0,
        billing_base_hours: float = 5.0,
        billing_max_hours: float = 24.0,
        storage_path: Union[Path, None, Literal[False]] = None,
    ) -> None:
        self._base_seconds = base_seconds
        self._multiplier = multiplier
        self._max_seconds = max_seconds
        self._billing_base_hours = billing_base_hours
        self._billing_max_hours = billing_max_hours
        self._states: Dict[str, _CooldownState] = {}
        self._lock = threading.Lock()
        if storage_path is False:
            self._storage_path: Optional[Path] = None
        elif storage_path is None:
            try:
                from hermes_constants import get_hermes_home
                self._storage_path = get_hermes_home() / "cooldowns.json"
            except Exception:
                self._storage_path = None
        else:
            self._storage_path = Path(storage_path)
        with self._storage_lock():
            self._load(persist_discarded=False)
            self._persist()

    @contextmanager
    def _storage_lock(self) -> Iterator[None]:
        """Serialize read-modify-write cycles across Hermes processes."""
        if self._storage_path is None:
            yield
            return
        lock_path = self._storage_path.with_name(f"{self._storage_path.name}.lock")
        deadline = time.monotonic() + 10.0
        while True:
            try:
                descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for cooldown lock: {lock_path}")
                time.sleep(0.01)
            else:
                break
        try:
            yield
        finally:
            os.close(descriptor)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def is_cooling(self, key: str) -> bool:
        with self._storage_lock():
            self._load(persist_discarded=False)
            with self._lock:
                state = self._states.get(key)
            return state is not None and time.monotonic() < state.until

    def mark_failure(
        self,
        key: str,
        reason: Literal["rate_limit", "billing"],
        cooldown_seconds: Optional[float] = None,
    ) -> float:
        # Read, calculate, and atomically publish while holding a cross-process
        # lock so separate Hermes processes preserve each other's cooldowns.
        with self._storage_lock():
            self._load(persist_discarded=False)
            with self._lock:
                state = self._states.setdefault(key, _CooldownState())
                if cooldown_seconds is None:
                    state.count += 1
                state.reason = reason
                if cooldown_seconds is not None:
                    cooldown_seconds = max(0.0, cooldown_seconds)
                elif reason == "billing":
                    cooldown_seconds = min(
                        self._billing_base_hours * (2 ** (state.count - 1)),
                        self._billing_max_hours,
                    ) * 3600.0
                else:
                    cooldown_seconds = min(
                        self._base_seconds * (self._multiplier ** (state.count - 1)),
                        self._max_seconds,
                    )
                state.until = time.monotonic() + cooldown_seconds
                count = state.count
            self._persist()
        logger.info("Cooldown: key=%r reason=%s count=%d duration=%.0fs", key, reason, count, cooldown_seconds)
        return cooldown_seconds

    def clear(self, key: str) -> None:
        with self._storage_lock():
            self._load(persist_discarded=False)
            with self._lock:
                self._states.pop(key, None)
            self._persist()

    def get_all_states(self) -> Dict[str, dict]:
        now = time.monotonic()
        with self._lock:
            return {
                key: {
                    "count": state.count,
                    "until": state.until,
                    "cooling": now < state.until,
                    "remaining_seconds": max(0.0, state.until - now),
                }
                for key, state in self._states.items()
            }

    def get_cooldown_status(self) -> dict:
        states = self.get_all_states()
        return {
            "total_tracked": len(states),
            "cooling": [key for key, state in states.items() if state["cooling"]],
            "expired": [key for key, state in states.items() if not state["cooling"] and state["count"] > 0],
            "details": states,
        }

    def _load(self, *, persist_discarded: bool = True) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            now_wall, now_mono = time.time(), time.monotonic()
            discarded_unsafe = False
            loaded_states: Dict[str, _CooldownState] = {}
            for key, entry in data.items():
                    if not _is_safe_persisted_key(key):
                        discarded_unsafe = True
                        continue
                    remaining = float(entry.get("until_wall", 0)) - now_wall
                    if remaining > 0:
                        loaded_states[key] = _CooldownState(
                            count=int(entry.get("count", 1)),
                            until=now_mono + remaining,
                            reason=str(entry.get("reason", "rate_limit")),
                        )
            with self._lock:
                self._states = loaded_states
            if discarded_unsafe and persist_discarded:
                self._persist()
        except Exception as exc:
            logger.warning("Failed to load cooldown state from %s: %s", self._storage_path, exc)

    def _persist(self) -> None:
        if self._storage_path is None:
            return
        try:
            now_mono, now_wall = time.monotonic(), time.time()
            with self._lock:
                data = {
                    key: {"reason": state.reason, "count": state.count, "until_wall": now_wall + remaining}
                    for key, state in self._states.items()
                    if _is_safe_persisted_key(key) and (remaining := state.until - now_mono) > 0
                }
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._storage_path.with_name(
                f"{self._storage_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp_path, self._storage_path)
        except Exception as exc:
            logger.warning("Failed to persist cooldown state to %s: %s", self._storage_path, exc)


_singleton: Optional[CooldownManager] = None
_singleton_lock = threading.Lock()


def get_cooldown_manager() -> CooldownManager:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = CooldownManager()
    return _singleton


def set_cooldown_manager(manager: CooldownManager) -> None:
    global _singleton
    with _singleton_lock:
        _singleton = manager
