"""Detect when the gateway is running stale code after a hot ``git pull``.

The gateway is a single long-lived process; its ``sys.modules`` is frozen at
boot. If the checkout is updated underneath it (a manual ``git pull``, or the
window before ``hermes update``'s graceful restart fires), a first-time lazy
import on a new code path can resolve a freshly-pulled consumer module against a
stale cached dependency -> ImportError (see
``tests/test_stale_utils_module_import.py`` for the exact failure).

We snapshot the checkout revision at gateway startup and compare on demand, so
risky callers (e.g. ``/model`` switching) can refuse with a clear "restart the
gateway" message instead of crashing on a cryptic import error.

If the revision can't be read (non-git install, IO error), the boot snapshot
stays ``None`` and skew detection no-ops — it never produces a false positive.

We also persist the boot fingerprint to ``HERMES_HOME`` so the next gateway
boot can detect that the checkout advanced while the previous gateway was
running, and purge stale ``__pycache__`` before any import picks up bytecode
compiled against the old source (see ``purge_stale_pycache``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_boot_fingerprint: str | None = None

logger = logging.getLogger("gateway.code_skew")

_PENDING_MODEL_SWITCH_VERSION = 1
_PENDING_MODEL_SWITCH_SOURCE_FIELDS = frozenset(
    {
        "platform",
        "chat_id",
        "chat_name",
        "chat_type",
        "user_id",
        "user_name",
        "thread_id",
        "chat_topic",
        "user_id_alt",
        "chat_id_alt",
        "scope_id",
        "guild_id",
        "parent_chat_id",
        "message_id",
        "profile",
    }
)


def _fingerprint_file() -> Path:
    """Resolve the boot-fingerprint path at call time for the active profile."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / ".gateway_boot_fingerprint"


def _pending_model_switches_file() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / ".pending_model_switches.json"


@dataclass(frozen=True)
class PendingModelSwitch:
    """A validated model selection that must wait for a fresh gateway process.

    The payload deliberately contains only routing and model-selection data.
    It never stores a callback, raw event, provider credentials, or arbitrary
    command text to replay after the reload.
    """

    source: dict[str, str]
    model_input: str
    provider: str | None
    persist_global: bool
    replay: bool = False
    intent_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        # Persist routing scalars only. Source dictionaries can otherwise carry
        # adapter-private nested metadata that this continuation never needs.
        object.__setattr__(
            self,
            "source",
            {
                key: value
                for key, value in self.source.items()
                if key in _PENDING_MODEL_SWITCH_SOURCE_FIELDS and isinstance(value, str)
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "source": {
                key: value
                for key, value in self.source.items()
                if key in _PENDING_MODEL_SWITCH_SOURCE_FIELDS
            },
            "model_input": self.model_input,
            "provider": self.provider,
            "persist_global": self.persist_global,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "PendingModelSwitch | None":
        if not isinstance(raw, dict):
            return None
        source = raw.get("source")
        if not isinstance(source, dict) or any(
            key not in _PENDING_MODEL_SWITCH_SOURCE_FIELDS or not isinstance(value, str)
            for key, value in source.items()
        ):
            return None
        model_input = raw.get("model_input")
        provider = raw.get("provider")
        persist_global = raw.get("persist_global")
        intent_id = raw.get("intent_id")
        if not isinstance(source, dict) or not isinstance(model_input, str):
            return None
        if provider is not None and not isinstance(provider, str):
            return None
        if not isinstance(persist_global, bool) or not isinstance(intent_id, str):
            return None
        if not isinstance(source.get("platform"), str) or not isinstance(source.get("chat_id"), str):
            return None
        if not intent_id or len(intent_id) > 128:
            return None
        if len(model_input) > 1024 or (provider is not None and len(provider) > 256):
            return None
        return cls(
            source={
                key: value
                for key, value in source.items()
                if key in _PENDING_MODEL_SWITCH_SOURCE_FIELDS and isinstance(value, str)
            },
            model_input=model_input,
            provider=provider,
            persist_global=persist_global,
            intent_id=intent_id,
        )


def _load_pending_model_switches() -> list[PendingModelSwitch]:
    try:
        raw = json.loads(_pending_model_switches_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict) or raw.get("version") != _PENDING_MODEL_SWITCH_VERSION:
        return []
    entries = raw.get("switches")
    if not isinstance(entries, list):
        return []
    return [intent for item in entries if (intent := PendingModelSwitch.from_dict(item)) is not None]


def _write_pending_model_switches(intents: list[PendingModelSwitch]) -> None:
    path = _pending_model_switches_file()
    if not intents:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _PENDING_MODEL_SWITCH_VERSION,
        "switches": [intent.to_dict() for intent in intents],
    }
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temp.replace(path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def enqueue_pending_model_switch(intent: PendingModelSwitch) -> None:
    """Durably enqueue a typed switch before requesting a code reload."""
    _write_pending_model_switches([*_load_pending_model_switches(), intent])


def peek_pending_model_switch() -> PendingModelSwitch | None:
    """Return the next continuation without removing its durable record."""
    intents = _load_pending_model_switches()
    return intents[0] if intents else None


def ack_pending_model_switch(intent_id: str) -> None:
    """Remove a continuation only after its replay task completed."""
    _write_pending_model_switches(
        [intent for intent in _load_pending_model_switches() if intent.intent_id != intent_id]
    )


def _fingerprint() -> str | None:
    """Current checkout fingerprint, reusing the CLI's git-rev reader.

    ``hermes_cli.main`` is always already imported in a gateway process (it's
    the entry point), so this import is free and avoids duplicating the
    worktree-aware ref resolution.
    """
    try:
        from hermes_cli.main import _read_git_revision_fingerprint

        return _read_git_revision_fingerprint(_PROJECT_ROOT)
    except Exception:
        return None


def record_boot_fingerprint() -> None:
    """Snapshot the checkout revision at gateway startup (idempotent).

    Also persists the fingerprint to ``HERMES_HOME`` so the next boot can
    detect drift and purge stale ``__pycache__`` before imports pick up
    bytecode compiled against the old source.
    """
    global _boot_fingerprint
    if _boot_fingerprint is None:
        _boot_fingerprint = _fingerprint()
    _persist_boot_fingerprint(_boot_fingerprint)


def _persist_boot_fingerprint(fingerprint: str | None) -> None:
    """Write the boot fingerprint to ``HERMES_HOME`` for next-boot comparison."""
    if fingerprint is None:
        return
    try:
        path = _fingerprint_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fingerprint, encoding="utf-8")
    except Exception:
        logger.debug("Failed to persist boot fingerprint", exc_info=True)


def purge_stale_pycache() -> bool:
    """Purge stale ``__pycache__`` directories if the checkout advanced.

    At gateway boot, compares the persisted fingerprint from the *previous*
    gateway run with the current checkout. If they differ, Python's
    ``__pycache__`` may contain bytecode compiled against the old source
    (``.pyc`` files whose header mtime matches the old ``.py`` because git
    operations can preserve mtime). Deleting the ``__pycache__`` dirs forces
    a clean recompile on first import.

    Returns ``True`` if a purge was performed, ``False`` otherwise.
    """
    current = _fingerprint()
    if current is None:
        return False
    try:
        previous = _fingerprint_file().read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return False
    if not previous or previous == current:
        return False

    purged = 0
    for pycache in _PROJECT_ROOT.rglob("__pycache__"):
        # Skip .venv and node_modules to avoid slow/unnecessary I/O
        if ".venv" in pycache.parts or "node_modules" in pycache.parts:
            continue
        try:
            for pyc in pycache.glob("*.pyc"):
                pyc.unlink()
                purged += 1
        except OSError:
            pass
    if purged:
        logger.info(
            "Purged %d stale .pyc file(s) — checkout advanced from %s to %s",
            purged,
            _short(previous),
            _short(current),
        )
    return purged > 0


def _short(fingerprint: str) -> str:
    """Render a ``git:<ref>:<sha>`` fingerprint as a compact label."""
    sha = fingerprint.rsplit(":", 1)[-1]
    if sha and sha != "unresolved" and len(sha) > 10:
        return sha[:10]
    return sha or fingerprint


def detect_code_skew() -> tuple[str, str] | None:
    """Return ``(boot_rev, disk_rev)`` short labels if the checkout drifted
    since boot, else ``None``."""
    if _boot_fingerprint is None:
        return None
    current = _fingerprint()
    if current is None or current == _boot_fingerprint:
        return None
    return _short(_boot_fingerprint), _short(current)
