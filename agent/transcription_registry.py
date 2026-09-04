"""
Transcription Provider Registry
================================

Central map of registered STT providers. Populated by plugins at
import-time via :meth:`PluginContext.register_transcription_provider`;
consumed by :mod:`tools.transcription_tools` to dispatch
:func:`transcribe_audio` calls to the active plugin backend **when**
the configured ``stt.provider`` name is not a built-in.

Built-ins-always-win
--------------------
Plugin names that collide with a built-in STT provider (``local``,
``local_command``, ``groq``, ``openai``, ``mistral``, ``xai``) are
rejected at registration with a warning. This invariant is also
re-checked at dispatch time in
:func:`tools.transcription_tools._dispatch_to_plugin_provider`.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider
from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)


# Names reserved for native built-in STT handlers. Plugins cannot
# register a name in this set — the registration call is rejected with
# a warning. **Kept in sync with ``BUILTIN_STT_PROVIDERS`` in
# :mod:`tools.transcription_tools`** — a regression test in
# ``tests/agent/test_transcription_registry.py::TestBuiltinSync``
# fails if the two lists drift. Importing from
# ``tools.transcription_tools`` directly would create a circular
# dependency (``tools.transcription_tools`` imports
# ``agent.transcription_registry`` for dispatch).
_BUILTIN_NAMES = frozenset({
    "local",
    "local_command",
    "groq",
    "openai",
    "mistral",
    "xai",
    "elevenlabs",
    "deepinfra",
})


_providers: Dict[str, TranscriptionProvider] = {}
_scoped_providers: Dict[str, Dict[str, TranscriptionProvider]] = {}
_lock = threading.Lock()


def register_provider(provider: TranscriptionProvider, *, scope: Optional[str] = None) -> None:
    """Register a transcription provider.

    Rejects:

    - Non-:class:`TranscriptionProvider` instances (raises :class:`TypeError`).
    - Empty/whitespace ``.name`` (raises :class:`ValueError`).
    - Names colliding with a built-in (logs a warning, silently
      ignores — built-ins-always-win invariant).

    Re-registration (same ``name``) overwrites the previous entry and
    logs a debug message — makes hot-reload scenarios (tests, dev
    loops) behave predictably.
    """
    if not isinstance(provider, TranscriptionProvider):
        raise TypeError(
            f"register_provider() expects a TranscriptionProvider instance, "
            f"got {type(provider).__name__}"
        )
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Transcription provider .name must be a non-empty string")
    key = name.strip().lower()
    if key in _BUILTIN_NAMES:
        logger.warning(
            "Transcription provider '%s' shadows a built-in name; registration "
            "ignored. Built-in STT providers (%s) always win — pick a different "
            "name.",
            key, ", ".join(sorted(_BUILTIN_NAMES)),
        )
        return
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
        existing = target.get(key)
        target[key] = provider
    if existing is not None:
        logger.debug(
            "Transcription provider '%s' re-registered (was %r)",
            key, type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered transcription provider '%s' (%s)",
            key, type(provider).__name__,
        )


def list_providers(*, scope: Optional[str] = None) -> List[TranscriptionProvider]:
    """Return all registered providers, sorted by name."""
    with _lock:
        merged = dict(_providers)
        merged.update(_scoped_providers.get(scope or hermes_home_key(), {}))
        items = list(merged.values())
    return sorted(items, key=lambda p: p.name)


def get_provider(name: str, *, scope: Optional[str] = None) -> Optional[TranscriptionProvider]:
    """Return the provider registered under *name*, or None.

    Name matching is case-insensitive and whitespace-tolerant — mirrors
    how ``tools.transcription_tools._get_provider`` normalizes the
    configured ``stt.provider`` value.
    """
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    with _lock:
        return _scoped_providers.get(scope or hermes_home_key(), {}).get(key) or _providers.get(key)


def snapshot_registration(
    name: str, *, scope: Optional[str] = None
) -> Optional[TranscriptionProvider]:
    key = name.strip().lower()
    with _lock:
        target = _providers if scope is None else _scoped_providers.get(scope, {})
        return target.get(key)


def restore_registration(
    name: str,
    current: TranscriptionProvider,
    previous: Optional[TranscriptionProvider],
    *,
    scope: Optional[str] = None,
) -> bool:
    """Restore a plugin registration only when *current* is still installed."""
    key = name.strip().lower()
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
        if target.get(key) is not current:
            return False
        if previous is None:
            target.pop(key, None)
        else:
            target[key] = previous
        if scope is not None and not target:
            _scoped_providers.pop(scope, None)
    return True


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    with _lock:
        _providers.clear()
        _scoped_providers.clear()
