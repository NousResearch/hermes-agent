"""A recoverable delivery must not be logged as if the work were lost.

``_build_process_event_source`` returns None for a raw (non-``agent:main:``)
session key because it cannot derive routing metadata from one. Its caller
then recovers: it pulls the raw session id back out and wakes the api_server
session via self-post. That path works — production logs show the wake firing
6 ms after the warning, followed by a full agent run.

The warning text ("Synthetic event source unresolvable") nevertheless reads as
dropped work, and was misread exactly that way during an audit. Raw api_server
keys have a documented fallback and belong at debug level; a structured key
that still fails to resolve has no fallback and must keep warning.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.run import _has_api_server_wake_fallback


def test_raw_session_key_has_a_fallback():
    """The api_server binds the raw X-Hermes-Session-Id as the session key."""
    assert _has_api_server_wake_fallback("20260812_002915_c9118f", {}) is True


def test_origin_session_id_has_a_fallback():
    """The caller prefers origin_session_id when the event carries one."""
    assert _has_api_server_wake_fallback("", {"origin_session_id": "20260812_002915_c9118f"}) is True


def test_structured_key_has_no_fallback():
    """A platform-qualified key that fails to resolve is genuinely unroutable."""
    assert _has_api_server_wake_fallback("agent:main:telegram:dm:12345", {}) is False


def test_empty_key_without_origin_has_no_fallback():
    """A CLI-origin event carries neither — nothing to wake."""
    assert _has_api_server_wake_fallback("", {}) is False


if __name__ == "__main__":  # pytest is not installed in the Hermes venv
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failing" if failures else "\nall green")
    sys.exit(1 if failures else 0)
