"""Tests for the Matrix ``cron_continuable_surface`` extra key and its pairing warning.

``cron_continuable_surface: in_channel`` (paired with ``session_scope: room`` — or
``auto_thread: false``) lets a continuable cron job deliver FLAT into a room — no
dedicated thread — so a plain room reply continues the job via the whole-room
session ``(matrix, room_id, None)``. Mirrors the Slack surface (specs/cron-inchannel-
continuable decisions D1/D4/D5/D6).

- ``_cron_continuable_surface`` resolves the key: default ``"thread"``, coerces
  any unrecognised value to ``"thread"`` (fail safe), only ``"in_channel"``
  opts in.
- ``supports_inchannel_continuable`` is True on Matrix (a send with no thread_id
  is a plain room event; the room bucket exists once session_scope pins it).
- ``_warn_if_inchannel_without_room_bucket`` warns (D5: warn, not hard-require)
  when ``in_channel`` is set while room messages still open per-message threads
  — the misconfig fails SAFE to a threaded continuation, so it is a warning,
  not a rejection.
"""

import logging
import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub mautrix if not installed (same pattern as test_matrix_approval_reaction_fail_closed.py)
# ---------------------------------------------------------------------------

def _stub_mautrix():
    stub = types.ModuleType("mautrix")
    for sub in ("mautrix.types", "mautrix.client", "mautrix.client.api",
                "mautrix.errors", "mautrix.crypto", "mautrix.util",
                "mautrix.util.config"):
        sys.modules.setdefault(sub, types.ModuleType(sub))
    sys.modules.setdefault("mautrix", stub)
    m = sys.modules["mautrix.types"]

    class EventType:
        ROOM_MESSAGE = "m.room.message"
        REACTION = "m.reaction"
        ROOM_ENCRYPTED = "m.room.encrypted"
        ROOM_NAME = "m.room.name"

    for attr in ("ContentURI", "EventID", "RoomID", "SyncToken", "UserID"):
        setattr(m, attr, str)
    m.EventType = EventType


_stub_mautrix()

from plugins.platforms.matrix.adapter import MatrixAdapter  # noqa: E402


def _make_adapter(extra, session_scope="auto", auto_thread=True):
    """object.__new__ skips __init__ (heavy setup) — established matrix-test
    pattern. Attach the config ``extra`` dict plus the two inbound-threading
    knobs __init__ derives (``_matrix_session_scope`` / ``_auto_thread``)."""
    adapter = object.__new__(MatrixAdapter)
    cfg = MagicMock()
    cfg.extra = dict(extra)
    adapter.config = cfg
    adapter._matrix_session_scope = session_scope
    adapter._auto_thread = auto_thread
    return adapter


# --- capability flag -------------------------------------------------------

def test_capability_flag_declared():
    """Matrix can host the flat continuable surface: D6 gate must pass through."""
    assert MatrixAdapter.supports_inchannel_continuable is True

def test_d6_gate_accepts_matrix_adapter():
    """The scheduler's _inchannel_surface_supported probe (D6) reads the class
    attribute on a native adapter — Matrix must no longer fail safe to thread."""
    from cron.scheduler_delivery import _inchannel_surface_supported
    adapter = _make_adapter({"cron_continuable_surface": "in_channel"})
    assert _inchannel_surface_supported(adapter, "matrix") is True

# --- surface resolver ------------------------------------------------------

def test_surface_default_is_thread():
    adapter = _make_adapter({})
    assert adapter._cron_continuable_surface() == "thread"

def test_surface_unrecognised_value_coerces_to_thread():
    """Fail safe: any value that isn't 'in_channel' resolves to 'thread'."""
    adapter = _make_adapter({"cron_continuable_surface": "bogus"})
    assert adapter._cron_continuable_surface() == "thread"

def test_surface_in_channel_resolved():
    adapter = _make_adapter({"cron_continuable_surface": "in_channel"})
    assert adapter._cron_continuable_surface() == "in_channel"

# --- pairing warning (D5: warn, not hard-require) --------------------------

def test_warn_inchannel_default_threading(caplog):
    """Default scope=auto + auto_thread=true: every room message opens its own
    thread, so a flat brief cannot continue on a plain reply — warn."""
    adapter = _make_adapter({"cron_continuable_surface": "in_channel"})
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.matrix.adapter"):
        adapter._warn_if_inchannel_without_room_bucket()
    assert "cron_continuable_surface=in_channel" in caplog.text

def test_warn_silent_when_session_scope_pins_room(caplog):
    """session_scope: room keys every inbound room message to the shared room
    bucket — the pairing in_channel needs; no warning."""
    adapter = _make_adapter({"cron_continuable_surface": "in_channel"},
                            session_scope="room", auto_thread=True)
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.matrix.adapter"):
        adapter._warn_if_inchannel_without_room_bucket()
    assert "cron_continuable_surface=in_channel" not in caplog.text

def test_warn_silent_when_auto_thread_disabled(caplog):
    """scope=auto + auto_thread=false also lands flat room replies in the
    room bucket — no warning."""
    adapter = _make_adapter({"cron_continuable_surface": "in_channel"},
                            session_scope="auto", auto_thread=False)
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.matrix.adapter"):
        adapter._warn_if_inchannel_without_room_bucket()
    assert "cron_continuable_surface=in_channel" not in caplog.text

def test_warn_scope_thread_warns_even_without_auto_thread(caplog):
    """session_scope: thread forces a synthetic thread per message regardless
    of auto_thread — still warn."""
    adapter = _make_adapter({"cron_continuable_surface": "in_channel"},
                            session_scope="thread", auto_thread=False)
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.matrix.adapter"):
        adapter._warn_if_inchannel_without_room_bucket()
    assert "cron_continuable_surface=in_channel" in caplog.text

def test_warn_silent_when_surface_is_thread(caplog):
    """No in_channel opt-in → nothing to pair, no warning even with the
    default per-message threading."""
    adapter = _make_adapter({})
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.matrix.adapter"):
        adapter._warn_if_inchannel_without_room_bucket()
    assert "cron_continuable_surface=in_channel" not in caplog.text
