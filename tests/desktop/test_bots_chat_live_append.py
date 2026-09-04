"""Regression tests for #93942 slice 1 — Bot Mode open chat pane never picks
up background deliveries.

Mechanism (traced on 85a55e2b30-era main, re-verified through 03b87d666d):

* Background DMs deliver via a separate ``hermes -p <profile> chat``
  subprocess (``tools/bot_relay.py::local_delivery_command``) that writes the
  turn straight to the session DB. No gateway stream event reaches the desktop.
* The gateway broadcasts ``sessions.changed`` on state.db mtime movement, and
  the desktop's tick handler calls ``requestActiveTranscriptRefresh(true)`` —
  but that refresh covers ONLY the MAIN pane's selection
  (``activeStoredSessionId = selectedStoredSessionId``, wiring.tsx).
* Bot canonical chats open as workspace tiles (``workspaceMode: 'bots'``,
  never the main selection), AND ``resolveActiveTranscriptSession()`` bails
  when the session is absent from ``$sessions``/``$messagingSessions`` — which
  bot chats always are (they carry the core ``hidden`` flag,
  plugin.js "Bot Mode sessions are ALWAYS hidden").

Double exclusion ⇒ the sessions.changed refresh silently no-ops for exactly
the conversation the user is staring at.

Fix contract (slice 1): the sessions.changed tick must also reconcile the
transcripts of VISIBLE workspace tiles — signature-gated like the main pane,
keyed off each tile's stored↔runtime id pair — without touching messaging
polls, roster cadence, or the main-pane path.
"""

from __future__ import annotations

import re
from pathlib import Path


def _bg_sync_source() -> str:
    return (
        Path(__file__).resolve().parents[2]
        / "apps"
        / "desktop"
        / "src"
        / "app"
        / "contrib"
        / "hooks"
        / "use-background-sync.ts"
    ).read_text(encoding="utf-8")


def test_sessions_changed_tick_covers_workspace_tiles():
    """The $sessionsChangeTick listener must refresh tile transcripts too, not
    only the main pane's active transcript."""
    src = _bg_sync_source()

    # Locate the tick listener block.
    m = re.search(r"\$sessionsChangeTick\.listen\(", src)
    assert m, "sessions.changed listener must exist"

    # The run() it invokes must include a per-tile reconciliation call in
    # addition to requestActiveTranscriptRefresh. Pre-fix, `run()` touches
    # refreshSessions/refreshMessagingSessions/requestActiveTranscriptRefresh
    # only — no tile path.
    run_start = src.rfind("const run = () => {", 0, m.start())
    body = src[run_start : src.find("}", m.start())]

    assert re.search(r"[Tt]ile", body), (
        "pre-fix state: sessions.changed refresh covers the main pane only; "
        "open workspace tiles (bot canonical chats) are never reconciled "
        "(#93942 scenario A)"
    )


def test_tile_reconciliation_is_signature_gated():
    """Tile refreshes must be signature-gated (no-change events must not churn
    every visible tile), matching the main pane's reconcile discipline."""
    src = _bg_sync_source()

    # The fix introduces a signature-gated tile reconciler; assert its guard
    # exists near the tile handling code.
    assert re.search(r"[Tt]ile[A-Za-z]*Signature|signature.*tile", src, re.IGNORECASE) or (
        "signatureRef" in src and "[Tt]ile" in src
    ), "tile reconciliation must be signature-gated like the main pane path"
