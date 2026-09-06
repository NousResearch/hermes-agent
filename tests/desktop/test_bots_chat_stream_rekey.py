"""Regression tests for #93942 slice 2 — open chat pane never follows a runtime
rebuild after a mid-conversation model switch.

Mechanism (traced on 41447a6d70):

* A mid-conversation model/provider switch rebuilds the agent runtime. The
  rebuilt runtime emits its ``session.info`` (and all later stream events)
  under a NEW explicit ``session_id``; the old id is dead.
* The pane's identity is the pair ``(activeSessionIdRef, selectedStoredSessionIdRef)``.
  After the rebuild, incoming events carry an explicit sid that no longer
  equals ``activeSessionIdRef.current``, so ``isActiveEvent`` is False for every
  subsequent event of the SAME conversation — view-scoped side effects stop,
  and the pane keeps listening on a dead runtime until a full resume.
* Existing machinery covers compression rotation only:
  ``ensureSessionState`` fires the stored-id rotation signal when the SAME
  runtime's stored id rotates — but a model-switch rebuild produces a NEW
  runtime with a NEW stored id, which is invisible to that path.

Fix contract: when a ``session.info`` event's lineage
(``sessionMatchesStoredId`` on ``stored_session_id``) matches the currently
selected conversation but its runtime id differs from the active one, re-bind
the pane: adopt the new runtime id as the active session id (keeping the same
durable selection), so live events keep flowing to the view without a resume.
Guarded to only fire when the old runtime is dead (no busy/streaming state) so
an overlapping turn is never hijacked mid-flight.

Together with #94255 (slice 1), this closes #93942.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "apps" / "desktop" / "src"


def _session_info_source() -> str:
    return (
        ROOT / "app" / "session" / "hooks" / "use-message-stream" / "gateway-event" / "session-info.ts"
    ).read_text(encoding="utf-8")


def test_session_info_rebinds_pane_to_rebuilt_runtime():
    """A session.info whose lineage matches the selected conversation but whose
    runtime id differs from the active one must trigger the re-bind path."""
    src = _session_info_source()

    # The fix adds a lineage-checked re-bind call in handleSessionInfoEvent,
    # distinct from the cwd-claim helper (#71254). Pre-fix there is exactly one
    # sessionInfoDescribesSelectedSession use site (cwd claim); post-fix a
    # second consumer exists for the re-bind.
    uses = len(re.findall(r"sessionInfoDescribesSelectedSession\(", src))

    assert uses >= 3, (
        "pre-fix state: session.info lineage matching is used only for the cwd "
        "claim; a rebuilt runtime (model switch) under a new session_id is "
        "never adopted back into the open pane (#93942 scenario B)"
    )


def test_rebind_is_guarded_against_live_turns():
    """The re-bind must not hijack a conversation while its OLD runtime is
    still streaming/busy — only a dead runtime may be adopted over."""
    src = _session_info_source()

    # Locate the rebind helper's body precisely, then assert its guard inside.
    fn = re.search(
        r"function maybeRebindPaneToRebuiltRuntime\([^)]*\): boolean \{", src
    )
    assert fn, (
        "pre-fix state: no maybeRebindPaneToRebuiltRuntime adoption path "
        "exists; the rebuilt runtime is never adopted into the open pane"
    )

    body_start = fn.end()
    next_fn = min(
        (i for i in (src.find("\nfunction ", body_start), src.find("\nexport function ", body_start)) if i != -1),
        default=-1,
    )
    assert next_fn != -1, "re-anchor needed: rebind helper is no longer followed by another function"

    body = src[body_start:next_fn]

    # The busy/awaiting/streaming guard must appear INSIDE the helper body —
    # not merely somewhere in the file (fixes the precedence hazard the
    # #94417 review flagged in the original draft of this assertion).
    guard = re.search(r"oldState\?\.(busy|awaitingResponse|streamId)", body)

    assert guard, (
        "the re-bind helper must check the outgoing runtime's busy/streaming "
        "state before switching; adopting over a live turn would split one "
        "conversation across two panes"
    )


def test_rebind_requires_stored_session_id_lineage():
    """The gateway always stamps stored_session_id on session.info (server.py:
    'stored_session_id': session_key or ''), but an empty string must NOT be
    adopted as lineage proof — only a real stored id may trigger the re-bind."""
    src = _session_info_source()

    fn = re.search(r"function maybeRebindPaneToRebuiltRuntime\([^)]*\): boolean \{", src)
    assert fn, "rebind helper missing"
    body_start = fn.end()
    body = src[body_start : min(
        (i for i in (src.find("\nfunction ", body_start), src.find("\nexport function ", body_start)) if i != -1),
        default=-1,
    )]

    # The early return guards on the field being a non-empty usable string via
    # the typeof check + downstream sessionInfoDescribesSelectedSession('' →
    # null → wildcard refusal).
    assert 'typeof payload?.stored_session_id !== "string"' in body or (
        "typeof payload?.stored_session_id !== 'string'" in body
    ), "rebind must refuse events without a string stored_session_id"

