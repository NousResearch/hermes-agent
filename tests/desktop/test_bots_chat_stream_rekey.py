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

    # The guard reads the previous runtime's cached state before adopting.
    m = re.search(r"rebind[A-Za-z]*|adoptRebuiltRuntime|sessionStateByRuntimeIdRef", src)

    assert m and "busy" in src[m.start() : m.start() + 2000] or "state?.busy" in src, (
        "pre-fix state: no guarded adoption path exists; the re-bind must check "
        "the outgoing runtime's busy/streaming state before switching"
    )
