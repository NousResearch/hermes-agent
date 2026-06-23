"""Multi-session replay probe — the closeout GATE for the 2026-06-22 hygiene
degrade fix (AC-9 / RC-D).

For each sampled real session in the local ``lcm.db`` it reconstructs the
session-hygiene populations exactly the way the gateway does — using the SHARED
``hygiene_eligible_msgs`` filter (OQ-F: import the real filter, never a copy) —
runs ``build_hygiene_stats`` over a realistic LCM ``[summary] + raw-tail``
compressed shape, and asserts the granular announce reconciles AND renders the
multi-line "Removed from live context" form (not the two-line degrade).

Selection rule (RC-D): the sample MUST include the known failing session
``20260619_175552_6a7a77e3`` AND the N most tool-heavy sessions available — never
cherry-picked easy ones.

This is SKIPPED when no local ``lcm.db`` is present (CI / fresh checkout); it is
the local closeout gate, with the organic live render as a follow-up.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from agent.compaction_stats import build_hygiene_stats, hygiene_eligible_msgs
from agent.conversation_compression import _format_compaction_announce
from agent.model_metadata import estimate_messages_tokens_rough

_LCM_DB = os.path.expanduser("~/.hermes/lcm.db")
_KNOWN_FAILING_SESSION = "20260619_175552_6a7a77e3"
_FRESH_TAIL = 32          # LCM fresh_tail_count
_MIN_ELIGIBLE = 4         # the gateway only compresses when eligible >= 4


def _load_session(con: sqlite3.Connection, sid: str) -> list[dict]:
    rows = con.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY store_id",
        (sid,),
    ).fetchall()
    return [{"role": r[0], "content": (r[1] or "")} for r in rows]


def _select_sessions(con: sqlite3.Connection, n_tool_heavy: int = 6) -> list[str]:
    """Known failing session + the N most tool-heavy sessions (RC-D selection rule)."""
    tool_heavy = con.execute(
        """
        SELECT session_id, SUM(CASE WHEN role='tool' THEN 1 ELSE 0 END) AS tools,
               COUNT(*) AS total
        FROM messages GROUP BY session_id
        HAVING total >= 50
        ORDER BY tools DESC
        LIMIT ?
        """,
        (n_tool_heavy,),
    ).fetchall()
    sids = [r[0] for r in tool_heavy]
    if _KNOWN_FAILING_SESSION not in sids:
        sids.insert(0, _KNOWN_FAILING_SESSION)
    return sids


def _hygiene_replay(history: list[dict], *, sanitize_tail: bool = False):
    """Reconstruct the gateway hygiene populations + a realistic LCM compressed
    shape, then build + validate stats. Returns (stats, rendered_line).

    ``sanitize_tail=True`` mutates the kept-tail copies so they no longer
    signature-match their raw originals — the REAL LCM behavior (it cleans
    assistant content / strips tool scaffolding). This exercises the
    two-population divergence (pre-side kept != comp-side kept) that the
    2026-06-22 token fix + the PR #101 message-axis fix both address.
    """
    # gateway: raw_history is a per-row shallow copy snapshot
    raw_history = [{**m} for m in history]
    # SHARED filter (OQ-F): identical to the live gateway filter
    eligible = hygiene_eligible_msgs(history)
    # LCM compresses the eligible set → [summary] + fresh-tail-of-eligible
    fresh_tail = [{**m} for m in eligible[-_FRESH_TAIL:]]
    if sanitize_tail:
        for m in fresh_tail:
            if m.get("role") == "assistant":
                m["content"] = (m.get("content") or "")[:40] + " [sanitized]"
    summary = [{"role": "user", "content": "[Recent Summary (d0, node 1)] folded chat ..."}]
    compressed = summary + fresh_tail
    stats = build_hygiene_stats(
        raw_history=raw_history,
        eligible_msgs=eligible,
        compressed=compressed,
        estimator=estimate_messages_tokens_rough,
    )
    line = _format_compaction_announce(
        engine_name="lcm", status="compacted",
        old_session_id="a", new_session_id="b",
        old_messages=stats.pre_messages, new_messages=stats.post_messages,
        pre_tokens=stats.pre_tokens, post_tokens=stats.post_tokens,
        model="claude-opus-4-8", provider="claude-app",
        trigger_reason="hygiene_messages", trigger_value=1000,
        stats=stats,
    )
    return stats, line, compressed


@pytest.mark.skipif(not os.path.exists(_LCM_DB), reason="no local lcm.db (CI / fresh checkout)")
def test_replay_real_sessions_reconcile_and_render_granular():
    con = sqlite3.connect(f"file:{_LCM_DB}?mode=ro", uri=True)
    try:
        sids = _select_sessions(con)
        assert _KNOWN_FAILING_SESSION in sids, "selection must include the known failing session"
        checked = 0
        for sid in sids:
            history = _load_session(con, sid)
            eligible = hygiene_eligible_msgs(history)
            if len(eligible) < _MIN_ELIGIBLE:
                continue  # the gateway wouldn't compress this; skip
            # Both the verbatim tail AND the sanitized tail (real LCM) must reconcile
            # AND render granular with the CORRECT kept count (the message-axis bug
            # rendered "kept 0 recent chat" on sanitized tails — PR #101).
            for sanitize in (False, True):
                stats, line, compressed = _hygiene_replay(history, sanitize_tail=sanitize)
                ok, reason = stats.validate()
                assert ok, f"session {sid} (sanitize={sanitize}) did NOT reconcile: {reason}"
                assert line is not None, f"session {sid} rendered no announce"
                assert "Removed from live context" in line, (
                    f"session {sid} (sanitize={sanitize}) degraded to two-line:\n{line}"
                )
                # the displayed kept count is the COMP-side kept tail, not pre-side 0
                expected_kept = sum(
                    1 for m in compressed
                    if m.get("role") != "system" and "[Recent Summary" not in (m.get("content") or "")
                )
                assert stats.kept_messages == expected_kept, (
                    f"session {sid} (sanitize={sanitize}): kept_messages "
                    f"{stats.kept_messages} != comp-side kept tail {expected_kept}"
                )
                assert f"kept {stats.kept_messages} recent chat" in line
                assert stats.post_messages == len(compressed)
            checked += 1
        assert checked >= 1, "no eligible real sessions were checked"
    finally:
        con.close()
