"""``hermes sessions`` legacy-session maintenance commands.

Hermes has evolved its session storage over many versions (``schema_version``
0..25, ``title_source`` provenance, FTS layout v0→v1→v23). Older installs
leave several classes of legacy data in ``state.db``:

1. **Orphaned compression segments** — when a long conversation hit context
   compression in an old version, the continuation segment was written as an
   *independent root* (no ``parent_session_id`` link, parent not marked
   ``end_reason='compression'``). The sidebar renders these as many separate
   same-titled entries that are really one conversation.
   → ``hermes sessions repair-chains``
2. **Missing/truncated titles** — rows with no ``title_source`` provenance
   (pre-5566379f5) whose title is a bare first-message truncation, and empty
   chain segments that never inherited a title.
   → ``hermes sessions retitle-missing``
3. **Fork compression chains** — sessions split into a head + linked
   segments (parent ``end_reason='compression'``). Flattening them back into
   one in-place session matches modern in-place compression.
   → ``hermes sessions merge-chains``

This module implements all three, honoring the official provenance and
compression-chain semantics in ``hermes_state``.

Safety
------
* Dry-run by default; pass ``--apply`` to write.
* Never touches ``title_source == 'user'`` rows.
* Chain-relink candidates are reported but **not** auto-written without
  ``--apply``; delegate/branch/tool children are always excluded.
* ``merge-chains`` takes an automatic timestamped state.db snapshot before
  writing and verifies message totals after.
* Uses official provenance API (``set_auto_title`` / ``set_session_title``),
  never raw SQL UPDATEs for titles.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from hermes_state_common import _sql_session_last_active


# ---------------------------------------------------------------------------
# Chain-relink detection
# ---------------------------------------------------------------------------

# Confirmation callback shared by the destructive commands. Callers pass
# ``(items, title)`` and may also pass ``selected=`` (pre-checked indices);
# returns the set of indices to process, or None/empty when cancelled.
ConfirmFn = Callable[..., Optional[set[int]]]

# Children that must never be treated as compression continuations.
_DELEGATE_EXPR = (
    "json_extract(COALESCE(s.model_config, '{}'), '$._delegate_from') IS NULL"
)
_BRANCH_EXPR = (
    "json_extract(COALESCE(s.model_config, '{}'), '$._branched_from') IS NULL"
)


def _title_key(title: Optional[str]) -> Optional[str]:
    """Normalize a title for orphan-grouping.

    Strips the `` #N`` dedupe suffix so ``"Plan review"`` and
    ``"Plan review #2"`` group together (they are segments of one
    conversation relinked with lineage dedupe).
    """
    if not title or not title.strip():
        return None
    return re.sub(r" #\d+$", "", title.strip())


def _first_message_content(db, session_id: str) -> Optional[str]:
    """Return the first message's content for a session, if any."""
    row = db._conn.execute(
        """
        SELECT content FROM messages
        WHERE session_id = ? AND content IS NOT NULL AND content != ''
        ORDER BY id LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    return row[0] if row else None


def _first_message_contents(db, session_ids: list[str]) -> dict[str, Optional[str]]:
    """Batch variant of :func:`_first_message_content` (avoids N+1 queries).

    Returns ``{session_id: first_nonempty_content}`` for every requested
    session; sessions with no non-empty message map to ``None``. Semantics
    are identical to calling :func:`_first_message_content` per session.
    """
    if not session_ids:
        return {}
    placeholders = ",".join("?" * len(session_ids))
    rows = db._conn.execute(
        f"""
        SELECT m.session_id, m.content
        FROM messages m
        JOIN (
            SELECT session_id, MIN(id) AS first_id
            FROM messages
            WHERE content IS NOT NULL AND content != ''
            GROUP BY session_id
        ) f ON m.id = f.first_id
        WHERE m.session_id IN ({placeholders})
        """,
        tuple(session_ids),
    ).fetchall()
    out: dict[str, Optional[str]] = {sid: None for sid in session_ids}
    for row in rows:
        out[row["session_id"]] = row["content"]
    return out


def _content_is_compaction_summary(text: str) -> bool:
    """True if *text* (a session's first message) is a compaction handoff."""
    stripped = text.lstrip()
    try:
        # Reuse the official matcher (keeps the historical-prefix frozen set
        # in sync automatically instead of duplicating it here).
        from agent.context_compressor import ContextCompressor

        return ContextCompressor._starts_with_summary_prefix(stripped)
    except Exception:  # noqa: BLE001 — optional import; fall back to prefixes
        try:
            from agent.context_compressor import (
                LEGACY_SUMMARY_PREFIX,
                SUMMARY_PREFIX,
                _HISTORICAL_SUMMARY_PREFIXES,
            )
        except Exception:  # noqa: BLE001
            LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"
            _HISTORICAL_SUMMARY_PREFIXES = ()
            SUMMARY_PREFIX = (
                "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns "
                "were compacted into the summary below."
            )
        if stripped.startswith(SUMMARY_PREFIX) or stripped.startswith(LEGACY_SUMMARY_PREFIX):
            return True
        return any(stripped.startswith(p) for p in _HISTORICAL_SUMMARY_PREFIXES)


def _starts_with_compaction_summary(db, session_id: str) -> bool:
    """True if the session's first message is a context-compaction handoff.

    The authoritative signal for "this root is really the continuation of an
    older conversation": old Hermes builds wrote compression continuations
    as independent roots whose first persisted message is the compaction
    handoff (``SUMMARY_PREFIX`` / ``LEGACY_SUMMARY_PREFIX`` / any
    ``_HISTORICAL_SUMMARY_PREFIXES`` entry — the frozen set of every wire
    prefix a shipped build persisted, maintained in
    ``agent.context_compressor``).

    This is far more reliable than grouping by same title: a Kanban task
    repeated under one title is NOT a continuation, but a root whose first
    message begins with the handoff prefix IS one, regardless of what title
    it carries.
    """
    content = _first_message_content(db, session_id)
    if not content:
        return False
    return _content_is_compaction_summary(content)


def find_orphaned_chain_candidates(db, *, min_group: int = 2) -> list[dict]:
    """Find root sessions that look like orphaned compression continuations.

    The hard signal is the compaction-handoff first message: an older build
    persisted a compression continuation as a NEW ROOT whose first message
    is the summary handoff (``[CONTEXT COMPACTION — REFERENCE ONLY]...`` /
    ``[CONTEXT SUMMARY]:...`` / any historical prefix), because at that time
    continuations were not linked via ``parent_session_id``.

    We also group roots sharing a normalized title as a secondary, weaker
    signal, but a root whose first message is a compaction handoff is
    reported even when it has no same-titled sibling (the strongest single
    indication it is a continuation of an earlier conversation).

    Children that are explicit delegates/branches/tools are always excluded
    — those are legitimate separate sessions.

    Returns a list of dicts:
    ``{"title": str, "sessions": [{id, started_at, message_count, title}],
      "signal": "handoff"|"same-title"|"both"}`` ordered oldest-first.
    This is *detection only* — relinking is destructive and requires human
    confirmation.
    """
    roots = db._conn.execute(
        f"""
        SELECT s.id, s.title, s.started_at, s.message_count, s.source,
               s.model_config
        FROM sessions s
        WHERE s.parent_session_id IS NULL
          AND COALESCE(s.title_source, '') != 'user'
          AND {_DELEGATE_EXPR}
          AND {_BRANCH_EXPR}
          AND COALESCE(s.source, '') != 'tool'
        ORDER BY s.started_at ASC
        """
    ).fetchall()

    by_key: dict[str, list[dict]] = {}
    handoff_ids: set[str] = set()
    first_contents = _first_message_contents(db, [r["id"] for r in roots])
    for r in roots:
        key = _title_key(r["title"])
        sess = {
            "id": r["id"],
            "title": r["title"],
            "started_at": r["started_at"],
            "message_count": r["message_count"],
        }
        if key is not None:
            by_key.setdefault(key, []).append(sess)
        content = first_contents.get(r["id"])
        if content and _content_is_compaction_summary(content):
            handoff_ids.add(r["id"])

    # Build candidates: handoff-signal roots always; same-title groups
    # (>= min_group) as the secondary signal.
    candidates: list[dict] = []
    seen_ids: set[str] = set()

    for title, sess in by_key.items():
        if len(sess) >= min_group:
            has_handoff = any(s["id"] in handoff_ids for s in sess)
            signal = "both" if has_handoff else "same-title"
            candidates.append(
                {"title": title, "sessions": sess, "signal": signal}
            )
            seen_ids.update(s["id"] for s in sess)

    for sid in sorted(handoff_ids):
        if sid in seen_ids:
            continue
        row = next(r for r in roots if r["id"] == sid)
        candidates.append(
            {
                "title": row["title"] or "(untitled)",
                "sessions": [
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "started_at": row["started_at"],
                        "message_count": row["message_count"],
                    }
                ],
                "signal": "handoff",
            }
        )

    return sorted(
        candidates, key=lambda g: g["sessions"][0]["started_at"]
    )


# ---------------------------------------------------------------------------
# Title repair
# ---------------------------------------------------------------------------


def _first_user_message(db, session_id: str) -> Optional[str]:
    """Return the first user message of a session, if any.

    Ordering matches the official preview subquery (``ORDER BY
    m.timestamp, m.id``) so truncation detection uses the same \"first\"
    message the sidebar previews.
    """
    row = db._conn.execute(
        """
        SELECT content FROM messages
        WHERE session_id = ? AND role = 'user'
          AND content IS NOT NULL AND content != ''
        ORDER BY timestamp, id LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    return row[0] if row else None


def _looks_truncated(title: Optional[str], first_user: Optional[str]) -> bool:
    """A title that is literally the first message truncated (old behavior).

    Old installs titled sessions with ``first_user_message[:~40]``, so the
    sidebar shows a mid-sentence slice. Those are regeneration candidates.
    """
    if not title or not title.strip():
        return True
    if not first_user:
        return False
    t = title.strip()
    fmc = first_user.strip()[:80]
    # Title starts like the message (>=12 chars so a short word is not a
    # false positive), or is a verbatim prefix of it.
    if len(t) >= 12 and (fmc.startswith(t[:25]) or t in fmc[:80]):
        return True
    return False


def _title_is_placeholder(title: Optional[str]) -> bool:
    """Chain segments often carry an empty or whitespace title."""
    return not title or not title.strip()


def _chain_ancestor_title(db, session_id: str, max_depth: int = 10) -> tuple[Optional[str], Optional[str]]:
    """Walk ``parent_session_id`` up the compression chain.

    Returns ``(ancestor_session_id, ancestor_title)`` for the nearest
    ancestor that has a title, or ``(None, None)`` if none exists.
    """
    seen = set()
    sid = session_id
    for _ in range(max_depth):
        if sid in seen or not sid:
            return None, None
        seen.add(sid)
        row = db._conn.execute(
            "SELECT parent_session_id, title FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        if row is None:
            return None, None
        if row[1]:
            return sid, row[1]
        sid = row[0]
    return None, None


def iter_missing_title_candidates(
    db,
    *,
    include_chain_segments: bool = True,
    include_legacy_truncated: bool = True,
    limit: int = 500,
) -> Iterable[dict]:
    """Yield candidate rows needing a title.

    * Chain segments (``parent_session_id`` set) with empty titles → inherit
      the nearest ancestor title (with ``#N`` dedupe).
    * Roots and stray sessions whose title is missing or truncated to the
      first message → regenerate via the LLM generator.
    * Pre-provenance rows (``title_source IS NULL``): official provenance
      treats NULL as ``user`` (a manual /title from that era is
      indistinguishable), so a non-empty legacy title is only repaired at
      user level — the default, because running this command is itself an
      explicit repair (``include_legacy_truncated=False`` opts out).
    """
    rows = db._conn.execute(
        """
        SELECT id, parent_session_id, title, title_source,
               started_at, message_count
        FROM sessions
        ORDER BY COALESCE(started_at, 0) ASC
        """
    ).fetchall()

    for row in rows:
        sid = row["id"]
        parent = row["parent_session_id"]
        title = row["title"]
        source = row["title_source"]

        if source == "user":
            continue

        # Chain segment with empty title → inheritance candidate.
        if parent and _title_is_placeholder(title):
            if include_chain_segments:
                yield {
                    "id": sid,
                    "kind": "inherit",
                    "title": title,
                    "title_source": source,
                }
            continue

        # Empty title (root or stray) → LLM candidate.
        if _title_is_placeholder(title):
            yield {
                "id": sid,
                "kind": "generate",
                "title": title,
                "title_source": source,
            }
            continue

        # Pre-provenance rows (title_source IS NULL from old installs):
        # official provenance treats NULL as ``user``. Refuse to overwrite a
        # non-empty legacy title unless the user explicitly opts in with
        # include_legacy_truncated.
        if source is None:
            if include_legacy_truncated:
                fm = _first_user_message(db, sid)
                if _looks_truncated(title, fm):
                    yield {
                        "id": sid,
                        "kind": "generate",
                        "title": title,
                        "title_source": source,
                        "legacy": True,
                    }
            continue

        # Provenance rows with a known source: derived/llm are repairable,
        # but only when the title itself is broken (truncated).
        if source in ("derived", "llm"):
            fm = _first_user_message(db, sid)
            if _looks_truncated(title, fm):
                yield {
                    "id": sid,
                    "kind": "generate",
                    "title": title,
                    "title_source": source,
                }
            continue


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def repair_chains(
    db,
    *,
    apply_changes: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    confirm: Optional[ConfirmFn] = None,
) -> dict:
    """Detect and repair orphaned compression chains. Returns a stats dict.

    Legacy builds wrote compression continuations as independent roots (no
    ``parent_session_id`` link). This reports groups of roots that look like
    orphaned segments; with ``apply_changes`` it relinks them under the
    oldest root (the head).

    Relinking is only performed for **strong-signal** groups — those where
    at least one root's first message is a compaction handoff (signal
    ``both``). Same-title-only groups (signal ``same-title``) are reported
    but never auto-relinked: identical titles also arise from legitimate
    repeated tasks (e.g. kanban subtasks), so without a handoff there is no
    evidence the roots are one conversation. Delegate/branch/tool children
    are always excluded.

    With ``apply_changes``, groups are confirmed via an interactive
    checklist (``confirm``) and a timestamped state.db snapshot is taken
    before any write. Strong-signal groups are pre-checked; weak-signal
    groups (``same-title`` / ``handoff``) are listed unchecked with a
    warning that the title match alone is not proof they are one
    conversation — the user may still check them to force the relink.
    """
    stats = {
        "orphaned_chain_groups": 0,
        "relinked": 0,
        "skipped": 0,
        "backup_path": None,
    }
    log = progress or (lambda msg: None)

    orphan_groups = find_orphaned_chain_candidates(db)
    stats["orphaned_chain_groups"] = len(orphan_groups)

    # Interactive confirmation before any write (only with --apply).
    chosen: Optional[set[int]] = None
    if apply_changes and confirm is not None and orphan_groups:
        strong = {
            i for i, g in enumerate(orphan_groups) if g["signal"] == "both"
        }
        items = []
        for i, g in enumerate(orphan_groups):
            if g["signal"] == "both":
                note = "compaction-handoff evidence"
            elif g["signal"] == "same-title":
                note = "⚠ title match only — may not be same conversation"
            else:  # "handoff"
                note = "⚠ one-sided handoff — may not be same conversation"
            items.append(
                f"{note}  {g['title'][:40]!r}  "
                f"({len(g['sessions'])} roots: "
                + ", ".join(s["id"][:10] for s in g["sessions"]) + ")"
            )
        # Pre-check only strong-signal groups; weak-signal groups are listed
        # unchecked but the user may toggle them on to force the relink.
        chosen = confirm(
            items,
            "Select orphan-chain groups to relink (SPACE toggle, ENTER confirm)",
            selected=strong,
        )
        if not chosen:
            # None (cancelled/EOF) or an empty set (ESC / nothing checked):
            # do NOTHING — the checked set is the exact list to process.
            log("Cancelled — nothing written.")
            return stats

    # Snapshot + transaction only if at least one group will actually be
    # relinked (a strong-signal group by default, or a weak-signal group the
    # user explicitly checked).
    def _will_relink(i: int) -> bool:
        if not apply_changes:
            return False
        if chosen is not None and i not in chosen:
            return False
        g = orphan_groups[i]
        if len(g["sessions"]) < 2:
            # No sibling to relink — never counts as a write.
            return False
        if g["signal"] != "both" and chosen is None:
            # No interactive confirm: weak-signal groups are never auto-
            # relinked (safe default).
            return False
        return True

    if any(_will_relink(i) for i in range(len(orphan_groups))):
        # Snapshot before any write.
        try:
            backup_path = _backup_before_mutation(db, "repair-chains")
            stats["backup_path"] = str(backup_path)
            log(f"✓ backup: {backup_path}")
        except Exception as exc:  # noqa: BLE001 — backup refusal is a HARD STOP
            log(f"✗ automatic backup failed: {exc}")
            raise RuntimeError(
                f"automatic backup failed ({exc}); refusing to relink without a backup"
            ) from exc
        # Batch the relinks in one transaction (SessionDB is autocommit;
        # without BEGIN each UPDATE would be durable independently).
        db._conn.execute("BEGIN IMMEDIATE")
    try:
        for i, g in enumerate(orphan_groups):
            ids = ", ".join(s["id"][:10] for s in g["sessions"])
            log(
                f"⚠ {len(g['sessions'])} roots share title {g['title']!r} "
                f"(orphaned compression segments?): {ids}"
            )
            if not apply_changes:
                continue
            # A single-root group has no sibling to relink — marking the root
            # compression-ended without a child would only corrupt its
            # end_reason semantics. Skip it entirely.
            if len(g["sessions"]) < 2:
                stats["skipped"] += 1
                continue
            if chosen is not None and i not in chosen:
                continue
            if g["signal"] != "both":
                # Weak-signal group: no hard evidence these are one
                # conversation. Skipped unless the user explicitly checked it
                # in the interactive confirm (forced relink).
                if chosen is None or i not in chosen:
                    stats["skipped"] += 1
                    continue
                log(
                    f"  ↳ user forced relink of weak-signal group "
                    f"(title match only — may not be the same conversation)"
                )
            head_id = g["sessions"][0]["id"]
            # A relinked child is only surfaced by the official list/chain
            # readers when the parent carries end_reason='compression' (the
            # compression-lineage edge). Without it the child becomes a
            # non-root, non-branch, non-reset row — invisible in the sidebar.
            # Mark the parent so the chain renders like a normal compression
            # continuation (one logical conversation, tip-projected).
            db._conn.execute(
                "UPDATE sessions SET end_reason='compression' WHERE id=?",
                (head_id,),
            )
            for s in g["sessions"][1:]:
                db._conn.execute(
                    "UPDATE sessions SET parent_session_id=? WHERE id=?",
                    (head_id, s["id"]),
                )
                stats["relinked"] += 1
        if any(_will_relink(i) for i in range(len(orphan_groups))):
            db._conn.commit()
    except Exception:
        if apply_changes:
            db._conn.execute("ROLLBACK")
        raise
    return stats


def retitle_missing(
    db,
    *,
    generate: Callable[[str], Optional[str]],
    apply_changes: bool = False,
    include_chain_segments: bool = True,
    include_legacy_truncated: bool = True,
    limit: int = 500,
    progress: Optional[Callable[[str], None]] = None,
    confirm: Optional[ConfirmFn] = None,
) -> dict:
    """Regenerate missing/truncated session titles. Returns a stats dict.

    ``generate`` receives a first-user-message string and returns a title or
    None. The live command wires ``agent.title_generator.generate_title``
    here; tests inject a stub. Roots get LLM-generated titles; empty chain
    segments inherit the nearest ancestor title (deduped with #N via
    ``get_next_title_in_lineage``). Never overwrites a user-titled row.

    Dry run (``apply_changes=False``) is side-effect free: it never calls
    ``generate``, and candidates that would need LLM generation are counted
    under ``would_generate`` in the stats dict. This keeps a preview from
    spending tokens or hitting the configured model.

    With ``apply_changes``, candidates are presented as an interactive
    checklist (``confirm``) before any write; only checked rows are
    processed, and a timestamped state.db snapshot is taken first.
    """
    stats = {
        "scanned": 0,
        "generated": 0,
        "would_generate": 0,
        "inherited": 0,
        "skipped_untouchable": 0,
        "failed": 0,
        "up_to_date": 0,
        "backup_path": None,
    }
    log = progress or (lambda msg: None)

    # --- Title repair ---
    candidates = list(iter_missing_title_candidates(
        db,
        include_chain_segments=include_chain_segments,
        include_legacy_truncated=include_legacy_truncated,
    ))
    if limit:
        candidates = candidates[:limit]
    stats["scanned"] = len(candidates)

    if not candidates:
        return stats

    # Interactive confirmation before any write (only with --apply).
    chosen: Optional[set[int]] = None
    if apply_changes and confirm is not None:
        items = [
            f"{c['id'][:8]}  {c.get('title') or '(no title)'}  ({c['kind']})"
            for c in candidates
        ]
        chosen = confirm(items, "Select sessions to re-title (SPACE toggle, ENTER confirm)")
        if not chosen:
            # None (cancelled/EOF) or an empty set (ESC / nothing checked):
            # do NOTHING — no backup, no writes.
            log("Cancelled — nothing written.")
            return stats

    # Snapshot before any write.
    if apply_changes:
        try:
            backup_path = _backup_before_mutation(db, "retitle-missing")
            stats["backup_path"] = str(backup_path)
            log(f"✓ backup: {backup_path}")
        except Exception as exc:  # noqa: BLE001 — backup refusal is a HARD STOP
            log(f"✗ automatic backup failed: {exc}")
            raise RuntimeError(
                f"automatic backup failed ({exc}); refusing to re-title without a backup"
            ) from exc

    db._conn.commit()  # commit any pending state before the loop
    try:
        for i, cand in enumerate(candidates, 1):
            if chosen is not None and (i - 1) not in chosen:
                continue
            sid = cand["id"]
            kind = cand["kind"]

            if kind == "inherit":
                anc_id, anc_title = _chain_ancestor_title(db, sid)
                if not anc_title:
                    stats["skipped_untouchable"] += 1
                    continue
                try:
                    deduped = db.get_next_title_in_lineage(anc_title)
                    log(f"[{i}/{len(candidates)}] {sid[:8]} ← inherit {anc_title!r} → {deduped!r}")
                    if apply_changes:
                        ok = db.set_auto_title(sid, deduped, source="derived")
                        if ok:
                            stats["inherited"] += 1
                        else:
                            stats["skipped_untouchable"] += 1
                    else:
                        stats["inherited"] += 1
                except Exception as e:  # noqa: BLE001 — repair should never crash
                    log(f"  ✗ {sid[:8]} inherit failed: {e}")
                    stats["failed"] += 1
                continue

            # kind == "generate"
            fm = _first_user_message(db, sid)
            if not fm:
                stats["skipped_untouchable"] += 1
                continue
            if not apply_changes:
                # Dry run: do NOT call the LLM — a preview must not spend
                # tokens or hit the configured model. Count the candidate so
                # the user knows what --apply would generate.
                log(f"[{i}/{len(candidates)}] {sid[:8]} would generate a new title (LLM; dry run — not called)")
                stats["would_generate"] += 1
                continue
            log(f"[{i}/{len(candidates)}] {sid[:8]} generating…")
            try:
                new_title = generate(fm)
            except Exception as e:  # noqa: BLE001
                log(f"  ✗ {sid[:8]} generation error: {e}")
                stats["failed"] += 1
                continue
            if not new_title:
                stats["failed"] += 1
                continue
            if new_title == cand["title"]:
                stats["up_to_date"] += 1
                continue
            log(f"  {cand['title']!r} → {new_title!r}")
            if apply_changes:
                try:
                    # A literal empty-string title (NOT NULL) is a quirk the
                    # official set_auto_title refuses to clobber ('' counts as
                    # an existing title), so write at user level — an explicit
                    # repair, not an auto-titler overwrite. title=NULL rows
                    # are fine through set_auto_title (it fills them).
                    if cand.get("legacy") or cand.get("title") == "":
                        ok = db.set_session_title(sid, new_title)
                    else:
                        ok = db.set_auto_title(sid, new_title, source="llm")
                    if ok:
                        stats["generated"] += 1
                    else:
                        stats["skipped_untouchable"] += 1
                except Exception as e:  # noqa: BLE001
                    log(f"  ✗ {sid[:8]} write failed: {e}")
                    stats["failed"] += 1
            else:
                stats["generated"] += 1

    except Exception:
        try:
            db._conn.execute("ROLLBACK")
        except Exception:  # noqa: BLE001 — no active transaction in autocommit
            pass
        raise

    return stats


# ---------------------------------------------------------------------------
# fork compression-chain flattening (hermes sessions merge-chains)
# ---------------------------------------------------------------------------

# Children that are legitimate continuations of a compression-ended parent.
# Mirrors get_compression_tip's exclusion set (branch/delegate/tool).
_MERGE_CHILD_SQL = (
    "json_extract(COALESCE(child.model_config, '{}'), '$._branched_from') IS NULL"
    " AND json_extract(COALESCE(child.model_config, '{}'), '$._delegate_from') IS NULL"
    " AND COALESCE(child.source, '') != 'tool'"
)


def _is_compression_child(db, session_id: str) -> bool:
    """True if this session is a child of a compression-ended parent
    (the authoritative fork-chain edge, mirroring get_compression_tip)."""
    row = db._conn.execute(
        f"""
        SELECT 1 FROM sessions parent
        JOIN sessions child ON child.parent_session_id = parent.id
        WHERE child.id = ?
          AND parent.end_reason = 'compression'
          AND {_MERGE_CHILD_SQL}
        """,
        (session_id,),
    ).fetchone()
    return row is not None


def _walk_compression_chain(db, head: str) -> list[str]:
    """Walk the compression-continuation chain forward from ``head``.

    Returns ``[head, seg2, seg3, ...]`` — every session in the fork chain
    (only compression edges; branch/delegate/tool children are skipped).
    """
    chain = [head]
    seen = {head}
    current = head
    for _ in range(100):  # defensive bound — pathological chains shouldn't happen
        row = db._conn.execute(
            f"""
            SELECT child.id FROM sessions parent
            JOIN sessions child ON child.parent_session_id = parent.id
            WHERE parent.id = ?
              AND parent.end_reason = 'compression'
              AND {_MERGE_CHILD_SQL}
            ORDER BY
              CASE WHEN child.end_reason = 'compression' THEN 0
                   WHEN child.ended_at IS NULL THEN 1
                   ELSE 2 END,
              {_sql_session_last_active('child')} DESC,
              child.started_at DESC, child.id DESC
            LIMIT 1
            """,
            (current,),
        ).fetchone()
        if not row:
            break
        child_id = row["id"]
        if child_id in seen:
            break
        seen.add(child_id)
        chain.append(child_id)
        current = child_id
    return chain


def find_merge_chain_candidates(db) -> list[dict]:
    """Find fork compression chains (compression-ended parent + linked child).

    Returns a list of dicts (oldest head first):
    ``{"head": str, "segments": [str, ...], "message_count": int,
      "head_title": str|None}`` for chains with at least one segment.
    This is *detection only* — flattening is destructive and requires
    ``--apply`` plus the automatic pre-write backup.
    """
    rows = db._conn.execute(
        f"""
        SELECT DISTINCT parent.id AS head
        FROM sessions parent
        JOIN sessions child ON child.parent_session_id = parent.id
        WHERE parent.end_reason = 'compression'
          AND {_MERGE_CHILD_SQL}
        ORDER BY parent.started_at ASC
        """
    ).fetchall()

    candidates = []
    seen_heads = set()
    for r in rows:
        head = r["head"]
        if head in seen_heads:
            continue
        # A chain head must not itself be a compression child (else it is a
        # middle segment of a longer chain — that longer chain's head owns it).
        if _is_compression_child(db, head):
            seen_heads.add(head)
            continue
        seen_heads.add(head)
        chain = _walk_compression_chain(db, head)
        if len(chain) <= 1:
            continue
        msg_count = 0
        for sid in chain:
            row = db._conn.execute(
                "SELECT COUNT(*) c FROM messages WHERE session_id = ?", (sid,)
            ).fetchone()
            msg_count += row["c"]
        title_row = db._conn.execute(
            "SELECT title FROM sessions WHERE id = ?", (head,)
        ).fetchone()
        candidates.append(
            {
                "head": head,
                "segments": chain[1:],
                "message_count": msg_count,
                "head_title": title_row["title"] if title_row else None,
            }
        )
    return candidates


def _merge_chain_stats(db, chain: list[str]) -> dict:
    """Aggregate token/cost counters for a chain's segments (excludes head)."""
    placeholders = ",".join("?" * len(chain[1:]))
    row = db._conn.execute(
        f"""
        SELECT
          COALESCE(SUM(input_tokens),0) it, COALESCE(SUM(output_tokens),0) ot,
          COALESCE(SUM(cache_read_tokens),0) cr, COALESCE(SUM(cache_write_tokens),0) cw,
          COALESCE(SUM(reasoning_tokens),0) rt,
          COALESCE(SUM(estimated_cost_usd),0) ec, COALESCE(SUM(actual_cost_usd),0) ac,
          COALESCE(SUM(tool_call_count),0) tc
        FROM sessions WHERE id IN ({placeholders})
        """,
        tuple(chain[1:]),
    ).fetchone()
    return {
        "input_tokens": row["it"],
        "output_tokens": row["ot"],
        "cache_read_tokens": row["cr"],
        "cache_write_tokens": row["cw"],
        "reasoning_tokens": row["rt"],
        "estimated_cost_usd": row["ec"],
        "actual_cost_usd": row["ac"],
        "tool_call_count": row["tc"],
    }


def _backup_before_mutation(db, label: str) -> Optional[Path]:
    """Timestamped full state.db snapshot before any destructive command.

    Reuses the official ``hermes_cli.backup.copy_db_and_verify`` (SQLite
    backup API — WAL-safe against a live connection — plus integrity
    verification of the destination). All ``--apply`` paths take a snapshot
    first so a mistake is always recoverable. Returns the snapshot path, or
    None when the DB path is missing/not writable (callers then decide
    whether to abort).
    """
    import datetime

    from hermes_cli.backup import copy_db_and_verify

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = db.db_path.with_name(f"{db.db_path.name}.pre-{label}-{stamp}")
    if not copy_db_and_verify(db.db_path, dest):
        raise RuntimeError(f"snapshot verification failed: {dest}")
    return dest


def _describe_title_model() -> str:
    """Human-readable description of the model used for title generation.

    Mirrors the official ``title_generation`` auxiliary routing: an explicit
    per-task config wins; ``provider: auto`` falls back to the user's main
    model. Used to confirm with the user (before any LLM spend) which model
    will generate titles — and whether that is a local or cloud endpoint.
    """
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly()
    aux = (config.get("auxiliary") or {}).get("title_generation") or {}
    model_cfg = config.get("model") or {}

    provider = str(aux.get("provider") or "").strip() or "auto"
    model = str(aux.get("model") or "").strip()
    base_url = str(aux.get("base_url") or "").strip()
    api_key = str(aux.get("api_key") or "").strip()
    enabled = aux.get("enabled", True)
    prefer_fast = bool(aux.get("prefer_fast_model", False))
    timeout = aux.get("timeout", 30)
    reasoning_effort = str(aux.get("reasoning_effort") or "").strip()

    lines = ["Title-generation model (auxiliary.title_generation):"]
    lines.append(f"  enabled        : {enabled}")
    lines.append(f"  provider       : {provider}")

    if provider == "auto":
        main_provider = str(model_cfg.get("provider") or "").strip() or "(unset)"
        main_model = str(model_cfg.get("model") or "").strip() or "(unset)"
        lines.append(f"  ↳ auto → main   : {main_provider} / {main_model}")
        if model:
            lines.append(f"  model          : {model} (explicit override)")
    else:
        lines.append(f"  model          : {model or '(provider default)'}")

    if base_url:
        lines.append(f"  base_url       : {base_url}")
    if api_key:
        lines.append(f"  api_key        : ***configured***")
    lines.append(f"  prefer_fast    : {prefer_fast}")
    if reasoning_effort:
        lines.append(f"  reasoning      : {reasoning_effort}")
    lines.append(f"  timeout        : {timeout}s")

    # Cost/endpoint classification for the confirmation prompt.
    import urllib.parse

    def _is_local_url(url: str) -> bool:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")

    if api_key or (base_url and not _is_local_url(base_url)):
        endpoint = "CLOUD API — may incur cost"
    else:
        endpoint = "local model — no cost"
    lines.append(f"  endpoint       : {endpoint}")

    return "\n".join(lines)


def _confirm_candidates(
    items: list[str],
    title: str,
    *,
    selected: Optional[set[int]] = None,
) -> Optional[set[int]]:
    """Interactive checklist confirmation for destructive edits.

    Renders ``items`` as a curses multi-select checklist (SPACE to toggle,
    ENTER to confirm, ESC to cancel). By default every row is pre-selected;
    pass ``selected`` to pre-check only those indices (e.g. strong-signal
    candidates) and leave the rest unchecked-but-selectable. Non-TTY
    environments fall back to a numbered toggle prompt; a non-interactive
    EOF returns ``None`` (caller aborts).

    Returns the set of selected indices, or ``None`` when cancelled. ESC and
    an empty selection both mean "do nothing": the caller must treat the
    returned set as the *exact* rows to process — nothing more, nothing less.
    """
    from hermes_cli.curses_ui import curses_checklist

    pre = set(range(len(items))) if selected is None else set(selected)
    try:
        chosen = curses_checklist(
            title,
            items,
            pre,
            # ESC must NOT fall back to the pre-selected rows (the official
            # default) — for destructive commands cancel means "do nothing",
            # otherwise pressing ESC would still process the pre-checks.
            cancel_returns=set(),
        )
    except (EOFError, KeyboardInterrupt):
        return None
    return chosen


def _confirm_snapshot(
    items: list[str],
    title: str,
) -> Optional[int]:
    """Single-select confirmation for choosing one snapshot to restore.

    Thin wrapper over the official ``curses_single_select`` (↑↓ navigate,
    ENTER confirm, ESC cancel; non-TTY falls back to a numbered prompt).
    Returns the selected index, or None on cancel.
    """
    from hermes_cli.curses_ui import curses_single_select

    try:
        return curses_single_select(title, items, default_index=0)
    except (EOFError, KeyboardInterrupt):
        return None


def merge_compression_chains(
    db,
    *,
    apply_changes: bool = False,
    backup: bool = True,
    progress: Optional[Callable[[str], None]] = None,
    confirm: Optional[ConfirmFn] = None,
) -> dict:
    """Flatten fork compression chains into single in-place sessions.

    For each chain (head + compression-linked segments), moves every segment
    message into the head session (message ids unchanged — FTS rowids stay
    valid, no index rebuild), merges ``session_model_usage`` counters,
    redirects orphaned children (reset/branch sessions whose parent was a
    removed segment) to the head, accumulates token/cost counters and gateway
    origin columns onto the head, then deletes the segment rows.

    With ``apply_changes``, chains are confirmed via an interactive
    checklist (``confirm``) before any write; a timestamped state.db
    snapshot is taken first (unless ``backup=False``).

    Returns a stats dict:
    ``{"chains": int, "segments": int, "messages_moved": int,
      "orphans_redirected": int, "usage_merged": int, "backup_path": str|None}``
    """
    log = progress or (lambda msg: None)
    candidates = find_merge_chain_candidates(db)
    stats = {
        "chains": len(candidates),
        "segments": sum(len(c["segments"]) for c in candidates),
        "messages_moved": 0,
        "orphans_redirected": 0,
        "usage_merged": 0,
        "backup_path": None,
        "verified": False,
        "verify_report": None,
    }

    if not candidates:
        return stats

    # Report (always)
    for c in candidates:
        ids = ", ".join(s[:10] for s in c["segments"])
        log(
            f"⚠ chain {c['head'][:10]} «{c['head_title'] or '(untitled)'}»: "
            f"{len(c['segments'])} segment(s), {c['message_count']} messages — {ids}"
        )

    if not apply_changes:
        return stats

    # Interactive confirmation before any write.
    chosen: Optional[set[int]] = None
    if confirm is not None:
        items = [
            f"{c['head'][:10]} «{c['head_title'] or '(untitled)'}» — "
            f"{len(c['segments'])} segment(s), {c['message_count']} messages"
            for c in candidates
        ]
        chosen = confirm(items, "Select chains to merge (SPACE toggle, ENTER confirm)")
        if not chosen:
            # None (cancelled/EOF) or an empty set (ESC / nothing checked):
            # do NOTHING — no backup, no writes.
            log("Cancelled — nothing written.")
            return stats
    else:
        chosen = set(range(len(candidates)))

    # Automatic backup before any write.
    if backup:
        try:
            backup_path = _backup_before_mutation(db, "merge-chains")
            stats["backup_path"] = str(backup_path)
            log(f"✓ backup: {backup_path}")
        except Exception as exc:  # noqa: BLE001 — backup refusal is a HARD STOP
            log(f"✗ automatic backup failed: {exc}")
            raise RuntimeError(
                f"automatic backup failed ({exc}); refusing to merge without a backup"
            ) from exc

    # One transaction for the whole merge: SessionDB connects in autocommit
    # (isolation_level=None), so without an explicit BEGIN every UPDATE/DELETE
    # below would be durable independently and a crash mid-loop would leave a
    # partially merged database. BEGIN IMMEDIATE takes the write lock up front
    # (mirrors the official write path) so the batch is all-or-nothing.
    db._conn.execute("BEGIN IMMEDIATE")
    try:
        for ci, c in enumerate(candidates):
            if ci not in chosen:
                continue
            head = c["head"]
            segs = c["segments"]
            placeholders = ",".join("?" * len(segs))
            segs_tuple = tuple(segs)

            # 1. Move segment messages into the head (ids unchanged).
            cur = db._conn.execute(
                f"UPDATE messages SET session_id=? WHERE session_id IN ({placeholders})",
                (head,) + segs_tuple,
            )
            stats["messages_moved"] += cur.rowcount

            # 2. Merge session_model_usage (UPSERT; prevents ON DELETE CASCADE loss).
            usage_rows = db._conn.execute(
                f"SELECT * FROM session_model_usage WHERE session_id IN ({placeholders})",
                segs_tuple,
            ).fetchall()
            for u in usage_rows:
                db._conn.execute(
                    """
                    INSERT INTO session_model_usage
                      (session_id, model, billing_provider, billing_base_url, billing_mode, task,
                       api_call_count, input_tokens, output_tokens, cache_read_tokens,
                       cache_write_tokens, reasoning_tokens, estimated_cost_usd,
                       actual_cost_usd, cost_status, cost_source, first_seen, last_seen)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(session_id, model, billing_provider, billing_base_url,
                                billing_mode, task)
                    DO UPDATE SET
                      api_call_count = session_model_usage.api_call_count + excluded.api_call_count,
                      input_tokens = session_model_usage.input_tokens + excluded.input_tokens,
                      output_tokens = session_model_usage.output_tokens + excluded.output_tokens,
                      cache_read_tokens = session_model_usage.cache_read_tokens + excluded.cache_read_tokens,
                      cache_write_tokens = session_model_usage.cache_write_tokens + excluded.cache_write_tokens,
                      reasoning_tokens = session_model_usage.reasoning_tokens + excluded.reasoning_tokens,
                      estimated_cost_usd = session_model_usage.estimated_cost_usd + excluded.estimated_cost_usd,
                      actual_cost_usd = session_model_usage.actual_cost_usd + excluded.actual_cost_usd,
                      last_seen = MAX(session_model_usage.last_seen, excluded.last_seen)
                    """,
                    (
                        head, u["model"], u["billing_provider"], u["billing_base_url"],
                        u["billing_mode"], u["task"], u["api_call_count"],
                        u["input_tokens"], u["output_tokens"], u["cache_read_tokens"],
                        u["cache_write_tokens"], u["reasoning_tokens"],
                        u["estimated_cost_usd"], u["actual_cost_usd"], u["cost_status"],
                        u["cost_source"], u["first_seen"], u["last_seen"],
                    ),
                )
                stats["usage_merged"] += 1
            db._conn.execute(
                f"DELETE FROM session_model_usage WHERE session_id IN ({placeholders})",
                segs_tuple,
            )

            # 3. Redirect orphaned children of removed segments to the head.
            cur = db._conn.execute(
                f"""
                UPDATE sessions SET parent_session_id=?
                WHERE parent_session_id IN ({placeholders})
                  AND id NOT IN ({placeholders})
                  AND id != ?
                """,
                (head,) + segs_tuple + segs_tuple + (head,),
            )
            stats["orphans_redirected"] += cur.rowcount

            # 4. Accumulate counters onto the head; inherit terminal end state.
            agg = _merge_chain_stats(db, [head] + segs)
            total_msgs = db._conn.execute(
                "SELECT COUNT(*) c FROM messages WHERE session_id=? AND active=1",
                (head,),
            ).fetchone()["c"]
            last_row = db._conn.execute(
                "SELECT * FROM sessions WHERE id=?",
                (chain_tip := segs[-1],),
            ).fetchone()
            head_row = db._conn.execute(
                "SELECT * FROM sessions WHERE id=?", (head,)
            ).fetchone()

            # Gateway origin inheritance: fill head's empty routing columns from the tip.
            origin_cols = [
                "user_id", "session_key", "chat_id", "chat_type", "thread_id",
                "display_name", "origin_json", "handoff_platform",
            ]
            origin_updates = [
                f"{col} = ?"
                for col in origin_cols
                if head_row[col] in (None, "") and last_row[col] not in (None, "")
            ]
            if origin_updates:
                origin_params = tuple(
                    last_row[c]
                    for c in origin_cols
                    if head_row[c] in (None, "") and last_row[c] not in (None, "")
                )
                db._conn.execute(
                    f"UPDATE sessions SET {', '.join(origin_updates)} WHERE id=?",
                    origin_params + (head,),
                )

            db._conn.execute(
                """
                UPDATE sessions SET
                  message_count=?,
                  input_tokens = COALESCE(input_tokens,0) + ?,
                  output_tokens = COALESCE(output_tokens,0) + ?,
                  cache_read_tokens = COALESCE(cache_read_tokens,0) + ?,
                  cache_write_tokens = COALESCE(cache_write_tokens,0) + ?,
                  reasoning_tokens = COALESCE(reasoning_tokens,0) + ?,
                  estimated_cost_usd = COALESCE(estimated_cost_usd,0) + ?,
                  actual_cost_usd = COALESCE(actual_cost_usd,0) + ?,
                  tool_call_count = COALESCE(tool_call_count,0) + ?,
                  ended_at = ?,
                  end_reason = ?
                WHERE id=?
                """,
                (
                    total_msgs, agg["input_tokens"], agg["output_tokens"],
                    agg["cache_read_tokens"], agg["cache_write_tokens"],
                    agg["reasoning_tokens"], agg["estimated_cost_usd"],
                    agg["actual_cost_usd"], agg["tool_call_count"],
                    last_row["ended_at"], last_row["end_reason"], head,
                ),
            )

            # 5. Delete segment rows (usage already merged — no cascade loss).
            db._conn.execute(
                f"DELETE FROM sessions WHERE id IN ({placeholders})", segs_tuple
            )
            log(
                f"  ✓ {head[:10]} ← merged {len(segs)} segment(s), "
                f"moved {stats['messages_moved']} messages so far"
            )
        db._conn.commit()
    except Exception:
        db._conn.execute("ROLLBACK")
        raise

    # --- Post-write file-level verification ---
    # Compare message/session/usage counts before vs after for the chains
    # actually merged. The only legitimate delta is the number of deleted
    # segment rows and (because a live gateway may append messages
    # mid-merge) any concurrent writes.
    merged_candidates = [
        c for i, c in enumerate(candidates) if i in chosen
    ]
    before_msgs = sum(c["message_count"] for c in merged_candidates)
    after_msgs = 0
    for c in merged_candidates:
        row = db._conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE session_id = ?",
            (c["head"],),
        ).fetchone()
        after_msgs += row["c"]
    orphan_rows = db._conn.execute(
        "SELECT COUNT(*) c FROM session_model_usage "
        "WHERE session_id NOT IN (SELECT id FROM sessions)"
    ).fetchone()["c"]
    verify = {
        "messages_before": before_msgs,
        "messages_after": after_msgs,
        "delta": after_msgs - before_msgs,
        "segments_deleted": sum(len(c["segments"]) for c in merged_candidates),
        "usage_orphans": orphan_rows,
    }
    # Messages are only re-homed (session_id changed), never copied or
    # deleted, so the total count must not shrink. A live gateway may append
    # messages mid-merge, so after >= before is the invariant. usage orphans
    # must be 0 (session_model_usage is ON DELETE CASCADE).
    stats["verified"] = verify["delta"] >= 0 and verify["usage_orphans"] == 0
    stats["verify_report"] = verify
    log(
        f"✓ verify: messages {verify['messages_before']} → "
        f"{verify['messages_after']} (Δ{verify['delta']:+d}, "
        f"deleted {verify['segments_deleted']} segment rows), "
        f"usage orphans={verify['usage_orphans']} "
        f"{'✅ OK' if stats['verified'] else '❌ MISMATCH'}"
    )
    return stats


# ---------------------------------------------------------------------------
# state.db restore (hermes sessions restore-db)
# ---------------------------------------------------------------------------


def _db_holder_matches(path: str, db_path: Path) -> bool:
    """True when an open file path refers to *db_path* or its WAL/SHM files."""
    try:
        p = Path(path)
    except (TypeError, ValueError):
        return False
    return (
        p == db_path
        or p == db_path.with_name(db_path.name + "-wal")
        or p == db_path.with_name(db_path.name + "-shm")
    )


def _find_state_db_holders(db_path: Path) -> list[int]:
    """PIDs of processes with *db_path* (or its WAL/SHM) open.

    Uses psutil's ``open_files()`` — same library the gateway and dashboard
    process management already depend on. The caller's own PID is excluded.
    """
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return []
    me = os.getpid()
    holders: list[int] = []
    for proc in psutil.process_iter(["pid"]):
        try:
            pid = proc.info["pid"]
            if pid == me:
                continue
            try:
                files = proc.open_files()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
            if any(_db_holder_matches(f.path, db_path) for f in files):
                holders.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(holders)


def _kill_processes(holders: list[int], log: Callable[[str], None]) -> tuple[list[int], list[tuple[int, str]]]:
    """Stop processes holding state.db (SIGTERM → 3s grace → SIGKILL).

    Mirrors the official dashboard-process teardown in
    ``hermes_cli.dashboard_procs``: a clean SIGTERM first so the process can
    flush, then SIGKILL survivors. Returns ``(killed, failed)``.
    """
    if not holders:
        return [], []
    killed: list[int] = []
    failed: list[tuple[int, str]] = []
    if sys.platform == "win32":
        for pid in holders:
            try:
                import subprocess

                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=10,
                )
                if result.returncode == 0:
                    killed.append(pid)
                else:
                    failed.append((pid, (result.stderr or result.stdout or "").strip()))
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
                failed.append((pid, str(e)))
        return killed, failed

    import signal as _signal

    import psutil

    for pid in holders:
        try:
            os.kill(pid, _signal.SIGTERM)
        except ProcessLookupError:
            killed.append(pid)
        except (PermissionError, OSError) as e:
            failed.append((pid, str(e)))

    deadline = time.monotonic() + 3.0
    pending = [p for p in holders if p not in killed and p not in {f[0] for f in failed}]
    while pending and time.monotonic() < deadline:
        time.sleep(0.1)
        still: list[int] = []
        for pid in pending:
            # psutil.pid_exists is the official recommendation: os.kill(pid, 0)
            # is NOT a no-op on Windows (CTRL_C_EVENT broadcast). psutil is a
            # core dependency (we already import it in _find_state_db_holders).
            if psutil.pid_exists(pid):
                still.append(pid)
            else:
                killed.append(pid)
        pending = still

    for pid in pending:
        try:
            os.kill(pid, _signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            killed.append(pid)
        except (PermissionError, OSError) as e:
            failed.append((pid, str(e)))
    return killed, failed


def _list_snapshot_candidates(db_path: Path) -> list[Path]:
    """Timestamped ``.pre-<label>-<stamp>`` snapshots next to *db_path*.

    Newest first, ordered by the ``YYYYMMDD_HHMMSS`` stamp parsed from the
    filename — NOT the raw basename, whose lexicographic order is dominated
    by the label (``repair-chains`` vs ``merge-chains``).
    """

    def _stamp(p: Path) -> str:
        name = p.name[len(f"{db_path.name}.pre-"):]
        parts = name.rsplit("-", 1)
        return parts[-1] if len(parts) == 2 else ""

    return sorted(
        (p for p in db_path.parent.glob(f"{db_path.name}.pre-*")
         if p.is_file() and p.name.startswith(f"{db_path.name}.pre-")),
        key=_stamp,
        reverse=True,
    )


def restore_state_db(
    db_path,
    snapshot: Optional[str] = None,
    *,
    force: bool = False,
    dry_run: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    confirm: Optional[Callable[[list[str], str], Optional[int]]] = None,
) -> dict:
    """Restore ``state.db`` from a ``.pre-*`` snapshot.

    Because every live Hermes process (gateway, dashboard backend, TUI, CLI
    sessions) holds ``state.db`` open — with WAL/SHM state and in-memory
    caches — restoring over a live DB corrupts it (stale WAL frames, cached
    writes re-persisted after the swap). This command therefore:

    1. Finds every process with the DB (or its WAL/SHM) open.
    2. Stops them (SIGTERM → 3s grace → SIGKILL), unless ``--dry-run``.
    3. Clears any leftover ``-wal``/``-shm`` files (they belong to the old DB).
    4. Copies the snapshot over ``state.db``.
    5. Verifies the restored file (``verify_sqlite_integrity``).

    ``snapshot`` names a ``.pre-*`` file (absolute path, or basename resolved
    next to the DB). When omitted and more than one snapshot exists, the
    ``confirm`` callback (``curses_single_select``) lets the user pick one —
    it returns the selected index, or None to cancel. With one snapshot it is
    chosen directly.

    The chosen snapshot is **always** integrity-verified before anything is
    written — a corrupted snapshot must never replace a live DB. A failed
    verification is a hard stop. The restored file is verified again after
    the swap.

    Stopped processes are **not** auto-restarted (the restored DB may need
    inspection first, and a freshly-spawned gateway would start writing to
    it immediately). Their exact argv is captured before the kill and the
    restart commands are printed, mirroring the official ``--stop``
    behaviour.

    With ``--dry-run`` nothing is killed or written.

    Returns a stats dict: ``{"snapshot": str, "holders": [...], "killed":
    [...], "failed": [...], "restored": bool, "verified": bool}``.
    """
    log = progress or (lambda msg: None)
    dbp = Path(db_path)

    # Resolve the snapshot.
    if snapshot:
        snap = Path(snapshot)
        if not snap.is_absolute():
            snap = dbp.parent / snap
        # Security: the snapshot must live next to the DB — restoring an
        # arbitrary file (e.g. another profile's state.db) could silently
        # replace this DB with the wrong data. Mirror the official snapshot
        # restore's traversal guard.
        db_dir = dbp.parent.resolve()
        try:
            snap.resolve().relative_to(db_dir)
        except ValueError:
            raise RuntimeError(
                f"refusing to restore: snapshot {snap} is outside {dbp.parent} "
                "(snapshots must live next to the database)"
            ) from None
        if not snap.is_file():
            raise RuntimeError(f"snapshot not found: {snap}")
    else:
        snaps = _list_snapshot_candidates(dbp)
        if not snaps:
            raise RuntimeError(
                f"no state.db snapshots found next to {dbp}; pass --snapshot"
            )
        if len(snaps) == 1:
            snap = snaps[0]
            log(f"ℹ snapshot: {snap.name}")
        elif confirm is not None:
            items = [f"{s.name} ({s.stat().st_size / 1024 / 1024:.1f} MB)" for s in snaps]
            idx = confirm(items, "Select a state.db snapshot to restore (↑↓ ENTER, ESC cancel)")
            if idx is None:
                log("Cancelled — nothing restored.")
                return {
                    "snapshot": None, "holders": [], "killed": [],
                    "failed": [], "restored": False, "verified": False,
                }
            snap = snaps[idx]
            log(f"ℹ selected snapshot: {snap.name}")
        else:
            snap = snaps[0]
            log(f"ℹ newest snapshot: {snap.name}")

    # The snapshot must be valid BEFORE anything is written — restoring a
    # corrupted snapshot over a live DB is irreversible damage. This check is
    # never skippable (no --no-verify flag).
    from hermes_cli.backup import verify_sqlite_integrity

    snap_integrity = verify_sqlite_integrity(snap, run_pragma=True)
    if not snap_integrity.get("valid"):
        raise RuntimeError(
            f"refusing to restore: snapshot {snap} failed integrity "
            f"verification ({snap_integrity.get('message') or 'invalid'})"
        )
    log(f"✓ snapshot integrity: {snap.name} OK")

    holders = _find_state_db_holders(dbp)
    stats: dict = {
        "snapshot": str(snap),
        "holders": holders,
        "killed": [],
        "failed": [],
        "restored": False,
        "verified": False,
    }
    holder_cmds: dict[int, list[str]] = {}

    log(f"⚠ {len(holders)} process(es) holding {dbp.name}: {', '.join(map(str, holders)) or 'none'}")
    if dry_run:
        log("dry run — no processes stopped, no files written.")
        return stats

    if holders and not force:
        raise RuntimeError(
            "refusing to restore over live processes; re-run with --force to "
            "stop them, or stop them manually and re-run"
        )

    if holders:
        # Capture each holder's argv BEFORE killing it so we can print the
        # exact restart command afterwards (official --stop behaviour: never
        # auto-restart a process that was holding a DB we just swapped, but
        # do tell the user how to bring services back).
        if sys.platform != "win32":
            try:
                from hermes_cli.main import _dashboard_cmdline_for_pid
            except Exception:  # noqa: BLE001 — best-effort only
                _dashboard_cmdline_for_pid = None
            if _dashboard_cmdline_for_pid is not None:
                for pid in holders:
                    argv = _dashboard_cmdline_for_pid(pid)
                    if argv:
                        holder_cmds[pid] = argv

        killed, failed = _kill_processes(holders, log)
        stats["killed"] = killed
        stats["failed"] = failed
        if failed and not force:
            raise RuntimeError(
                f"failed to stop {len(failed)} process(es): {failed}"
            )
        if killed:
            log(f"✓ stopped {len(killed)} process(es): {', '.join(map(str, killed))}")
        # Give the OS a beat to release file handles.
        time.sleep(0.3)

    # Remove stale WAL/SHM belonging to the pre-restore DB.
    for suffix in ("-wal", "-shm"):
        side = dbp.with_name(dbp.name + suffix)
        if side.exists():
            side.unlink()
            log(f"✓ removed stale {side.name}")

    # Swap the snapshot in.
    import shutil

    tmp = dbp.with_name(f".{dbp.name}.restore_tmp")
    shutil.copy2(snap, tmp)
    dbp.unlink(missing_ok=True)
    shutil.move(str(tmp), str(dbp))
    stats["restored"] = True
    log(f"✓ restored {dbp.name} ← {snap.name}")

    # Verify the restored file — also never skippable.
    integrity = verify_sqlite_integrity(dbp, run_pragma=True)
    stats["verified"] = bool(integrity.get("valid"))
    log(
        f"✓ verify: {dbp.name} {'✅ OK' if stats['verified'] else '❌ INVALID'} "
        f"({integrity.get('message') or 'integrity check passed'})"
    )
    if not stats["verified"]:
        raise RuntimeError(f"restored {dbp.name} failed integrity verification")

    # Tell the user how to bring the stopped processes back (official --stop
    # style: print the restart command, never auto-restart — the restored DB
    # may need inspection first, and a freshly-spawned gateway would start
    # writing to it immediately).
    if stats["killed"]:
        log("")
        log("  Restart stopped processes when you're ready (nothing was auto-restarted):")
        seen_cmds: set[str] = set()
        for pid in stats["killed"]:
            argv = holder_cmds.get(pid)
            if argv:
                display = " ".join(argv)
                if display not in seen_cmds:
                    seen_cmds.add(display)
                    log(f"    {display}")
            else:
                log(f"    (PID {pid}: could not recover its command line; restart it manually)")
        if not seen_cmds:
            log("    hermes gateway run   # gateway")
            log("    hermes dashboard --port 0   # desktop backend")
    return stats
