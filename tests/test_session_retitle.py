"""Tests for ``hermes sessions retitle-missing`` (hermes_cli.session_migration).

The command migrates legacy session data to the current storage format.
Key behaviors under test:

* Orphaned-chain detection: multiple roots sharing a normalized title are
  reported as relink candidates; delegate/branch/tool children are excluded.
* Dry-run default: nothing is written without ``apply_changes=True``.
* ``title_source == 'user'`` rows are never touched.
* Old pre-provenance rows (``title_source IS NULL``) with a truncated
  first-message title are regenerated only with the explicit
  ``include_legacy_truncated`` opt-in.
* Empty chain segments inherit the nearest ancestor title, deduped with
  ``#N`` via the official ``get_next_title_in_lineage``.
* LLM failures are skipped without crashing.
"""

import sqlite3

import pytest

from hermes_cli.session_migration import (
    find_merge_chain_candidates,
    find_orphaned_chain_candidates,
    iter_missing_title_candidates,
    merge_compression_chains,
    repair_chains,
    retitle_missing,
    _chain_ancestor_title,
    _looks_truncated,
)
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "test_state.db")
    yield d
    d.close()


def _mk(db, sid, *, parent=None, title=None, source=None, msg=None, model_config=None):
    """Create a session row with an optional first user message."""
    kwargs = {}
    if parent is not None:
        kwargs["parent_session_id"] = parent
    if model_config is not None:
        kwargs["model_config"] = model_config
    db.create_session(sid, source="cli", **kwargs)
    if title is not None or source is not None:
        if title is None:
            title = ""
        db._conn.execute(
            "UPDATE sessions SET title=?, title_source=? WHERE id=?",
            (title, source, sid),
        )
        db._conn.commit()
    if msg:
        db.append_message(sid, "user", content=msg)
    return sid


def _generate_stub(mapping):
    """Return a generate() stub: fm -> title from mapping, else None."""

    def gen(fm):
        return mapping.get(fm)

    return gen


# ---------------------------------------------------------------------------
# Orphaned-chain detection
# ---------------------------------------------------------------------------


class TestOrphanedChainDetection:
    def test_same_title_roots_grouped(self, db):
        _mk(db, "r1", title="Plan review", source="llm", msg="hello")
        _mk(db, "r2", title="Plan review #2", source="llm", msg="hello again")
        _mk(db, "r3", title="Plan review #3", source="llm", msg="hello third")
        groups = find_orphaned_chain_candidates(db)
        assert len(groups) == 1
        assert groups[0]["title"] == "Plan review"
        assert len(groups[0]["sessions"]) == 3

    def test_single_root_not_reported(self, db):
        _mk(db, "r1", title="Plan review", source="llm", msg="hello")
        groups = find_orphaned_chain_candidates(db)
        assert groups == []

    def test_user_titled_roots_excluded(self, db):
        _mk(db, "r1", title="My manual title", source="user", msg="hello")
        _mk(db, "r2", title="My manual title #2", source="user", msg="hello again")
        groups = find_orphaned_chain_candidates(db)
        assert groups == []

    def test_delegate_children_excluded(self, db):
        # A subagent/delegate session with the same title as its parent is
        # NOT an orphan — it is a legitimate separate session.
        _mk(db, "r1", title="SSH host check", source="llm", msg="hello")
        _mk(db, "r2", title="SSH host check #2", source="llm",
            msg="hello again", model_config={"_delegate_from": "parent"})
        groups = find_orphaned_chain_candidates(db)
        assert groups == []

    def test_handoff_first_message_reported_alone(self, db):
        # A root whose FIRST message is a compaction handoff is a
        # continuation of an earlier conversation even with no same-titled
        # sibling — the authoritative signal. LEGACY_SUMMARY_PREFIX is the
        # short historical prefix; the current SUMMARY_PREFIX is a long
        # 1954-char block that also starts with "[CONTEXT COMPACTION...".
        _mk(db, "h1", title="Anything", source="llm",
            msg="[CONTEXT SUMMARY]: prior conversation contents")
        groups = find_orphaned_chain_candidates(db)
        assert len(groups) == 1
        assert groups[0]["signal"] == "handoff"
        assert groups[0]["sessions"][0]["id"] == "h1"

    def test_current_summary_prefix_recognized(self, db):
        # The current (long) prefix is matched too — use the real constant.
        from agent.context_compressor import SUMMARY_PREFIX

        _mk(db, "c1", title="Anything", source="llm",
            msg=f"{SUMMARY_PREFIX}\nsummary body")
        groups = find_orphaned_chain_candidates(db)
        assert len(groups) == 1
        assert groups[0]["signal"] == "handoff"

    def test_normal_root_not_flagged_by_handoff(self, db):
        # A normal conversation whose first message is a user request is
        # not a continuation, even if a sibling shares its title.
        _mk(db, "n1", title="Deploy API", source="llm",
            msg="deploy the new API to staging")
        _mk(db, "n2", title="Deploy API #2", source="llm",
            msg="deploy follow-up")
        groups = find_orphaned_chain_candidates(db)
        # Same-title group exists (secondary signal) but neither has a
        # handoff first message.
        assert len(groups) == 1
        assert groups[0]["signal"] == "same-title"


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


class TestCandidateSelection:
    def test_user_titled_never_candidate(self, db):
        _mk(db, "u1", title="My manual title", source="user", msg="hello world")
        cands = list(iter_missing_title_candidates(db))
        assert cands == []

    def test_empty_title_root_is_candidate(self, db):
        _mk(db, "e1", msg="what is the plan")
        cands = list(iter_missing_title_candidates(db))
        assert [c["id"] for c in cands] == ["e1"]
        assert cands[0]["kind"] == "generate"

    def test_truncated_title_pre_provenance_repaired_by_default(self, db):
        # Old installs: title is the first message cut off, no title_source.
        # Full-authority repair is the default: legacy truncated titles are
        # candidates (--no-legacy-truncated opts out).
        fm = "we need to review the quarterly budget report before the meeting"
        _mk(db, "t1", title=fm[:40], source=None, msg=fm)
        cands = list(iter_missing_title_candidates(db))
        assert [c["id"] for c in cands] == ["t1"]
        assert cands[0]["kind"] == "generate"
        assert cands[0].get("legacy") is True

    def test_truncated_title_pre_provenance_skipped_with_no_legacy(self, db):
        fm = "we need to review the quarterly budget report before the meeting"
        _mk(db, "t1", title=fm[:40], source=None, msg=fm)
        cands = list(iter_missing_title_candidates(
            db, include_legacy_truncated=False
        ))
        assert cands == []

    def test_empty_string_title_is_candidate(self, db):
        # Real-world quirk: some rows carry title='' (empty string, NOT NULL).
        # These are placeholders and must be candidates too.
        _mk(db, "e1", title="", source=None, msg="what is the plan")
        cands = list(iter_missing_title_candidates(db))
        assert [c["id"] for c in cands] == ["e1"]
        assert cands[0]["kind"] == "generate"

    def test_plausible_title_pre_provenance_left_alone(self, db):
        _mk(db, "p1", title="Review Q3 budget", source=None, msg="we need to review the quarterly budget")
        cands = list(iter_missing_title_candidates(db))
        assert cands == []

    def test_llm_title_good_left_alone(self, db):
        _mk(db, "l1", title="Review Q3 budget", source="llm", msg="we need to review the quarterly budget")
        cands = list(iter_missing_title_candidates(db))
        assert cands == []

    def test_chain_segment_empty_is_inherit_candidate(self, db):
        _mk(db, "root", title="Root title", source="llm")
        _mk(db, "seg1", parent="root", msg="continuation")
        cands = list(iter_missing_title_candidates(db))
        assert cands and cands[0]["id"] == "seg1"
        assert cands[0]["kind"] == "inherit"

    def test_no_chain_inherit_flag_excludes_segments(self, db):
        _mk(db, "root", title="Root title", source="llm")
        _mk(db, "seg1", parent="root", msg="continuation")
        cands = list(iter_missing_title_candidates(
            db, include_chain_segments=False
        ))
        assert cands == []


# ---------------------------------------------------------------------------
# Chain inheritance
# ---------------------------------------------------------------------------


class TestChainInheritance:
    def test_ancestor_walk(self, db):
        _mk(db, "root", title="Root title", source="llm")
        _mk(db, "seg1", parent="root")
        _mk(db, "seg2", parent="seg1")
        anc_id, anc_title = _chain_ancestor_title(db, "seg2")
        assert anc_id == "root"
        assert anc_title == "Root title"

    def test_ancestor_none_when_no_title(self, db):
        _mk(db, "root")
        _mk(db, "seg1", parent="root")
        anc_id, anc_title = _chain_ancestor_title(db, "seg1")
        assert anc_id is None
        assert anc_title is None


# ---------------------------------------------------------------------------
# _looks_truncated
# ---------------------------------------------------------------------------


class TestLooksTruncated:
    def test_none_or_empty(self):
        assert _looks_truncated(None, "hello")
        assert _looks_truncated("", "hello")

    def test_first_message_prefix(self):
        fm = "this is a fairly long first user message about something"
        assert _looks_truncated(fm[:30], fm)
        assert _looks_truncated("this is a fairly long", fm)

    def test_proper_title_not_truncated(self):
        fm = "this is a fairly long first user message about something"
        assert not _looks_truncated("Plan review", fm)
        assert not _looks_truncated("Short", fm)

    def test_short_title_not_flagged(self):
        # <12 chars should not be treated as truncation even if it appears
        # at the start of the message (e.g. the whole message is short).
        fm = "hi"
        assert not _looks_truncated("hi", fm)


# ---------------------------------------------------------------------------
# Runner (dry-run / apply / safety)
# ---------------------------------------------------------------------------


class TestRunner:
    def test_dry_run_writes_nothing_and_skips_llm(self, db):
        _mk(db, "e1", msg="what is the plan")
        _mk(db, "root", title="Root title", source="llm")
        _mk(db, "seg1", parent="root", msg="continuation")
        stub = _generate_stub({"what is the plan": "Plan discussion"})
        calls = []
        counting_stub = lambda fm: (calls.append(fm), stub(fm))[1]

        stats = retitle_missing(db, generate=counting_stub, apply_changes=False)

        row = db.get_session("e1")
        assert row["title"] is None
        # Dry run must not call the LLM at all — zero token spend.
        assert calls == []
        assert stats["generated"] == 0
        assert stats["would_generate"] == 1
        assert stats["inherited"] == 1

    def test_apply_writes_and_marks_provenance(self, db):
        _mk(db, "e1", msg="what is the plan")
        stub = _generate_stub({"what is the plan": "Plan discussion"})

        stats = retitle_missing(db, generate=stub, apply_changes=True)

        assert stats["generated"] == 1
        row = db.get_session("e1")
        assert row["title"] == "Plan discussion"
        assert row["title_source"] == "llm"

    def test_apply_inherits_chain_with_dedupe(self, db):
        _mk(db, "root", title="Root title", source="llm")
        _mk(db, "seg1", parent="root", msg="first continuation")
        _mk(db, "seg2", parent="seg1", msg="second continuation")
        stub = _generate_stub({})

        stats = retitle_missing(db, generate=stub, apply_changes=True)

        assert stats["inherited"] == 2
        seg1 = db.get_session("seg1")
        seg2 = db.get_session("seg2")
        assert seg1["title"] == "Root title #2"
        assert seg2["title"] == "Root title #3"
        assert seg1["title_source"] == "derived"

    def test_user_title_never_overwritten(self, db):
        _mk(db, "u1", title="My manual title", source="user", msg="whatever")
        stub = _generate_stub({"whatever": "New title"})

        stats = retitle_missing(db, generate=stub, apply_changes=True)

        assert stats["scanned"] == 0
        row = db.get_session("u1")
        assert row["title"] == "My manual title"

    def test_generation_failure_skipped(self, db):
        _mk(db, "e1", msg="what is the plan")
        stub = _generate_stub({})  # returns None for everything

        stats = retitle_missing(db, generate=stub, apply_changes=True)

        assert stats["failed"] == 1
        assert stats["generated"] == 0

    def test_generate_exception_skipped(self, db):
        _mk(db, "e1", msg="what is the plan")

        def boom(fm):
            raise RuntimeError("provider down")

        stats = retitle_missing(db, generate=boom, apply_changes=True)

        assert stats["failed"] == 1
        row = db.get_session("e1")
        assert row["title"] is None

    def test_mixed_historical_db_end_to_end(self, db):
        """Simulate a real post-migration DB with mixed storage generations.

        * root1: user-titled (untouchable)
        * root2: old pre-provenance truncated title — repaired by default
          (full-authority repair is the default; --no-legacy-truncated opts out)
        * root3: empty title (generate)
        * seg under root2: empty chain segment (inherit + dedupe)
        * seg under root3: empty chain segment (inherit)
        """
        _mk(db, "r1", title="My manual title", source="user", msg="hello")
        fm2 = "we need to review the quarterly budget before the board meeting"
        _mk(db, "r2", title=fm2[:40], source=None, msg=fm2)
        _mk(db, "r3", msg="deploy the new API to staging")
        _mk(db, "s2", parent="r2", msg="continuation of budget work")
        _mk(db, "s3", parent="r3", msg="deploy follow-up")

        stub = _generate_stub({
            fm2: "Budget review",
            "deploy the new API to staging": "Deploy API to staging",
        })

        stats = retitle_missing(db, generate=stub, apply_changes=True)

        assert stats["generated"] == 2  # r2 (legacy, now default) + r3
        assert stats["inherited"] == 2  # s2, s3
        # r1 untouched
        assert db.get_session("r1")["title"] == "My manual title"
        # r2 repaired at user level (pre-provenance row)
        assert db.get_session("r2")["title"] == "Budget review"
        assert db.get_session("r2")["title_source"] == "user"
        # r3 generated
        assert db.get_session("r3")["title"] == "Deploy API to staging"
        assert db.get_session("r3")["title_source"] == "llm"
        # segments inherited: s2 = "Budget review #2" (from repaired r2),
        # s3 = "Deploy API to staging #2"
        assert db.get_session("s2")["title"] == "Budget review #2"
        assert db.get_session("s2")["title_source"] == "derived"
        assert db.get_session("s3")["title"] == "Deploy API to staging #2"

    def test_legacy_opt_in_repairs_truncated_pre_provenance(self, db):
        fm2 = "we need to review the quarterly budget before the board meeting"
        _mk(db, "r2", title=fm2[:40], source=None, msg=fm2)
        _mk(db, "s2", parent="r2", msg="continuation of budget work")
        stub = _generate_stub({fm2: "Budget review"})

        stats = retitle_missing(
            db,
            generate=stub,
            apply_changes=True,
            include_legacy_truncated=True,
        )

        # r2 now repaired (user-level write) AND segment inherits new title
        assert stats["generated"] == 1
        assert db.get_session("r2")["title"] == "Budget review"
        assert db.get_session("s2")["title"] == "Budget review #2"

    def test_empty_string_title_repaired_at_user_level(self, db):
        # title='' (empty string, NOT NULL) is a placeholder the official
        # set_auto_title refuses to clobber ('' counts as an existing title),
        # so the repair must write at user level.
        _mk(db, "e1", title="", source=None, msg="what is the plan")
        stub = _generate_stub({"what is the plan": "Plan discussion"})

        stats = retitle_missing(db, generate=stub, apply_changes=True)

        assert stats["generated"] == 1
        row = db.get_session("e1")
        assert row["title"] == "Plan discussion"
        assert row["title_source"] == "user"

    def test_confirm_callback_filters_rows(self, db):
        _mk(db, "a1", msg="msg 0")
        _mk(db, "a2", msg="msg 1")
        _mk(db, "a3", msg="msg 2")
        stub = _generate_stub({"msg 0": "T0", "msg 1": "T1", "msg 2": "T2"})

        def confirm(items, title):
            # only the first row
            return {0}

        stats = retitle_missing(
            db, generate=stub, apply_changes=True, confirm=confirm
        )

        assert stats["generated"] == 1
        assert db.get_session("a1")["title"] == "T0"
        assert db.get_session("a2")["title"] is None
        assert db.get_session("a3")["title"] is None
        assert stats["backup_path"] is not None

    def test_apply_reports_would_generate_zero(self, db):
        """would_generate is dry-run-only; --apply routes candidates to
        generated (or failed), never to would_generate."""
        _mk(db, "e1", msg="what is the plan")
        stub = _generate_stub({"what is the plan": "Plan discussion"})

        stats = retitle_missing(db, generate=stub, apply_changes=True)

        assert stats["would_generate"] == 0
        assert stats["generated"] == 1


class TestDescribeTitleModel:
    def test_cloud_provider_flags_cost(self, monkeypatch):
        import hermes_cli.config as config_mod
        from hermes_cli.session_migration import _describe_title_model

        monkeypatch.setattr(
            config_mod, "load_config_readonly",
            lambda: {
                "model": {"provider": "deepseek", "model": "deepseek-chat"},
                "auxiliary": {
                    "title_generation": {
                        "provider": "deepseek",
                        "model": "deepseek-chat",
                        "api_key": "sk-test",
                    }
                },
            },
        )
        desc = _describe_title_model()
        assert "provider       : deepseek" in desc
        assert "deepseek-chat" in desc
        assert "CLOUD API — may incur cost" in desc
        assert "api_key        : ***configured***" in desc

    def test_local_provider_no_cost(self, monkeypatch):
        import hermes_cli.config as config_mod
        from hermes_cli.session_migration import _describe_title_model

        monkeypatch.setattr(
            config_mod, "load_config_readonly",
            lambda: {
                "model": {"provider": "deepseek", "model": "deepseek-chat"},
                "auxiliary": {
                    "title_generation": {
                        "provider": "lmstudio",
                        "model": "qwen3-27b",
                        "base_url": "http://127.0.0.1:1234/v1",
                        "api_key": "",
                    }
                },
            },
        )
        desc = _describe_title_model()
        assert "provider       : lmstudio" in desc
        assert "qwen3-27b" in desc
        assert "local model — no cost" in desc

    def test_auto_falls_back_to_main_model(self, monkeypatch):
        import hermes_cli.config as config_mod
        from hermes_cli.session_migration import _describe_title_model

        monkeypatch.setattr(
            config_mod, "load_config_readonly",
            lambda: {
                "model": {"provider": "deepseek", "model": "deepseek-chat"},
                "auxiliary": {"title_generation": {"provider": "auto"}},
            },
        )
        desc = _describe_title_model()
        assert "provider       : auto" in desc
        assert "deepseek / deepseek-chat" in desc


# ---------------------------------------------------------------------------
# repair-chains: orphaned compression-chain relink
# ---------------------------------------------------------------------------


class TestRepairChains:
    def _mk_handoff(self, db, sid, title):
        """Create a root whose first message is a compaction handoff —
        the strong evidence it is a continuation of an earlier chat."""
        _mk(
            db,
            sid,
            title=title,
            source="llm",
            msg="[CONTEXT SUMMARY]: legacy continuation of an earlier chat",
        )
        return sid

    def test_dry_run_detects_but_does_not_relink(self, db):
        # One handoff root + one same-titled root = "both" group; dry run
        # reports it but never writes.
        self._mk_handoff(db, "r1", "Plan review")
        _mk(db, "r2", title="Plan review #2", source="llm", msg="second")

        stats = repair_chains(db, apply_changes=False)

        assert stats["orphaned_chain_groups"] == 1
        assert stats["relinked"] == 0
        r2 = db._conn.execute(
            "SELECT parent_session_id FROM sessions WHERE id='r2'"
        ).fetchone()
        assert r2["parent_session_id"] is None

    def test_apply_relinks_both_group_under_oldest_root(self, db):
        # Both = handoff root + same-titled sibling: hard evidence of one
        # conversation, so --apply relinks the later root under the oldest
        # and marks the parent as compression-ended so the chain renders
        # like a normal compression continuation.
        self._mk_handoff(db, "r1", "Plan review")
        _mk(db, "r2", title="Plan review #2", source="llm", msg="second")

        stats = repair_chains(db, apply_changes=True)

        assert stats["relinked"] == 1
        r2 = db._conn.execute(
            "SELECT parent_session_id FROM sessions WHERE id='r2'"
        ).fetchone()
        assert r2["parent_session_id"] == "r1"
        r1 = db._conn.execute(
            "SELECT end_reason FROM sessions WHERE id='r1'"
        ).fetchone()
        assert r1["end_reason"] == "compression"
        # Official projection: the chain tip surfaces as one list entry.
        tips = [r["id"] for r in db.list_sessions_rich(limit=100)]
        assert "r2" in tips
        assert "r1" not in tips

    def test_apply_does_not_relink_same_title_only_group(self, db):
        # Same-title without any handoff is a weak signal (kanban subtasks
        # legitimately repeat titles) — report but never relink.
        _mk(db, "r1", title="Plan review", source="llm", msg="first")
        _mk(db, "r2", title="Plan review #2", source="llm", msg="second")

        stats = repair_chains(db, apply_changes=True)

        assert stats["relinked"] == 0
        assert stats["skipped"] >= 1
        r2 = db._conn.execute(
            "SELECT parent_session_id FROM sessions WHERE id='r2'"
        ).fetchone()
        assert r2["parent_session_id"] is None

    def test_apply_weak_signal_relinks_when_user_checks_it(self, db):
        # Same-title-only group is weak signal (skipped by default) but the
        # interactive checklist lists it (unchecked) and the user may check
        # it to force the relink.
        _mk(db, "r1", title="Plan review", source="llm", msg="first")
        _mk(db, "r2", title="Plan review #2", source="llm", msg="second")

        captured = {}

        def confirm(items, title, **kw):
            captured["items"] = items
            # user checks row 0 (the weak-signal group)
            return {0}

        stats = repair_chains(db, apply_changes=True, confirm=confirm)

        assert stats["relinked"] == 1
        assert stats["skipped"] == 0
        r2 = db._conn.execute(
            "SELECT parent_session_id FROM sessions WHERE id='r2'"
        ).fetchone()
        assert r2["parent_session_id"] == "r1"

    def test_apply_weak_signal_unchecked_stays_skipped(self, db):
        # Weak-signal group listed but NOT checked by the user: skipped.
        _mk(db, "r1", title="Plan review", source="llm", msg="first")
        _mk(db, "r2", title="Plan review #2", source="llm", msg="second")

        def confirm(items, title, **kw):
            # user confirms with nothing checked
            return set()

        stats = repair_chains(db, apply_changes=True, confirm=confirm)

        assert stats["relinked"] == 0
        assert stats["skipped"] == 0  # unchecked rows are simply not selected
        r2 = db._conn.execute(
            "SELECT parent_session_id FROM sessions WHERE id='r2'"
        ).fetchone()
        assert r2["parent_session_id"] is None

    def test_confirm_menu_labels_weak_signal_warning(self, db):
        # The interactive menu must label weak-signal groups so the user is
        # not misled into thinking a title match proves one conversation.
        _mk(db, "r1", title="Plan review", source="llm", msg="first")
        _mk(db, "r2", title="Plan review #2", source="llm", msg="second")

        captured = {}

        def confirm(items, title, **kw):
            captured["items"] = items
            captured["selected"] = kw.get("selected")
            return set()

        repair_chains(db, apply_changes=True, confirm=confirm)

        assert any("title match only" in it for it in captured["items"])
        # weak-signal group is NOT pre-checked
        assert captured["selected"] == set()

    def test_confirm_menu_prechecks_strong_signal(self, db):
        self._mk_handoff(db, "r1", "Plan review")
        _mk(db, "r2", title="Plan review #2", source="llm", msg="second")

        captured = {}

        def confirm(items, title, **kw):
            captured["items"] = items
            captured["selected"] = kw.get("selected")
            return {0}

        repair_chains(db, apply_changes=True, confirm=confirm)

        # strong-signal group carries the evidence label and is pre-checked
        assert any("compaction-handoff evidence" in it for it in captured["items"])
        assert captured["selected"] == {0}

    def test_esc_cancel_does_nothing_even_with_preselected(self, db):
        # ESC must not fall back to the pre-checked rows: a strong-signal
        # group is pre-checked, but if the user cancels (empty set returned)
        # NOTHING may be relinked — the checked set is the exact contract.
        self._mk_handoff(db, "r1", "Plan review")
        _mk(db, "r2", title="Plan review #2", source="llm", msg="second")

        def confirm(items, title, **kw):
            # ESC → empty selection, even though strong groups were pre-checked
            return set()

        stats = repair_chains(db, apply_changes=True, confirm=confirm)

        assert stats["relinked"] == 0
        assert stats["backup_path"] is None  # no backup either
        r2 = db._conn.execute(
            "SELECT parent_session_id FROM sessions WHERE id='r2'"
        ).fetchone()
        assert r2["parent_session_id"] is None

    def test_esc_cancel_retitle_no_backup(self, db):
        _mk(db, "e1", msg="what is the plan")
        stub = _generate_stub({"what is the plan": "Plan discussion"})

        def confirm(items, title, **kw):
            return set()  # ESC → nothing selected

        stats = retitle_missing(
            db, generate=stub, apply_changes=True, confirm=confirm
        )

        assert stats["generated"] == 0
        assert stats["backup_path"] is None
        assert db.get_session("e1")["title"] is None

    def test_no_groups_when_single_root(self, db):
        _mk(db, "r1", title="Alpha", source="llm", msg="first")
        _mk(db, "r2", title="Beta", source="llm", msg="second")

        stats = repair_chains(db, apply_changes=False)

        assert stats["orphaned_chain_groups"] == 0

    def test_single_handoff_root_forced_relink_is_noop(self, db):
        """A lone handoff root (no sibling) forced via the checklist must
        NOT be marked compression-ended — without a child that would corrupt
        its end_reason semantics."""
        _mk(db, "r1", title="Lone", source="llm",
            msg="[CONTEXT SUMMARY]: handoff body")

        # Orphan detection reports the single handoff root as a group.
        groups = find_orphaned_chain_candidates(db)
        assert len(groups) == 1
        assert groups[0]["signal"] == "handoff"
        assert len(groups[0]["sessions"]) == 1

        # User forces the relink of that group; still nothing may be written.
        stats = repair_chains(
            db, apply_changes=True, confirm=lambda items, title, **kw: {0},
        )

        assert stats["relinked"] == 0
        assert stats["skipped"] == 1
        # The root must NOT have been marked compression-ended.
        row = db._conn.execute(
            "SELECT end_reason FROM sessions WHERE id='r1'"
        ).fetchone()
        assert row[0] != "compression"


class TestMergeAtomicity:
    def test_merge_rolls_back_on_failure(self, db):
        """A mid-merge failure must not leave a partially merged database:
        SessionDB runs in autocommit, so the whole merge is wrapped in an
        explicit transaction (BEGIN IMMEDIATE ... ROLLBACK on error)."""
        # Two chains, each head + one segment.
        for h, seg in (("h1", "s1"), ("h2", "s2")):
            _mk(db, h, title=f"Chain {h}", source="llm", msg="first")
            _mk(db, seg, parent=h, title=f"Chain {h} #2", source="llm", msg="second")
            db._conn.execute(
                "UPDATE sessions SET end_reason='compression' WHERE id=?",
                (h,),
            )
            db._conn.commit()

        # Force a failure mid-merge by dropping a column the merge reads.
        db._conn.execute(
            "ALTER TABLE sessions DROP COLUMN tool_call_count"
        )
        db._conn.commit()

        from hermes_cli.session_migration import merge_compression_chains

        with pytest.raises(Exception):
            merge_compression_chains(db, apply_changes=True, backup=False)

        # No partial merge: both segment rows still exist with messages intact.
        for seg in ("s1", "s2"):
            row = db._conn.execute(
                "SELECT parent_session_id FROM sessions WHERE id=?", (seg,)
            ).fetchone()
            assert row["parent_session_id"] is not None
            cnt = db._conn.execute(
                "SELECT COUNT(*) c FROM messages WHERE session_id=?", (seg,)
            ).fetchone()["c"]
            assert cnt == 1


# ---------------------------------------------------------------------------
# Phase 3: fork compression-chain flattening (merge-chains)
# ---------------------------------------------------------------------------


def _mk_fork(db, sid, *, parent=None, end_reason=None, msg=None, title=None,
             model_config=None, source=None, user_id=None, session_key=None,
             chat_id=None, chat_type=None, input_tokens=0, ended_at=None):
    """Create a session row; parent + end_reason let us build fork chains.

    ``end_reason='compression'`` on a parent marks the fork edge
    (mirrors how real compression writes the DB).
    """
    kwargs = {}
    if parent is not None:
        kwargs["parent_session_id"] = parent
    if model_config is not None:
        # create_session JSON-serializes model_config; pass a dict.
        import json
        kwargs["model_config"] = json.loads(model_config) if isinstance(model_config, str) else model_config
    db.create_session(sid, source=source or "cli", **kwargs)
    sets = []
    params = []
    if title is not None:
        sets.append("title=?")
        params.append(title)
        sets.append("title_source='llm'")
    if end_reason is not None:
        sets.append("end_reason=?")
        params.append(end_reason)
    if user_id is not None:
        sets.append("user_id=?")
        params.append(user_id)
    if session_key is not None:
        sets.append("session_key=?")
        params.append(session_key)
    if chat_id is not None:
        sets.append("chat_id=?")
        params.append(chat_id)
    if chat_type is not None:
        sets.append("chat_type=?")
        params.append(chat_type)
    if input_tokens:
        sets.append("input_tokens=?")
        params.append(input_tokens)
    if ended_at is not None:
        sets.append("ended_at=?")
        params.append(ended_at)
    if sets:
        params.append(sid)
        db._conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id=?", params)
    if msg:
        db.append_message(sid, "user", content=msg)
    db._conn.commit()
    return sid


class TestMergeChainCandidates:
    def test_detects_single_fork_chain(self, db):
        _mk_fork(db, "root", end_reason="compression", title="T", msg="first")
        _mk_fork(db, "seg1", parent="root", end_reason="compression", msg="second")
        _mk_fork(db, "seg2", parent="seg1", msg="third")

        cands = find_merge_chain_candidates(db)

        assert len(cands) == 1
        assert cands[0]["head"] == "root"
        assert cands[0]["segments"] == ["seg1", "seg2"]

    def test_excludes_delegate_branch_tool_children(self, db):
        _mk_fork(db, "root", end_reason="compression", msg="first")
        _mk_fork(db, "ok", parent="root", msg="second")
        _mk_fork(db, "del", parent="root", msg="delegate",
                 model_config='{"_delegate_from": "root"}')
        _mk_fork(db, "br", parent="root", msg="branch",
                 model_config='{"_branched_from": "root"}')
        _mk_fork(db, "tool", parent="root", msg="tool", source="tool")

        cands = find_merge_chain_candidates(db)

        assert len(cands) == 1
        # Only the plain child is a segment; delegate/branch/tool are not.
        assert cands[0]["segments"] == ["ok"]

    def test_middle_segment_not_treated_as_head(self, db):
        _mk_fork(db, "root", end_reason="compression", msg="first")
        _mk_fork(db, "seg1", parent="root", end_reason="compression", msg="second")
        _mk_fork(db, "seg2", parent="seg1", msg="third")

        cands = find_merge_chain_candidates(db)

        assert len(cands) == 1
        assert cands[0]["head"] == "root"
        assert cands[0]["segments"] == ["seg1", "seg2"]

    def test_no_chain_when_parent_not_compression(self, db):
        _mk_fork(db, "root", end_reason="cli_close", msg="first")
        _mk_fork(db, "child", parent="root", msg="second")

        cands = find_merge_chain_candidates(db)

        assert cands == []


class TestMergeCompressionChains:
    def test_dry_run_writes_nothing_and_no_backup(self, db, tmp_path):
        _mk_fork(db, "root", end_reason="compression", title="T", msg="first")
        _mk_fork(db, "seg1", parent="root", msg="second")

        stats = merge_compression_chains(db, apply_changes=False, backup=True)

        assert stats["chains"] == 1
        assert stats["backup_path"] is None
        # Nothing written: segment still exists, messages not moved.
        assert db.get_session("seg1") is not None
        assert db.get_session("root")["message_count"] == 1
        assert not list(tmp_path.glob("*.pre-merge-chains-*"))

    def test_apply_moves_messages_and_deletes_segments(self, db):
        _mk_fork(db, "root", end_reason="compression", title="T", msg="m1")
        _mk_fork(db, "seg1", parent="root", end_reason="compression", msg="m2")
        _mk_fork(db, "seg2", parent="seg1", msg="m3")

        stats = merge_compression_chains(db, apply_changes=True, backup=False)

        assert stats["chains"] == 1
        assert stats["segments"] == 2
        assert stats["messages_moved"] == 2
        # Segment rows gone; head owns all messages.
        assert db.get_session("seg1") is None
        assert db.get_session("seg2") is None
        msgs = db.get_messages_as_conversation("root", include_ancestors=False)
        contents = [m["content"] for m in msgs]
        assert "m1" in contents and "m2" in contents and "m3" in contents
        assert db.get_session("root")["message_count"] == 3

    def test_apply_accumulates_counters_and_inherits_terminal_state(self, db):
        _mk_fork(db, "root", end_reason="compression", msg="m1", input_tokens=10)
        _mk_fork(db, "seg1", parent="root", msg="m2", input_tokens=20,
                 ended_at=1780000000.0, end_reason=None)
        # seg1 has no end_reason; root inherits its ended_at but not a
        # 'compression' marker (that would mislead the projection).

        stats = merge_compression_chains(db, apply_changes=True, backup=False)

        assert stats["usage_merged"] == 0  # no session_model_usage rows
        root = db.get_session("root")
        assert root["input_tokens"] == 30
        assert root["ended_at"] == 1780000000.0
        assert root["end_reason"] is None

    def test_apply_redirects_orphan_children(self, db):
        # seg1 will be deleted; child "orphan" is a reset child of seg1.
        _mk_fork(db, "root", end_reason="compression", msg="m1")
        _mk_fork(db, "seg1", parent="root", msg="m2")
        _mk_fork(db, "orphan", parent="seg1", msg="m3",
                 model_config='{"_reset_from": "seg1"}')

        stats = merge_compression_chains(db, apply_changes=True, backup=False)

        assert stats["orphans_redirected"] == 1
        orphan = db._conn.execute(
            "SELECT parent_session_id FROM sessions WHERE id='orphan'"
        ).fetchone()
        assert orphan["parent_session_id"] == "root"

    def test_apply_merges_session_model_usage(self, db):
        _mk_fork(db, "root", end_reason="compression", msg="m1")
        _mk_fork(db, "seg1", parent="root", msg="m2")
        db._conn.execute(
            """
            INSERT INTO session_model_usage
              (session_id, model, billing_provider, billing_base_url, billing_mode,
               task, api_call_count, input_tokens, output_tokens)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            ("seg1", "m1", "p", "b", "bm", "t", 2, 100, 50),
        )
        db._conn.execute(
            """
            INSERT INTO session_model_usage
              (session_id, model, billing_provider, billing_base_url, billing_mode,
               task, api_call_count, input_tokens, output_tokens)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            ("root", "m1", "p", "b", "bm", "t", 1, 10, 5),
        )
        db._conn.commit()

        stats = merge_compression_chains(db, apply_changes=True, backup=False)

        assert stats["usage_merged"] == 1
        rows = db._conn.execute(
            "SELECT api_call_count, input_tokens, output_tokens "
            "FROM session_model_usage WHERE session_id='root'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["api_call_count"] == 3
        assert rows[0]["input_tokens"] == 110
        assert rows[0]["output_tokens"] == 55

    def test_apply_inherits_gateway_origin(self, db):
        _mk_fork(db, "root", end_reason="compression", msg="m1")
        _mk_fork(db, "seg1", parent="root", msg="m2", user_id="u1",
                 session_key="k1", chat_id="c1", chat_type="dm")

        stats = merge_compression_chains(db, apply_changes=True, backup=False)

        root = db.get_session("root")
        assert root["user_id"] == "u1"
        assert root["session_key"] == "k1"
        assert root["chat_id"] == "c1"
        assert root["chat_type"] == "dm"

    def test_apply_takes_automatic_backup(self, db, tmp_path):
        _mk_fork(db, "root", end_reason="compression", msg="m1")
        _mk_fork(db, "seg1", parent="root", msg="m2")

        stats = merge_compression_chains(db, apply_changes=True, backup=True)

        assert stats["backup_path"] is not None
        from pathlib import Path
        assert Path(stats["backup_path"]).exists()
        assert "pre-merge-chains" in stats["backup_path"]

    def test_apply_idempotent(self, db):
        _mk_fork(db, "root", end_reason="compression", msg="m1")
        _mk_fork(db, "seg1", parent="root", msg="m2")

        first = merge_compression_chains(db, apply_changes=True, backup=False)
        second = merge_compression_chains(db, apply_changes=True, backup=False)

        assert first["chains"] == 1
        assert second["chains"] == 0

    def test_apply_verify_report(self, db):
        _mk_fork(db, "root", end_reason="compression", msg="m1")
        _mk_fork(db, "seg1", parent="root", end_reason="compression", msg="m2")
        _mk_fork(db, "seg2", parent="seg1", msg="m3")

        stats = merge_compression_chains(db, apply_changes=True, backup=False)

        assert stats["verified"] is True
        v = stats["verify_report"]
        assert v["messages_before"] == 3
        assert v["messages_after"] == 3
        assert v["delta"] == 0
        assert v["segments_deleted"] == 2
        assert v["usage_orphans"] == 0

    def test_no_chains_returns_empty_stats(self, db):
        _mk_fork(db, "root", end_reason="cli_close", msg="m1")

        stats = merge_compression_chains(db, apply_changes=True, backup=False)

        assert stats["chains"] == 0
        assert stats["segments"] == 0


# ---------------------------------------------------------------------------
# restore-db (hermes sessions restore-db)
# ---------------------------------------------------------------------------


class TestRestoreState:
    def test_db_holder_matches_main_wal_shm(self, tmp_path):
        from hermes_cli.session_migration import _db_holder_matches

        dbp = tmp_path / "state.db"
        assert _db_holder_matches(str(dbp), dbp)
        assert _db_holder_matches(str(dbp.with_name("state.db-wal")), dbp)
        assert _db_holder_matches(str(dbp.with_name("state.db-shm")), dbp)
        assert not _db_holder_matches(str(tmp_path / "other.db"), dbp)
        assert not _db_holder_matches("", dbp)

    def test_list_snapshot_candidates_sorted(self, tmp_path):
        from hermes_cli.session_migration import _list_snapshot_candidates

        dbp = tmp_path / "state.db"
        (tmp_path / "state.db.pre-repair-chains-20260815_010000").touch()
        (tmp_path / "state.db.pre-merge-chains-20260815_020000").touch()
        (tmp_path / "state.db").touch()  # not a snapshot
        (tmp_path / "other.pre-x").touch()  # different prefix

        snaps = _list_snapshot_candidates(dbp)
        # Newest first: 02:00 stamp sorts after 01:00, then reversed.
        assert [p.name for p in snaps] == [
            "state.db.pre-merge-chains-20260815_020000",
            "state.db.pre-repair-chains-20260815_010000",
        ]

    def test_restore_dry_run_reports_holders_only(self, tmp_path):
        from hermes_cli.session_migration import restore_state_db

        import sqlite3

        dbp = tmp_path / "state.db"
        conn = sqlite3.connect(dbp)
        conn.execute("CREATE TABLE t (v)")
        conn.commit()
        conn.close()
        before = dbp.read_bytes()

        snap = tmp_path / "state.db.pre-test-20260815_000000"
        sconn = sqlite3.connect(snap)
        sconn.execute("CREATE TABLE t (v)")
        sconn.commit()
        sconn.close()

        stats = restore_state_db(dbp, snapshot=snap.name, dry_run=True)

        assert stats["snapshot"] == str(snap)
        assert stats["restored"] is False
        assert stats["killed"] == []
        # Dry run must not touch the DB or the snapshot.
        assert dbp.read_bytes() == before

    def test_restore_no_snapshot_raises(self, tmp_path):
        from hermes_cli.session_migration import restore_state_db

        dbp = tmp_path / "state.db"
        dbp.write_bytes(b"x")

        with pytest.raises(RuntimeError, match="no state.db snapshots"):
            restore_state_db(dbp, dry_run=True)

    def test_restore_missing_explicit_snapshot_raises(self, tmp_path):
        from hermes_cli.session_migration import restore_state_db

        dbp = tmp_path / "state.db"
        dbp.write_bytes(b"x")

        with pytest.raises(RuntimeError, match="snapshot not found"):
            restore_state_db(dbp, snapshot="nope.db", dry_run=True)

    def test_restore_rejects_snapshot_outside_db_dir(self, tmp_path):
        from hermes_cli.session_migration import restore_state_db

        dbp = tmp_path / "state.db"
        dbp.write_bytes(b"x")

        # A snapshot outside the DB's directory (e.g. another profile's DB)
        # must be refused — mirroring the official snapshot restore
        # traversal guard. tmp_path.parent is guaranteed outside tmp_path.
        other = tmp_path.parent / f"outside-{tmp_path.name}.db"
        other.write_bytes(b"y")
        try:
            with pytest.raises(RuntimeError, match="outside"):
                restore_state_db(dbp, snapshot=str(other), dry_run=True)
        finally:
            other.unlink(missing_ok=True)

        # Absolute path inside the DB dir is fine (existence checked next).
        with pytest.raises(RuntimeError, match="snapshot not found"):
            restore_state_db(
                dbp, snapshot=str(tmp_path / "state.db.pre-x-20260815_000000"),
                dry_run=True,
            )

    def test_restore_swaps_db_and_clears_wal(self, tmp_path):
        from hermes_cli.session_migration import restore_state_db

        # Real SQLite DB so verification passes.
        import sqlite3

        dbp = tmp_path / "state.db"
        conn = sqlite3.connect(dbp)
        conn.execute("CREATE TABLE t (v)")
        conn.execute("INSERT INTO t VALUES ('before')")
        conn.commit()
        conn.close()

        # Snapshot with different content.
        snap = tmp_path / "state.db.pre-test-20260815_000000"
        sconn = sqlite3.connect(snap)
        sconn.execute("CREATE TABLE t (v)")
        sconn.execute("INSERT INTO t VALUES ('after')")
        sconn.commit()
        sconn.close()

        # Stale WAL/SHM that must be removed.
        (tmp_path / "state.db-wal").write_bytes(b"stale")
        (tmp_path / "state.db-shm").write_bytes(b"stale")

        stats = restore_state_db(dbp, snapshot=snap.name, force=True)

        assert stats["restored"] is True
        assert stats["verified"] is True
        assert not (tmp_path / "state.db-wal").exists()
        assert not (tmp_path / "state.db-shm").exists()
        # Restored DB contains the snapshot's data.
        conn = sqlite3.connect(dbp)
        val = conn.execute("SELECT v FROM t").fetchone()[0]
        conn.close()
        assert val == "after"

    def test_restore_refuses_live_holders_without_force(self, tmp_path, monkeypatch):
        from hermes_cli.session_migration import restore_state_db

        import sqlite3

        dbp = tmp_path / "state.db"
        conn = sqlite3.connect(dbp)
        conn.execute("CREATE TABLE t (v)")
        conn.execute("INSERT INTO t VALUES ('before')")
        conn.commit()
        conn.close()

        snap = tmp_path / "state.db.pre-test-20260815_000000"
        sconn = sqlite3.connect(snap)
        sconn.execute("CREATE TABLE t (v)")
        sconn.execute("INSERT INTO t VALUES ('after')")
        sconn.commit()
        sconn.close()

        monkeypatch.setattr(
            "hermes_cli.session_migration._find_state_db_holders",
            lambda db_path: [12345],
        )

        with pytest.raises(RuntimeError, match="refusing to restore"):
            restore_state_db(dbp, snapshot=snap.name, force=False)
        conn = sqlite3.connect(dbp)
        val = conn.execute("SELECT v FROM t").fetchone()[0]
        conn.close()
        assert val == "before"  # untouched

    def test_restore_rejects_corrupt_snapshot(self, tmp_path):
        from hermes_cli.session_migration import restore_state_db

        dbp = tmp_path / "state.db"
        dbp.write_bytes(b"not-a-db")
        snap = tmp_path / "state.db.pre-test-20260815_000000"
        snap.write_bytes(b"also-not-a-db")

        with pytest.raises(RuntimeError, match="failed integrity verification"):
            restore_state_db(dbp, snapshot=snap.name, force=True)
        assert dbp.read_bytes() == b"not-a-db"  # untouched

    def test_restore_picks_newest_with_multiple_snapshots(self, tmp_path, monkeypatch):
        from hermes_cli.session_migration import restore_state_db

        import sqlite3

        def _mk_snap(name, value):
            p = tmp_path / name
            conn = sqlite3.connect(p)
            conn.execute("CREATE TABLE t (v)")
            conn.execute("INSERT INTO t VALUES (?)", (value,))
            conn.commit()
            conn.close()
            return p

        dbp = tmp_path / "state.db"
        conn = sqlite3.connect(dbp)
        conn.execute("CREATE TABLE t (v)")
        conn.execute("INSERT INTO t VALUES ('live')")
        conn.commit()
        conn.close()

        _mk_snap("state.db.pre-merge-chains-20260815_020000", "newest")
        _mk_snap("state.db.pre-repair-chains-20260815_010000", "old")

        # Multiple snapshots, no confirm callback → newest (by stamp) wins.
        stats = restore_state_db(dbp, force=True)

        assert stats["restored"] is True
        assert "020000" in stats["snapshot"]  # newest stamp
        conn = sqlite3.connect(dbp)
        val = conn.execute("SELECT v FROM t").fetchone()[0]
        conn.close()
        assert val == "newest"

    def test_restore_single_select_with_confirm(self, tmp_path, monkeypatch):
        from hermes_cli.session_migration import restore_state_db

        import sqlite3

        def _mk_snap(name, value):
            p = tmp_path / name
            conn = sqlite3.connect(p)
            conn.execute("CREATE TABLE t (v)")
            conn.execute("INSERT INTO t VALUES (?)", (value,))
            conn.commit()
            conn.close()
            return p

        dbp = tmp_path / "state.db"
        conn = sqlite3.connect(dbp)
        conn.execute("CREATE TABLE t (v)")
        conn.execute("INSERT INTO t VALUES ('live')")
        conn.commit()
        conn.close()

        _mk_snap("state.db.pre-a-20260815_010000", "A")
        _mk_snap("state.db.pre-b-20260815_020000", "B")

        # confirm picks index 0 (the "B" snapshot — newest first in items).
        seen = {}

        def _fake_confirm(items, title):
            seen["items"] = items
            return 0

        stats = restore_state_db(dbp, force=True, confirm=_fake_confirm)

        assert stats["restored"] is True
        assert seen["items"], "confirm must be offered the snapshot list"
        conn = sqlite3.connect(dbp)
        val = conn.execute("SELECT v FROM t").fetchone()[0]
        conn.close()
        assert val == "B"

    def test_restore_cancel_returns_empty_stats(self, tmp_path, monkeypatch):
        from hermes_cli.session_migration import restore_state_db

        import sqlite3

        def _mk_snap(name, value):
            p = tmp_path / name
            conn = sqlite3.connect(p)
            conn.execute("CREATE TABLE t (v)")
            conn.execute("INSERT INTO t VALUES (?)", (value,))
            conn.commit()
            conn.close()
            return p

        dbp = tmp_path / "state.db"
        conn = sqlite3.connect(dbp)
        conn.execute("CREATE TABLE t (v)")
        conn.execute("INSERT INTO t VALUES ('live')")
        conn.commit()
        conn.close()

        _mk_snap("state.db.pre-a-20260815_010000", "A")
        _mk_snap("state.db.pre-b-20260815_020000", "B")

        stats = restore_state_db(
            dbp, force=True, confirm=lambda items, title: None
        )

        assert stats["restored"] is False
        assert stats["snapshot"] is None
        conn = sqlite3.connect(dbp)
        val = conn.execute("SELECT v FROM t").fetchone()[0]
        conn.close()
        assert val == "live"  # untouched

    def test_restore_kills_holders_with_force(self, tmp_path, monkeypatch):
        from hermes_cli.session_migration import restore_state_db

        import sqlite3

        dbp = tmp_path / "state.db"
        conn = sqlite3.connect(dbp)
        conn.execute("CREATE TABLE t (v)")
        conn.execute("INSERT INTO t VALUES ('before')")
        conn.commit()
        conn.close()

        snap = tmp_path / "state.db.pre-test-20260815_000000"
        sconn = sqlite3.connect(snap)
        sconn.execute("CREATE TABLE t (v)")
        sconn.execute("INSERT INTO t VALUES ('after')")
        sconn.commit()
        sconn.close()

        monkeypatch.setattr(
            "hermes_cli.session_migration._find_state_db_holders",
            lambda db_path: [12345],
        )
        monkeypatch.setattr(
            "hermes_cli.session_migration._kill_processes",
            lambda holders, log: ([12345], []),
        )

        stats = restore_state_db(dbp, snapshot=snap.name, force=True)

        assert stats["killed"] == [12345]
        assert stats["restored"] is True
        conn = sqlite3.connect(dbp)
        val = conn.execute("SELECT v FROM t").fetchone()[0]
        conn.close()
        assert val == "after"

    def test_restore_prints_restart_hints_after_kill(self, tmp_path, monkeypatch):
        from hermes_cli.session_migration import restore_state_db

        import sqlite3

        dbp = tmp_path / "state.db"
        conn = sqlite3.connect(dbp)
        conn.execute("CREATE TABLE t (v)")
        conn.execute("INSERT INTO t VALUES ('before')")
        conn.commit()
        conn.close()

        snap = tmp_path / "state.db.pre-test-20260815_000000"
        sconn = sqlite3.connect(snap)
        sconn.execute("CREATE TABLE t (v)")
        sconn.execute("INSERT INTO t VALUES ('after')")
        sconn.commit()
        sconn.close()

        monkeypatch.setattr(
            "hermes_cli.session_migration._find_state_db_holders",
            lambda db_path: [12345],
        )
        monkeypatch.setattr(
            "hermes_cli.session_migration._kill_processes",
            lambda holders, log: ([12345], []),
        )
        # Simulate the official argv capture for PID 12345 (the function is
        # imported from hermes_cli.main inside restore_state_db, so patch
        # it at its definition site).
        monkeypatch.setattr(
            "hermes_cli.main._dashboard_cmdline_for_pid",
            lambda pid: ["python", "-m", "hermes_cli.main", "gateway", "run"],
        )

        lines: list[str] = []
        stats = restore_state_db(
            dbp, snapshot=snap.name, force=True,
            progress=lambda msg: lines.append(msg),
        )

        assert stats["killed"] == [12345]
        joined = "\n".join(lines)
        assert "Restart stopped processes" in joined
        assert "hermes_cli.main gateway run" in joined
        # Nothing was auto-restarted (no respawn call).
        assert "auto-restarted" in joined

    def test_kill_processes_handles_empty(self):
        from hermes_cli.session_migration import _kill_processes

        killed, failed = _kill_processes([], lambda m: None)
        assert killed == []
        assert failed == []
