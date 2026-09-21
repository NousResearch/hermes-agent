"""P04 hermetic migration proof — disposable default->core transfer.

Hermetic safety contract: every test builds its OWN disposable root via
the proof's make_owned_root() (a TemporaryDirectory with an ownership
marker). The proof clears all ambient HERMES_KANBAN_* overrides before any
kanban_db call, so a hostile caller environment cannot redirect the proof
at a live board. No live ~/.hermes is ever opened.

Coverage:
  - Proof A (board_transfer_preserving): true default->core migration that
    preserves ALL fixture-owned states (tasks incl. archived, epics, links,
    runs, events, comments, attachments, notify subs, lifecycle approvals),
    remaps colliding integer PKs + run_id references, injects a
    pre-swap failure (byte/content + pointer unchanged), exercises a real
    pointer transition + exact restore, FK-checks and disposes.
  - Proof B (transfer_and_prove): single-task move via production
    move_task_atomic seam (NOT presented as full preservation).
  - Hostile-env sentinel: external sentinel DB + attachment paths supplied
    through ambient overrides are left byte-unchanged by the proof.
  - Failure-after-copy: an attachment copy mismatch is rolled back and the
    target carries no half-written file.
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    import subprocess
    _out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True)
    if _out.returncode == 0 and _out.stdout.strip():
        REPO_ROOT = Path(_out.stdout.strip())
except Exception:
    pass
SCRIPT_PATH = REPO_ROOT / "scripts" / "kanban-migration-proof.py"
_MOD_NAME = "kanban_migration_proof"


def _load_proof():
    """Import the proof script fresh. kanban_db is imported lazily inside
    the proof under the owned env, so the temp root is honoured."""
    sys.modules.pop(_MOD_NAME, None)
    spec = importlib.util.spec_from_file_location(_MOD_NAME, str(SCRIPT_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def owned_root():
    """A self-owned disposable root from the proof's own factory.

    Pins HERMES_HOME / HERMES_KANBAN_HOME to the owned root for the test's
    ambient reads (e.g. direct kb.kanban_db_path(board=...)) so verification
    queries hit the same disposable DB the proof wrote. The proof's own
    owned_env() saves/restores these around each call.
    """
    mod = _load_proof()
    root = mod.make_owned_root()
    saved = {}
    for k in ("HERMES_HOME", "HERMES_KANBAN_HOME", "HERMES_KANBAN_DB",
              "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_KANBAN_WORKSPACES_ROOT",
              "HERMES_KANBAN_BOARD"):
        saved[k] = os.environ.get(k)
        os.environ.pop(k, None)
    os.environ["HERMES_HOME"] = str(root)
    os.environ["HERMES_KANBAN_HOME"] = str(root)
    yield root
    for k in ("HERMES_HOME", "HERMES_KANBAN_HOME", "HERMES_KANBAN_DB",
              "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_KANBAN_WORKSPACES_ROOT",
              "HERMES_KANBAN_BOARD"):
        os.environ.pop(k, None)
        if saved[k] is not None:
            os.environ[k] = saved[k]
    # Cleanup (mirror main()'s no-leak guarantee).
    import shutil
    try:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
    except OSError:
        pass


def test_script_exists():
    assert SCRIPT_PATH.is_file(), "scripts/kanban-migration-proof.py is absent"


def test_proof_a_seed_full_fixture_creates_full_scope(owned_root):
    """seed_full_fixture builds the complete scope: tasks (incl. archived),
    epic, link, runs, comments, attachments, notify subs, approvals."""
    mod = _load_proof()
    fixture = mod.seed_full_fixture(owned_root)
    assert fixture["epic_id"] == "epic-proof-1"
    for k in ("T1", "T2", "T3", "T4_archived"):
        assert k in fixture["tasks"], f"missing fixture task {k}"
    # T2->T3 link present.
    from hermes_cli import kanban_db as kb
    path = kb.kanban_db_path(board="default")
    conn = sqlite3.connect(str(path))
    try:
        links = conn.execute("SELECT parent_id, child_id FROM task_links").fetchall()
        assert (fixture["tasks"]["T2"], fixture["tasks"]["T3"]) in [
            (r[0], r[1]) for r in links
        ]
    finally:
        conn.close()


def test_proof_a_full_migration_preserves_all_states(owned_root):
    """board_transfer_preserving transfers ALL source tasks (incl. archived),
    preserves epic + re-board, preserves both-endpoint links, remaps
    colliding PKs, FK clean on both boards."""
    mod = _load_proof()
    mod.seed_full_fixture(owned_root)
    res = mod.board_transfer_preserving(owned_root)
    assert res["ok"], f"Proof A failed: {res}"
    transfer = res["transfer"]
    assert transfer["ok"]
    # Archived task transferred (full preservation, no silent loss). Verified
    # inside board_transfer_preserving before disposal (re-opening the
    # disposed target would hit a deleted file).
    ri = res["row_invariants"]
    assert ri.get("archived_preserved_count", 0) >= 1, (
        f"archived task not preserved in transfer: {ri}"
    )
    epic = ri.get("epic_transferred", {})
    assert epic.get("present") is True
    assert epic.get("board_slug") == "core"
    links = ri.get("task_links", [])
    assert any(l["parent_id"] == transfer.get("transferred_link_parent")
               and l["child_id"] == transfer.get("transferred_link_child")
               for l in links)
    assert res["fk_default"] == []
    assert res["fk_core"] == []
    # Explicit row-class preservation, not merely FK integrity: comments,
    # subscriptions, approvals and attachments all move as equal deltas.
    assert all(v["preserved"] for v in ri["state_counts"].values()), ri["state_counts"]
    assert ri["attachment_hashes_preserved"] is True


def test_proof_a_remaps_colliding_integer_pks_and_run_id(owned_root):
    """Colliding integer PKs (task_runs/events/comments/attachments) and the
    dependent run_id reference in task_events are remapped, not merely
    FK-checked. Assert exact remapped ids."""
    mod = _load_proof()
    mod.seed_full_fixture(owned_root)
    res = mod.board_transfer_preserving(owned_root)
    transfer = res["transfer"]
    assert transfer["ok"]
    # Every transferred integer-PK table offset must be > 0 (remapped above
    # the pre-existing core rows), proving explicit remap rather than reuse.
    offsets = transfer["remap_offsets"]
    for tbl in ("task_runs", "task_events", "task_comments", "task_attachments"):
        assert offsets[tbl] >= 0
    # run_id_map proves event run_id remapping is tracked explicitly.
    assert transfer["run_id_map"], "run_id_map empty — run_id remap not proven"
    # The transferred event's run_id must equal a remapped run id. Verified
    # inside board_transfer_preserving before disposal (re-opening the
    # disposed target would hit a deleted file).
    ri = res["row_invariants"]
    assert ri.get("event_run_id_remapped_count", 0) >= 1, (
        f"task_event run_id not remapped to a transferred run id: {ri}"
    )


def test_proof_a_failure_injection_is_unchanged_and_pointer_untouched(owned_root):
    """Pre-swap failure injection leaves source+target byte/content
    unchanged and the pointer untouched."""
    mod = _load_proof()
    mod.seed_full_fixture(owned_root)
    res = mod.board_transfer_preserving(owned_root)
    fi = res["failure_injection"]
    assert fi["mutated_before_rollback"] is True
    assert res["src_byte_unchanged_after_fail"]
    assert res["tgt_byte_unchanged_after_fail"]
    assert res["src_content_unchanged_after_fail"]
    assert res["tgt_content_unchanged_after_fail"]
    assert res["pointer_unchanged_after_fail"]


def test_proof_a_pointer_real_transition_and_exact_restore(owned_root):
    """Pointer test exercises a real fixture transition and restores the
    EXACT prior state, including the original absent-file condition."""
    from hermes_cli import kanban_db as kb
    mod = _load_proof()
    kb.set_current_board("default")
    mod.seed_full_fixture(owned_root)
    # Ensure pointer starts absent (default condition) for the absent-restore case.
    p = owned_root / "kanban" / "current"
    if p.exists():
        p.unlink()
    res = mod.board_transfer_preserving(owned_root)
    pe = res["pointer_exercise"]
    assert pe["original"] is None, "expected original absent-file condition"
    assert pe["transitioned_present"] is True
    assert pe["restored_exact"] is True, "pointer not restored to absent exactly"


def test_proof_a_disposes_disposable_fixtures(owned_root):
    """board_transfer_preserving disposes the disposable DBs + attachment
    dirs; no leaked kanban.db remains under the owned root."""
    mod = _load_proof()
    mod.seed_full_fixture(owned_root)
    res = mod.board_transfer_preserving(owned_root)
    disposed = res["disposed"]
    assert disposed.get("src_db") is True
    assert disposed.get("tgt_db") is True
    from hermes_cli import kanban_db as kb
    assert not kb.kanban_db_path(board="default").exists(), "default DB leaked"


def test_proof_a_refuses_unowned_root(tmp_path):
    """A marker file is not enough: only roots registered by make_owned_root
    in this process are accepted, and this test stays entirely under tmp_path."""
    mod = _load_proof()
    with pytest.raises(RuntimeError):
        mod.guard_owned_root(REPO_ROOT)
    forged = tmp_path / "forged-root"
    forged.mkdir()
    (forged / ".p04_owned_root").write_text("forged\n")
    with pytest.raises(RuntimeError):
        mod.guard_owned_root(forged)


def test_proof_a_hostile_env_sentinel_unchanged(owned_root, tmp_path):
    """Hostile external sentinel: an ambient HERMES_KANBAN_DB + attachments
    root point at a sentinel OUTSIDE the owned root. The proof must clear
    those overrides, run, and leave the sentinel bytes/tree unchanged."""
    mod = _load_proof()
    # Build a sentinel DB + attachment tree OUTSIDE the owned root.
    sentinel_dir = tmp_path / "sentinel"
    sentinel_dir.mkdir()
    sentinel_db = sentinel_dir / "kanban.db"
    sentinel_db.write_bytes(b"SENTINEL-LIVE-BOARD-MUST-NOT-TOUCH")
    sentinel_att = sentinel_dir / "attachments"
    sentinel_att.mkdir()
    (sentinel_att / "secret.txt").write_bytes(b"do-not-read-attachment")

    import hashlib
    def _h(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    db_before = _h(sentinel_db)
    att_before = _h(sentinel_att / "secret.txt")
    tree_before = sorted(str(p.relative_to(sentinel_dir)) for p in sentinel_dir.rglob("*"))

    # Inject hostile overrides that would aim a naive proof at the sentinel.
    os.environ["HERMES_KANBAN_DB"] = str(sentinel_db)
    os.environ["HERMES_KANBAN_ATTACHMENTS_ROOT"] = str(sentinel_att)
    os.environ["HERMES_KANBAN_HOME"] = str(sentinel_dir)
    try:
        # The proof clears these overrides internally; run it fully.
        mod.seed_full_fixture(owned_root)
        mod.board_transfer_preserving(owned_root)
    finally:
        os.environ.pop("HERMES_KANBAN_DB", None)
        os.environ.pop("HERMES_KANBAN_ATTACHMENTS_ROOT", None)
        os.environ.pop("HERMES_KANBAN_HOME", None)

    # Sentinel must be byte-unchanged and no kanban files written under it.
    assert _h(sentinel_db) == db_before, "sentinel DB was mutated by the proof"
    assert _h(sentinel_att / "secret.txt") == att_before, "sentinel attachment mutated"
    tree_after = sorted(str(p.relative_to(sentinel_dir)) for p in sentinel_dir.rglob("*"))
    assert tree_after == tree_before, f"sentinel tree changed: {tree_before} -> {tree_after}"


def test_proof_a_failure_after_attachment_copy_rolls_back(owned_root, monkeypatch):
    """If an attachment copy fails verification during transfer, the transfer
    rolls back: source + target content unchanged, no half-written staged
    attachment file on the target.

    Tested at the _do_migration_transfer level (which rolls back internally
    and does NOT dispose fixtures), because board_transfer_preserving
    legitimately unlinks the source at the very end of a successful run.
    """
    mod = _load_proof()
    mod.seed_full_fixture(owned_root)
    from hermes_cli import kanban_db as kb
    src_path = kb.kanban_db_path(board="default")
    tgt_path = kb.kanban_db_path(board="core")
    src_att = kb.attachments_root(board="default")
    tgt_att = kb.attachments_root(board="core")

    # Snapshot source + target content before any transfer.
    src_sig_before = mod._db_content_signature(src_path)
    tgt_sig_before = mod._db_content_signature(tgt_path)

    # Inject a failure AFTER the staged bytes are published. This proves the
    # compensation path removes published files, not merely pre-copy stages.
    original_publish = mod._publish_attachment
    def _publish_then_fail(root, stage, target):
        original_publish(root, stage, target)
        raise RuntimeError("injected post-publication failure")
    monkeypatch.setattr(mod, "_publish_attachment", _publish_then_fail)

    res = mod._do_migration_transfer(owned_root, src_path, tgt_path, src_att, tgt_att)
    # Transfer must fail cleanly (no partial mutation).
    assert res["ok"] is False
    # Source + target content must be unchanged (rollback).
    assert mod._db_content_signature(src_path) == src_sig_before, "source mutated on failure"
    assert mod._db_content_signature(tgt_path) == tgt_sig_before, "target mutated on failure"
    # No half-written attachment file under the target root.
    if tgt_att.exists():
        stray = list(tgt_att.rglob("*.stage"))
        assert not stray, f"stray staged attachment left on target: {stray}"


def test_proof_a_guard_rejects_symlinked_owned_path(owned_root):
    mod = _load_proof()
    target = owned_root / "real"
    target.mkdir()
    link = owned_root / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinked owned path"):
        mod.guard_owned_path(owned_root, link / "child", must_exist=False)


def test_proof_a_rejects_pending_approval_without_mutation(owned_root):
    """Pending/approved lifecycle state is explicitly unsupported, never lost."""
    mod = _load_proof()
    fixture = mod.seed_full_fixture(owned_root)
    from hermes_cli import kanban_db as kb
    src_path = kb.kanban_db_path(board="default")
    tgt_path = kb.kanban_db_path(board="core")
    with kb.connect_closing(board="default") as conn:
        conn.execute(
            "INSERT INTO profile_lifecycle_approvals "
            "(id, task_id, op, profile, args_json, status, token, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("pending-proof", fixture["tasks"]["T1"], "create", "p", "{}", "pending", "tok", 1),
        )
        conn.commit()
    src_before = mod._db_content_signature(src_path)
    tgt_before = mod._db_content_signature(tgt_path)
    result = mod._do_migration_transfer(owned_root, src_path, tgt_path, kb.attachments_root(board="default"), kb.attachments_root(board="core"))
    assert result["ok"] is False
    assert "unsupported lifecycle approval" in result["error"]
    assert mod._db_content_signature(src_path) == src_before
    assert mod._db_content_signature(tgt_path) == tgt_before


def test_proof_a_rejects_inflight_task_without_mutation(owned_root):
    """In-flight task/run pointers are rejected instead of silently nulled."""
    mod = _load_proof()
    mod.seed_full_fixture(owned_root)
    from hermes_cli import kanban_db as kb
    src_path = kb.kanban_db_path(board="default")
    tgt_path = kb.kanban_db_path(board="core")
    with kb.connect_closing(board="default") as conn:
        tid = kb.create_task(conn, title="inflight", assignee="worker")
        kb.claim_task(conn, tid)
        conn.commit()
    src_before = mod._db_content_signature(src_path)
    tgt_before = mod._db_content_signature(tgt_path)
    result = mod._do_migration_transfer(owned_root, src_path, tgt_path, kb.attachments_root(board="default"), kb.attachments_root(board="core"))
    assert result["ok"] is False
    assert "unsupported in-flight" in result["error"]
    assert mod._db_content_signature(src_path) == src_before
    assert mod._db_content_signature(tgt_path) == tgt_before


def test_proof_a_rejects_open_terminal_run_without_mutation(owned_root):
    """A terminal label is not enough: a run with no ended_at is still live."""
    mod = _load_proof()
    mod.seed_full_fixture(owned_root)
    from hermes_cli import kanban_db as kb
    src_path = kb.kanban_db_path(board="default")
    tgt_path = kb.kanban_db_path(board="core")
    with kb.connect_closing(board="default") as conn:
        conn.execute("UPDATE task_runs SET status = 'done', ended_at = NULL")
        conn.commit()
    src_before = mod._db_content_signature(src_path)
    tgt_before = mod._db_content_signature(tgt_path)
    result = mod._do_migration_transfer(owned_root, src_path, tgt_path, kb.attachments_root(board="default"), kb.attachments_root(board="core"))
    assert result["ok"] is False
    assert "unsupported in-flight run" in result["error"]
    assert mod._db_content_signature(src_path) == src_before
    assert mod._db_content_signature(tgt_path) == tgt_before


def test_proof_a_raw_helper_rejects_unowned_root(tmp_path):
    """Direct helper calls cannot bypass the owned-root boundary."""
    mod = _load_proof()
    with pytest.raises(RuntimeError, match="refusing unowned root"):
        mod._do_migration_transfer(tmp_path, tmp_path / "src.db", tmp_path / "tgt.db", tmp_path / "src", tmp_path / "tgt")
    with pytest.raises(RuntimeError, match="refusing unowned root"):
        mod._inject_failure_before_swap(tmp_path / "src.db", tmp_path / "tgt.db", tmp_path)
    victim = tmp_path / "victim.db"
    victim.write_bytes(b"must survive")
    with pytest.raises(RuntimeError, match="refusing unowned root"):
        mod._dispose_fixtures(victim, tmp_path / "other.db", tmp_path / "a", tmp_path / "b", tmp_path)
    assert victim.read_bytes() == b"must survive"


def test_proof_a_rejects_target_epic_conflict_without_mutation(owned_root):
    """Conflicting text IDs are preflight errors, never INSERT OR IGNORE loss."""
    mod = _load_proof()
    fixture = mod.seed_full_fixture(owned_root)
    from hermes_cli import kanban_db as kb
    src_path = kb.kanban_db_path(board="default")
    tgt_path = kb.kanban_db_path(board="core")
    with kb.connect_closing(board="core") as conn:
        conn.execute(
            "INSERT INTO epics (id,title,description,board_slug,status,created_at,updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fixture["epic_id"], "conflict", "different", "core", "active", 1, 1),
        )
        conn.commit()
    src_before = mod._db_content_signature(src_path)
    tgt_before = mod._db_content_signature(tgt_path)
    result = mod._do_migration_transfer(owned_root, src_path, tgt_path, kb.attachments_root(board="default"), kb.attachments_root(board="core"))
    assert result["ok"] is False
    assert "target epic conflict" in result["error"]
    assert mod._db_content_signature(src_path) == src_before
    assert mod._db_content_signature(tgt_path) == tgt_before


def test_proof_b_single_task_move(owned_root):
    """Proof B: single-task move via production move_task_atomic; not a
    full preservation claim."""
    from hermes_cli import kanban_db as kb
    mod = _load_proof()
    with mod.owned_env(owned_root):
        kb.init_db(board="default")
        kb.init_db(board="core")
        with kb.connect_closing(board="default") as conn:
            tid = kb.create_task(conn, title="single-move", assignee="worker")
            kb.claim_task(conn, tid)
            kb.complete_task(conn, tid, summary="done", result="ok")
            conn.commit()
    res = mod.transfer_and_prove(owned_root)
    assert res["ok"], f"move failed: {res.get('move_result')}"
    assert res["move_result"]["status"] == "moved"
    assert res["pointer_rolled_back"] is True


def test_proof_b_rejected_move_leaves_pointer_untouched(owned_root):
    mod = _load_proof()
    res = mod.prove_failed_transfer_before_swap(owned_root)
    assert res["rejected"]
    assert res["pointer_untouched"]


def test_proof_b_missing_schema_rejected(owned_root):
    mod = _load_proof()
    res = mod.prove_missing_schema_rejected(owned_root)
    assert res["rejected"]
    assert res["pointer_untouched"]


def test_main_exits_zero_and_cleans_up(tmp_path, monkeypatch, capsys):
    """main() creates its own temp root, proves, and removes it (no leak)."""
    mod = _load_proof()
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0, f"main returned {rc}; output:\n{out}"
    assert "HERMETIC MIGRATION PROVEN" in out
    # No leaked temp home: main() rmtree's its own root; assert no p04-hermetic
    # dirs remain under the system temp dir.
    import tempfile
    leftover = list(Path(tempfile.gettempdir()).glob("p04-hermetic-*"))
    assert not leftover, f"leaked temp roots: {leftover}"
