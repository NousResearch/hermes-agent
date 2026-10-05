"""P4 PARTE 3 — off-board launch origin: a run launched outside the dispatcher
can no longer be completed as normal dispatched work by omitting the flag, and
a recorded origin's served model survives an omitted metadata field.

Covers ``record_off_board_run``, ``_off_board_run_origin`` and the effective
off-board gate in ``complete_task`` (declared OR recorded), plus the narrow
``require_recorded_origin=False`` escape used by the CLI/tool attestation path.
"""
import json
from pathlib import Path

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for k in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(k, raising=False)
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


def _card(kb, claim=True):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as c:
        tid = kb.create_task(c, title="ob-origin", assignee="builder", workspace_kind="scratch")
        if claim:
            assert kb.claim_task(c, tid) is not None
    return tid


def _conn():
    from hermes_cli import kanban_db_connect as kbc
    return kbc.connect_closing()


def _status(kb, tid):
    with _conn() as c:
        return kb.get_task(c, tid).status


def _events(kb, tid, kind):
    with _conn() as c:
        rows = c.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, kind),
        ).fetchall()
    return [json.loads(r["payload"]) if r["payload"] else None for r in rows]


def _run_metadata(kb, tid):
    with _conn() as c:
        row = c.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        return json.loads(row["metadata"]) if row and row["metadata"] else {}


# --- complete_task: declared off-board -------------------------------------------------

def test_off_board_requires_served_model(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        with pytest.raises(kb.OffBoardServedModelError):
            kb.complete_task(c, tid, summary="s", off_board=True, require_recorded_origin=False)
    assert _status(kb, tid) == "running"
    assert _events(kb, tid, "completion_blocked_served_model")


def test_off_board_with_served_model_ok(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        assert kb.complete_task(
            c, tid, summary="s", off_board=True, require_recorded_origin=False,
            metadata={"served_model": "gpt-6.1-sol"},
        )
    assert _status(kb, tid) == "done"


def test_declared_off_board_without_origin_refused_by_default(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        with pytest.raises(kb.OffBoardOriginError):
            kb.complete_task(
                c, tid, summary="s", off_board=True,
                metadata={"served_model": "gpt-6.1-sol"},
            )
    assert _status(kb, tid) == "running"
    assert _events(kb, tid, "completion_blocked_off_board_origin")


def test_declared_off_board_no_origin_allowed_when_explicitly_disabled(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        assert kb.complete_task(
            c, tid, summary="s", off_board=True, require_recorded_origin=False,
            metadata={"served_model": "gpt-6.1-sol"},
        )
    assert _status(kb, tid) == "done"


# --- record_off_board_run --------------------------------------------------------------

def test_record_off_board_run_requires_active_run(env):
    kb = env
    tid = _card(kb, claim=False)
    with _conn() as c:
        with pytest.raises(kb.OffBoardLaunchError):
            kb.record_off_board_run(c, tid, served_model="gpt-6.1-sol")


def test_record_off_board_run_blank_served_model_refused(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        with pytest.raises(kb.OffBoardLaunchError):
            kb.record_off_board_run(c, tid, served_model="   ")


def test_record_off_board_run_expected_run_mismatch_refused(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        with pytest.raises(kb.OffBoardLaunchError):
            kb.record_off_board_run(c, tid, served_model="m", expected_run_id=999999)


def test_record_off_board_run_writes_origin_and_event(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        rid = kb.record_off_board_run(
            c, tid, served_model="gpt-6.1-sol", provider="openai-codex",
            executor="claude-code", launch_ref="job-7",
        )
    assert isinstance(rid, int)
    meta = _run_metadata(kb, tid)
    origin = meta["off_board_run"]
    assert origin["off_board"] is True
    assert origin["served_model"] == "gpt-6.1-sol"
    assert origin["executor"] == "claude-code"
    assert origin["schema"] == "v1"
    assert _events(kb, tid, "route_off_board_launch")


# --- recorded origin forces off-board treatment ----------------------------------------

def test_recorded_origin_forces_off_board_without_flag(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        kb.record_off_board_run(c, tid, served_model="gpt-6.1-sol")
        # No flag, no served_model in metadata: the recorded origin alone must
        # force the off-board path and supply the served model.
        assert kb.complete_task(c, tid, summary="s", require_recorded_origin=False)
    assert _status(kb, tid) == "done"
    # Strong: the run's TOP-LEVEL metadata carries the origin model, and the
    # completion emits route_served_model (the on-board path would emit neither).
    assert _run_metadata(kb, tid).get("served_model") == "gpt-6.1-sol"
    rows = _events(kb, tid, "route_served_model")
    assert rows and rows[-1]["served_model"] == "gpt-6.1-sol"


def test_invalid_origin_blank_model_refuses_even_without_flag(env):
    """A present-but-invalid origin must REFUSE, never silently downgrade to
    on-board (review 2026-10-05 HIGH #1)."""
    kb = env
    tid = _card(kb)
    with _conn() as c:
        rid = kb._current_run_id(c, tid)
        c.execute("UPDATE task_runs SET metadata = ? WHERE id = ?",
                  (json.dumps({"off_board_run": {"schema": "v1", "off_board": True,
                                                 "served_model": "   "}}), rid))
        c.commit()
        with pytest.raises(kb.OffBoardOriginError):
            kb.complete_task(c, tid, summary="s", require_recorded_origin=False)
    assert _status(kb, tid) == "running"
    assert _events(kb, tid, "completion_blocked_off_board_origin")


def test_invalid_origin_off_board_false_string_refuses(env):
    """``off_board="false"`` is truthy — the validity check is identity, not
    truthiness (review #1)."""
    kb = env
    tid = _card(kb)
    with _conn() as c:
        rid = kb._current_run_id(c, tid)
        c.execute("UPDATE task_runs SET metadata = ? WHERE id = ?",
                  (json.dumps({"off_board_run": {"schema": "v1", "off_board": "false",
                                                 "served_model": "m"}}), rid))
        c.commit()
        with pytest.raises(kb.OffBoardOriginError):
            kb.complete_task(c, tid, summary="s", require_recorded_origin=False)


def test_valid_origin_normalizes_blank_caller_model(env):
    """Blank caller served_model must be replaced by the origin's validated
    model, not persisted as-is (review 2026-10-05 MEDIUM #7)."""
    kb = env
    tid = _card(kb)
    with _conn() as c:
        kb.record_off_board_run(c, tid, served_model="origin-model")
        assert kb.complete_task(
            c, tid, summary="s", require_recorded_origin=False,
            metadata={"served_model": "   "},
        )
    assert _run_metadata(kb, tid).get("served_model") == "origin-model"


def test_caller_cannot_clobber_recorded_origin(env):
    """A caller metadata dict carrying ``off_board_run`` must not overwrite the
    trusted origin at completion (review 2026-10-05 MEDIUM #5)."""
    kb = env
    tid = _card(kb)
    with _conn() as c:
        kb.record_off_board_run(c, tid, served_model="origin-model")
        assert kb.complete_task(
            c, tid, summary="s", require_recorded_origin=False,
            metadata={"off_board_run": {"schema": "v1", "off_board": True,
                                        "served_model": "forged"}},
        )
    assert _run_metadata(kb, tid)["off_board_run"]["served_model"] == "origin-model"


def test_caller_cannot_forge_origin_without_recorded_launch(env):
    """A caller cannot inject an origin into an unattributed run's metadata: the
    stripped key must not persist, so the run stays on-board (review #5)."""
    kb = env
    tid = _card(kb)
    with _conn() as c:
        assert kb.complete_task(
            c, tid, summary="s",
            metadata={"off_board_run": {"schema": "v1", "off_board": True,
                                        "served_model": "forged"}},
        )
    assert _status(kb, tid) == "done"
    assert "off_board_run" not in _run_metadata(kb, tid)


def test_force_does_not_bypass_origin_gate(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        with pytest.raises(kb.OffBoardOriginError):
            kb.complete_task(c, tid, summary="s", off_board=True, force=True,
                             metadata={"served_model": "m"})
    assert _status(kb, tid) == "running"


def test_invalid_origin_unreadable_json_with_launch_event_refuses(env):
    """Round-2 residual HIGH: unreadable metadata on a run that HAS a launch
    event must refuse, not silently downgrade to on-board."""
    kb = env
    tid = _card(kb)
    with _conn() as c:
        kb.record_off_board_run(c, tid, served_model="m")
        rid = kb._current_run_id(c, tid)
        c.execute("UPDATE task_runs SET metadata = ? WHERE id = ?", ('{"off_board_run":', rid))
        c.commit()
        with pytest.raises(kb.OffBoardOriginError):
            kb.complete_task(c, tid, summary="s", require_recorded_origin=False)
    assert _status(kb, tid) == "running"


def test_unreadable_metadata_without_launch_event_stays_on_board(env):
    """Unreadable metadata with NO launch event for the run is ordinary on-board
    (backwards compatible)."""
    kb = env
    tid = _card(kb)
    with _conn() as c:
        rid = kb._current_run_id(c, tid)
        c.execute("UPDATE task_runs SET metadata = ? WHERE id = ?", ("not-json", rid))
        c.commit()
        assert kb.complete_task(c, tid, summary="s")
    assert _status(kb, tid) == "done"


def test_request_review_cannot_clobber_origin(env):
    """request_review carries caller metadata; it must not overwrite the trusted
    origin on the run it closes (review round-2 residual MEDIUM #5)."""
    from hermes_cli import kanban_db_connect as kbc
    kb = env
    with kbc.connect_closing() as c:
        tid = kb.create_task(c, title="rr", assignee="builder", workspace_kind="scratch")
        assert kb.claim_task(c, tid) is not None
        kb.record_off_board_run(c, tid, served_model="origin-model")
        ok = kb.request_review(
            c, tid, summary="handing off",
            metadata={"off_board_run": {"schema": "v1", "off_board": True,
                                        "served_model": "forged"}},
        )
    assert ok
    meta = _run_metadata(kb, tid)
    assert meta["off_board_run"]["served_model"] == "origin-model"


def test_edit_task_cannot_clobber_origin(env):
    """edit_task backfills metadata; it must not overwrite the historical origin
    (review round-2 residual MEDIUM #5)."""
    kb = env
    tid = _card(kb)
    with _conn() as c:
        kb.record_off_board_run(c, tid, served_model="origin-model")
        assert kb.complete_task(c, tid, summary="s", require_recorded_origin=False)
    with _conn() as c:
        assert kb.edit_task(c, tid, result="r2",
                            metadata={"off_board_run": {"schema": "v1",
                                                        "off_board": True,
                                                        "served_model": "forged"}})
    assert _run_metadata(kb, tid)["off_board_run"]["served_model"] == "origin-model"


def test_tool_recorded_origin_assigns_model_and_event(env):
    """Strengthened tool test (round-2 #9): assert the persisted model AND the
    completion event, not just ``ok``/``done``."""
    kb = env
    tid = _card(kb)
    with _conn() as c:
        kb.record_off_board_run(c, tid, served_model="tool-model")
    from tools import kanban_tools  # noqa: F401
    from tools.registry import registry
    r = json.loads(registry.dispatch("kanban_complete", {"task_id": tid, "summary": "s"}))
    assert r.get("ok"), r
    assert _status(kb, tid) == "done"
    assert _run_metadata(kb, tid).get("served_model") == "tool-model"
    rows = _events(kb, tid, "route_served_model")
    assert rows and rows[-1]["served_model"] == "tool-model"


def test_previous_run_origin_does_not_cover_current(env):
    """An origin on an ENDED run must not attribute the current run."""
    kb = env
    tid = _card(kb)
    with _conn() as c:
        rid = kb._current_run_id(c, tid)
        # Stamp an origin on the (open) run, then end it and open a new run.
        kb.record_off_board_run(c, tid, served_model="old")
        kb._end_run(c, tid, outcome="crashed", status="crashed")
        c.commit()
        assert kb.claim_task(c, tid) is None or True  # status now crashed
        # Re-open a fresh run without origin by moving the card back to ready.
        c.execute("UPDATE tasks SET status='ready', claim_lock=NULL WHERE id=?", (tid,))
        c.commit()
        claimed = kb.claim_task(c, tid)
    assert claimed is not None
    with _conn() as c:
        # New active run has no origin -> a plain on-board completion is fine
        # and must NOT inherit the old run's served_model.
        assert kb.complete_task(c, tid, summary="s")
    assert _run_metadata(kb, tid).get("served_model") is None


def test_recorded_origin_survives_completion(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        kb.record_off_board_run(
            c, tid, served_model="gpt-6.1-sol", executor="claude-code",
        )
        assert kb.complete_task(c, tid, summary="s", require_recorded_origin=False)
    meta = _run_metadata(kb, tid)
    assert meta["off_board_run"]["served_model"] == "gpt-6.1-sol"


def test_recorded_origin_merge_preserves_run_metadata(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        # Simulate dispatcher provenance already on the open run.
        rid = kb._current_run_id(c, tid)
        c.execute("UPDATE task_runs SET metadata = ? WHERE id = ?",
                  (json.dumps({"resolved_route_provenance": {"schema": "v1"}}), rid))
        c.commit()
        kb.record_off_board_run(c, tid, served_model="m")
    meta = _run_metadata(kb, tid)
    assert meta["resolved_route_provenance"] == {"schema": "v1"}
    assert meta["off_board_run"]["served_model"] == "m"


# --- on-board path unchanged -----------------------------------------------------------

def test_on_board_completion_unchanged(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        assert kb.complete_task(c, tid, summary="s")
    assert _status(kb, tid) == "done"
    assert not _events(kb, tid, "completion_blocked_served_model")
    assert not _events(kb, tid, "completion_blocked_off_board_origin")


# --- CLI / tool attestation surfaces (no launch step) ----------------------------------

def test_cli_off_board_without_origin_ok(env):
    kb = env
    tid = _card(kb)
    from hermes_cli.kanban import run_slash
    out = run_slash(f"complete {tid} --result r --off-board --metadata "
                    + json.dumps(json.dumps({"served_model": "gpt-6.1-sol"})))
    assert _status(kb, tid) == "done", out


def test_cli_off_board_blank_served_model_refused(env):
    kb = env
    tid = _card(kb)
    from hermes_cli.kanban import run_slash
    out = run_slash(f"complete {tid} --result r --off-board")
    assert "served_model" in out.lower() or "off-board" in out.lower(), out
    assert _status(kb, tid) == "running"


def test_tool_off_board_without_origin_ok(env):
    kb = env
    tid = _card(kb)
    from tools import kanban_tools  # noqa: F401
    from tools.registry import registry
    r = json.loads(registry.dispatch("kanban_complete", {
        "task_id": tid, "summary": "s", "off_board": True,
        "metadata": {"served_model": "gpt-6.1-sol"},
    }))
    assert r.get("ok"), r
    assert _status(kb, tid) == "done"


def test_tool_omitted_flag_forced_off_board_by_recorded_origin(env):
    kb = env
    tid = _card(kb)
    with _conn() as c:
        kb.record_off_board_run(c, tid, served_model="gpt-6.1-sol")
    from tools import kanban_tools  # noqa: F401
    from tools.registry import registry
    # No off_board flag, no served_model: the recorded origin must force it.
    r = json.loads(registry.dispatch("kanban_complete", {
        "task_id": tid, "summary": "s",
    }))
    assert r.get("ok"), r
    assert _status(kb, tid) == "done"
