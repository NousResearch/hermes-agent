"""P4 route provenance (audit 2026-10-03 F-05 / PARTE III P4).

The dispatcher resolves the model route at spawn time: a per-task
``model_override``/``provider_override`` (policy-checked by
``assert_model_route_allowed``) or the assignee profile's own default model
from its ``config.yaml``. Until P4 that decision was invisible afterwards —
the only record was the ad-hoc ``kanban_routing_dispatches.jsonl`` written by
an older watchdog, stalled at 2026-08-14 — so NF-11 could only snapshot 21/662
runs by hand.

Schema v1 (the same record BOTH destinations carry):

  task_runs.metadata["resolved_route_provenance"] == {
      "schema": "v1",
      "dispatch_ts": <unix int>,
      "selected_model": <str or None>,
      "selected_provider": <str or None>,
      "profile_default_slug": <str or None>,
      "policy_source": "model_override" | "profile_default",
      "dispatch_lock_held": True,
  }

plus one raw line appended to ``<kanban_home>/runtime/kanban_routing_dispatches.jsonl``
per dispatched spawn. Both writes are DISPATCHER-owned, best-effort for the
JSONL (a provenance-record failure must never break a dispatch), durable for
the run-metadata write (same board DB, same write txn discipline). Off-board
completions (no dispatcher spawn) stay out of scope here: they must carry an
explicit ``served_model`` (next stage of P4, not this subagent).

Sandbox: every fixture pins HERMES_KANBAN_HOME at tmp_path (write-guard
fence stays anchored to the real root and never lists the sandbox DB) and
drops the delegated-child marker (sibling P2 sandbox convention), so
``init_db`` may open the throwaway DB in write mode. Boards reais READ-ONLY.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME/kanban home with an empty kanban DB."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "2000")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _patch_profile_default_model(monkeypatch, model, provider):
    """Point the provenance resolution at a deterministic profile default."""
    from hermes_cli import profiles as profiles_mod
    monkeypatch.setattr(
        profiles_mod, "_read_config_model", lambda *a, **k: (model, provider),
    )


def _run_metadata(kanban_home, monkeypatch, run_id):
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()
    assert row is not None and row["metadata"], "run metadata must carry the provenance"
    return json.loads(row["metadata"])


def _jsonl_lines(kanban_home):
    from hermes_cli import kanban_db as kb

    jsonl_path = kb.kanban_home() / "runtime" / "kanban_routing_dispatches.jsonl"
    assert jsonl_path.exists(), jsonl_path
    return [
        json.loads(ln)
        for ln in jsonl_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def _dispatch_one(kanban_home, all_assignees_spawnable, monkeypatch, **task_kwargs):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="route probe", assignee="alice", **task_kwargs)
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
        assert res.spawned, res
        run_id = kb.get_task(conn, tid).current_run_id
    assert run_id
    return tid, run_id


def test_dispatch_run_metadata_carries_provenance_v1(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A dispatcher spawn writes resolved_route_provenance (schema v1) onto the
    run row and an appended JSONL line — profile_default policy."""
    from hermes_cli import kanban_db as kb

    _patch_profile_default_model(monkeypatch, "glm-5.3-flash", "openai-codex")
    tid, run_id = _dispatch_one(kanban_home, all_assignees_spawnable, monkeypatch)

    prov = _run_metadata(kanban_home, monkeypatch, run_id)["resolved_route_provenance"]
    assert prov["schema"] == "v1"
    assert prov["policy_source"] == "profile_default"
    assert prov["selected_model"] == "glm-5.3-flash"
    assert prov["selected_provider"] == "openai-codex"
    assert prov["profile_default_slug"] == "glm-5.3-flash"
    assert prov["dispatch_lock_held"] is True
    assert isinstance(prov["dispatch_ts"], int) and 0 < prov["dispatch_ts"] <= time.time() + 5

    match = [ln for ln in _jsonl_lines(kanban_home) if ln.get("task_id") == tid]
    assert len(match) == 1, match
    line = match[0]
    assert line["schema"] == "v1"
    assert line["policy_source"] == "profile_default"
    assert line["selected_model"] == "glm-5.3-flash"
    assert line["selected_provider"] == "openai-codex"
    assert line["run_id"] == run_id
    assert line["dispatch_lock_held"] is True
    assert line["dispatch_ts"] == prov["dispatch_ts"]


def test_dispatch_model_override_wins_policy_source(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """model_override is the policy-checked route (assert_model_route_allowed
    inside _worker_argv); provenance must name it with
    policy_source == 'model_override' and keep the profile default visible
    under profile_default_slug."""
    _patch_profile_default_model(monkeypatch, "glm-5.3-flash", "openai-codex")
    tid, run_id = _dispatch_one(
        kanban_home, all_assignees_spawnable, monkeypatch,
        model_override="qwen-max", provider_override="openrouter",
    )
    prov = _run_metadata(kanban_home, monkeypatch, run_id)["resolved_route_provenance"]
    assert prov["policy_source"] == "model_override"
    assert prov["selected_model"] == "qwen-max"
    assert prov["selected_provider"] == "openrouter"
    assert prov["profile_default_slug"] == "glm-5.3-flash"

    match = [ln for ln in _jsonl_lines(kanban_home) if ln.get("task_id") == tid]
    assert match[0]["policy_source"] == "model_override"


def test_jsonl_append_never_breaks_dispatch(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The JSONL is best-effort: a failing append must not fail the spawn NOR
    the run-metadata provenance write."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(_kbd(), "_append_route_jsonl", _boom)
    with kbc_connect_conn() as conn:
        tid = kb.create_task(conn, title="broken jsonl", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    assert res.spawned, "spawn survives the JSONL failure"
    run_id = kb.get_task(conn, tid).current_run_id
    prov = _run_metadata(kanban_home, monkeypatch, run_id)["resolved_route_provenance"]
    assert prov["schema"] == "v1"
    match = [ln for ln in _jsonl_lines_allow_missing(kanban_home) if ln.get("task_id") == tid]
    assert match == []


def _kbd():
    from hermes_cli import kanban_db_dispatch as kbd

    return kbd


def kbc_connect_conn():
    from hermes_cli import kanban_db_connect as kbc

    return kbc.connect()


def _jsonl_lines_allow_missing(kanban_home):
    from hermes_cli import kanban_db as kb

    jsonl_path = kb.kanban_home() / "runtime" / "kanban_routing_dispatches.jsonl"
    if not jsonl_path.exists():
        return []
    return [
        json.loads(ln)
        for ln in jsonl_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def test_provenance_resolution_failure_never_breaks_dispatch(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Provenance is observability, not a gate: a resolver crash (no profile,
    config boom) must degrade to selected_model=None with policy_source
    still stamped, and the dispatch must succeed."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import profiles as profiles_mod

    def _boom(*a, **k):
        raise RuntimeError("profile gone")

    monkeypatch.setattr(profiles_mod, "_read_config_model", _boom)
    with kbc_connect_conn() as conn:
        tid = kb.create_task(conn, title="resolver boom", assignee="alice")
        res = _kbd().dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    assert res.spawned, res
    run_id = kb.get_task(conn, tid).current_run_id
    prov = _run_metadata(kanban_home, monkeypatch, run_id)["resolved_route_provenance"]
    assert prov["schema"] == "v1"
    assert prov["policy_source"] == "profile_default"
    assert prov["selected_model"] is None


class TestS6LifecycleProvenanceSurvival:
    """Review 6 (2026-10-04) S6-01: the dispatcher's provenance is written on
    the OPEN run; every closure used to REPLACE run metadata — completion,
    spawn_failed and reclaim wiped the record, breaking the P4 acceptance
    ("audit query em 1 linha" counted themed runs on task_runs). Closure now
    MERGES: closer fields win on collision, dispatcher-controlled keys
    survive."""

    def test_provenance_survives_completion(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        _patch_profile_default_model(monkeypatch, "glm-5.3-flash", "openai-codex")
        tid, run_id = _dispatch_one(kanban_home, all_assignees_spawnable, monkeypatch)
        with kbc.connect() as conn:
            claimed = kb.get_task(conn, tid)
            ok = kb.complete_task(
                conn, tid, result="r", summary="s",
                metadata={"artifacts": []}, expected_run_id=claimed.current_run_id,
            )
        assert ok is True
        meta = _run_metadata(kanban_home, monkeypatch, run_id)
        assert meta["resolved_route_provenance"]["schema"] == "v1"
        assert meta["artifacts"] == []

    def test_provenance_survives_spawn_failed(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        _patch_profile_default_model(monkeypatch, "glm-5.3-flash", "openai-codex")
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="s6 spawnfail", assignee="alice")

            def boom(task, workspace, board=None):
                raise RuntimeError("host refused")

            res = kbd.dispatch_once(conn, spawn_fn=boom, failure_limit=999)
            # The spawn_failed closure already cleared the task's
            # current_run_id — the run row is found via task_runs.
            run_row = conn.execute(
                "SELECT id FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
            ).fetchone()
        assert not res.spawned
        run_id = run_row["id"]
        meta = _run_metadata(kanban_home, monkeypatch, run_id)
        prov = meta["resolved_route_provenance"]
        assert prov["schema"] == "v1"
        assert prov["selected_model"] == "glm-5.3-flash"
        assert "failures" in meta  # closer's own payload fields still present

    def test_provenance_survives_stale_reclaim(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        _patch_profile_default_model(monkeypatch, "glm-5.3-flash", "openai-codex")
        tid, run_id = _dispatch_one_no_pid(kanban_home, all_assignees_spawnable, monkeypatch)
        with kbc.connect() as conn:
            conn.execute(
                "UPDATE tasks SET claim_expires = ? WHERE id = ?",
                (int(time.time()) - 100, tid),
            )
            conn.commit()
            kb.release_stale_claims(conn, failure_limit=99)
        meta = _run_metadata(kanban_home, monkeypatch, run_id)
        prov = meta["resolved_route_provenance"]
        assert prov["schema"] == "v1"

    def test_edit_task_metadata_keeps_provenance(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        """A post-completion edit REPLACES prose metadata but must keep the
        dispatcher-controlled provenance key."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        _patch_profile_default_model(monkeypatch, "glm-5.3-flash", "openai-codex")
        tid, run_id = _dispatch_one(kanban_home, all_assignees_spawnable, monkeypatch)
        with kbc.connect() as conn:
            claimed = kb.get_task(conn, tid)
            kb.complete_task(
                conn, tid, result="r", summary="s", expected_run_id=claimed.current_run_id,
            )
            assert kb.edit_task(
                conn, tid, result="edited result", summary="edited",
                metadata={"artifacts": ["x"]},
            )
        meta = _run_metadata(kanban_home, monkeypatch, run_id)
        assert meta["resolved_route_provenance"]["schema"] == "v1"
        assert meta["artifacts"] == ["x"]

    def test_closer_fields_win_on_collision(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        """The merge is caller-preferred: a closure key colliding with a
        post-dispatch writer's key keeps the CLOSER's value."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        _patch_profile_default_model(monkeypatch, "glm-5.3-flash", "openai-codex")
        tid, run_id = _dispatch_one(kanban_home, all_assignees_spawnable, monkeypatch)
        with kbc.connect() as conn:
            # A concurrent post-dispatch writer MERGES a colliding key onto
            # the open run (the way every system writer does — direct REPLACE
            # is a rogue manual edit outside the merge contract and would wipe
            # the provenance key it never wrote).
            with kb.write_txn(conn):
                prow = conn.execute(
                    "SELECT metadata FROM task_runs WHERE id = ?", (run_id,)
                ).fetchone()
                prior = json.loads(prow["metadata"]) if prow["metadata"] else {}
                prior["failures"] = "staged-by-writer"
                conn.execute(
                    "UPDATE task_runs SET metadata = ? WHERE id = ?",
                    (json.dumps(prior), run_id),
                )
            kb.complete_task(
                conn, tid, result="r", summary="s", metadata={"failures": "from-closer"},
                expected_run_id=kb.get_task(conn, tid).current_run_id,
            )
        meta = _run_metadata(kanban_home, monkeypatch, run_id)
        assert meta["failures"] == "from-closer"
        assert meta["resolved_route_provenance"]["schema"] == "v1"


def _dispatch_one_no_pid(kanban_home, all_assignees_spawnable, monkeypatch):
    """A dispatch whose spawn_fn returns 0 (no PID recorded) — used where the
    closure would otherwise walk live-pid kill paths on the fake PID."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="route probe no-pid", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0)
        assert res.spawned, res
        run_id = kb.get_task(conn, tid).current_run_id
    assert run_id
    return tid, run_id
