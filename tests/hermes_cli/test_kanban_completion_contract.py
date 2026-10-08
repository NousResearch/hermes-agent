"""P3 completion-contract gate (audit 2026-10-04 plan v1.1.1).

A card whose ``tasks.completion_contract`` is a JSON dict of requirements
(``{"required_artifacts": [...]}``) must not complete unless the declared
artifacts exist on disk inside managed scratch storage — or outside it, in
which case the completion records ``external_artifact_recorded`` evidence.
Missing artifacts refuse completion with ``ContractCompletionError`` after an
auditable ``completion_blocked_contract`` event.

Backwards compatibility pinned here:
  * ``''`` / ``None`` / ``'local-only'`` (all 16 live rows today) — no extra
    enforcement; ``complete_task`` behaves exactly as before P3.
  * a contract that is not parseable as a JSON dict fails OPEN with an audible
    ``contract_unparseable`` event (legacy contracts are strings, not dicts;
    failing closed would strand every pre-P3 card).

Also pinned (P3 item 2): ``_persist_scratch_completion_artifacts`` records an
auditable event for every artifact it touches — ``external_artifact_missing``
when a declared OUTSIDE-scratch path does not exist (previously: silent
string-only persistence), ``external_artifact_recorded`` when it does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The rc=0 violation regression test claims a task and expects an
    # immediate ``detect_crashed_workers`` reclaim: disable the 30 s
    # multi-dispatcher crash grace here like the other crash tests do.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _set_contract(conn, task_id: str, contract) -> None:
    """Point ``tasks.completion_contract`` at a raw value (JSON dict / str / '')."""
    value = contract if isinstance(contract, str) else json.dumps(contract)
    conn.execute(
        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
        (value, task_id),
    )
    conn.commit()


def _scratch_task(conn, kanban_home, title: str):
    """A task with a real managed-scratch workspace directory."""
    t = kb.create_task(conn, title=title)
    task = kb.get_task(conn, t)
    ws = kbw.resolve_workspace(task)
    kbw.set_workspace_path(conn, t, ws)
    return t, ws


def _event_kinds(conn, task_id: str) -> list[str]:
    return [
        r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
    ]


def _event_payloads(conn, task_id: str, kind: str) -> list[dict]:
    return [
        json.loads(r["payload"]) if isinstance(r["payload"], str) else (r["payload"] or {})
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
            (task_id, kind),
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# 1. No contract (or the 16 live 'local-only' rows) — zero extra enforcement
# ---------------------------------------------------------------------------


def test_no_contract_completion_unchanged(kanban_home):
    """A card with a NULL contract completes on the pre-P3 evidence rules."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "null contract")
        assert kb.complete_task(conn, tid, result="done") is True
    assert _event_kinds(conn, tid).count("completion_blocked_contract") == 0
    assert _event_kinds(conn, tid).count("contract_unparseable") == 0


def test_empty_contract_completion_unchanged(kanban_home):
    """``completion_contract = ''`` behaves as no contract."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "empty contract")
        _set_contract(conn, tid, "")
        assert kb.complete_task(conn, tid, result="done") is True
    assert "completion_blocked_contract" not in _event_kinds(conn, tid)


def test_local_only_contract_completion_unchanged(kanban_home):
    """'local-only' (the 16 real rows on study_10/target_026) is inert."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "local-only card")
        _set_contract(conn, tid, "local-only")
        assert kb.complete_task(conn, tid, summary="substantive local summary") is True
        assert _event_kinds(conn, tid).count("completion_blocked_contract") == 0
        assert _event_kinds(conn, tid).count("contract_unparseable") == 0


def test_local_only_contract_with_empty_evidence_still_refused(kanban_home):
    """Inert contract does NOT disable the P2 empty-completion gate."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "local-only silent")
        _set_contract(conn, tid, "local-only")
        with pytest.raises(kb.EmptyCompletionError):
            kb.complete_task(conn, tid)
    assert _status(conn, tid) == "ready"


def _status(conn, task_id: str) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0]


# ---------------------------------------------------------------------------
# 2. JSON-dict contract: required artifacts are enforced
# ---------------------------------------------------------------------------


def test_contract_artifact_in_scratch_completes(kanban_home):
    """A declared artifact that exists inside managed scratch passes the gate."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "contract ok")
        artifact = ws / "report.md"
        artifact.write_text("# report", encoding="utf-8")
        _set_contract(conn, tid, {"required_artifacts": [str(artifact)]})
        assert kb.complete_task(conn, tid, result="with artifact") is True
    assert _status(conn, tid) == "done"
    assert "completion_blocked_contract" not in _event_kinds(conn, tid)


def test_contract_missing_artifact_refused(kanban_home):
    """A declared artifact that does not exist refuses the completion."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "contract missing")
        missing = ws / "never_written.md"
        _set_contract(conn, tid, {"required_artifacts": [str(missing)]})
        with pytest.raises(kb.ContractCompletionError):
            kb.complete_task(conn, tid, result="substantive result")
    assert _status(conn, tid) == "ready"
    kinds = _event_kinds(conn, tid)
    assert "completion_blocked_contract" in kinds
    assert "completed" not in kinds
    payloads = _event_payloads(conn, tid, "completion_blocked_contract")
    assert any(
        str(missing) in json.dumps(p) for p in payloads
    ), "the refusal event must name the missing artifact"


def test_contract_artifact_is_directory_refused(kanban_home):
    """Only regular files satisfy the contract — a dir named as artifact fails."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "contract dir")
        (ws / "results_dir").mkdir()
        _set_contract(conn, tid, {"required_artifacts": [str(ws / "results_dir")]})
        with pytest.raises(kb.ContractCompletionError):
            kb.complete_task(conn, tid, result="substantive result")
    assert _status(conn, tid) == "ready"


def test_contract_external_artifact_recorded(kanban_home):
    """An existing artifact OUTSIDE scratch passes with an auditable record."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "contract external ok")
        external = Path(kanban_home) / "external_evidence.md"
        external.write_text("real evidence", encoding="utf-8")
        _set_contract(conn, tid, {"required_artifacts": [str(external)]})
        assert kb.complete_task(conn, tid, result="external evidence exists") is True
    kinds = _event_kinds(conn, tid)
    assert "completion_blocked_contract" not in kinds
    assert "external_artifact_recorded" in kinds
    payloads = _event_payloads(conn, tid, "external_artifact_recorded")
    assert any(str(external) == p.get("artifact") for p in payloads)


def test_contract_empty_requirements_list_completes(kanban_home):
    """``required_artifacts: []`` declares nothing — no refusal."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "contract empty list")
        _set_contract(conn, tid, {"required_artifacts": []})
        assert kb.complete_task(conn, tid, result="nothing required") is True
    assert "completion_blocked_contract" not in _event_kinds(conn, tid)


def test_contract_missing_artifact_ignores_unrelated_declared_artifacts(kanban_home):
    """The gate validates the CONTRACT's artifacts, not metadata['artifacts']."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "contract vs metadata")
        real = ws / "real.txt"
        real.write_text("x", encoding="utf-8")
        _set_contract(conn, tid, {"required_artifacts": [str(ws / "ghost.txt")]})
        with pytest.raises(kb.ContractCompletionError):
            kb.complete_task(conn, tid, result="r", metadata={"artifacts": [str(real)]})


# ---------------------------------------------------------------------------
# 3. Non-parseable contract: fail-open with an audible event
# ---------------------------------------------------------------------------


def test_unparseable_json_contract_fails_open_with_event(kanban_home):
    """Legacy string contracts are not dicts: complete, but leave a marker."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "unparseable json")
        _set_contract(conn, tid, "{not json at all")
        assert kb.complete_task(conn, tid, result="legacy contract") is True
    kinds = _event_kinds(conn, tid)
    assert "contract_unparseable" in kinds
    assert "completion_blocked_contract" not in kinds


def test_non_dict_json_contract_fails_open_with_event(kanban_home):
    """A JSON list/string contract is not a requirements dict: fail-open."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "json list contract")
        _set_contract(conn, tid, '["just", "a", "list"]')
        assert kb.complete_task(conn, tid, result="legacy shape") is True
    assert "contract_unparseable" in _event_kinds(conn, tid)


def test_repo_contract_not_claimed_by_artifact_gate(kanban_home):
    """OWNER/REPO is a legal contract shape owned by the PR-acceptance store:
    the artifact gate neither enforces nor marks it unparseable (the refusal
    comes from pr_acceptance demanding metadata.published_pr — pre-P3)."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "repo contract")
        _set_contract(conn, tid, "owner/repo")
        assert kb.complete_task(conn, tid, result="pr contract shape") is False
    kinds = _event_kinds(conn, tid)
    assert "contract_unparseable" not in kinds
    assert "completion_blocked_contract" not in kinds


def test_contract_rejection_does_not_consume_or_block_retry(kanban_home):
    """After fixing the artifact, the same card completes: the refusal is not
    sticky and does not corrupt the failure counter."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "retry after fix")
        missing = ws / "late.txt"
        _set_contract(conn, tid, {"required_artifacts": [str(missing)]})
        with pytest.raises(kb.ContractCompletionError):
            kb.complete_task(conn, tid, result="premature")
        assert _status(conn, tid) == "ready"
        missing.write_text("now it exists", encoding="utf-8")
        assert kb.complete_task(conn, tid, result="artifact now exists") is True
        task = kb.get_task(conn, tid)
        assert task.consecutive_failures == 0


# ---------------------------------------------------------------------------
# 4. External artifact auditing in _persist_scratch_completion_artifacts
# ---------------------------------------------------------------------------


def test_missing_external_artifact_emits_audible_event(kanban_home):
    """Declared metadata artifact outside scratch that does not exist:
    completion still succeeds (P2 behavior) but the omission is now audible."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "external missing")
        ghost = Path(kanban_home) / "ghost.bin"
        assert kb.complete_task(
            conn, tid, result="r", metadata={"artifacts": [str(ghost)]}
        ) is True
    assert _status(conn, tid) == "done"
    payloads = _event_payloads(conn, tid, "external_artifact_missing")
    assert payloads, "missing external artifact must be auditable"
    assert any(str(ghost) == p.get("artifact") for p in payloads)


def test_existing_external_artifact_emits_recorded_event(kanban_home):
    """A real external file persists as a string and gains a recorded event."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "external recorded")
        external = Path(kanban_home) / "kept.txt"
        external.write_text("persist me", encoding="utf-8")
        assert kb.complete_task(
            conn, tid, result="r", metadata={"artifacts": [str(external)]}
        ) is True
    assert _status(conn, tid) == "done"
    payloads = _event_payloads(conn, tid, "external_artifact_recorded")
    assert any(str(external) == p.get("artifact") for p in payloads)
    # P2 behavior preserved: the path stays as the (unvalidated-existence) string
    completed = _event_payloads(conn, tid, "completed")
    assert any(str(external) in json.dumps(p) for p in completed)


def test_scratch_artifacts_do_not_gain_external_events(kanban_home):
    """Inside-scratch artifacts keep the P2 staging path — no external noise."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "scratch only")
        artifact = ws / "chart.png"
        artifact.write_bytes(b"png")
        assert kb.complete_task(
            conn, tid, result="r", metadata={"artifacts": [str(artifact)]}
        ) is True
    kinds = _event_kinds(conn, tid)
    assert "external_artifact_missing" not in kinds
    assert "external_artifact_recorded" not in kinds


# ---------------------------------------------------------------------------
# 5. Protocol violation vs transient (P3 item 3): rc=0 clean-exit violations
#    get a bounded STREAK budget and a sticky trip — the dispatcher must not
#    blindly respawn them across ticks.
# ---------------------------------------------------------------------------


def _drive_worker_exit(conn, tid, fake_pid, raw_status):
    """Claim ``tid``, record ``raw_status`` for its dead worker pid, reap."""
    from hermes_cli import kanban_db_dispatch as _kbd
    host_prefix = kb._claimer_id().split(":", 1)[0]
    claimed = kb.claim_task(conn, tid, claimer=f"{host_prefix}:mock")
    assert claimed is not None, "task was not claimable for the next attempt"
    _kbd._set_worker_pid(conn, tid, fake_pid)
    _kbd._record_worker_exit(fake_pid, raw_status)
    original_alive = kb._pid_alive
    kb._pid_alive = lambda p: False
    try:
        return _kbd.detect_crashed_workers(conn)
    finally:
        kb._pid_alive = original_alive


def test_rc0_protocol_violation_burst_trips_sticky_and_stays_blocked(kanban_home):
    """Three consecutive rc=0 clean-exit protocol violations trip the bounded
    budget into a STICKY block; the dispatcher's next ticks (recompute_ready)
    must not respawn the card — while a single below-budget violation stays
    retryable (transient, not sticky)."""
    from hermes_cli import kanban_db_dispatch as _kbd
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rc0 burst", assignee="worker")

        # One violation: below budget → back at ready (transient retry).
        _drive_worker_exit(conn, tid, 993001, 0)
        assert kb.get_task(conn, tid).status == "ready"

        # Two more: the streak hits _PROTOCOL_VIOLATION_FAILURE_LIMIT → blocked.
        _drive_worker_exit(conn, tid, 993002, 0)
        _drive_worker_exit(conn, tid, 993003, 0)
        assert kb.get_task(conn, tid).status == "blocked"
        gave_up = [e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]
        assert len(gave_up) == 1
        assert (gave_up[0].payload or {}).get("sticky") is True
        assert (gave_up[0].payload or {}).get("protocol_violations") == (
            _kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT
        )

        # The dispatcher must NOT blindly respawn it: the sticky gave_up holds
        # through recompute_ready ticks (what the burst loop would otherwise
        # turn into an infinite rc=0 respawn).
        for _ in range(5):
            assert kb.recompute_ready(conn) == 0
            assert kb.get_task(conn, tid).status == "blocked"
