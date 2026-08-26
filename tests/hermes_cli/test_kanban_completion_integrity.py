"""Behavioral tests for Kanban completion integrity (Phase 1A + bounded 1C).

These tests prove controller-owned verification, typed review_approved
gates, and immutable Historian certification. They do not mutate a real
LocoMail repository or production Kanban DB.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_completion_integrity as kci
from hermes_cli import kanban_db as kb


FAKE_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(path: Path, *, message: str = "init") -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(path, "config", "user.email", "kanban@example.com")
    _git(path, "config", "user.name", "Kanban Test")
    (path / "README.md").write_text(f"{message}\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", message)
    return _git(path, "rev-parse", "HEAD").stdout.strip()


def _commit_file(repo: Path, name: str, body: str, message: str) -> str:
    (repo / name).write_text(body, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _git_contract(repo: Path, expected_base: str | None = None) -> dict:
    contract = {
        "schema_version": 1,
        "type": "git_revision",
        "repository": str(repo),
        "worktree": str(repo),
        "git_common_dir": kci.resolve_git_common_dir(str(repo)),
    }
    if expected_base:
        contract["expected_base"] = expected_base
        contract["max_commits"] = 8
    return contract


def _review_contract(repo: Path, required_sha: str | None = None) -> dict:
    contract = {
        "schema_version": 1,
        "type": "review",
        "repository": str(repo),
        "worktree": str(repo),
        "git_common_dir": kci.resolve_git_common_dir(str(repo)),
    }
    if required_sha:
        contract["required_sha"] = required_sha
    return contract


def _historian_contract(repo: Path, required_sha: str) -> dict:
    return {
        "schema_version": 1,
        "type": "historian_certify",
        "repository": str(repo),
        "required_sha": required_sha,
        "git_common_dir": kci.resolve_git_common_dir(str(repo)),
    }


def _review_gate(sha: str) -> dict:
    return {
        "schema_version": 1,
        "type": "review_approved",
        "reviewed_sha": sha,
    }


def _create_and_claim(conn, title: str, *, contract: dict, assignee: str = "implementer") -> tuple[str, str]:
    tid = kb.create_task(
        conn,
        title=title,
        assignee=assignee,
        completion_contract=contract,
    )
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.attempt_id
    return tid, task.attempt_id


def _complete_revision(conn, tid: str, *, attempt_id: str, repo: Path, sha: str, verdict: str | None = None):
    payload = {
        "attempt_id": attempt_id,
        "repository": str(repo),
        "worktree": str(repo),
        "commit_sha": sha,
    }
    if verdict is not None:
        payload["verdict"] = verdict
        payload["reviewed_sha"] = sha
    return kb.complete_task(
        conn,
        tid,
        summary=f"completed {sha[:8]}",
        metadata={"terminal_result": payload},
    )


def _status(conn, tid: str) -> str:
    task = kb.get_task(conn, tid)
    assert task is not None
    return task.status


def _row(conn, tid: str):
    return conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()


def test_a_narrative_only_completion_is_not_done(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    with kb.connect() as conn:
        engineer, _ = _create_and_claim(conn, "engineer", contract=_git_contract(repo, base))
        historian = kb.create_task(
            conn,
            title="historian",
            assignee="historian",
            parents=[engineer],
            completion_contract=_historian_contract(repo, FAKE_SHA),
        )
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            kb.complete_task(conn, engineer, summary="I finished the work in prose")
        assert exc.value.code == kci.REASON_NARRATIVE_ONLY
        row = _row(conn, engineer)
        assert row["status"] != "done"
        assert row["awaiting_verification"] == 1
        assert row["verification_code"] == kci.REASON_NARRATIVE_ONLY
        kb.recompute_ready(conn)
        assert _status(conn, historian) == "todo"


def test_b_missing_terminal_result_is_not_adopted(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    _commit_file(repo, "work.txt", "done\n", "work")
    with kb.connect() as conn:
        engineer, _ = _create_and_claim(conn, "engineer", contract=_git_contract(repo, base))
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            kb.complete_task(conn, engineer, result="exit 0")
        assert exc.value.code in {
            kci.REASON_MISSING_TERMINAL_RESULT,
            kci.REASON_NARRATIVE_ONLY,
        }
        assert _status(conn, engineer) != "done"
        # Clean-exit reclaim must not adopt unreported work as done.
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, claim_lock = ? WHERE id = ?",
            (os.getpid() + 10_000, f"{kb._claimer_id().split(':', 1)[0]}:dead", engineer),
        )
        kb.detect_crashed_workers(conn)
        assert _status(conn, engineer) != "done"


def test_c_fabricated_sha_is_rejected(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(conn, "engineer", contract=_git_contract(repo, base))
        historian = kb.create_task(
            conn, title="historian", assignee="historian", parents=[engineer],
        )
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(conn, engineer, attempt_id=attempt, repo=repo, sha=FAKE_SHA)
        assert exc.value.code == kci.REASON_OBJECT_MISSING
        row = _row(conn, engineer)
        assert row["status"] != "done"
        assert row["verification_status"] == kci.STATUS_REJECTED
        kb.recompute_ready(conn)
        assert _status(conn, historian) == "todo"


def test_d_wrong_repository_or_base_is_rejected(kanban_home, tmp_path):
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    base_a = _init_repo(repo_a, message="repo-a")
    _init_repo(repo_b, message="repo-b")
    foreign = _commit_file(repo_b, "x.txt", "nope\n", "foreign")
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(
            conn, "engineer", contract=_git_contract(repo_a, base_a),
        )
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(conn, engineer, attempt_id=attempt, repo=repo_b, sha=foreign)
        assert exc.value.code in {kci.REASON_WRONG_REPOSITORY, kci.REASON_OBJECT_MISSING}
        assert _status(conn, engineer) != "done"


def test_d_non_ancestor_base_is_rejected(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    side = tmp_path / "side"
    side_sha = _init_repo(side, message="unrelated")
    work = _commit_file(repo, "w.txt", "w\n", "work")
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(
            conn, "engineer", contract=_git_contract(repo, side_sha),
        )
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(conn, engineer, attempt_id=attempt, repo=repo, sha=work)
        assert exc.value.code == kci.REASON_BASE_MISMATCH
        assert _status(conn, engineer) != "done"


def test_e_valid_commit_reaches_verified_done(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(conn, "engineer", contract=_git_contract(repo, base))
        assert _complete_revision(conn, engineer, attempt_id=attempt, repo=repo, sha=sha_b) is True
        row = _row(conn, engineer)
        assert row["status"] == "done"
        assert row["verification_status"] == kci.STATUS_VERIFIED
        assert row["verified_revision"] == sha_b
        assert row["awaiting_verification"] == 0


def test_f_rejected_review_does_not_satisfy_gate(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        engineer, e_attempt = _create_and_claim(conn, "engineer", contract=_git_contract(repo, base))
        _complete_revision(conn, engineer, attempt_id=e_attempt, repo=repo, sha=sha_b)
        reviewer, r_attempt = _create_and_claim(
            conn, "reviewer", contract=_review_contract(repo, sha_b), assignee="reviewer",
        )
        historian = kb.create_task(
            conn,
            title="historian",
            assignee="historian",
            parents=[reviewer],
            completion_contract=_historian_contract(repo, sha_b),
        )
        kb.link_tasks(conn, reviewer, historian, semantic_gate=_review_gate(sha_b))
        assert _complete_revision(
            conn, reviewer, attempt_id=r_attempt, repo=repo, sha=sha_b, verdict="REJECTED",
        ) is True
        assert _status(conn, reviewer) == "done"
        assert _row(conn, reviewer)["verified_verdict"] == "REJECTED"
        kb.recompute_ready(conn)
        assert _status(conn, historian) == "todo"
        hist = kb.certify_historian_revision(conn, historian, repository=str(repo), commit_sha=sha_b)
        # Object may exist, but the historian child is not semantically eligible.
        assert _status(conn, historian) == "todo"
        assert hist.ok  # object exists; eligibility is the gate, not object presence


def test_g_missing_review_verdict_is_blocked(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        reviewer, attempt = _create_and_claim(
            conn, "reviewer", contract=_review_contract(repo, sha_b), assignee="reviewer",
        )
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(conn, reviewer, attempt_id=attempt, repo=repo, sha=sha_b)
        assert exc.value.code == kci.REASON_VERDICT_MISSING
        assert _status(conn, reviewer) != "done"


def test_h_malformed_review_verdict_fails_closed(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        reviewer, attempt = _create_and_claim(
            conn, "reviewer", contract=_review_contract(repo, sha_b), assignee="reviewer",
        )
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(
                conn, reviewer, attempt_id=attempt, repo=repo, sha=sha_b, verdict="LGTM",
            )
        assert exc.value.code == kci.REASON_VERDICT_MALFORMED
        assert _status(conn, reviewer) != "done"


def test_i_approval_for_wrong_sha_does_not_satisfy_gate(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    sha_c = _commit_file(repo, "c.txt", "C\n", "commit-C")
    with kb.connect() as conn:
        reviewer, attempt = _create_and_claim(
            conn, "reviewer", contract=_review_contract(repo), assignee="reviewer",
        )
        historian = kb.create_task(
            conn, title="historian", assignee="historian", parents=[reviewer],
        )
        kb.link_tasks(conn, reviewer, historian, semantic_gate=_review_gate(sha_b))
        assert _complete_revision(
            conn, reviewer, attempt_id=attempt, repo=repo, sha=sha_c, verdict="APPROVED",
        ) is True
        kb.recompute_ready(conn)
        assert _status(conn, historian) == "todo"


def test_j_approval_for_exact_b_satisfies_gate(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        reviewer, attempt = _create_and_claim(
            conn, "reviewer", contract=_review_contract(repo, sha_b), assignee="reviewer",
        )
        historian = kb.create_task(
            conn, title="historian", assignee="historian", parents=[reviewer],
        )
        kb.link_tasks(conn, reviewer, historian, semantic_gate=_review_gate(sha_b))
        assert _complete_revision(
            conn, reviewer, attempt_id=attempt, repo=repo, sha=sha_b, verdict="PASS",
        ) is True
        kb.recompute_ready(conn)
        assert _status(conn, historian) == "ready"


def test_k_approval_for_b_does_not_transfer_to_c(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        engineer, e_attempt = _create_and_claim(conn, "engineer", contract=_git_contract(repo, base))
        _complete_revision(conn, engineer, attempt_id=e_attempt, repo=repo, sha=sha_b)
        reviewer, r_attempt = _create_and_claim(
            conn, "reviewer", contract=_review_contract(repo), assignee="reviewer",
        )
        historian = kb.create_task(
            conn, title="historian", assignee="historian", parents=[reviewer],
        )
        kb.link_tasks(conn, reviewer, historian, semantic_gate=_review_gate(sha_b))
        _complete_revision(
            conn, reviewer, attempt_id=r_attempt, repo=repo, sha=sha_b, verdict="APPROVED",
        )
        sha_c = _commit_file(repo, "c.txt", "C\n", "commit-C")
        engineer2, e2_attempt = _create_and_claim(
            conn, "engineer-c", contract=_git_contract(repo, sha_b),
        )
        _complete_revision(conn, engineer2, attempt_id=e2_attempt, repo=repo, sha=sha_c)
        assert _row(conn, engineer2)["verified_revision"] == sha_c
        historian_c = kb.create_task(
            conn, title="historian-c", assignee="historian", parents=[reviewer],
        )
        kb.link_tasks(conn, reviewer, historian_c, semantic_gate=_review_gate(sha_c))
        kb.recompute_ready(conn)
        assert _status(conn, historian_c) == "todo"


def test_l_and_m_historian_certifies_c_without_head_equality(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    sha_c = _commit_file(repo, "c.txt", "C\n", "commit-C")
    with kb.connect() as conn:
        reviewer, attempt = _create_and_claim(
            conn, "reviewer", contract=_review_contract(repo, sha_c), assignee="reviewer",
        )
        historian = kb.create_task(
            conn,
            title="historian",
            assignee="historian",
            parents=[reviewer],
            completion_contract=_historian_contract(repo, sha_c),
        )
        kb.link_tasks(conn, reviewer, historian, semantic_gate=_review_gate(sha_c))
        _complete_revision(
            conn, reviewer, attempt_id=attempt, repo=repo, sha=sha_c, verdict="APPROVED",
        )
        kb.recompute_ready(conn)
        assert _status(conn, historian) == "ready"
        later = _commit_file(repo, "d.txt", "D\n", "move-head")
        head = _git(repo, "rev-parse", "HEAD").stdout.strip()
        assert head == later
        assert head != sha_c
        certified = kb.certify_historian_revision(
            conn, historian, repository=str(repo), commit_sha=sha_c,
        )
        assert certified.ok
        assert certified.details.get("head") == later
        assert certified.details.get("head_policy") == kci.REASON_HEAD_NOT_REQUIRED
        assert certified.details.get("require_head_match") is False
        del base, sha_b


def test_n_identical_duplicate_is_idempotent(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(conn, "engineer", contract=_git_contract(repo, base))
        assert _complete_revision(conn, engineer, attempt_id=attempt, repo=repo, sha=sha_b) is True
        first = _row(conn, engineer)
        assert _complete_revision(conn, engineer, attempt_id=attempt, repo=repo, sha=sha_b) is True
        second = _row(conn, engineer)
        assert second["terminal_result_hash"] == first["terminal_result_hash"]
        assert second["verified_revision"] == sha_b
        assert second["needs_reconciliation"] == 0
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? AND kind = 'terminal_result_duplicate'",
            (engineer,),
        ).fetchall()
        assert events


def test_o_conflicting_duplicate_needs_reconciliation(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    sha_c = _commit_file(repo, "c.txt", "C\n", "commit-C")
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(conn, "engineer", contract=_git_contract(repo, base))
        child = kb.create_task(conn, title="child", assignee="worker", parents=[engineer])
        assert _complete_revision(conn, engineer, attempt_id=attempt, repo=repo, sha=sha_b) is True
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(conn, engineer, attempt_id=attempt, repo=repo, sha=sha_c)
        assert exc.value.code == kci.REASON_CONFLICTING_RESULT
        row = _row(conn, engineer)
        assert row["verified_revision"] == sha_b
        assert row["needs_reconciliation"] == 1
        kb.recompute_ready(conn)
        assert _status(conn, child) == "todo"


def test_p_force_promotion_cannot_bypass_semantic_gate(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        reviewer, attempt = _create_and_claim(
            conn, "reviewer", contract=_review_contract(repo, sha_b), assignee="reviewer",
        )
        historian = kb.create_task(
            conn, title="historian", assignee="historian", parents=[reviewer],
        )
        kb.link_tasks(conn, reviewer, historian, semantic_gate=_review_gate(sha_b))
        _complete_revision(
            conn, reviewer, attempt_id=attempt, repo=repo, sha=sha_b, verdict="REJECTED",
        )
        ok, err = kb.promote_task(conn, historian, actor="operator", force=True)
        assert ok is False
        assert err is not None
        assert "review" in err.lower() or "VERDICT" in err or "unsatisfied" in err
        assert _status(conn, historian) == "todo"


def test_q_direct_complete_cannot_bypass_semantic_gate(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        reviewer, attempt = _create_and_claim(
            conn, "reviewer", contract=_review_contract(repo, sha_b), assignee="reviewer",
        )
        historian = kb.create_task(
            conn, title="historian", assignee="historian", parents=[reviewer],
        )
        kb.link_tasks(conn, reviewer, historian, semantic_gate=_review_gate(sha_b))
        _complete_revision(
            conn, reviewer, attempt_id=attempt, repo=repo, sha=sha_b, verdict="REJECTED",
        )
        # Direct complete of the historian without an approved gate/result.
        ok = kb.complete_task(conn, historian, summary="certify anyway")
        assert ok is False
        assert _status(conn, historian) != "done"


def test_r_request_review_cannot_bypass_required_verification(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    with kb.connect() as conn:
        engineer, _ = _create_and_claim(conn, "engineer", contract=_git_contract(repo, base))
        ok = kb.request_review(conn, engineer, summary="please review my narrative")
        assert ok is False
        assert _status(conn, engineer) == "running"
        assert _row(conn, engineer)["awaiting_verification"] == 1


def test_s_unsatisfied_parent_does_not_release_grandchildren(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        reviewer, attempt = _create_and_claim(
            conn, "reviewer", contract=_review_contract(repo, sha_b), assignee="reviewer",
        )
        historian = kb.create_task(
            conn, title="historian", assignee="historian", parents=[reviewer],
        )
        grandchild = kb.create_task(
            conn, title="grandchild", assignee="ops", parents=[historian],
        )
        kb.link_tasks(conn, reviewer, historian, semantic_gate=_review_gate(sha_b))
        _complete_revision(
            conn, reviewer, attempt_id=attempt, repo=repo, sha=sha_b, verdict="REJECTED",
        )
        kb.recompute_ready(conn)
        assert _status(conn, historian) == "todo"
        assert _status(conn, grandchild) == "todo"


def test_t_ungated_legacy_parent_child_remains_compatible(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="legacy-parent", assignee="worker")
        child = kb.create_task(conn, title="legacy-child", assignee="worker", parents=[parent])
        assert _status(conn, child) == "todo"
        kb.claim_task(conn, parent)
        assert kb.complete_task(conn, parent, summary="plain legacy done") is True
        kb.recompute_ready(conn)
        assert _status(conn, child) == "ready"


def test_u_ordinary_review_lifecycle_remains_compatible(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ordinary-review", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.request_review(
            conn, tid, summary="ready for review",
            expected_run_id=claimed.current_run_id,
        ) is True
        assert _status(conn, tid) == "review"
        assert kb.claim_review_task(conn, tid) is not None
        assert kb.complete_task(conn, tid, summary="approved by human") is True
        assert _status(conn, tid) == "done"


def test_declaration_surfaces_accept_typed_contract_and_gate(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        parent = kb.create_task(
            conn,
            title="reviewer",
            assignee="reviewer",
            completion_contract=_review_contract(repo, sha_b),
        )
        child = kb.create_task(conn, title="historian", assignee="historian")
        kb.link_tasks(conn, parent, child, semantic_gate=_review_gate(sha_b))
        gate = conn.execute(
            "SELECT semantic_gate FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent, child),
        ).fetchone()["semantic_gate"]
        parsed = json.loads(gate)
        assert parsed["type"] == "review_approved"
        assert parsed["reviewed_sha"] == sha_b
        assert kb.get_task(conn, parent).completion_contract["type"] == "review"
        del base


def test_negative_path_orchestration_certification(kanban_home, tmp_path):
    """Isolated A->M plus N/O/T certification walk."""
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    receipt = {
        "baseline": "9dbb8868e8bdd8f2ca24c33e0a3b58ef2b605bc8",
        "steps": [],
    }
    with kb.connect() as conn:
        engineer, e_attempt = _create_and_claim(
            conn, "engineer", contract=_git_contract(repo, base),
        )
        reviewer = kb.create_task(
            conn,
            title="reviewer",
            assignee="reviewer",
            completion_contract=_review_contract(repo),
        )
        historian = kb.create_task(
            conn,
            title="historian",
            assignee="historian",
            parents=[reviewer],
            completion_contract=_historian_contract(repo, FAKE_SHA),
        )
        legacy_parent = kb.create_task(conn, title="legacy-parent", assignee="worker")
        legacy_child = kb.create_task(
            conn, title="legacy-child", assignee="worker", parents=[legacy_parent],
        )

        with pytest.raises(kci.CompletionIntegrityError) as exc:
            kb.complete_task(conn, engineer, summary="narrative only")
        receipt["steps"].append({"step": "narrative", "code": exc.value.code})
        assert _status(conn, engineer) != "done"

        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(conn, engineer, attempt_id=e_attempt, repo=repo, sha=FAKE_SHA)
        receipt["steps"].append({"step": "fake_sha", "code": exc.value.code})

        other = tmp_path / "other"
        _init_repo(other, message="other")
        foreign = _commit_file(other, "x.txt", "x\n", "foreign")
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(conn, engineer, attempt_id=e_attempt, repo=other, sha=foreign)
        receipt["steps"].append({"step": "wrong_repo", "code": exc.value.code})

        sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
        assert _complete_revision(conn, engineer, attempt_id=e_attempt, repo=repo, sha=sha_b)
        receipt["steps"].append({"step": "valid_b", "sha": sha_b, "status": _status(conn, engineer)})

        kb.claim_task(conn, reviewer)
        r_attempt = kb.get_task(conn, reviewer).attempt_id
        kb.link_tasks(conn, reviewer, historian, semantic_gate=_review_gate(sha_b))
        assert _complete_revision(
            conn, reviewer, attempt_id=r_attempt, repo=repo, sha=sha_b, verdict="REJECTED",
        )
        kb.recompute_ready(conn)
        receipt["steps"].append({
            "step": "review_rejected_b",
            "historian": _status(conn, historian),
        })
        assert _status(conn, historian) == "todo"

        sha_c = _commit_file(repo, "c.txt", "C\n", "commit-C")
        engineer2, e2_attempt = _create_and_claim(
            conn, "engineer-c", contract=_git_contract(repo, sha_b),
        )
        assert _complete_revision(conn, engineer2, attempt_id=e2_attempt, repo=repo, sha=sha_c)
        reviewer2, r2_attempt = _create_and_claim(
            conn, "reviewer-c", contract=_review_contract(repo, sha_c), assignee="reviewer",
        )
        # The same Historian now waits on the corrected review of C, not the
        # rejected review of B. Approval for B must not transfer.
        kb.unlink_tasks(conn, reviewer, historian)
        kb.link_tasks(conn, reviewer2, historian, semantic_gate=_review_gate(sha_c))
        conn.execute(
            "UPDATE tasks SET completion_contract = ? WHERE id = ?",
            (kci.persist_json(_historian_contract(repo, sha_c)), historian),
        )
        assert _complete_revision(
            conn, reviewer2, attempt_id=r2_attempt, repo=repo, sha=sha_c, verdict="APPROVED",
        )
        kb.recompute_ready(conn)
        receipt["steps"].append({
            "step": "review_approved_c",
            "historian": _status(conn, historian),
        })
        assert _status(conn, historian) == "ready"

        later = _commit_file(repo, "moved.txt", "away\n", "move-head")
        certified = kb.certify_historian_revision(
            conn, historian, repository=str(repo), commit_sha=sha_c,
        )
        receipt["steps"].append({
            "step": "historian_certify_c",
            "ok": certified.ok,
            "head": certified.details.get("head"),
            "later": later,
        })
        assert certified.ok
        assert certified.details.get("head") == later

        assert _complete_revision(conn, engineer, attempt_id=e_attempt, repo=repo, sha=sha_b)
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(conn, engineer, attempt_id=e_attempt, repo=repo, sha=sha_c)
        receipt["steps"].append({"step": "conflict", "code": exc.value.code})
        assert _row(conn, engineer)["needs_reconciliation"] == 1

        kb.claim_task(conn, legacy_parent)
        assert kb.complete_task(conn, legacy_parent, summary="legacy ok")
        kb.recompute_ready(conn)
        receipt["steps"].append({"step": "legacy", "child": _status(conn, legacy_child)})
        assert _status(conn, legacy_child) == "ready"

        structural = kb.create_task(conn, title="force-structural", assignee="worker")
        force_child = kb.create_task(
            conn, title="force-child", assignee="worker",
            parents=[structural, engineer],
        )
        _link_in_order(conn, force_child, [structural, engineer])
        ok, err = kb.promote_task(conn, force_child, actor="operator", force=True)
        receipt["steps"].append({
            "step": "multi_parent_force",
            "ok": ok,
            "error": err,
            "status": _status(conn, force_child),
        })
        assert ok is False
        assert _status(conn, force_child) == "todo"

        mid = kb.create_task(
            conn, title="archive-mid", assignee="worker",
            completion_contract=_git_contract(repo, sha_c),
        )
        leaf = kb.create_task(conn, title="archive-leaf", assignee="worker", parents=[mid])
        assert kb.archive_task(conn, mid) is True
        kb.recompute_ready(conn)
        receipt["steps"].append({
            "step": "archive_unexecuted_intermediate",
            "leaf": _status(conn, leaf),
        })
        assert _status(conn, leaf) == "todo"

        replay, replay_a = _create_and_claim(
            conn, "replay", contract=_git_contract(repo, sha_c),
        )
        assert kb.reclaim_task(conn, replay, reason="retry") is True
        claimed = kb.claim_task(conn, replay)
        assert claimed is not None
        replay_task = kb.get_task(conn, replay)
        assert replay_task is not None
        replay_b = replay_task.attempt_id
        assert replay_b != replay_a
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(conn, replay, attempt_id=replay_a, repo=repo, sha=sha_c)
        receipt["steps"].append({"step": "stale_attempt_replay", "code": exc.value.code})
        assert exc.value.code == kci.REASON_WRONG_ATTEMPT

        rec_child = kb.create_task(conn, title="rec-child", assignee="worker", parents=[engineer2])
        rec_grand = kb.create_task(conn, title="rec-grand", assignee="worker", parents=[rec_child])
        kb.recompute_ready(conn)
        kb.claim_task(conn, rec_child)
        assert kb.complete_task(conn, rec_child, summary="rec child done")
        kb.recompute_ready(conn)
        assert _status(conn, rec_grand) == "ready"
        with pytest.raises(kci.CompletionIntegrityError):
            _complete_revision(conn, engineer2, attempt_id=e2_attempt, repo=repo, sha=sha_b)
        receipt["steps"].append({
            "step": "reconciliation_invalidates_grandchild",
            "grandchild": _status(conn, rec_grand),
        })
        assert _status(conn, rec_grand) == "todo"

    receipt_path = tmp_path / "kanban-completion-integrity-certification.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    assert receipt["steps"]


# ---------------------------------------------------------------------------
# Independent Sol rejection regressions (must fail on 4269ffae)
# ---------------------------------------------------------------------------


def _complete_without_attempt(conn, tid: str, *, repo: Path, sha: str):
    return kb.complete_task(
        conn,
        tid,
        summary=f"completed {sha[:8]}",
        metadata={
            "terminal_result": {
                "repository": str(repo),
                "worktree": str(repo),
                "commit_sha": sha,
            }
        },
    )


def test_sol1_historian_ordinary_completion_binds_required_sha(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    sha_c = _commit_file(repo, "c.txt", "C\n", "commit-C")
    with kb.connect() as conn:
        historian, attempt = _create_and_claim(
            conn,
            "historian",
            contract=_historian_contract(repo, sha_c),
            assignee="historian",
        )
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(
                conn, historian, attempt_id=attempt, repo=repo, sha=sha_b,
            )
        assert exc.value.code == kci.REASON_SHA_MISMATCH
        assert _status(conn, historian) != "done"
        assert _complete_revision(
            conn, historian, attempt_id=attempt, repo=repo, sha=sha_c,
        ) is True
        assert _status(conn, historian) == "done"


def test_sol1_historian_direct_certify_binds_required_sha(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    sha_c = _commit_file(repo, "c.txt", "C\n", "commit-C")
    with kb.connect() as conn:
        historian = kb.create_task(
            conn,
            title="historian",
            assignee="historian",
            completion_contract=_historian_contract(repo, sha_c),
        )
        certified = kb.certify_historian_revision(
            conn, historian, repository=str(repo), commit_sha=sha_b,
        )
        assert certified.ok is False
        assert certified.code == kci.REASON_SHA_MISMATCH
        certified_c = kb.certify_historian_revision(
            conn, historian, repository=str(repo), commit_sha=sha_c,
        )
        assert certified_c.ok is True


def test_sol2_worker_cannot_choose_repository_identity(kanban_home, tmp_path):
    authorized = tmp_path / "authorized"
    worker = tmp_path / "worker"
    _init_repo(authorized, message="authorized")
    worker_sha = _init_repo(worker, message="worker-chosen")
    contract = {
        "schema_version": 1,
        "type": "git_revision",
    }
    with kb.connect() as conn:
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            kb.create_task(
                conn,
                title="engineer",
                assignee="implementer",
                completion_contract=contract,
            )
        assert exc.value.code == kci.REASON_INVALID
        engineer, attempt = _create_and_claim(
            conn, "engineer", contract=_git_contract(authorized),
        )
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(
                conn, engineer, attempt_id=attempt, repo=worker, sha=worker_sha,
            )
        assert exc.value.code in {
            kci.REASON_INVALID,
            kci.REASON_WRONG_REPOSITORY,
            kci.REASON_OBJECT_MISSING,
        }
        assert _status(conn, engineer) != "done"


def test_sol3_missing_attempt_id_is_rejected(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(
            conn, "engineer", contract=_git_contract(repo, base),
        )
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_without_attempt(conn, engineer, repo=repo, sha=sha_b)
        assert exc.value.code in {
            getattr(kci, "REASON_MISSING_ATTEMPT", "MISSING_ATTEMPT"),
            "MISSING_ATTEMPT",
        }
        assert _status(conn, engineer) != "done"
        assert attempt


def test_sol3_stale_attempt_id_replay_is_rejected(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        engineer, attempt_a = _create_and_claim(
            conn, "engineer", contract=_git_contract(repo, base),
        )
        assert kb.reclaim_task(conn, engineer, reason="retry") is True
        claimed = kb.claim_task(conn, engineer)
        assert claimed is not None
        task_b = kb.get_task(conn, engineer)
        assert task_b is not None
        attempt_b = task_b.attempt_id
        assert attempt_b
        assert attempt_b != attempt_a
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(
                conn, engineer, attempt_id=attempt_a, repo=repo, sha=sha_b,
            )
        assert exc.value.code == kci.REASON_WRONG_ATTEMPT
        assert _status(conn, engineer) != "done"


def test_sol4_max_commits_without_expected_base_is_rejected(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    contract = {
        "schema_version": 1,
        "type": "git_revision",
        "repository": str(repo),
        "worktree": str(repo),
        "git_common_dir": kci.resolve_git_common_dir(str(repo)),
        "max_commits": 2,
    }
    with kb.connect() as conn:
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            kb.create_task(
                conn, title="engineer", assignee="implementer", completion_contract=contract,
            )
        assert exc.value.code == kci.REASON_INVALID


def test_sol4_unknown_dirty_policy_is_rejected(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    contract = _git_contract(repo, base)
    contract["dirty_policy"] = "yolo"
    with kb.connect() as conn:
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            kb.create_task(
                conn, title="engineer", assignee="implementer", completion_contract=contract,
            )
        assert exc.value.code == kci.REASON_INVALID


def test_sol4_git_status_nonzero_is_verification_error(kanban_home, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    real_run = kci._run_git

    def _status_fails(args, *, cwd=None):
        if args and args[0] == "status":
            return subprocess.CompletedProcess(
                args=["git", *args],
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository",
            )
        return real_run(args, cwd=cwd)

    monkeypatch.setattr(kci, "_run_git", _status_fails)
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(
            conn, "engineer", contract=_git_contract(repo, base),
        )
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(
                conn, engineer, attempt_id=attempt, repo=repo, sha=sha_b,
            )
        assert exc.value.code == kci.REASON_VERIFICATION_ERROR
        assert _status(conn, engineer) != "done"


def test_sol4_replace_objects_do_not_influence_inspection(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    _git(repo, "checkout", "--orphan", "decoy")
    (repo / "decoy.txt").write_text("decoy\n", encoding="utf-8")
    _git(repo, "add", "decoy.txt")
    _git(repo, "commit", "-m", "decoy")
    decoy = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "main")
    _git(repo, "replace", decoy, sha_b)
    replaced = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", base, decoy],
        capture_output=True,
        text=True,
    )
    assert replaced.returncode == 0
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(
            conn, "engineer", contract=_git_contract(repo, base),
        )
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(
                conn, engineer, attempt_id=attempt, repo=repo, sha=decoy,
            )
        assert exc.value.code == kci.REASON_BASE_MISMATCH
        assert _status(conn, engineer) != "done"


def test_sol4_remote_url_text_is_not_sufficient_provenance(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    _git(repo, "remote", "add", "origin", "https://example.invalid/authorized.git")
    contract = _git_contract(repo, base)
    contract["require_remote"] = "https://example.invalid/authorized.git"
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(conn, "engineer", contract=contract)
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            _complete_revision(
                conn, engineer, attempt_id=attempt, repo=repo, sha=sha_b,
            )
        assert exc.value.code == kci.REASON_REMOTE_MISMATCH
        assert _status(conn, engineer) != "done"


def test_sol5_worker_context_exposes_governance_evidence(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(
            conn, "engineer", contract=_git_contract(repo, base),
        )
        ctx = kb.build_worker_context(conn, engineer)
        assert attempt in ctx
        assert "completion_contract" in ctx
        assert "terminal_result" in ctx
        assert "git_revision" in ctx


def test_sol5_cli_and_tool_surfaces_expose_governance_fields(kanban_home, tmp_path, monkeypatch):
    from hermes_cli.kanban import _task_to_dict
    from tools import kanban_tools as kt

    repo = tmp_path / "repo"
    base = _init_repo(repo)
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(
            conn, "engineer", contract=_git_contract(repo, base),
        )
        task = kb.get_task(conn, engineer)
        cli_payload = _task_to_dict(task)
        assert cli_payload["attempt_id"] == attempt
        assert cli_payload["completion_contract"]["type"] == "git_revision"
        assert "terminal_result" in cli_payload
        assert "verified_verdict" in cli_payload

    monkeypatch.setenv("HERMES_KANBAN_TASK", engineer)
    shown = json.loads(kt._handle_show({"task_id": engineer}))
    assert shown["task"]["attempt_id"] == attempt
    assert shown["task"]["completion_contract"]["type"] == "git_revision"
    assert "terminal_result" in shown["task"]


def test_sol5_dashboard_api_accepts_and_returns_governance_fields(kanban_home, tmp_path):
    import importlib.util
    import sys

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_sol5", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/kanban")
    client = TestClient(app)
    created = client.post(
        "/api/plugins/kanban/tasks",
        json={
            "title": "engineer-api",
            "assignee": "implementer",
            "completion_contract": _git_contract(repo, base),
        },
    )
    assert created.status_code == 200, created.text
    task = created.json()["task"]
    tid = task["id"]
    assert task["completion_contract"]["type"] == "git_revision"
    with kb.connect() as conn:
        kb.claim_task(conn, tid)
        attempt = kb.get_task(conn, tid).attempt_id
    shown = client.get(f"/api/plugins/kanban/tasks/{tid}")
    assert shown.status_code == 200, shown.text
    body = shown.json()["task"]
    assert body["attempt_id"] == attempt
    assert body["completion_contract"]["type"] == "git_revision"
    patched = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={
            "status": "done",
            "summary": "api complete",
            "terminal_result": {
                "attempt_id": attempt,
                "repository": str(repo),
                "worktree": str(repo),
                "commit_sha": sha_b,
            },
        },
    )
    assert patched.status_code == 200, patched.text
    with kb.connect() as conn:
        assert _status(conn, tid) == "done"
        assert _row(conn, tid)["verified_revision"] == sha_b


def test_sol6_force_retains_legacy_structural_override(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="legacy-parent", assignee="worker")
        child = kb.create_task(
            conn, title="legacy-child", assignee="worker", parents=[parent],
        )
        ok, err = kb.promote_task(conn, child, actor="operator", force=True)
        assert ok is True
        assert err is None
        assert _status(conn, child) == "ready"


def test_sol6_force_cannot_bypass_required_verification(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    with kb.connect() as conn:
        engineer, _ = _create_and_claim(
            conn, "engineer", contract=_git_contract(repo, base),
        )
        ok = kb.request_review(
            conn, engineer, summary="force the narrative", force=True,
        )
        assert ok is False
        assert _status(conn, engineer) == "running"


def test_sol6_force_cannot_bypass_reconciliation(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    sha_c = _commit_file(repo, "c.txt", "C\n", "commit-C")
    with kb.connect() as conn:
        engineer, attempt = _create_and_claim(
            conn, "engineer", contract=_git_contract(repo, base),
        )
        child = kb.create_task(conn, title="child", assignee="worker", parents=[engineer])
        assert _complete_revision(
            conn, engineer, attempt_id=attempt, repo=repo, sha=sha_b,
        ) is True
        with pytest.raises(kci.CompletionIntegrityError):
            _complete_revision(
                conn, engineer, attempt_id=attempt, repo=repo, sha=sha_c,
            )
        ok, err = kb.promote_task(conn, child, actor="operator", force=True)
        assert ok is False
        assert err is not None
        assert _status(conn, child) == "todo"


# ---------------------------------------------------------------------------
# Second-round Sol corrections
# ---------------------------------------------------------------------------


def _link_in_order(conn, child: str, parents: list[str]) -> None:
    """Force parent evaluation order by deleting and reinserting links."""
    conn.execute("DELETE FROM task_links WHERE child_id = ?", (child,))
    for parent in parents:
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent, child),
        )


def test_sol7_force_structural_first_reconciliation_second_is_blocked(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    sha_c = _commit_file(repo, "c.txt", "C\n", "commit-C")
    with kb.connect() as conn:
        structural = kb.create_task(conn, title="structural", assignee="worker")
        governed, attempt = _create_and_claim(
            conn, "governed", contract=_git_contract(repo, base),
        )
        child = kb.create_task(
            conn, title="child", assignee="worker", parents=[structural, governed],
        )
        _complete_revision(conn, governed, attempt_id=attempt, repo=repo, sha=sha_b)
        with pytest.raises(kci.CompletionIntegrityError):
            _complete_revision(conn, governed, attempt_id=attempt, repo=repo, sha=sha_c)
        _link_in_order(conn, child, [structural, governed])
        ok, err = kb.promote_task(conn, child, actor="operator", force=True)
        assert ok is False
        assert err is not None
        assert "NEEDS_RECONCILIATION" in err or "reconciliation" in err.lower()
        assert _status(conn, child) == "todo"


def test_sol7_force_reconciliation_first_structural_second_is_blocked(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    sha_c = _commit_file(repo, "c.txt", "C\n", "commit-C")
    with kb.connect() as conn:
        structural = kb.create_task(conn, title="structural", assignee="worker")
        governed, attempt = _create_and_claim(
            conn, "governed", contract=_git_contract(repo, base),
        )
        child = kb.create_task(
            conn, title="child", assignee="worker", parents=[governed, structural],
        )
        _complete_revision(conn, governed, attempt_id=attempt, repo=repo, sha=sha_b)
        with pytest.raises(kci.CompletionIntegrityError):
            _complete_revision(conn, governed, attempt_id=attempt, repo=repo, sha=sha_c)
        _link_in_order(conn, child, [governed, structural])
        ok, err = kb.promote_task(conn, child, actor="operator", force=True)
        assert ok is False
        assert err is not None
        assert "NEEDS_RECONCILIATION" in err or "reconciliation" in err.lower()
        assert _status(conn, child) == "todo"


def test_sol7_force_multiple_ordinary_structural_parents_still_allowed(kanban_home):
    with kb.connect() as conn:
        first = kb.create_task(conn, title="first", assignee="worker")
        second = kb.create_task(conn, title="second", assignee="worker")
        child = kb.create_task(
            conn, title="child", assignee="worker", parents=[first, second],
        )
        ok, err = kb.promote_task(conn, child, actor="operator", force=True)
        assert ok is True
        assert err is None
        assert _status(conn, child) == "ready"


def test_sol8_archived_unexecuted_governed_parent_does_not_release_child(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    with kb.connect() as conn:
        parent, _ = _create_and_claim(conn, "parent", contract=_git_contract(repo, base))
        child = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
        assert kb.archive_task(conn, parent) is True
        kb.recompute_ready(conn)
        assert _status(conn, parent) == "archived"
        assert _status(conn, child) == "todo"
        assert kb.claim_task(conn, child) is None


def test_sol8_archived_governed_intermediate_does_not_release_leaf(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    with kb.connect() as conn:
        parent, _ = _create_and_claim(conn, "parent", contract=_git_contract(repo, base))
        mid = kb.create_task(
            conn, title="mid", assignee="worker", parents=[parent],
            completion_contract=_git_contract(repo, base),
        )
        leaf = kb.create_task(conn, title="leaf", assignee="worker", parents=[mid])
        assert kb.archive_task(conn, mid) is True
        kb.recompute_ready(conn)
        assert _status(conn, leaf) == "todo"
        assert kb.claim_task(conn, leaf) is None


def test_sol8_verified_then_archived_governed_parent_still_satisfies(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    with kb.connect() as conn:
        parent, attempt = _create_and_claim(
            conn, "parent", contract=_git_contract(repo, base),
        )
        child = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
        assert _complete_revision(conn, parent, attempt_id=attempt, repo=repo, sha=sha_b)
        assert kb.archive_task(conn, parent) is True
        kb.recompute_ready(conn)
        assert _status(conn, child) == "ready"


def test_sol8_legacy_ungated_archive_still_releases_child(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="legacy-parent", assignee="worker")
        child = kb.create_task(conn, title="legacy-child", assignee="worker", parents=[parent])
        assert kb.archive_task(conn, parent) is True
        kb.recompute_ready(conn)
        assert _status(conn, child) == "ready"


def test_sol9_reconciliation_recursively_invalidates_ready_grandchild(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    sha_b = _commit_file(repo, "b.txt", "B\n", "commit-B")
    sha_c = _commit_file(repo, "c.txt", "C\n", "commit-C")
    with kb.connect() as conn:
        ancestor, attempt = _create_and_claim(
            conn, "ancestor", contract=_git_contract(repo, base),
        )
        child = kb.create_task(conn, title="child", assignee="worker", parents=[ancestor])
        grandchild = kb.create_task(
            conn, title="grandchild", assignee="worker", parents=[child],
        )
        assert _complete_revision(conn, ancestor, attempt_id=attempt, repo=repo, sha=sha_b)
        kb.recompute_ready(conn)
        kb.claim_task(conn, child)
        assert kb.complete_task(conn, child, summary="child done") is True
        kb.recompute_ready(conn)
        assert _status(conn, grandchild) == "ready"
        with pytest.raises(kci.CompletionIntegrityError):
            _complete_revision(conn, ancestor, attempt_id=attempt, repo=repo, sha=sha_c)
        assert _row(conn, ancestor)["needs_reconciliation"] == 1
        assert _status(conn, child) == "done"
        assert _status(conn, grandchild) == "todo"
        assert kb.claim_task(conn, grandchild) is None


def test_sol10_integer_max_commits_is_accepted(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    contract = _git_contract(repo, base)
    contract["max_commits"] = 2
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="engineer", assignee="implementer", completion_contract=contract,
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        stored = task.completion_contract
        assert stored is not None
        assert stored["max_commits"] == 2


@pytest.mark.parametrize(
    "raw",
    ["two", "1.5", "-1", 0, True, {"n": 2}, [2], 1.5, "2", 2.0],
)
def test_sol10_malformed_max_commits_is_invalid_contract(kanban_home, tmp_path, raw):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    contract = _git_contract(repo, base)
    contract["max_commits"] = raw
    with kb.connect() as conn:
        with pytest.raises(kci.CompletionIntegrityError) as exc:
            kb.create_task(
                conn, title="engineer", assignee="implementer", completion_contract=contract,
            )
        assert exc.value.code == kci.REASON_INVALID


def test_sol11_model_tool_schema_publishes_terminal_result_fields():
    from tools import kanban_tools as kt

    props = kt.KANBAN_COMPLETE_SCHEMA["parameters"]["properties"]["terminal_result"]
    assert props["type"] == "object"
    published = props["properties"]
    for key in ("attempt_id", "commit_sha", "reviewed_sha", "repository", "worktree", "verdict"):
        assert key in published
    create_desc = kt.KANBAN_CREATE_SCHEMA["parameters"]["properties"]["completion_contract"]["description"]
    assert '"type": "historian_certify"' in create_desc
    assert '"repository": "/abs/repo"' in create_desc
    assert '"required_sha"' in create_desc


def test_sol11_cli_create_and_show_parse_governance_contract(kanban_home, tmp_path):
    import argparse

    from hermes_cli import kanban as kc

    repo = tmp_path / "repo"
    base = _init_repo(repo)
    args = argparse.Namespace(
        title="engineer",
        body=None,
        assignee="implementer",
        created_by=None,
        workspace="scratch",
        branch=None,
        project=None,
        tenant=None,
        priority=0,
        parent=None,
        triage=False,
        idempotency_key=None,
        max_runtime=None,
        skills=None,
        max_retries=None,
        model_override=None,
        provider_override=None,
        goal_mode=False,
        goal_max_turns=None,
        initial_status="running",
        json=True,
        completion_contract=json.dumps(_git_contract(repo, base)),
    )
    rc = kc._cmd_create(args)
    assert rc == 0
    with kb.connect() as conn:
        tasks = kb.list_tasks(conn)
        assert len(tasks) == 1
        tid = tasks[0].id
        assert tasks[0].completion_contract["type"] == "git_revision"
    shown = json.loads(kc.run_slash(f"show {tid} --json"))
    assert shown["task"]["completion_contract"]["type"] == "git_revision"
    assert "attempt_id" in shown["task"]
    assert "terminal_result" in shown["task"]
