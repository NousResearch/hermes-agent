from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_worker_receipts import (
    _worktree_fingerprint,
    begin_worker_receipt,
    finalize_worker_receipt,
)
from hermes_state import SessionDB


def test_worker_lifecycle_records_route_context_and_usage_delta(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="instrument", body="measure this run", assignee="implementer",
            skills=["test-driven-development"],
        )
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None and claimed.current_run_id is not None
        run_id = claimed.current_run_id

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("HERMES_PROFILE", "implementer")

    session_db = SessionDB(home / "state.db")
    session_db.create_session("session-1", source="kanban")
    session_db.update_token_counts(
        "session-1", input_tokens=10, output_tokens=2, model="effective-model",
        billing_provider="effective-provider", api_call_count=1,
    )
    agent = SimpleNamespace(
        session_id="session-1", provider="effective-provider", model="effective-model",
        reasoning_config={"enabled": True, "effort": "high"},
        _cached_system_prompt="trusted system prompt",
    )
    cli = SimpleNamespace(
        agent=agent, session_id="session-1", _session_db=session_db,
        requested_provider="requested-provider", model="effective-model",
        reasoning_config={"effort": "medium"}, enabled_toolsets=["file", "terminal"],
        _resumed=True,
    )

    state = begin_worker_receipt(cli)
    assert state is not None
    with kbc.connect() as conn:
        receipt = kb.get_run_receipt(conn, run_id)
        assert receipt is not None
        assert receipt.runner == "herdr"
        assert receipt.worker_session_id == "session-1"
        assert receipt.profile == "implementer"
        assert receipt.requested_provider == "requested-provider"
        assert receipt.effective_provider == "effective-provider"
        assert receipt.effective_model == "effective-model"
        assert receipt.requested_reasoning == "medium"
        assert receipt.effective_reasoning == "high"
        assert receipt.fresh_or_resumed == "resumed"
        assert receipt.context_chars > 0
        assert len(receipt.context_fingerprint or "") == 64
        assert len(receipt.system_prompt_hash or "") == 64
        assert len(receipt.toolset_hash or "") == 64
        assert len(receipt.skills_hash or "") == 64

    session_db.update_token_counts(
        "session-1", input_tokens=90, output_tokens=18, cache_read_tokens=40,
        reasoning_tokens=5, estimated_cost_usd=0.12, model="effective-model",
        billing_provider="effective-provider", api_call_count=2,
    )
    session_db.create_session(
        "session-2", source="compression", parent_session_id="session-1",
    )
    session_db.update_token_counts(
        "session-2", input_tokens=7, output_tokens=3, cache_read_tokens=50,
        reasoning_tokens=2, estimated_cost_usd=0.03, model="effective-model",
        billing_provider="effective-provider", api_call_count=1,
    )
    agent.session_id = "session-2"
    with kbc.connect() as conn:
        kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)

    assert finalize_worker_receipt(cli, state)
    with kbc.connect() as conn:
        receipt = kb.get_run_receipt(conn, run_id)
        assert receipt is not None
        assert receipt.receipt_completeness == "complete"
        assert receipt.api_call_delta == 3
        assert receipt.input_token_delta == 97
        assert receipt.output_token_delta == 21
        assert receipt.cache_read_token_delta == 90
        assert receipt.reasoning_token_delta == 7

    session_db.close()


def test_worktree_fingerprint_tracks_head_diff_and_untracked_content(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "receipt@example.test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Receipt Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)

    clean = _worktree_fingerprint(repo)
    assert clean is not None and clean.startswith("git-v1:")
    tracked.write_text("two\n", encoding="utf-8")
    dirty = _worktree_fingerprint(repo)
    assert dirty is not None and dirty != clean
    untracked = repo / "new.txt"
    untracked.write_text("first\n", encoding="utf-8")
    first_untracked = _worktree_fingerprint(repo)
    untracked.write_text("second\n", encoding="utf-8")
    assert _worktree_fingerprint(repo) != first_untracked
