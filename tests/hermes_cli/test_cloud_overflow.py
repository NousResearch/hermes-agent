"""Focused tests for the READY-only cloud-overflow preparation seam."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.cloud_overflow import (
    EXCLUDED_CLASSES,
    BoardSnapshot,
    ClaudeCloudAdapter,
    CodexCloudAdapter,
    CommentWriteError,
    CursorCloudAdapter,
    LaunchResult,
    OverflowState,
    ProviderRefused,
    TaskSnapshot,
    eligibility,
    exponential_backoff,
    load_fixture,
    record_launch,
    run_tick,
    sanitize_environment,
    sanitize_receipt,
    source_revision,
)


@pytest.fixture
def task() -> TaskSnapshot:
    return TaskSnapshot(id="t_docs", title="Documentation draft", skills=("docs",))


def test_only_explicit_docs_or_research_is_eligible(task):
    assert eligibility(task) == (True, "eligible", "docs")
    assert eligibility(TaskSnapshot(id="t_r", title="Research", skills=("research",)))[0]
    assert eligibility(TaskSnapshot(id="t_amb", title="Research", skills=()))[1] == "work_class_not_explicit"
    assert eligibility(TaskSnapshot(id="t_both", title="Mixed", skills=("docs", "research")))[1] == "work_class_ambiguous"
    assert eligibility(TaskSnapshot(id="t_meta", title="Metadata", metadata={"work_class": "documentation"}))[0]


@pytest.mark.parametrize("excluded", sorted(EXCLUDED_CLASSES))
def test_every_exclusion_fails_closed(excluded):
    value = TaskSnapshot(id="t_x", title="Plain title", skills=("docs",), metadata={"risk": excluded, "labels": [excluded]})
    # risk alone is not a supported classification field, but labels are.
    value = TaskSnapshot(id=value.id, title=value.title, skills=value.skills, labels=(excluded,))
    assert eligibility(value)[0] is False
    assert eligibility(value)[1] == f"excluded:{excluded}"


def test_ready_parent_and_claim_gates(task):
    assert eligibility(TaskSnapshot(**{**task.__dict__, "status": "running"}))[1] == "source_not_ready"
    assert eligibility(TaskSnapshot(**{**task.__dict__, "claim_lock": "leased"}))[1] == "source_claimed"
    assert eligibility(TaskSnapshot(**{**task.__dict__, "parents_satisfied": False}))[1] == "parents_not_satisfied"


def test_saturation_and_max_one_tick(tmp_path, task):
    state = OverflowState(tmp_path / "state.sqlite3", max_concurrency=3)
    adapters = {"cursor-cloud": CursorCloudAdapter("cursor-cloud", plan_authenticated=True, isolated_checkout="fixture")}
    board = BoardSnapshot("board-a", running=3, max_spawn=3, tasks=(task,))
    first = run_tick((board,), state=state, adapters=adapters, now=100)
    second = run_tick((board,), state=state, adapters=adapters, now=101)
    assert first.status == "planned"
    assert first.action == "prepare-only"
    assert second.reason == "duplicate_lease"
    assert state.get(first.idempotency_key)["status"] == "planned"
    assert run_tick((BoardSnapshot("board-a", 2, 3, (task,)),), state=state, adapters=adapters).reason == "no_eligible_candidate"


def test_toctou_reread_rejects_changed_source(tmp_path, task):
    state = OverflowState(tmp_path / "state.sqlite3")
    changed = TaskSnapshot(id=task.id, title="Changed", skills=task.skills)
    board = BoardSnapshot("board-a", 3, 3, (task,), reread=lambda _: changed)
    result = run_tick((board,), state=state, adapters={"cursor-cloud": CursorCloudAdapter("cursor-cloud", plan_authenticated=True, isolated_checkout="x")})
    assert result.reason == "no_eligible_candidate"


def test_atomic_idempotency_under_contention(tmp_path, task):
    path = tmp_path / "state.sqlite3"
    def acquire():
        store = OverflowState(path)
        return store.acquire(board="b", task=task, provider="cursor-cloud", now=1)[0]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: acquire(), range(8)))
    assert sum(results) == 1


def test_saturation_dry_run_cli_is_fixture_only(tmp_path):
    fixture = Path(__file__).parent.parent / "fixtures" / "cloud_overflow.json"
    state = tmp_path / "cli-state.sqlite3"
    cmd = [sys.executable, "-m", "hermes_cli.main", "kanban", "cloud-overflow", "--fixture", str(fixture), "--state", str(state), "--dry-run", "--json"]
    first = subprocess.run(cmd, capture_output=True, text=True, check=False)
    second = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["action"] == "prepare-only"
    assert json.loads(second.stdout)["reason"] == "duplicate_lease"
    assert "cloud-agent" not in first.stdout
    assert "claude --cloud" not in first.stdout
    assert "codex cloud exec" not in first.stdout


def test_provider_argv_timeout_and_parsed_identity(monkeypatch, task):
    calls = []
    def runner(argv, *, timeout, env):
        calls.append((tuple(argv), timeout, dict(env)))
        return SimpleNamespace(stdout='{"session_id":"sess-1", "url":"https://example.test/sess-1"}')
    adapter = ClaudeCloudAdapter("claude-cloud", plan_authenticated=True, isolated_checkout="/repo/wt", timeout=17, runner=runner)
    result = adapter.launch(task)
    assert result == LaunchResult("sess-1", "https://example.test/sess-1", ("claude", "--cloud", "--worktree", "/repo/wt", "ISO HOLD; draft PR only; task t_docs; title: Documentation draft. No merge, deploy, live trading, credential access, or schedule activation."))
    assert calls[0][1] == 17
    assert calls[0][0][0] == "claude"
    assert "shell" not in calls[0][2]


def test_provider_output_regex_and_env_sanitization(task):
    captured = {}
    def runner(argv, *, timeout, env):
        captured.update(env)
        return SimpleNamespace(stdout="session_id: sess-2 https://example.test/sess-2")
    adapter = CursorCloudAdapter("cursor-cloud", plan_authenticated=True, isolated_checkout="x", runner=runner)
    assert adapter.launch(task).session_id == "sess-2"
    clean = sanitize_environment({"PATH": "/bin", "HOME": "/tmp", "API_KEY": "do-not-pass", "TOKEN": "do-not-pass"})
    assert clean == {"PATH": "/bin", "HOME": "/tmp"}
    assert "API_KEY" not in captured


@pytest.mark.parametrize("env_id,approval", [(None, "exact"), ("env_1", None)])
def test_codex_requires_both_gates(env_id, approval, task):
    adapter = CodexCloudAdapter(env_id=env_id, exact_spend_approval=approval, plan_authenticated=True, isolated_checkout="x")
    assert not adapter.available
    with pytest.raises(ProviderRefused):
        adapter.build_argv(task)


def test_codex_argv_is_explicit_and_never_falls_back(task):
    adapter = CodexCloudAdapter(env_id="env_exact", exact_spend_approval="approval-exact", plan_authenticated=True, isolated_checkout="x")
    assert adapter.build_argv(task)[:5] == ("codex", "cloud", "exec", "--env", "env_exact")


def test_receipt_is_allowlisted_and_no_prompt_or_secret():
    receipt = sanitize_receipt(provider="claude-cloud", session_id="s", url="https://example.test/s", branch="wt/x", workspace="/tmp/x", status="launched", idempotency_key="b:t:r:p")
    assert set(receipt) == {"provider", "external_session_id", "external_session_url", "isolated_branch", "isolated_workspace", "status", "idempotency_key"}
    assert "prompt" not in json.dumps(receipt).lower()


def test_comment_failure_marks_launch_unresolved(tmp_path, task):
    state = OverflowState(tmp_path / "state.sqlite3")
    ok, _, key = state.acquire(board="b", task=task, provider="claude-cloud", now=1)
    assert ok
    def runner(argv, *, timeout, env):
        return SimpleNamespace(stdout='{"session_id":"s"}')
    adapter = ClaudeCloudAdapter("claude-cloud", plan_authenticated=True, isolated_checkout="x", runner=runner)
    with pytest.raises(CommentWriteError):
        record_launch(state=state, lease_key=key, board="b", task=task, adapter=adapter, comment_writer=lambda *_: (_ for _ in ()).throw(RuntimeError("no comment")), approved=True, now=2)
    assert state.get(key)["status"] == "unresolved"
    assert state.get(key)["veto"] == "receipt_comment_failed"


def test_backoff_is_exponential_and_bounded():
    assert exponential_backoff(0, base_seconds=3, cap_seconds=10) == 3
    assert exponential_backoff(2, base_seconds=3, cap_seconds=10) == 10
    assert exponential_backoff(99, base_seconds=3, cap_seconds=10) == 10


def test_pauses_and_kill_switch_do_not_lease(tmp_path, task):
    state = OverflowState(tmp_path / "state.sqlite3")
    adapter = {"cursor-cloud": CursorCloudAdapter("cursor-cloud", plan_authenticated=True, isolated_checkout="x")}
    board = BoardSnapshot("b", 3, 3, (task,))
    assert run_tick((board,), state=state, adapters=adapter, fleet_paused=True).status == "blocked"
    assert run_tick((board,), state=state, adapters=adapter, kill_switch=True).status == "blocked"
    assert not list((tmp_path).glob("*.sqlite3-wal")) or state.get("unused") is None


def test_fixture_shape_and_revision_are_deterministic(task):
    fixture = Path(__file__).parent.parent / "fixtures" / "cloud_overflow.json"
    boards = load_fixture(fixture)
    assert boards[0].saturated
    assert source_revision(task) == source_revision(task)
