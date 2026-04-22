import importlib.util
import os
import subprocess
from pathlib import Path

import pytest


SCRIPT_ROOT = Path(os.environ.get("HERMES_SCRIPT_ROOT", str(Path.home() / ".hermes" / "scripts")))
FANOUT_SCRIPT_PATH = SCRIPT_ROOT / "minos_fanout.py"
fanout_spec = importlib.util.spec_from_file_location("minos_fanout", FANOUT_SCRIPT_PATH)
minos_fanout = importlib.util.module_from_spec(fanout_spec)
fanout_spec.loader.exec_module(minos_fanout)

WORKTREE_SCRIPT_PATH = SCRIPT_ROOT / "minos_worktree_run.py"
worktree_spec = importlib.util.spec_from_file_location("minos_worktree_run", WORKTREE_SCRIPT_PATH)
minos_worktree_run = importlib.util.module_from_spec(worktree_spec)
worktree_spec.loader.exec_module(minos_worktree_run)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "backend.txt").write_text("backend\n", encoding="utf-8")
    (repo / "frontend.txt").write_text("frontend\n", encoding="utf-8")
    _git(repo, "add", "backend.txt", "frontend.txt")
    _git(repo, "commit", "-m", "initial")
    return repo


def _task_pack(repo: Path, lane_id: str, allowed: str) -> Path:
    path = repo / ".hermes" / "task-packs" / f"{lane_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Task pack\n\nRun id: placeholder\n\n## Workspace\n- Allowed paths: {allowed}\n\n## Verifier commands\n- git diff --stat\n",
        encoding="utf-8",
    )
    return path


def _commit_task_packs(repo: Path) -> None:
    _git(repo, 'add', '.hermes/task-packs')
    _git(repo, 'commit', '-m', 'add task packs')


def test_prepare_fanout_plan_rejects_overlapping_disjoint_scopes(clean_repo: Path):
    repo = clean_repo

    tasks = [
        {"lane_id": "a", "allowed_paths": ["src/app.py"]},
        {"lane_id": "b", "allowed_paths": ["src/app.py"]},
    ]

    with pytest.raises(ValueError, match="overlap"):
        minos_fanout.prepare_fanout_plan(repo, "run-200", tasks)



def test_prepare_fanout_plan_rejects_duplicate_lane_ids(clean_repo: Path):
    repo = clean_repo
    tasks = [
        {"lane_id": "dup", "allowed_paths": ["backend.txt"]},
        {"lane_id": "dup", "allowed_paths": ["frontend.txt"]},
    ]
    with pytest.raises(ValueError, match="Duplicate lane_id"):
        minos_fanout.prepare_fanout_plan(repo, "run-dup", tasks)


def test_prepare_fanout_plan_allows_independent_approach_overlap(clean_repo: Path):
    repo = clean_repo

    tasks = [
        {"lane_id": "a", "allowed_paths": ["src/app.py"], "mode": "independent"},
        {"lane_id": "b", "allowed_paths": ["src/app.py"], "mode": "independent"},
    ]

    plan = minos_fanout.prepare_fanout_plan(repo, "run-200", tasks)
    assert len(plan["lanes"]) == 2


def test_prepare_fanout_plan_creates_separate_worktree_and_artifact_roots(clean_repo: Path):
    repo = clean_repo

    tasks = [
        {"lane_id": "backend", "allowed_paths": ["backend/"]},
        {"lane_id": "frontend", "allowed_paths": ["frontend/"]},
    ]

    plan = minos_fanout.prepare_fanout_plan(repo, "run-201", tasks)

    lane_a, lane_b = plan["lanes"]
    assert lane_a["run_id"] != lane_b["run_id"]
    assert lane_a["worktree_path"] != lane_b["worktree_path"]
    assert lane_a["artifact_dir"] != lane_b["artifact_dir"]
    assert lane_a["worktree_path"] == str(minos_worktree_run.build_worktree_path(repo, lane_a["run_id"]))
    assert lane_b["artifact_dir"] == str(minos_worktree_run.build_artifact_dir(repo, lane_b["run_id"]))


def test_prepare_fanout_plan_disables_automatic_merge(clean_repo: Path):
    repo = clean_repo

    tasks = [
        {"lane_id": "backend", "allowed_paths": ["backend/"]},
        {"lane_id": "frontend", "allowed_paths": ["frontend/"]},
    ]

    plan = minos_fanout.prepare_fanout_plan(repo, "run-202", tasks)
    assert plan["auto_merge"] is False
    assert plan["requires_explicit_review"] is True
    assert plan["gate_each_lane"] is True


def test_execute_fanout_plan_uses_separate_worktrees_without_shared_mutation(clean_repo: Path):
    repo = clean_repo
    tasks = [
        {"lane_id": "backend", "allowed_paths": ["backend.txt"], "task_pack_path": str(_task_pack(repo, 'backend', 'backend.txt'))},
        {"lane_id": "frontend", "allowed_paths": ["frontend.txt"], "task_pack_path": str(_task_pack(repo, 'frontend', 'frontend.txt'))},
    ]
    _commit_task_packs(repo)
    plan = minos_fanout.prepare_fanout_plan(repo, "run-300", tasks)
    execution = minos_fanout.execute_fanout_plan(
        plan,
        {
            'backend': ['python3', '-c', "from pathlib import Path; p=Path('backend.txt'); p.write_text('backend changed\\n', encoding='utf-8')"],
            'frontend': ['python3', '-c', "from pathlib import Path; p=Path('frontend.txt'); p.write_text('frontend changed\\n', encoding='utf-8')"],
        },
    )
    backend_status = Path(execution['lane_results'][0]['builder']['git_status_path']).read_text(encoding='utf-8')
    frontend_status = Path(execution['lane_results'][1]['builder']['git_status_path']).read_text(encoding='utf-8')
    assert 'backend.txt' in backend_status
    assert 'frontend.txt' not in backend_status
    assert 'frontend.txt' in frontend_status
    assert 'backend.txt' not in frontend_status


def test_execute_fanout_plan_allows_lane_failures_independently(clean_repo: Path):
    repo = clean_repo
    tasks = [
        {"lane_id": "success", "allowed_paths": ["backend.txt"], "task_pack_path": str(_task_pack(repo, 'success', 'backend.txt'))},
        {"lane_id": "failure", "allowed_paths": ["frontend.txt"], "task_pack_path": str(_task_pack(repo, 'failure', 'frontend.txt'))},
    ]
    _commit_task_packs(repo)
    plan = minos_fanout.prepare_fanout_plan(repo, "run-301", tasks)
    execution = minos_fanout.execute_fanout_plan(
        plan,
        {
            'success': ['python3', '-c', "from pathlib import Path; Path('backend.txt').write_text('ok\\n', encoding='utf-8')"],
            'failure': ['python3', '-c', "import sys; sys.exit(4)"],
        },
    )
    by_lane = {lane['lane_id']: lane for lane in execution['lane_results']}
    assert by_lane['success']['builder']['exit_code'] == 0
    assert by_lane['failure']['builder']['exit_code'] == 4
    assert by_lane['success']['scope_ok'] is True
    assert by_lane['failure']['scope_ok'] is True



def test_execute_fanout_plan_flags_out_of_scope_mutation(clean_repo: Path):
    repo = clean_repo
    tasks = [
        {"lane_id": "backend", "allowed_paths": ["backend.txt"], "task_pack_path": str(_task_pack(repo, 'backend', 'backend.txt'))},
        {"lane_id": "frontend", "allowed_paths": ["frontend.txt"], "task_pack_path": str(_task_pack(repo, 'frontend', 'frontend.txt'))},
    ]
    _commit_task_packs(repo)
    plan = minos_fanout.prepare_fanout_plan(repo, "run-303", tasks)
    execution = minos_fanout.execute_fanout_plan(
        plan,
        {
            'backend': ['python3', '-c', "from pathlib import Path; Path('frontend.txt').write_text('bad\\n', encoding='utf-8')"],
            'frontend': ['python3', '-c', "from pathlib import Path; Path('frontend.txt').write_text('ok\\n', encoding='utf-8')"],
        },
    )
    backend = next(l for l in execution['lane_results'] if l['lane_id'] == 'backend')
    assert backend['scope_ok'] is False
    assert 'frontend.txt' in backend['disallowed_paths']


def test_compare_lane_outcomes_requires_per_lane_gate_before_publish(clean_repo: Path):
    repo = clean_repo
    tasks = [
        {"lane_id": "a", "allowed_paths": ["backend.txt"], "task_pack_path": str(_task_pack(repo, 'a', 'backend.txt'))},
        {"lane_id": "b", "allowed_paths": ["frontend.txt"], "task_pack_path": str(_task_pack(repo, 'b', 'frontend.txt'))},
    ]
    _commit_task_packs(repo)
    plan = minos_fanout.prepare_fanout_plan(repo, "run-302", tasks)
    execution = minos_fanout.execute_fanout_plan(
        plan,
        {
            'a': ['python3', '-c', "from pathlib import Path; Path('backend.txt').write_text('ok\\n', encoding='utf-8')"],
            'b': ['python3', '-c', "from pathlib import Path; Path('frontend.txt').write_text('ok\\n', encoding='utf-8')"],
        },
    )
    comparison = minos_fanout.compare_lane_outcomes(
        execution,
        {'a': {'gate_passed': True}, 'b': {'gate_passed': False}},
    )
    assert comparison['publishable'] is False
    assert comparison['auto_merge'] is False
    assert comparison['requires_explicit_review'] is True
    assert all('scope_ok' in lane for lane in comparison['lane_summaries'])
