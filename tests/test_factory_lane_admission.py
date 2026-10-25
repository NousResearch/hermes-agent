import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "factory_lane.py"


def run_lane(registry, *args, check=False, cwd=None):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--registry", str(registry), *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result


def load_factory_lane():
    spec = importlib.util.spec_from_file_location("factory_lane_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_git_worktree(path: Path):
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("ok\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def read_owner(registry: Path, key: str):
    return json.loads((registry / "locks" / key / "owner.json").read_text())


def test_different_issues_cannot_claim_same_realpath_via_trailing_slash_or_symlink(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()
    link = tmp_path / "repo-link"
    link.symlink_to(worktree, target_is_directory=True)

    first = run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "s1", "--worktree", f"{worktree}/")
    assert first.returncode == 0, first.stderr

    second = run_lane(registry, "claim", "HER-96", "--agent", "hermes-immo", "--session", "s2", "--worktree", str(link))
    assert second.returncode != 0
    assert "worktree already claimed" in second.stderr
    assert not (registry / "locks" / "HER-96" / "owner.json").exists()


def test_owner_reentrant_claim_refreshes_heartbeat_without_conflict(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()

    assert run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "s1", "--worktree", str(worktree)).returncode == 0
    before = read_owner(registry, "HER-95")["heartbeat_at"]
    time.sleep(0.01)
    again = run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "s1", "--worktree", str(worktree))
    assert again.returncode == 0, again.stderr
    after = read_owner(registry, "HER-95")["heartbeat_at"]
    assert after > before


def test_reviewer_admission_coexists_without_touching_owner(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()
    assert run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "owner", "--worktree", str(worktree)).returncode == 0
    before = read_owner(registry, "HER-95")

    result = run_lane(
        registry,
        "admit",
        "HER-96",
        "--mode",
        "reviewer",
        "--agent",
        "opus-reviewer",
        "--session",
        "review",
        "--worktree",
        str(worktree),
        "--json",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["decision"] == "reviewer_allowed"
    assert read_owner(registry, "HER-95") == before
    assert not (registry / "locks" / "HER-96" / "owner.json").exists()


def test_hard_admission_refuses_second_live_owner_before_lane_mutation(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()
    assert run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "owner", "--worktree", str(worktree)).returncode == 0

    result = run_lane(
        registry,
        "admit",
        "HER-96",
        "--mode",
        "owner",
        "--hard",
        "--agent",
        "hermes-immo",
        "--session",
        "intruder",
        "--worktree",
        str(worktree),
    )

    assert result.returncode != 0
    assert "worktree already claimed" in result.stderr
    assert not (registry / "lanes" / "HER-96.jsonl").exists()


def test_stale_dead_owner_can_be_reclaimed_only_after_ttl_and_inactive_worktree(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()
    assert run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "old", "--worktree", str(worktree)).returncode == 0
    owner_file = registry / "locks" / "HER-95" / "owner.json"
    owner = json.loads(owner_file.read_text())
    owner.update({"pid": 987654321, "heartbeat_at": time.time() - 7200, "ttl_hours": 0.001})
    owner_file.write_text(json.dumps(owner), encoding="utf-8")
    old = time.time() - (25 * 3600)
    os.utime(worktree, (old, old))

    result = run_lane(
        registry,
        "claim",
        "HER-96",
        "--agent",
        "default",
        "--session",
        "new",
        "--worktree",
        str(worktree),
        "--reclaim-worktree",
        "--ttl-hours",
        "0.001",
    )

    assert result.returncode == 0, result.stderr
    assert not owner_file.exists()
    assert read_owner(registry, "HER-96")["session_id"] == "new"


def test_fresh_or_active_ownerless_dirty_worktree_is_refused_before_build(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_worktree(worktree)
    (worktree / "dirty.txt").write_text("untracked\n", encoding="utf-8")

    result = run_lane(
        registry,
        "admit",
        "HER-95",
        "--mode",
        "owner",
        "--hard",
        "--agent",
        "default",
        "--session",
        "s1",
        "--worktree",
        str(worktree),
    )

    assert result.returncode != 0
    assert "dirty ownerless worktree" in result.stderr
    assert not (registry / "locks" / "HER-95" / "owner.json").exists()


def test_toctou_parallel_hard_admission_has_exactly_one_winner(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()
    commands = [
        [sys.executable, str(SCRIPT), "--registry", str(registry), "admit", key, "--mode", "owner", "--hard", "--agent", agent, "--session", agent, "--worktree", str(worktree)]
        for key, agent in (("HER-95", "default"), ("HER-96", "hermes-immo"))
    ]

    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for cmd in commands]
    results = [p.communicate(timeout=20) + (p.returncode,) for p in procs]

    assert sorted(r[2] for r in results) == [0, 1]
    owners = list((registry / "locks").glob("*/owner.json"))
    assert len(owners) == 1


def test_advisory_hook_fails_open_when_registry_is_corrupt(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_worktree(worktree)
    owner_dir = registry / "locks" / "HER-95"
    owner_dir.mkdir(parents=True)
    (owner_dir / "owner.json").write_text("{not-json", encoding="utf-8")

    result = run_lane(registry, "hook-session-start", "--repo", str(worktree))

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_advisory_hook_prints_stop_for_different_live_owner(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_worktree(worktree)
    assert run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "owner", "--worktree", str(worktree)).returncode == 0

    result = run_lane(
        registry,
        "hook-session-start",
        "--repo",
        str(worktree),
        "--agent",
        "hermes-immo",
        "--session",
        "continue",
    )

    assert result.returncode == 0
    assert "STOP: worktree already owned" in result.stdout
    assert "HER-95" in result.stdout
    assert "default/owner" in result.stdout


def test_business_profile_cannot_hard_admit_product_lane_outside_allowed_prefixes(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()

    result = run_lane(
        registry,
        "admit",
        "SCA-740",
        "--mode",
        "owner",
        "--hard",
        "--agent",
        "hermes-immo",
        "--profile",
        "hermes-immo",
        "--domain-prefixes",
        "JYI,HER",
        "--session",
        "continue",
        "--worktree",
        str(worktree),
    )

    assert result.returncode != 0
    assert "profile hermes-immo cannot own lane SCA-740" in result.stderr
    assert not (registry / "locks" / "SCA-740" / "owner.json").exists()


def test_process_start_time_mismatch_marks_owner_as_reused(monkeypatch):
    module = load_factory_lane()
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(module, "_get_process_state_char", lambda pid: "S")
    monkeypatch.setattr(module, "_get_process_start_time", lambda pid: "new-start")

    assert module.determine_process_state({"pid": 123, "process_start_time": "old-start"}) == "reused"


def test_close_releases_owner_after_handoff_flow(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()
    assert run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "s1", "--worktree", str(worktree)).returncode == 0

    result = run_lane(registry, "close", "HER-95")

    assert result.returncode == 0, result.stderr
    assert not (registry / "locks" / "HER-95" / "owner.json").exists()
