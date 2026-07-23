"""HER-95 — RED probes for the AppSec / concurrency blockers raised by the two
exact-head reviews of PR #69955.

Each test reproduces one blocker before the fix and pins the fail-closed
behavior after it:

1. ``process_start_time=None`` must never be classified ``reused`` (a live owner
   with no recorded start baseline stays ``alive`` and non-reclaimable).
2. Same-session rebind from an owned worktree W1 onto a worktree W2 that is
   already owned by another lane must refuse — never rewrite the owner, never
   create two owners for W2.
3. A secret-like ``gateway_session_key`` (e.g. ``bot_token=sekret-123``) must be
   rejected before any owner file is written; a benign routing key survives.
4. An ancestor-symlink swap of ``registry/locks`` between validation and the
   write must never let an owner.json escape the registry.
"""

import importlib.util
import json
import os
import shutil
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
        timeout=30,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result


def load_factory_lane():
    spec = importlib.util.spec_from_file_location("factory_lane_appsec_uut", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_owner(registry: Path, key: str):
    return json.loads((registry / "locks" / key / "owner.json").read_text())


def owner_worktrees(registry: Path):
    return [
        json.loads(p.read_text()).get("worktree")
        for p in (registry / "locks").glob("*/owner.json")
    ]


# ---------------------------------------------------------------------------
# Blocker 4 — process_start_time=None must never be "reused"
# ---------------------------------------------------------------------------

def test_missing_recorded_start_time_is_alive_not_reused(monkeypatch):
    """A live PID whose owner has no recorded process_start_time cannot be
    proven to be a reused PID, so it must be treated as ``alive`` — the current
    code wrongly returns ``reused`` because ``None != "<real-start>"``."""
    module = load_factory_lane()
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(module, "_get_process_state_char", lambda pid: "S")
    monkeypatch.setattr(module, "_get_process_start_time", lambda pid: "real-start")

    state = module.determine_process_state({"pid": 4321, "process_start_time": None})
    assert state == "alive", state


def test_live_owner_with_missing_start_time_is_not_reclaimable():
    """End-to-end: a genuinely live owner PID with no recorded start time must
    not be reclaimable, even with an expired heartbeat and an idle worktree."""
    import tempfile

    registry = Path(tempfile.mkdtemp()) / "registry"
    worktree = Path(tempfile.mkdtemp()) / "repo"
    worktree.mkdir(parents=True)

    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run_lane(registry, "claim", "HER-95", "--agent", "default",
                 "--session", "old", "--worktree", str(worktree), check=True)
        owner_file = registry / "locks" / "HER-95" / "owner.json"
        owner = json.loads(owner_file.read_text())
        owner.update({
            "pid": holder.pid,
            "process_start_time": None,
            "heartbeat_at": time.time() - 7200,
            "ttl_hours": 0.001,
        })
        owner_file.write_text(json.dumps(owner), encoding="utf-8")
        idle = time.time() - (25 * 3600)
        os.utime(worktree, (idle, idle))

        result = run_lane(
            registry, "claim", "HER-96", "--agent", "default", "--session", "new",
            "--worktree", str(worktree), "--reclaim-worktree", "--ttl-hours", "0.001",
        )
        assert result.returncode != 0, "live owner without start baseline was reclaimed"
        assert owner_file.exists()
        assert not (registry / "locks" / "HER-96" / "owner.json").exists()
    finally:
        holder.terminate()
        holder.wait(timeout=10)


# ---------------------------------------------------------------------------
# Blocker 1 — same-session rebind onto an already-owned worktree
# ---------------------------------------------------------------------------

def test_same_session_rebind_onto_owned_worktree_is_refused(tmp_path):
    registry = tmp_path / "registry"
    w1 = tmp_path / "w1"
    w1.mkdir()
    w2 = tmp_path / "w2"
    w2.mkdir()

    # Another lane already owns W2.
    run_lane(registry, "claim", "HER-96", "--agent", "other", "--session", "sX",
             "--worktree", str(w2), check=True)
    # Our session owns HER-95 on W1.
    run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "s1",
             "--worktree", str(w1), check=True)

    # Same session tries to rebind HER-95 from W1 onto the already-owned W2.
    result = run_lane(registry, "claim", "HER-95", "--agent", "default",
                      "--session", "s1", "--worktree", str(w2))

    assert result.returncode != 0, "rebind onto an owned worktree was allowed"
    assert "already claimed" in result.stderr
    # HER-95 must remain on W1 — no silent owner rewrite.
    assert read_owner(registry, "HER-95")["worktree"] == os.path.realpath(str(w1))
    # Exactly one owner for W2, and it is still HER-96.
    w2_real = os.path.realpath(str(w2))
    holders = [wt for wt in owner_worktrees(registry) if wt == w2_real]
    assert len(holders) == 1
    assert read_owner(registry, "HER-96")["worktree"] == w2_real


def test_same_session_reentrant_same_worktree_still_heartbeats(tmp_path):
    """The rebind guard must not break the legitimate reentrant path: the same
    session re-claiming the SAME worktree only refreshes the heartbeat."""
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()

    run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "s1",
             "--worktree", str(worktree), check=True)
    before = read_owner(registry, "HER-95")["heartbeat_at"]
    time.sleep(0.02)
    again = run_lane(registry, "claim", "HER-95", "--agent", "default",
                     "--session", "s1", "--worktree", str(worktree))
    assert again.returncode == 0, again.stderr
    assert read_owner(registry, "HER-95")["heartbeat_at"] > before


# ---------------------------------------------------------------------------
# Blocker 3 — gateway_session_key must not persist secret-like values
# ---------------------------------------------------------------------------

def test_secret_like_gateway_session_key_is_rejected(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()

    result = run_lane(
        registry, "claim", "HER-95", "--agent", "default", "--session", "s1",
        "--worktree", str(worktree), "--gateway-session-key", "bot_token=sekret-123",
    )

    assert result.returncode != 0, "secret-like gateway_session_key was accepted"
    assert "secret" in result.stderr.lower()
    owner_file = registry / "locks" / "HER-95" / "owner.json"
    if owner_file.exists():
        assert "sekret-123" not in owner_file.read_text()


def test_benign_gateway_session_key_is_preserved(tmp_path):
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()

    result = run_lane(
        registry, "claim", "HER-95", "--agent", "default", "--session", "s1",
        "--worktree", str(worktree), "--gateway-session-key", "telegram:12345:67",
    )
    assert result.returncode == 0, result.stderr
    assert read_owner(registry, "HER-95")["gateway_session_key"] == "telegram:12345:67"


# ---------------------------------------------------------------------------
# Blocker 2 — ancestor symlink swap must never escape the registry
# ---------------------------------------------------------------------------

def test_ancestor_symlink_swap_during_claim_never_escapes_registry(tmp_path, monkeypatch):
    """Adversarial TOCTOU probe: swap ``registry/locks`` for a symlink pointing
    OUTSIDE the registry after path validation but before the owner write. The
    write must never land under the attacker-controlled target. ``O_NOFOLLOW``
    on the leaf alone does not stop this — the fix must resolve every ancestor
    via dirfd/openat and fail closed."""
    module = load_factory_lane()

    registry = tmp_path / "registry"
    (registry / "locks").mkdir(parents=True)
    (registry / "lanes").mkdir(parents=True)
    worktree = tmp_path / "repo"
    worktree.mkdir()
    evil = tmp_path / "evil"
    (evil / "HER-95").mkdir(parents=True)

    root = module._safe_registry_root(str(registry))

    original_build_owner = module._build_owner
    fired = {"done": False}

    def swap_then_build(*args, **kwargs):
        # Runs after _safe_subdir validation, immediately before the owner is
        # written: rename the real locks/ aside and drop a symlink to `evil`.
        if not fired["done"]:
            fired["done"] = True
            shutil.move(str(registry / "locks"), str(registry / "locks-real"))
            os.symlink(str(evil), str(registry / "locks"), target_is_directory=True)
        return original_build_owner(*args, **kwargs)

    monkeypatch.setattr(module, "_build_owner", swap_then_build)

    try:
        module._claim_under_gate(
            root, "HER-95", "default", "s1", str(worktree), False, 72.0,
        )
    except module.RegistryError:
        # Failing closed is an acceptable outcome; escaping the registry is not.
        pass

    assert not (evil / "HER-95" / "owner.json").exists(), (
        "owner.json escaped the registry through a swapped ancestor symlink"
    )
