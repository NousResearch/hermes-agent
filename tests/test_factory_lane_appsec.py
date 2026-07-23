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

import pytest


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

    assert not (evil / "HER-95" / "owner.json").exists()


def test_runtime_owner_scan_rejects_locks_swap_after_open(tmp_path, monkeypatch):
    """A locks directory swapped after its no-follow open cannot become an
    ownerless result: the runtime guard must detect the inode change and block."""
    module = load_factory_lane()
    registry = tmp_path / "registry"
    locks = registry / "locks"
    locks.mkdir(parents=True)
    root = module._safe_registry_root(str(registry))
    original_listdir = module.os.listdir
    swapped = {"done": False}

    def list_then_swap(fd):
        names = original_listdir(fd)
        if not swapped["done"]:
            swapped["done"] = True
            shutil.move(str(locks), str(registry / "locks-old"))
            locks.mkdir()
        return names

    monkeypatch.setattr(module.os, "listdir", list_then_swap)

    with pytest.raises(module.RegistryError, match="changed during owner scan"):
        module._find_claim_for_worktree(root, str(tmp_path / "repo"))


def test_atomic_owner_replace_cannot_escape_after_text_path_swap(tmp_path, monkeypatch):
    """A text-path replace after fstat/stat can be redirected by a post-check
    swap.  Owner writes must instead rename relative to the open directory fd."""
    module = load_factory_lane()
    registry = tmp_path / "registry"
    parent = registry / "locks" / "HER-95"
    parent.mkdir(parents=True)
    evil = tmp_path / "evil"
    evil.mkdir()
    external_owner = evil / "owner.json"
    external_owner.write_text('{"external": true}\n', encoding="utf-8")
    root = module._safe_registry_root(str(registry))
    parent_fd = module._open_dir_chain(str(root), ("locks", "HER-95"), create=False)
    original_replace = module.os.replace
    original_rename = module.os.rename
    swapped = False

    def swap_parent():
        nonlocal swapped
        if swapped:
            return
        swapped = True
        (registry / "locks-real").mkdir()
        original_rename(str(parent), str(registry / "locks-real" / "HER-95"))
        os.symlink(str(evil), str(parent), target_is_directory=True)

    def swap_then_replace(source, target, *args, **kwargs):
        swap_parent()
        (evil / Path(source).name).write_text('{"attacker": true}\n', encoding="utf-8")
        return original_replace(source, target, *args, **kwargs)

    def swap_then_rename(source, target, *args, **kwargs):
        swap_parent()
        return original_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(module.os, "replace", swap_then_replace)
    monkeypatch.setattr(module.os, "rename", swap_then_rename)
    try:
        module._write_json_at(parent_fd, str(parent), "owner.json", {"safe": True})
    finally:
        os.close(parent_fd)

    assert external_owner.read_text(encoding="utf-8") == '{"external": true}\n'


def test_context_output_cannot_escape_after_post_stat_parent_swap(tmp_path, monkeypatch):
    """Registry context writes must use the same dirfd rename discipline as
    owner records: a swapped ``contexts/`` path cannot redirect the output."""
    module = load_factory_lane()
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_repo(worktree)
    root = module._safe_registry_root(str(registry))
    contexts = registry / "contexts"
    contexts.mkdir()
    evil = tmp_path / "evil"
    evil.mkdir()
    external_output = evil / "HER-95.md"
    external_output.write_text("external\n", encoding="utf-8")
    external_before = external_output.read_bytes()
    original_replace = module.os.replace
    original_rename = module.os.rename
    swapped = False

    def swap_contexts():
        nonlocal swapped
        if swapped:
            return
        swapped = True
        original_rename(str(contexts), str(registry / "contexts-real"))
        os.symlink(str(evil), str(contexts), target_is_directory=True)

    def swap_then_replace(source, target, *args, **kwargs):
        swap_contexts()
        (evil / Path(source).name).write_text("attacker\n", encoding="utf-8")
        return original_replace(source, target, *args, **kwargs)

    def swap_then_rename(source, target, *args, **kwargs):
        swap_contexts()
        return original_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(module.os, "replace", swap_then_replace)
    monkeypatch.setattr(module.os, "rename", swap_then_rename)
    try:
        module.cmd_context(root, "HER-95", str(worktree), str(tmp_path / "vault"), None, None)
    except module.RegistryError:
        pass

    assert external_output.read_bytes() == external_before


def test_fifo_journal_refuses_without_waiting(tmp_path):
    """Existing journal FIFOs are malformed registry state, never a blocking
    read or a timeout-derived allow path."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO unsupported")
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    worktree.mkdir()
    run_lane(registry, "claim", "HER-95", "--agent", "default", "--session", "s1",
             "--worktree", str(worktree), check=True)
    lane_file = registry / "lanes" / "HER-95.jsonl"
    lane_file.unlink()
    os.mkfifo(lane_file)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--registry", str(registry), "event", "HER-95", "ci_green"],
        capture_output=True, text=True, timeout=1,
    )

    assert result.returncode != 0
    assert "regular file" in result.stderr


def make_git_repo(path: Path):
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=path, check=True)
    (path / "README.md").write_text("ok\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


@pytest.mark.parametrize("operation", ["event", "handoff", "close", "capture", "reconcile"])
def test_registry_mutations_reject_ancestor_symlink_swap_before_external_write(
    tmp_path, monkeypatch, operation,
):
    """Every mutating registry command must re-open its directories by dirfd.

    Swapping both ``locks`` and ``lanes`` after their initial Path preflight
    used to redirect owner and journal writes outside the registry. The command
    must now fail closed, leaving attacker-controlled records byte-for-byte
    unchanged.
    """
    module = load_factory_lane()
    key = "HER-95"
    registry = tmp_path / "registry"
    worktree = tmp_path / "repo"
    make_git_repo(worktree)
    root = module._safe_registry_root(str(registry))
    module.cmd_claim(root, key, "default", "s1", str(worktree), False, 72.0)

    handoffs_dir = registry / "handoffs"
    handoffs_dir.mkdir()
    handoff_path = handoffs_dir / f"{key}-input.handoff.json"
    handoff_path.write_text(json.dumps({"issue": key, "repo": str(worktree.resolve())}), encoding="utf-8")

    evil = tmp_path / "evil"
    evil_owner = evil / "locks" / key / "owner.json"
    evil_owner.parent.mkdir(parents=True)
    evil_owner.write_text('{"external": "owner"}\n', encoding="utf-8")
    evil_lane = evil / "lanes" / f"{key}.jsonl"
    evil_lane.parent.mkdir(parents=True)
    evil_lane.write_text('{"external": "journal"}\n', encoding="utf-8")
    external_before = {path: path.read_bytes() for path in (evil_owner, evil_lane)}

    original_safe_subdir = module._safe_subdir
    swapped = False

    def swap_after_locks_preflight(root_arg, name):
        nonlocal swapped
        result = original_safe_subdir(root_arg, name)
        if name == "locks" and not swapped:
            swapped = True
            shutil.move(str(registry / "locks"), str(registry / "locks-real"))
            shutil.move(str(registry / "lanes"), str(registry / "lanes-real"))
            os.symlink(str(evil / "locks"), str(registry / "locks"), target_is_directory=True)
            os.symlink(str(evil / "lanes"), str(registry / "lanes"), target_is_directory=True)
        return result

    monkeypatch.setattr(module, "_safe_subdir", swap_after_locks_preflight)

    with pytest.raises(module.RegistryError):
        if operation == "event":
            module.cmd_event(root, key, "ci_passed", None, None)
        elif operation == "handoff":
            module.cmd_handoff(root, key, "blocked", "repair the gate", None)
        elif operation == "close":
            module.cmd_close(root, key)
        elif operation == "capture":
            module.cmd_capture_handoff(root, key, str(worktree), None)
        else:
            module.cmd_reconcile(root, key, str(worktree), str(handoff_path))

    assert {path: path.read_bytes() for path in external_before} == external_before
