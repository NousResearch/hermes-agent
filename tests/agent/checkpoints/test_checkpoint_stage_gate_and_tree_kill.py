"""Real-Git contracts for the pre-stage byte gate and the timeout process-tree kill.

Covers the two 2026-10-11 failure modes that the file-count guard cannot see:
a workdir of *few huge* files (one unignored 1.2 TB VHDX) and an orphaned
``git add`` that outlives its timeout and keeps writing the shared store.
"""
import os
import subprocess

from agent.checkpoints import manager as cm

_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _repo(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    for args in (["init", "-q"], ["config", "user.name", "Test"],
                 ["config", "user.email", "test@example.invalid"]):
        subprocess.run(["git", *args], cwd=work, env=_ENV, check=True, capture_output=True)
    return work


def test_stage_bytes_gate_rejects_unignored_oversize_workdir(tmp_path, monkeypatch):
    work = _repo(tmp_path)
    (work / "big.bin").write_bytes(b"\x00" * (3 * 1024**2))
    monkeypatch.setattr(cm, "_MAX_STAGE_BYTES", 1024**2)
    mgr = cm.CheckpointManager(enabled=True)
    assert mgr.ensure_checkpoint(str(work), "oversize gate") is False
    # The gate fires before staging: no checkpoint commit may exist.
    assert mgr.list_checkpoints(str(work)) == []


def test_stage_bytes_gate_ignores_gitignored_data(tmp_path, monkeypatch):
    work = _repo(tmp_path)
    (work / "big.bin").write_bytes(b"\x00" * (3 * 1024**2))
    (work / ".gitignore").write_text("big.bin\n", encoding="utf-8")
    (work / "keep.txt").write_text("project state\n", encoding="utf-8")
    monkeypatch.setattr(cm, "_MAX_STAGE_BYTES", 1024**2)
    mgr = cm.CheckpointManager(enabled=True)
    # The scan must mirror what `git add -A` would stage: gitignored data does
    # not count, so a legitimate workdir still checkpoints.
    assert mgr.ensure_checkpoint(str(work), "ignored data allowed") is True


def test_run_git_timeout_kills_process_tree(tmp_path, monkeypatch):
    work = _repo(tmp_path)
    (work / "warmup.txt").write_text("seed\n", encoding="utf-8")
    store = cm._store_path()
    cm._init_store(store, str(work))
    # Incompressible data: zlib cannot beat the 1 s timeout on any machine.
    (work / "slow.bin").write_bytes(os.urandom(256 * 1024**2))
    index = store / "indexes" / "timeout-test.bin"
    index.parent.mkdir(parents=True, exist_ok=True)
    called = []
    real_kill = cm.kill_process_tree

    def spy(proc):
        called.append(proc)
        real_kill(proc)

    monkeypatch.setattr(cm, "kill_process_tree", spy)
    ok, _, err = cm._run_git(["add", "-A"], store, str(work),
                             timeout=1, index_file=index)
    assert ok is False
    assert "timed out" in err
    # The timed-out git must be reaped with its tree, not left orphaned.
    assert len(called) == 1
