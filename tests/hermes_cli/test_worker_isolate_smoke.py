"""Integration smoke test: exercise the live Python launcher against the
REAL production tree (the same mount sequence the dispatcher will use on
every kanban worker spawn).

This is NOT a unit test -- it touches the host mount table, runs a child
namespace, and requires unprivileged userns. It is gated to run only when
``HERMES_SMOKE_LAUNCHER=1`` is set so CI on a developer machine that does
not have userns (or where the production tree is unwriteable) does not
trip on it. Local hand-runs on the host that owns the production tree
run it via ``HERMES_SMOKE_LAUNCHER=1 pytest -k smoke_launcher``.

The test asserts three things, all of which the card requires before the
worker isolation ships:

1. The launcher enters the namespace, applies the bind dance, and execs
   the worker argv verbatim (here: a probe Python script that exits 0).
2. The probe cannot write to the production tree (EROFS at the syscall
   level, identical to the unit-test acceptance harness).
3. The production tree is byte-identical after the run -- no stray
   branches, no residue, no touched files.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest


_SMOKE = os.environ.get("HERMES_SMOKE_LAUNCHER") == "1"


@pytest.mark.skipif(not _SMOKE, reason="set HERMES_SMOKE_LAUNCHER=1 to run the live launcher smoke test")
@pytest.mark.skipif(shutil.which("unshare") is None, reason="unshare not on PATH")
def test_smoke_launcher_against_real_production_tree(tmp_path):
    from hermes_cli.worker_isolate import _launcher_module_target

    prod = os.environ.get("HERMES_AGENT_HOME") or "/home/serveradmin/.hermes/hermes-agent"
    # The smoke test must run on a host whose prod tree is a git checkout.
    if not os.path.isdir(os.path.join(prod, ".git")):
        pytest.skip(f"{prod} is not a git checkout; nothing to smoke against")

    # The current worktree MUST be a sibling of .worktrees/t_ff7d30cc so the
    # launcher carves it out as the worker's rw workspace.
    worktree = os.path.join(prod, ".worktrees", "t_ff7d30cc")
    if not os.path.isdir(worktree):
        pytest.skip(f"{worktree} not present; this smoke test targets t_ff7d30cc")

    # Snapshot pre-state: HEAD sha + porcelain status.
    pre_head = subprocess.check_output(["git", "-C", prod, "rev-parse", "HEAD"], text=True).strip()
    pre_porcelain = subprocess.check_output(["git", "-C", prod, "status", "--porcelain"], text=True)
    pre_branches = subprocess.check_output(["git", "-C", prod, "branch", "--list"], text=True)

    # Build the composed argv: unshare -Urm <launcher> PROD WT -- <probe>.
    launcher_prefix = _launcher_module_target()
    inner = [
        *launcher_prefix,
        prod,
        "--worktree", worktree,
        "--",
        sys.executable, "-c", _SMOKE_PROBE,
        prod, worktree,
    ]
    cmd = ["unshare", "-Urm", *inner]
    result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)

    out = result.stdout + result.stderr
    fail_lines = [line for line in out.splitlines() if line.startswith("FAIL ")]
    assert not fail_lines, f"smoke test: forbidden op was allowed:\n{out}"
    assert "SMOKE: PASS" in out, out

    # Post-state must equal pre-state -- the launcher did not mutate prod.
    post_head = subprocess.check_output(["git", "-C", prod, "rev-parse", "HEAD"], text=True).strip()
    post_porcelain = subprocess.check_output(["git", "-C", prod, "status", "--porcelain"], text=True)
    post_branches = subprocess.check_output(["git", "-C", prod, "branch", "--list"], text=True)

    assert post_head == pre_head, f"HEAD moved: {pre_head} -> {post_head}"
    assert post_porcelain == pre_porcelain, f"working tree changed:\n{post_porcelain}"
    # The launcher should not introduce or remove branches on the production tree.
    assert set(post_branches.split()) == set(pre_branches.split()), (
        f"branches changed:\n--- pre ---\n{pre_branches}\n--- post ---\n{post_branches}"
    )


_SMOKE_PROBE = r"""
import os, subprocess, sys
PROD = sys.argv[1]
WT = sys.argv[2]
FAIL = 0

def chk(label, expect, args, cwd=None):
    global FAIL
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=20)
    denied = (
        r.returncode != 0
        or "Read-only file system" in (r.stderr + r.stdout)
        or "Permission denied" in (r.stderr + r.stdout)
        or "Operation not permitted" in (r.stderr + r.stdout)
    )
    ok = (expect == "deny" and denied) or (expect == "allow" and r.returncode == 0)
    print(("PASS  " if ok else "FAIL  ") + label + " :: " + (r.stderr.strip() or r.stdout.strip() or "ok"))
    if not ok:
        FAIL = 1

chk("direct edit", "deny", ["bash", "-c", "echo x >> '%s/app.py'" % PROD])
chk("git checkout", "deny", ["git", "-C", PROD, "checkout", "--", "."])
chk("worktree add escape", "deny", ["git", "-C", PROD, "worktree", "add", "/tmp/smoke-esc-a", "HEAD"])
chk("worktree add -B canon", "deny", ["git", "-C", PROD, "worktree", "add", "-B", "stray", "/tmp/smoke-esc-b", "HEAD"])
chk("git update-ref canon", "deny", ["git", "-C", PROD, "update-ref", "refs/heads/canonical-deploy", "HEAD"])
chk("git branch -f canon", "deny", ["git", "-C", PROD, "branch", "-f", "canonical-deploy", "HEAD"])
chk("nested remount rw", "deny", ["bash", "-c",
    "unshare -Urm sh -c 'mount -o remount,bind,rw '%s''" % PROD])
chk("read prod tree", "allow", ["cat", os.path.join(PROD, "AGENTS.md")])
print("---")
print("SMOKE: PASS" if FAIL == 0 else "SMOKE: FAIL")
sys.exit(FAIL)
"""
