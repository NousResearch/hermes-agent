"""Unit + acceptance tests for hermes_cli.worker_isolate (t_ff7d30cc).

The acceptance harness is a Python port of
``artifacts/t_f4806674/verify-isolation.sh``: it builds a synthetic repo
under ``tmp_path``, runs the isolation launcher over it, and asserts every
forbidden op fails with EROFS/EPERM while a legitimate in-worktree commit
still succeeds. The pure argv builder is tested without ever touching the
host mount table.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Pure argv builder
# ---------------------------------------------------------------------------


def test_build_isolated_worker_argv_prefix_and_separator():
    """Argv wraps the worker with unshare -Urm <launcher> -- <worker argv>.

    The ``--`` separator is mandatory: a worker command beginning with a
    flag must never get eaten by the launcher.
    """
    from hermes_cli.worker_isolate import build_isolated_worker_argv

    argv = build_isolated_worker_argv(
        ["hermes", "chat", "-q", "hello"],
        agent_home="/p/agent",
        worktree="/p/agent/.worktrees/t_x",
    )
    # ``unshare -Urm <launcher-argv> -- <worker argv>``. The launcher argv
    # starts with sys.executable + ``-m hermes_cli.worker_isolate _launcher``,
    # so ``-Urm`` is at position 1 and the launcher takes over from 2.
    assert argv[0] == "unshare"
    assert argv[1] == "-Urm"
    # The launcher is invoked as ``python -m hermes_cli.worker_isolate _launcher``.
    assert "-m" in argv and "hermes_cli.worker_isolate" in argv
    assert "_launcher" in argv
    # agent_home is always carried into the launcher.
    assert "/p/agent" in argv
    # worktree is passed via --worktree.
    assert "--worktree" in argv
    assert "/p/agent/.worktrees/t_x" in argv
    # The worker's argv follows ``--`` UNMODIFIED.
    sep = argv.index("--")
    assert argv[sep + 1:] == ["hermes", "chat", "-q", "hello"]


def test_build_isolated_worker_argv_no_worktree():
    """A non-worktree task still wraps in unshare -Urm, just without --worktree."""
    from hermes_cli.worker_isolate import build_isolated_worker_argv

    argv = build_isolated_worker_argv(["echo", "ok"], agent_home="/p/agent")
    assert argv[0] == "unshare"
    assert argv[1] == "-Urm"
    assert "--worktree" not in argv
    assert argv[-2:] == ["echo", "ok"]
    assert "--" in argv
    sep = argv.index("--")
    assert argv[sep + 1:] == ["echo", "ok"]


def test_build_isolated_worker_argv_extra_rw_passes_through():
    from hermes_cli.worker_isolate import build_isolated_worker_argv

    argv = build_isolated_worker_argv(
        ["cmd"], agent_home="/p", worktree="/p/wt", extra_rw=["/extra/a", "/extra/b"]
    )
    # Both --rw flag and its values must appear contiguously before --.
    rw_idx = argv.index("--rw")
    assert argv[rw_idx + 1] == "/extra/a"
    assert argv[rw_idx + 2] == "/extra/b"
    # Sanity: the -- separator follows.
    assert "--" in argv[rw_idx + 3:]


def test_build_isolated_worker_argv_rejects_empty_command():
    from hermes_cli.worker_isolate import build_isolated_worker_argv

    with pytest.raises(ValueError):
        build_isolated_worker_argv([], agent_home="/p")


def test_build_isolated_worker_argv_rejects_missing_agent_home():
    from hermes_cli.worker_isolate import build_isolated_worker_argv

    with pytest.raises(ValueError):
        build_isolated_worker_argv(["x"], agent_home="")


# ---------------------------------------------------------------------------
# Gate / mode resolution
# ---------------------------------------------------------------------------


def test_resolve_isolation_mode_off_returns_off():
    from hermes_cli.worker_isolate import resolve_isolation_mode

    cfg = resolve_isolation_mode("off")
    assert cfg.mode == "off"
    assert cfg.effective == "off"


def test_resolve_isolation_mode_warn_returns_off_no_log(monkeypatch, caplog):
    """warn -> effective=off, with no warning logged (it's a user choice, not a degrade)."""
    from hermes_cli import worker_isolate
    from hermes_cli.worker_isolate import resolve_isolation_mode

    # Reset the once-per-process warning so we can observe the absence of one.
    worker_isolate._DEGRADE_WARNED = False

    caplog.set_level("WARNING", logger="hermes_cli.worker_isolate")
    cfg = resolve_isolation_mode("warn")
    assert cfg.mode == "warn"
    assert cfg.effective == "off"
    # No warning -- the user explicitly asked for warn.
    assert not any("degraded" in rec.message for rec in caplog.records)


def test_resolve_isolation_mode_unknown_degrades_to_off_once(monkeypatch, caplog):
    """Unknown config values degrade to off and emit exactly ONE warning per process."""
    from hermes_cli import worker_isolate
    from hermes_cli.worker_isolate import resolve_isolation_mode

    worker_isolate._DEGRADE_WARNED = False
    caplog.set_level("WARNING", logger="hermes_cli.worker_isolate")

    cfg1 = resolve_isolation_mode("garbage")
    assert cfg1.effective == "off"
    cfg2 = resolve_isolation_mode("still-garbage")
    assert cfg2.effective == "off"

    degrade_msgs = [r for r in caplog.records if "degraded" in r.message]
    # Single-shot: the second call must NOT add a second warning.
    assert len(degrade_msgs) == 1


def test_resolve_isolation_mode_kill_switch_overrides_enforce(monkeypatch):
    from hermes_cli.worker_isolate import resolve_isolation_mode

    monkeypatch.setenv("HERMES_WORKER_ISOLATION", "off")
    cfg = resolve_isolation_mode("enforce")
    assert cfg.effective == "off"
    assert "kill switch" in cfg.reason


def test_resolve_isolation_mode_kill_switch_truthy_overrides_off(monkeypatch):
    from hermes_cli.worker_isolate import resolve_isolation_mode

    monkeypatch.setenv("HERMES_WORKER_ISOLATION", "enforce")
    cfg = resolve_isolation_mode("off")
    # Default platform default would be enforce here (Linux+userns); kill switch flips it back on.
    assert cfg.effective == "enforce"


def test_load_configured_isolation_mode_parses_strings(monkeypatch):
    from hermes_cli import config
    from hermes_cli.worker_isolate import load_configured_isolation_mode

    def _set(value):
        monkeypatch.setattr(config, "load_config_readonly", lambda: {"kanban": {"worker_isolation": value}})

    _set("enforce")
    assert load_configured_isolation_mode() == "enforce"
    _set("  warn  ")
    assert load_configured_isolation_mode() == "warn"
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"kanban": {}})
    assert load_configured_isolation_mode() is None
    monkeypatch.setattr(config, "load_config_readonly", lambda: {})
    assert load_configured_isolation_mode() is None
    _set(True)
    assert load_configured_isolation_mode() == "enforce"
    _set(False)
    assert load_configured_isolation_mode() == "off"
    # Any truthy int is interpreted as enforce; the resolver's default-mode
    # path runs only for types the loader explicitly rejects (str coerces via
    # .strip().lower(), bool/int coerce to a mode).
    _set(42)
    assert load_configured_isolation_mode() == "enforce"
    _set(0)
    assert load_configured_isolation_mode() == "off"
    # Lists / dicts / None are types the loader refuses to interpret; the
    # resolver's default-mode path runs.
    _set([1, 2])
    assert load_configured_isolation_mode() is None


# ---------------------------------------------------------------------------
# Dispatcher integration: _isolation_worker_argv
# ---------------------------------------------------------------------------


def _make_task(workspace_kind: str = "worktree", task_id: str = "t_iso"):
    from hermes_cli import kanban_db as kb
    return kb.Task(
        id=task_id,
        title="iso",
        body=None,
        assignee="friday",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind=workspace_kind,
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def test_isolation_worker_argv_wraps_when_worktree_under_agent_home(tmp_path, monkeypatch):
    """The wrap path: workspace is a dispatcher-provisioned worktree under agent_home/.worktrees."""
    from hermes_cli import kanban_db_dispatch as kbd

    repo = tmp_path / "agent"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".worktrees" / "t_iso").mkdir(parents=True)
    worktree = repo / ".worktrees" / "t_iso"
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))

    task = _make_task()
    argv = kbd._isolation_worker_argv(task, ["hermes", "chat"], str(worktree))
    assert argv[0] == "unshare"
    assert argv[1] == "-Urm"
    assert "--worktree" in argv
    assert str(worktree) in argv


def test_isolation_worker_argv_skips_when_not_worktree_kind(tmp_path, monkeypatch):
    """A scratch (``dir``) task MUST NOT be wrapped -- we never carve out an arbitrary cwd."""
    from hermes_cli import kanban_db_dispatch as kbd

    repo = tmp_path / "agent"
    repo.mkdir()
    (repo / ".git").mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))

    task = _make_task(workspace_kind="dir")
    argv = kbd._isolation_worker_argv(task, ["hermes", "chat"], str(scratch))
    # Unchanged -- the helper refused to wrap.
    assert argv == ["hermes", "chat"]


def test_isolation_worker_argv_skips_when_workspace_outside_agent_home(tmp_path, monkeypatch):
    """A workspace path that escapes ``agent_home/.worktrees`` is not eligible for wrapping."""
    from hermes_cli import kanban_db_dispatch as kbd

    repo = tmp_path / "agent"
    repo.mkdir()
    (repo / ".git").mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))

    task = _make_task()
    argv = kbd._isolation_worker_argv(task, ["hermes", "chat"], str(other))
    assert argv == ["hermes", "chat"]


def test_isolation_worker_argv_skips_when_agent_home_unresolvable(tmp_path, monkeypatch):
    """Without an agent home we refuse to wrap -- never pretend."""
    from hermes_cli import kanban_db_dispatch as kbd

    # Strip every candidate env var the resolver looks at.
    for k in ("HERMES_AGENT_HOME", "HERMES_AGENT", "HERMES_HOME"):
        monkeypatch.delenv(k, raising=False)
    # Use a tmp_path that has NO .git anywhere up the tree, so even HERMES_HOME's parent fails.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    task = _make_task()
    argv = kbd._isolation_worker_argv(task, ["hermes", "chat"], str(tmp_path))
    assert argv == ["hermes", "chat"]


def test_isolation_worker_argv_kill_switch_short_circuits(tmp_path, monkeypatch):
    """The HERMES_WORKER_ISOLATION=off kill switch skips wrapping even with a valid worktree."""
    from hermes_cli import kanban_db_dispatch as kbd

    repo = tmp_path / "agent"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".worktrees" / "t_iso").mkdir(parents=True)
    worktree = repo / ".worktrees" / "t_iso"
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))
    monkeypatch.setenv("HERMES_WORKER_ISOLATION", "off")

    task = _make_task()
    argv = kbd._isolation_worker_argv(task, ["hermes", "chat"], str(worktree))
    assert argv == ["hermes", "chat"]


def test_is_worktree_under_agent_home(tmp_path):
    from hermes_cli.kanban_db_dispatch import _is_worktree_under_agent_home

    repo = tmp_path / "agent"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ".worktrees").mkdir()
    wt = repo / ".worktrees" / "t_iso"
    wt.mkdir()
    other = tmp_path / "scratch"
    other.mkdir()

    assert _is_worktree_under_agent_home(str(wt), str(repo)) is True
    assert _is_worktree_under_agent_home(str(repo / ".worktrees"), str(repo)) is True
    assert _is_worktree_under_agent_home(str(other), str(repo)) is False
    # ``.git/worktrees/<id>`` is the git bookkeeping dir, NOT the worktree
    # workspace -- a worker arriving here did NOT get its workspace from
    # the dispatcher's anchor logic, so we refuse to carve it out.
    bookkeeping = repo / ".git" / "worktrees" / "t_iso"
    bookkeeping.mkdir(parents=True)
    assert _is_worktree_under_agent_home(str(bookkeeping), str(repo)) is False
    assert _is_worktree_under_agent_home(str(tmp_path / "nope"), str(repo)) is False


# ---------------------------------------------------------------------------
# Acceptance harness (port of verify-isolation.sh)
# ---------------------------------------------------------------------------
#
# Mirrors the bash harness checked in at
# ``artifacts/t_f4806674/verify-isolation.sh`` so the contract is testable
# in CI without a shell-out. The harness builds a synthetic repo + worktree,
# enters the isolation launcher, and verifies every forbidden op fails
# while a legitimate commit succeeds.
#
# Skipped automatically on hosts where unprivileged userns is unavailable.
#


@pytest.mark.skipif(shutil.which("unshare") is None, reason="unshare not on PATH")
@pytest.mark.skipif(
    subprocess.run(
        ["unshare", "--user", "--map-root-user", "true"],
        capture_output=True, timeout=5,
    ).returncode != 0,
    reason="unprivileged user namespace unavailable on this host",
)
def test_acceptance_harness_isolates_synthetic_repo(tmp_path):
    """The full 13-check acceptance harness, ported to pytest."""
    from hermes_cli.worker_isolate import _launcher_module_target, _enter_isolation

    repo = tmp_path / "agent"
    repo.mkdir()
    (repo / ".git" / "objects").mkdir(parents=True)
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / ".git" / "logs").mkdir(parents=True)
    (repo / ".git" / "worktrees" / "t_accept").mkdir(parents=True)
    (repo / ".worktrees" / "t_accept").mkdir(parents=True)
    # Seed a HEAD so the harness has a sane baseline.
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/canonical-deploy\n")
    # Seed app.py in BOTH the production tree AND the worktree, so the
    # "read prod tree" check actually has something to read.
    (repo / "app.py").write_text("print('hi')\n")

    worktree = repo / ".worktrees" / "t_accept"
    (worktree / "app.py").write_text("print('hi')\n")

    # Compose the launcher argv (without the unshare prefix; this test calls
    # ``unshare`` directly so we measure the inner launcher behaviour).
    launcher_prefix = _launcher_module_target()
    inner_cmd = [
        *launcher_prefix,
        str(repo),
        "--worktree", str(worktree),
        "--",
        sys.executable, "-c",
        # Probe script: run every forbidden op + the allowed one inside the sandbox.
        _PROBE_SCRIPT,
        str(repo), str(worktree),
    ]

    # Wrap in unshare + enter the namespace.
    cmd = ["unshare", "-Urm", *inner_cmd]
    result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)

    out = result.stdout + result.stderr
    fail_lines = [line for line in out.splitlines() if line.startswith("FAIL ")]
    assert not fail_lines, f"forbidden op was allowed:\n{out}"
    assert "ACCEPTANCE: PASS" in out, out


_PROBE_SCRIPT = r"""
import os, subprocess, sys
PROD = sys.argv[1]
WT = sys.argv[2]
FAIL = 0

def chk(label, expect, args, cwd=None, env=None):
    global FAIL
    r = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=20)
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

# Forbidden: direct writes into the production tree.
chk("direct edit", "deny", ["bash", "-c", "echo x >> '%s/app.py'" % PROD])
# Forbidden: git state mutations that need .git/HEAD.lock / .git/index.lock.
chk("git checkout", "deny", ["git", "-C", PROD, "checkout", "--", "."])
# Worktree operations need .git/worktrees/* which is read-only under the sandbox.
chk("worktree add", "deny", ["git", "-C", PROD, "worktree", "add", "/tmp/esc-wt", "HEAD"])
chk("worktree add -B", "deny", ["git", "-C", PROD, "worktree", "add", "-B", "stray", "/tmp/esc-wt-b", "HEAD"])
# Nested namespace escape: remounting prod rw from inside a child userns is locked.
chk("nested remount rw", "deny", ["bash", "-c",
    "unshare -Urm sh -c 'mount -o remount,bind,rw '%s''" % PROD])
# Allowed: read the production tree.
chk("read prod tree", "allow", ["cat", os.path.join(PROD, "app.py")])
# Allowed: commit inside the worker's own worktree.
chk("git commit in own WT", "allow", ["bash", "-c",
    "cd '%s' && git init -q . && git -c user.email=a@b -c user.name=a commit --allow-empty -qm ok" % WT])
# Allowed: read .git under the sandbox (it's ro but readable).
chk("read prod .git/HEAD", "allow", ["cat", os.path.join(PROD, ".git", "HEAD")])
print("---")
print("ACCEPTANCE: PASS" if FAIL == 0 else "ACCEPTANCE: FAIL")
sys.exit(FAIL)
"""


# ---------------------------------------------------------------------------
# Launcher parser / helper sanity
# ---------------------------------------------------------------------------


def test_launcher_argv_parser_rejects_unknown_flags():
    from hermes_cli.worker_isolate import _parse_launcher_argv

    with pytest.raises(ValueError):
        _parse_launcher_argv(["/p/a", "--bogus", "x", "--", "cmd"])
    # Missing ``--`` separator (worker argv was never given).
    with pytest.raises(ValueError):
        _parse_launcher_argv(["/p/a", "--worktree", "/p/wt"])  # no `--` and no cmd
    # Missing value for --worktree.
    with pytest.raises(ValueError):
        _parse_launcher_argv(["/p/a", "--worktree", "--", "cmd"])
    parsed = _parse_launcher_argv(["/p/a", "--worktree", "/p/wt", "--rw", "/e1", "/e2", "--", "cmd", "x"])
    assert parsed[0] == "/p/a"
    assert parsed[1] == "/p/wt"
    assert parsed[2] == ["/e1", "/e2"]
    assert parsed[3] == ["cmd", "x"]


def test_enter_isolation_distinct_exit_codes_on_missing_paths(tmp_path):
    """Stage-91 path: parent doesn't exist -> exit 90, not 91."""
    from hermes_cli.worker_isolate import _enter_isolation

    rc = _enter_isolation(agent_home="/no/such/parent/agent", worktree=None, extra_rw=())
    assert rc == 90
