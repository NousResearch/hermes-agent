"""Self-test for the live-system guard fixture in tests/conftest.py.

This file is the canary. If anyone removes a guard or weakens it, these
tests fail. If anyone adds a NEW kill primitive to the codebase without
adding it to the guard, the corresponding test added here will fail too.

The guard exists to protect the developer's live ``hermes-gateway`` process
from being SIGTERMed by tests. See PR #23397 for the original incident
(5+ live gateway kills in 3 days). Per Teknium 2026-05-10:

  > "You better do such a deep scan and scrub of the tests that this
  >  never is possible ever again for all eternity."

Every primitive that can deliver a signal to a foreign process or mutate
the live systemd unit MUST be exercised below. Adding a new primitive to
the guard? Add a test here too.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import types

import pytest

import tests.conftest as _conftest
from tests.conftest import PROJECT_ROOT

# A guaranteed-foreign PID: PID 1 (init).  Owned by root, not us, and
# always exists. A sane guard refuses to signal it.
FOREIGN_PID = 1


# ──────────────────── fail-closed self-protection ──────────────
#
# This file executes REAL kill primitives — os.kill(-1, SIGTERM), os.killpg,
# pkill -f python — and depends entirely on the autouse ``_live_system_guard``
# fixture in tests/conftest.py to intercept them. That makes the canary
# fail-OPEN: in any collection context where this file is present but its home
# conftest is not, the primitives fire for real and ``os.kill(-1, SIGTERM)``
# SIGTERMs every process the invoking user owns (a full desktop-session kill was
# reported in the field — see issue #68311). Such contexts are not exotic:
# published sdists that ship ``tests/`` but not ``tests/conftest.py``, trees
# assembled by copying ``test*.py`` files (that glob does NOT match
# ``conftest.py``), ``pytest --noconftest``, or running from a foreign rootdir.
#
# The fixture below makes the canary fail-CLOSED instead: it refuses to run any
# test in this file unless the guard is provably active, so no collection
# context can ever detonate the primitives. The one thing the canary can detect
# about its own safety is that the guard monkeypatches ``os.kill`` with a plain
# Python function, whereas the unguarded primitive is a C builtin.


def _live_system_guard_is_active() -> bool:
    """True iff tests/conftest.py's ``_live_system_guard`` has patched os.kill.

    The guard replaces ``os.kill`` with a plain Python function; the raw,
    unguarded primitive is a C builtin (``types.BuiltinFunctionType``). If
    ``os.kill`` is still the builtin, the guard never loaded and every kill
    primitive in this file would fire for real.
    """
    return not isinstance(os.kill, types.BuiltinFunctionType)


@pytest.fixture(autouse=True)
def _refuse_to_fire_live_weapons(request):
    """Fail closed: refuse to run a canary test unless the guard is active.

    Tests genuinely marked ``@pytest.mark.live_system_guard_bypass`` opt out
    (they run the raw primitive deliberately and harmlessly, e.g. a signal-0
    liveness probe of our own PID), matching the guard's own bypass contract.
    """
    if request.node.get_closest_marker("live_system_guard_bypass"):
        yield
        return
    if not _live_system_guard_is_active():
        pytest.fail(
            "REFUSING TO RUN: the live-system guard from tests/conftest.py is "
            "not active in this interpreter (os.kill is still the raw C "
            "builtin). This canary file executes real kill primitives — "
            "os.kill(-1, SIGTERM), os.killpg, pkill -f python — and relies on "
            "the guard to intercept them; unguarded, they SIGTERM every process "
            "the current user owns. This usually means the file was collected "
            "without its home tests/conftest.py (note: a test*.py copy glob "
            "does NOT match conftest.py). See issue #68311.",
            pytrace=False,
        )
    yield


def test_fail_closed_probe_reports_guard_active():
    """In the real suite the guard is loaded, so the probe reports active and
    ``_refuse_to_fire_live_weapons`` stays out of the way (no false positives
    that would wedge CI)."""
    assert _live_system_guard_is_active() is True


def test_fail_closed_probe_classifies_raw_builtin_as_unguarded():
    """The probe's discriminator, exercised against real objects: a raw C
    builtin the guard never touches (``os.getpid``) is exactly what an
    unguarded ``os.kill`` looks like and must read as 'guard not active', while
    the loaded guard's ``os.kill`` is a plain Python function."""
    assert isinstance(os.getpid, types.BuiltinFunctionType)
    assert not isinstance(os.kill, types.BuiltinFunctionType)


# ──────────────────── kill primitives ─────────────────────────


def test_os_kill_blocks_foreign_pid():
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.kill(FOREIGN_PID, signal.SIGTERM)


def test_os_kill_blocks_negative_one():
    """``os.kill(-1, sig)`` signals every process we can reach. Must be blocked."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.kill(-1, signal.SIGTERM)


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="killpg POSIX-only")
def test_os_killpg_blocks_foreign_pgid():
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.killpg(FOREIGN_PID, signal.SIGTERM)


# ──────────────────── subprocess regex bypasses ────────────────


def test_subprocess_run_systemctl_restart_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["systemctl", "--user", "restart", "hermes-gateway"])


def test_subprocess_run_full_path_systemctl_blocked():
    """``/usr/bin/systemctl`` (full path) must be blocked too."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["/usr/bin/systemctl", "--user", "stop", "hermes-gateway"])


def test_subprocess_run_sudo_systemctl_blocked():
    """``sudo systemctl ...`` defeated the old head==systemctl check."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["sudo", "systemctl", "restart", "hermes-gateway"])


def test_subprocess_run_env_systemctl_blocked():
    """``env systemctl ...`` similarly defeated the old head check."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["env", "systemctl", "--user", "restart", "hermes-gateway"])


def test_subprocess_run_bash_c_systemctl_blocked():
    """``bash -c "systemctl ..."`` must also be caught."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["bash", "-c", "systemctl --user restart hermes-gateway"])


def test_subprocess_run_sh_c_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["sh", "-c", "systemctl --user stop hermes-gateway"])


def test_subprocess_run_setsid_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["setsid", "systemctl", "kill", "hermes-gateway"])


def test_subprocess_run_string_shell_true_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(
            "systemctl --user restart hermes-gateway",
            shell=True,
        )


def test_subprocess_popen_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.Popen(["systemctl", "--user", "stop", "hermes-gateway"])


def test_subprocess_call_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.call(["systemctl", "--user", "restart", "hermes-gateway"])


def test_subprocess_check_call_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.check_call(["systemctl", "--user", "restart", "hermes-gateway"])


def test_subprocess_check_output_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.check_output(["systemctl", "--user", "restart", "hermes-gateway"])


def test_subprocess_getoutput_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.getoutput("systemctl --user restart hermes-gateway")


def test_subprocess_getstatusoutput_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.getstatusoutput("systemctl --user restart hermes-gateway")


# ──────────────────── os.system / os.popen ────────────────────


def test_os_system_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.system("systemctl --user restart hermes-gateway")


def test_os_popen_systemctl_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.popen("systemctl --user restart hermes-gateway")


# ──────────────────── pty.spawn ────────────────────────────────


def test_pty_spawn_systemctl_blocked():
    import pty
    with pytest.raises(RuntimeError, match="live-system guard"):
        pty.spawn(["systemctl", "--user", "restart", "hermes-gateway"])


# ──────────────────── asyncio.create_subprocess_* ──────────────


def test_asyncio_create_subprocess_exec_systemctl_blocked():
    import asyncio

    async def _attempt():
        await asyncio.create_subprocess_exec(
            "systemctl", "--user", "restart", "hermes-gateway"
        )

    with pytest.raises(RuntimeError, match="live-system guard"):
        asyncio.run(_attempt())


def test_asyncio_create_subprocess_shell_systemctl_blocked():
    import asyncio

    async def _attempt():
        await asyncio.create_subprocess_shell(
            "systemctl --user restart hermes-gateway"
        )

    with pytest.raises(RuntimeError, match="live-system guard"):
        asyncio.run(_attempt())


# ──────────────────── pkill / killall / taskkill ───────────────


def test_subprocess_pkill_hermes_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["pkill", "-f", "hermes"])


def test_subprocess_pkill_hermes_gateway_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["pkill", "-f", "hermes-gateway"])


def test_subprocess_pkill_python_dash_f_blocked():
    """``pkill -f python`` matches the gateway's "python -m hermes_cli.main"."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["pkill", "-f", "python"])


def test_subprocess_killall_hermes_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["killall", "hermes"])


# ──────────────── destructive git vs. PROJECT_ROOT ─────────────
#
# A test that forgets to mock subprocess for a raw ``git checkout`` /
# ``reset`` / ``clean`` / ``switch`` call must never let it run for real
# against PROJECT_ROOT — this repo's own live checkout. That flips the
# developer's actual branch or discards real uncommitted work; it happened
# for real on 2026-08-11 (see Operations.md's isolated-clone-first rule).
# The ``hermes update`` check above only catches commands that go through
# ``hermes update`` itself — this catches the raw git command directly.


def test_subprocess_run_git_checkout_project_root_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["git", "checkout", "main"], cwd=PROJECT_ROOT)


def test_subprocess_run_git_reset_hard_project_root_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["git", "reset", "--hard"], cwd=PROJECT_ROOT)


def test_subprocess_run_git_clean_project_root_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["git", "clean", "-fd"], cwd=PROJECT_ROOT)


def test_subprocess_run_git_switch_project_root_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["git", "switch", "main"], cwd=PROJECT_ROOT)


def test_subprocess_run_bash_c_git_reset_project_root_blocked():
    """``bash -c "git reset --hard"`` must also be caught."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(
            ["bash", "-c", "git reset --hard"], cwd=PROJECT_ROOT
        )


def test_subprocess_run_git_reset_no_cwd_kwarg_uses_process_cwd(monkeypatch):
    """No explicit ``cwd=`` means git runs against the process's actual
    cwd — must still be caught when that happens to be PROJECT_ROOT."""
    monkeypatch.chdir(PROJECT_ROOT)
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["git", "reset", "--hard"])


def test_subprocess_popen_git_checkout_project_root_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.Popen(["git", "checkout", "main"], cwd=PROJECT_ROOT)


# ``git stash push`` is the verb that actually caused BOTH real incidents
# (2026-08-11 and 2026-08-12), via `hermes update`'s autostash reaching
# PROJECT_ROOT from inside a test. It leaves `git status` clean and writes
# `reset: moving to HEAD` to the reflog, so it reads exactly like a hard
# reset — the work is recoverable from `git stash list`, but only if you
# know to look. It was missing from the original verb list.


def test_subprocess_run_git_stash_push_project_root_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(
            ["git", "stash", "push", "--include-untracked", "-m", "x"],
            cwd=PROJECT_ROOT,
        )


def test_subprocess_run_bare_git_stash_project_root_blocked():
    """Bare ``git stash`` is ``git stash push`` — must be caught."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["git", "stash"], cwd=PROJECT_ROOT)


def test_subprocess_run_git_stash_pop_project_root_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["git", "stash", "pop"], cwd=PROJECT_ROOT)


def test_subprocess_run_git_restore_project_root_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["git", "restore", "."], cwd=PROJECT_ROOT)


# `git -C <path> ...` retargets the command regardless of cwd. A cwd-only
# check misses it entirely — and hermes_cli/mcp_catalog.py genuinely builds
# `git -C <dest> checkout <ref>`, so this shape exists in production code.


def test_git_dash_c_checkout_targeting_project_root_blocked(tmp_path):
    """cwd is an innocent tmp dir; `-C` aims it at PROJECT_ROOT."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "checkout", "main"], cwd=tmp_path
        )


def test_git_work_tree_flag_targeting_project_root_blocked(tmp_path):
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(
            ["git", f"--work-tree={PROJECT_ROOT}", "reset", "--hard"], cwd=tmp_path
        )


def test_git_dash_c_stash_targeting_project_root_blocked(tmp_path):
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "stash", "push"], cwd=tmp_path
        )


# ─────────────── import-time layer (no fixture required) ───────────────
#
# _live_system_guard is function-scoped, so it is inactive during collection,
# during session/module-scoped fixture setup+teardown, and after the last
# test. The import-time wrappers installed by conftest cover those windows.


def test_import_time_git_guard_is_installed_without_any_fixture():
    """Importing conftest alone must install the wrappers.

    Checked in a FRESH interpreter on purpose: inside a running test the
    autouse fixture has layered its own wrappers on top, so inspecting
    subprocess.run here would only prove the fixture ran. A clean process
    that merely imports conftest is exactly the collection-time situation
    this layer exists to cover.
    """
    probe = (
        "import sys; sys.path.insert(0, %r);"
        "import tests.conftest, subprocess, os;"
        "print(getattr(subprocess.run, '_git_guarded', False),"
        "      getattr(subprocess.Popen, '_git_guarded', False),"
        "      getattr(os.system, '_git_guarded', False))"
    ) % str(PROJECT_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "True", "True"], result.stdout


def test_import_time_predicate_flags_destructive_git():
    """The shared predicate — the one consulted in the fixture-less
    collection window — classifies correctly."""
    flag = _conftest._destructive_git_against_project_root
    assert flag(["git", "stash", "push"], {"cwd": PROJECT_ROOT})
    assert flag(["git", "reset", "--hard"], {"cwd": PROJECT_ROOT})
    assert flag(["bash", "-c", "git reset --hard"], {"cwd": PROJECT_ROOT})
    assert flag(["git", "-C", str(PROJECT_ROOT), "checkout", "main"], {})


def test_import_time_predicate_allows_safe_calls(tmp_path):
    flag = _conftest._destructive_git_against_project_root
    assert not flag(["git", "stash", "list"], {"cwd": PROJECT_ROOT})
    assert not flag(["git", "status"], {"cwd": PROJECT_ROOT})
    assert not flag(["git", "reset", "--hard"], {"cwd": tmp_path})
    assert not flag(["echo", "git reset --hard"], {"cwd": PROJECT_ROOT})


def test_os_system_destructive_git_blocked(monkeypatch):
    # Either layer may catch it first (the fixture wraps on top of the
    # import-time wrapper); both refuse, which is all that matters here.
    monkeypatch.chdir(PROJECT_ROOT)
    with pytest.raises(RuntimeError, match="guard"):
        os.system("git reset --hard")


# ──────────────────── pass-through cases (must NOT raise) ──────


def test_git_dash_c_checkout_isolated_dir_allowed(tmp_path):
    """`-C` aimed somewhere harmless must still pass through."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    result = subprocess.run(
        ["git", "-C", str(tmp_path), "checkout", "-b", "throwaway"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_git_stash_list_project_root_allowed():
    """``git stash list``/``show`` only READ — blocking them would break
    legitimate tests and diagnostics for no safety gain."""
    result = subprocess.run(
        ["git", "stash", "list"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_git_stash_push_allowed_when_cwd_is_isolated(tmp_path):
    # A real repo WITH an initial commit: `git stash` refuses outright on a
    # repo that has none ("You do not have the initial commit yet"), which
    # would pass the guard for the wrong reason.
    def git(*args, **kwargs):
        return subprocess.run(["git", *args], cwd=tmp_path, **kwargs)

    git("init", "-q", check=True)
    git("config", "user.email", "test@example.com", check=True)
    git("config", "user.name", "Test", check=True)
    (tmp_path / "f.txt").write_text("x")
    git("add", "f.txt", check=True)
    git("commit", "-qm", "initial", check=True)

    (tmp_path / "f.txt").write_text("modified")
    result = git("stash", "push", "-m", "throwaway", capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_git_checkout_allowed_when_cwd_is_isolated(tmp_path):
    # Real git repo at tmp_path so the underlying command runs cleanly
    # instead of just failing with "not a git repository" — that proves
    # the GUARD let it through, not that git itself refused for some
    # unrelated reason.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    result = subprocess.run(
        ["git", "checkout", "-b", "throwaway"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_git_rev_parse_project_root_allowed():
    # Read-only git commands (status, rev-parse, log, ...) are not the
    # risk this guard exists for — only checkout/reset/clean/switch are.
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr














# ──────────────────── real gateway runtime spawn ─────────────────


def test_subprocess_popen_real_gateway_restart_blocked():
    """``python -m hermes_cli.main gateway restart`` is a detached child that
    inherits the pytest-tmp HERMES_HOME, resolves the developer's real
    ``hermes-gateway`` unit, and outlives the test (39 six-day orphans squatted
    the webhook port, 2026-09-03). Blocked at the spawn primitive."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.Popen(
            [sys.executable, "-m", "hermes_cli.main", "gateway", "restart"],
            start_new_session=True,
        )


def test_subprocess_run_gateway_status_passes_through():
    """Only lifecycle verbs are blocked: ``gateway status`` (and every other
    read-only subcommand) must still spawn — via the canonical matcher, not an
    argv substring."""
    result = subprocess.run(
        [sys.executable, "-c", "import sys; print(sys.argv[1:])", "-m", "hermes_cli.main", "gateway", "status"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


# ──────────────────── bypass marker ─────────────────────────────


@pytest.mark.live_system_guard_bypass
def test_bypass_marker_disables_guard():
    """The bypass marker exists for tests that genuinely need real signal delivery
    (e.g. PTY tests SIGINTing their own child). Verify it works.

    We use it harmlessly here by signaling our own PID 0 (own group) so we
    don't actually kill anything — but the call goes through real os.kill.
    """
    # With bypass, the guard yields without installing the monkeypatch,
    # so we get the real os.kill. Calling os.kill(os.getpid(), 0) just
    # checks that the PID exists — harmless.
    os.kill(os.getpid(), 0)  # No exception — guard is OFF.
