"""The autouse live-system guard: no real kills, systemctl writes, gateway spawns or checkout writes.

Imported into ``tests/conftest.py`` so pytest registers the fixture there;
``pytest_plugins`` is not an option because ``tests/conftest.py`` is not the
rootdir conftest (the rootdir is the repo root, where ``pyproject.toml`` lives).
"""
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ── Live-system guard ──────────────────────────────────────────────────────
#
# Several test files exercise the gateway-restart / kill code paths
# (``cmd_update``, ``kill_gateway_processes``, ``stop_profile_gateway``).
# When a single test forgets to mock either ``os.kill`` or the global
# ``find_gateway_pids`` helper, the real call leaks out of the hermetic
# environment and finds the developer's live ``hermes-gateway`` process
# via ``psutil`` — sending it SIGTERM mid-test. The shutdown forensics in
# PR #23285 caught this happening 5+ times in 3 days, every time
# correlated with a ``tests/hermes_cli/`` pytest run starting up.
#
# This fixture makes the leak impossible by intercepting the two
# primitives that actually do damage:
#
#  • ``os.kill`` rejects any PID outside the test process subtree with
#    a hard ``RuntimeError`` so the offending test gets a stack trace
#    instead of silently murdering the real gateway.
#  • ``subprocess.run`` / ``subprocess.Popen`` / ``call`` / ``check_call`` /
#    ``check_output`` reject any ``systemctl ... <verb> hermes-gateway``
#    invocation that would mutate the live unit. Read-only systemctl
#    calls (``status``, ``show``, ``list-units``) still pass through.
#
# We intentionally do NOT stub ``find_gateway_pids`` / ``_scan_gateway_pids``
# here — tests of those functions themselves need the real implementation.
# Even if a test gets the live gateway PID back from a real scan, the
# ``os.kill`` guard above catches the actual signal call, and the
# ``systemctl`` guard catches the systemd path. Discovery without
# delivery is harmless.

_LIVE_SYSTEM_GUARD_BYPASS_MARK = "live_system_guard_bypass"
_GATEWAY_LOOKALIKE_MARK = "spawns_gateway_lookalike"

# Tests may designate a temporary repo to exercise the real guard safely.
_LIVE_GUARD_PROTECTED_GIT_ROOTS = (PROJECT_ROOT,)


@pytest.fixture(autouse=True)
def _live_system_guard(request, monkeypatch):
    """Block real os.kill / systemctl / gateway-pid scans during tests.

    See block comment above for the why. Tests that genuinely need
    real signal delivery (e.g. PTY tests that SIGINT their own child)
    can opt out with ``@pytest.mark.live_system_guard_bypass``.

    Coverage (every primitive that can deliver a signal to or otherwise
    terminate a foreign process):
      • os.kill, os.killpg (POSIX)
      • subprocess.run / Popen / call / check_call / check_output
      • subprocess.getoutput / getstatusoutput
      • os.system / os.popen
      • pty.spawn
      • asyncio.create_subprocess_exec / create_subprocess_shell
    Subprocess inspection looks at the WHOLE command string (not just
    tokens[0]), so ``bash -c "systemctl restart hermes-gateway"``,
    ``sudo systemctl ...``, ``env systemctl ...``, ``setsid systemctl ...``
    are all caught. ``pkill``/``killall``/``taskkill`` invocations
    targeting hermes/python patterns are also blocked. Bare ``git``
    commands may not mutate a protected checkout. Git writes against
    temporary repositories and read-only Git commands remain allowed.
    """
    if request.node.get_closest_marker(_LIVE_SYSTEM_GUARD_BYPASS_MARK):
        yield
        return

    import os as _os
    import shlex as _shlex
    import subprocess as _subprocess

    test_pid = _os.getpid()
    lookalike_ok = request.node.get_closest_marker(_GATEWAY_LOOKALIKE_MARK) is not None
    # Capture the test process's existing children at fixture start —
    # any *new* children spawned by the test are also allowlisted via
    # the live psutil walk below. Static set keeps the fast path cheap.
    try:
        import psutil as _psutil
        _initial_children = {
            c.pid for c in _psutil.Process(test_pid).children(recursive=True)
        }
    except Exception:
        _psutil = None
        _initial_children = set()

    def _is_own_subtree(pid: int) -> bool:
        # PID 0 means "our own process group"; -1 means "every process we
        # can signal". Both are dangerous when paired with SIGTERM/SIGKILL,
        # but pid 0 is technically scoped to our group so allow it; pid -1
        # is treated as foreign (refuse).
        if pid == 0:
            return True
        if pid < 0:
            return False
        if pid == test_pid or pid in _initial_children:
            return True
        if _psutil is None:
            return False
        try:
            walker = _psutil.Process(pid)
        except Exception:
            # Stale PID — kill would be a no-op anyway, allow it.
            return True
        try:
            for parent in walker.parents():
                if parent.pid == test_pid:
                    return True
        except Exception:
            return False
        return False

    real_kill = _os.kill

    def _guarded_kill(pid, sig, *args, **kwargs):
        # Signal 0 is a pure liveness probe — it cannot terminate anything.
        # psutil.pid_exists() uses os.kill(pid, 0) on POSIX, and probing a
        # just-killed grandchild that was reparented to init (zombie with a
        # foreign parent chain) must not trip the guard. Flaked in CI on
        # test_entire_tree_is_sigkilled_not_just_parent.
        if int(sig) == 0:
            return real_kill(pid, sig, *args, **kwargs)
        if _is_own_subtree(int(pid)):
            return real_kill(pid, sig, *args, **kwargs)
        raise RuntimeError(
            f"tests/conftest.py live-system guard: blocked os.kill("
            f"{pid}, {sig}) — PID is outside the test process subtree. "
            "If this fired in CI it means the test reached a real "
            "kill_gateway_processes / stop_profile_gateway / cmd_update "
            "code path without mocking find_gateway_pids and os.kill. "
            "Mock both, or mark the test with "
            "@pytest.mark.live_system_guard_bypass if real signal "
            "delivery is genuinely required."
        )

    monkeypatch.setattr(_os, "kill", _guarded_kill)

    # ``os.killpg`` is the same risk class — sends a signal to every
    # process in a group. The gateway is a session leader (its own
    # PGID == its PID), so killpg(gateway_pid, SIGTERM) is a one-shot
    # kill of the live process. Allow it only when the target PGID is
    # the test process's own group.
    if hasattr(_os, "killpg"):
        real_killpg = _os.killpg
        own_pgid = _os.getpgrp()

        def _guarded_killpg(pgid, sig, *args, **kwargs):
            # Signal 0 is a pure liveness probe — never destructive.
            if int(sig) == 0:
                return real_killpg(pgid, sig, *args, **kwargs)
            if int(pgid) == own_pgid or _is_own_subtree(int(pgid)):
                return real_killpg(pgid, sig, *args, **kwargs)
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"os.killpg({pgid}, {sig}) — PGID is outside the test "
                "process group. See _live_system_guard for the why."
            )

        monkeypatch.setattr(_os, "killpg", _guarded_killpg)

    # ── Subprocess command-string inspection (whole-line) ──────────
    _HERMES_TOKENS = (
        "hermes-gateway",
        "hermes.service",
        "hermes_cli.main gateway",
        "hermes_cli/main.py gateway",
        "gateway/run.py",
        "hermes gateway",
    )
    _MUTATING_VERBS = (
        "restart", "start", "stop", "kill", "reload",
        "reset-failed", "enable", "disable", "mask", "unmask",
        "daemon-reload", "try-restart", "reload-or-restart",
    )
    _PROCESS_KILLERS = ("pkill", "killall", "taskkill", "skill", "fuser")
    _CONTAINER_RUNTIMES = ("docker", "podman", "nerdctl")

    def _first_token_basename(cmd_str: str) -> str:
        try:
            tokens = _shlex.split(cmd_str)
        except ValueError:
            tokens = cmd_str.split()
        return tokens[0].rsplit("/", 1)[-1].lower() if tokens else ""
    # Shell/launcher executables whose arguments are themselves commands —
    # argv[0]-only scanning must not exempt what they wrap.
    _WRAPPER_COMMANDS = (
        "sh", "bash", "zsh", "dash", "env", "nohup", "setsid",
        "timeout", "sudo", "xargs", "nice", "ionice", "stdbuf", "flock",
    )

    def _cmd_to_string(cmd) -> str:
        if cmd is None:
            return ""
        if isinstance(cmd, (bytes, bytearray)):
            try:
                return bytes(cmd).decode(errors="replace")
            except Exception:
                return ""
        if isinstance(cmd, str):
            return cmd
        if isinstance(cmd, (list, tuple)):
            try:
                return " ".join(str(t) for t in cmd)
            except Exception:
                return ""
        return str(cmd)

    def _matches_hermes_gateway(cmd_str: str) -> bool:
        low = cmd_str.lower()
        return any(tok in low for tok in _HERMES_TOKENS)

    def _is_blocked_systemctl(cmd) -> bool:
        cmd_str = _cmd_to_string(cmd)
        if "systemctl" not in cmd_str:
            return False
        if not _matches_hermes_gateway(cmd_str):
            return False
        try:
            tokens = _shlex.split(cmd_str)
        except ValueError:
            tokens = cmd_str.split()
        return any(verb in tokens for verb in _MUTATING_VERBS)

    def _is_process_killer(cmd) -> bool:
        cmd_str = _cmd_to_string(cmd)
        try:
            tokens = _shlex.split(cmd_str)
        except ValueError:
            tokens = cmd_str.split()
        if not tokens:
            return False

        # For argv-style calls only argv[0] is the executable; scanning every
        # argument blocked innocent commands like ``cat /tmp/.../skill``
        # ("skill" is in _PROCESS_KILLERS).  Wrapper executables still get
        # full-token scanning so ``["bash", "-c", "pkill ..."]`` stays caught.
        if isinstance(cmd, (list, tuple)):
            head0 = tokens[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            killer_tokens = tokens if head0 in _WRAPPER_COMMANDS else tokens[:1]
        else:
            killer_tokens = tokens
        for tok in killer_tokens:
            head = tok.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if head in _PROCESS_KILLERS:
                low = cmd_str.lower()
                # pkill -f pattern: catch hermes-themed patterns + a
                # plain "python" -f which would catch the live gateway
                # whose cmdline contains "python -m hermes_cli.main".
                if (
                    "hermes" in low
                    or "gateway" in low
                    or ("python" in low and "-f" in tokens)
                ):
                    return True
        return False

    from tests.git_safety import blocked_git_mutation

    def _check_subprocess_cmd(name, cmd, kwargs=None):
        git_verb = blocked_git_mutation(cmd, kwargs, _LIVE_GUARD_PROTECTED_GIT_ROOTS)
        if git_verb is not None:
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"subprocess.{name}({cmd!r}) — `git {git_verb}` would mutate "
                "the protected checkout. Use a temporary repository or mock "
                "the update boundary; live_system_guard_bypass is for deliberate live tests."
            )
        if _is_blocked_systemctl(cmd):
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"subprocess.{name}({cmd!r}) — would mutate the "
                "live hermes-gateway systemd unit. Mock "
                "subprocess.run / _run_systemctl in the test, or "
                "mark with @pytest.mark.live_system_guard_bypass."
            )
        if _is_process_killer(cmd):
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"subprocess.{name}({cmd!r}) — process-killer command "
                "targeting hermes/python could hit the live gateway. "
                "Mark with @pytest.mark.live_system_guard_bypass if "
                "intentional."
            )
        # Block any subprocess that would run `hermes update` (or the
        # equivalent `python -m hermes_cli.main update`).  These commands
        # run `git fetch origin + git pull` against the REAL checkout,
        # overwriting files like pyproject.toml mid-test-run and corrupting
        # every subsequent subprocess that reads them.  The corruption is
        # especially insidious because the spawned process uses setsid/
        # start_new_session=True, making it invisible to pytest's process
        # tree (PPid=1) and nearly impossible to trace without explicit
        # inotify/SHA watchdogs.  Any test that legitimately needs to exercise
        # the update-spawn path must mock subprocess.Popen explicitly.
        cmd_str = _cmd_to_string(cmd)
        low = cmd_str.lower()
        if "update" in low and (
            # hermes update / hermes update --gateway / setsid bash -c ... hermes update
            ("hermes" in low and "update" in low.split())
            or
            # python -m hermes_cli.main update --gateway
            ("hermes_cli" in low and "update" in low.split())
            or
            # venv/bin/hermes update  (absolute path variant used in tests)
            (".venv/bin/hermes" in low and "update" in low)
        ):
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"subprocess.{name}({cmd!r}) — this command would run "
                "`hermes update` against the real checkout, fetching "
                "from origin and overwriting repo files (e.g. "
                "pyproject.toml) mid-test-run. This corrupts every "
                "subsequent subprocess in the same runner. "
                "Mock subprocess.Popen (and subprocess.run if used) "
                "in the test instead, or mark with "
                "@pytest.mark.live_system_guard_bypass if genuinely "
                "needed (e.g. an integration test testing the update "
                "flow against a dedicated throwaway repo)."
            )
        # Block spawning a REAL gateway runtime (``python -m hermes_cli.main
        # gateway run|start|restart``). ``_spawn_hermes_action`` launches it
        # with start_new_session=True, so it outlives the pytest worker; the
        # child inherits the pytest-tmp HERMES_HOME, resolves the DEVELOPER's
        # ``hermes-gateway`` systemd unit (a tmp home hashes to no profile
        # suffix), restarts the live gateway, and the survivors squat the
        # webhook port. 2026-09-03: 39 such orphans lived 6 days after a
        # sibling refactor moved the spawn seam and left tests patching the
        # facade. The canonical matcher, never an argv substring.
        from gateway.status import gateway_spawn_intent_subcommand
        # A gateway launched INSIDE a container (`docker exec … hermes gateway start`) cannot
        # reach the host's systemd unit or webhook port; tests/docker/ exists to exercise it.
        in_container = _first_token_basename(cmd_str) in _CONTAINER_RUNTIMES
        if (
            not lookalike_ok
            and not in_container
            and gateway_spawn_intent_subcommand(cmd_str) in ("run", "start", "restart")
        ):
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"subprocess.{name}({cmd!r}) — this would spawn a REAL "
                "hermes gateway runtime that outlives the test (it is "
                "detached), restarts the developer's live gateway, and "
                "holds the webhook port. Patch the spawn seam where "
                "production reads it (hermes_cli.web_server_gateway."
                "_spawn_hermes_action), or mark with "
                "@pytest.mark.spawns_gateway_lookalike a test that spawns "
                "and reaps its own stub child."
            )

    def _wrap_subprocess(name, real):
        def _guarded(cmd, *args, **kwargs):
            _check_subprocess_cmd(name, cmd, kwargs)
            return real(cmd, *args, **kwargs)
        _guarded.__name__ = f"_guarded_{name}"
        # Make the wrapper subscriptable like the wrapped callable when
        # the wrapped object is. ``subprocess.Popen[bytes]`` is used as
        # a type annotation in third-party packages (mcp, etc.); replacing
        # ``Popen`` with a plain function breaks ``Popen[bytes]`` at
        # import time. Defer ``__class_getitem__`` to the original.
        if hasattr(real, "__class_getitem__"):
            _guarded.__class_getitem__ = real.__class_getitem__
        return _guarded

    def _wrap_popen():
        """Subclass Popen so isinstance checks AND Popen[bytes] still work."""
        real = _subprocess.Popen

        class _GuardedPopen(real):  # type: ignore[misc, valid-type]
            def __init__(self, cmd, *args, **kwargs):
                _check_subprocess_cmd("Popen", cmd, kwargs)
                super().__init__(cmd, *args, **kwargs)

        _GuardedPopen.__name__ = "Popen"
        _GuardedPopen.__qualname__ = "Popen"
        return _GuardedPopen

    real_run = _subprocess.run
    real_popen = _subprocess.Popen
    real_call = _subprocess.call
    real_check_call = _subprocess.check_call
    real_check_output = _subprocess.check_output
    real_getoutput = _subprocess.getoutput
    real_getstatusoutput = _subprocess.getstatusoutput

    monkeypatch.setattr(_subprocess, "run", _wrap_subprocess("run", real_run))
    monkeypatch.setattr(_subprocess, "Popen", _wrap_popen())
    monkeypatch.setattr(_subprocess, "call", _wrap_subprocess("call", real_call))
    monkeypatch.setattr(
        _subprocess, "check_call", _wrap_subprocess("check_call", real_check_call)
    )
    monkeypatch.setattr(
        _subprocess,
        "check_output",
        _wrap_subprocess("check_output", real_check_output),
    )
    monkeypatch.setattr(
        _subprocess, "getoutput", _wrap_subprocess("getoutput", real_getoutput)
    )
    monkeypatch.setattr(
        _subprocess,
        "getstatusoutput",
        _wrap_subprocess("getstatusoutput", real_getstatusoutput),
    )

    # os.system / os.popen — same risk class, completely unwrapped before.
    real_os_system = _os.system
    real_os_popen = _os.popen

    def _guarded_os_system(command):
        _check_subprocess_cmd("os.system", command)
        return real_os_system(command)

    def _guarded_os_popen(cmd, *args, **kwargs):
        _check_subprocess_cmd("os.popen", cmd, kwargs)
        return real_os_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(_os, "system", _guarded_os_system)
    monkeypatch.setattr(_os, "popen", _guarded_os_popen)

    # pty.spawn — POSIX-only.
    try:
        import pty as _pty
        if hasattr(_pty, "spawn"):
            real_pty_spawn = _pty.spawn

            def _guarded_pty_spawn(argv, *args, **kwargs):
                _check_subprocess_cmd("pty.spawn", argv, kwargs)
                return real_pty_spawn(argv, *args, **kwargs)

            monkeypatch.setattr(_pty, "spawn", _guarded_pty_spawn)
    except Exception:
        pass

    # asyncio.create_subprocess_* — bypasses subprocess module entirely.
    try:
        import asyncio as _asyncio
        real_async_exec = _asyncio.create_subprocess_exec
        real_async_shell = _asyncio.create_subprocess_shell

        async def _guarded_async_exec(program, *args, **kwargs):
            _check_subprocess_cmd(
                "asyncio.create_subprocess_exec", [program, *args], kwargs
            )
            return await real_async_exec(program, *args, **kwargs)

        async def _guarded_async_shell(cmd, *args, **kwargs):
            _check_subprocess_cmd("asyncio.create_subprocess_shell", cmd, kwargs)
            return await real_async_shell(cmd, *args, **kwargs)

        monkeypatch.setattr(_asyncio, "create_subprocess_exec", _guarded_async_exec)
        monkeypatch.setattr(
            _asyncio, "create_subprocess_shell", _guarded_async_shell
        )
    except Exception:
        pass

    yield
