"""Profile delete/rename stop the backends bound to the profile (``hermes_cli.profiles_process_stop``).

The scan binds by ``--profile`` selector or ``HERMES_HOME`` and acts only on processes the spawn
ledger positively identifies (purpose + ``(pid, create_time)``), so an interactive ``chat`` whose
argv mentions backend words, another profile's backend, or a recycled PID is never signalled.
"""

import os
import sys
import types
from pathlib import Path

import pytest

from hermes_cli import profiles_process_stop as process_stop
from hermes_cli.profiles import create_profile, get_profile_dir


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Path.home() -> tmp_path and HERMES_HOME -> tmp_path/.hermes, so profiles land in the temp dir."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


class TestProfileBackendStop:
    def test_backend_scan_only_matches_this_profile(self, profile_env, monkeypatch):
        """The backend PID scan binds by --profile selector and skips self."""
        create_profile("coder", no_alias=True)
        profile_dir = get_profile_dir("coder")

        class FakeProc:
            def __init__(self, pid, cmdline, username="me"):
                self.pid = pid
                self.info = {
                    "pid": pid,
                    "name": "python",
                    "username": username,
                    "cmdline": cmdline,
                    "create_time": float(pid),
                }

            def parent(self):
                return None

            def username(self):
                return "me"

            def environ(self):
                return {}

        self_pid = os.getpid()
        procs = [
            # Backend bound to coder → matched.
            FakeProc(101, ["python", "-m", "hermes_cli.main", "--profile", "coder", "serve"]),
            # Browser-hosted Desktop is the same profile-bound backend class.
            FakeProc(104, ["python", "-m", "hermes_cli.main", "--profile", "coder", "webapp"]),
            # Interactive chat for coder → NOT a backend subcommand, skipped.
            FakeProc(102, ["python", "-m", "hermes_cli.main", "--profile", "coder", "chat"]),
            # Prompt text containing backend words is still interactive and
            # has no positive ledger identity — never kill it.
            FakeProc(105, ["python", "-m", "hermes_cli.main", "--profile", "coder", "chat",
                           "-q", "please debug the webapp serve command"]),
            # Backend for a different profile → skipped.
            FakeProc(103, ["python", "-m", "hermes_cli.main", "--profile", "other", "serve"]),
            # PID was reused after the ledger entry — skipped despite backend argv.
            FakeProc(106, ["python", "-m", "hermes_cli.main", "--profile", "coder", "serve"]),
            # Ledger identity without create_time is insufficient — skipped.
            FakeProc(107, ["python", "-m", "hermes_cli.main", "--profile", "coder", "serve"]),
            # This very process → skipped even if it matched.
            FakeProc(self_pid, ["python", "-m", "hermes_cli.main", "--profile", "coder", "serve"]),
        ]
        for proc in procs:
            proc.info["create_time"] = float(proc.pid)

        fake_psutil = types.SimpleNamespace(
            process_iter=lambda attrs=None: iter(procs),
            Process=lambda pid=None: FakeProc(self_pid, []),
            NoSuchProcess=Exception,
            AccessDenied=Exception,
            ZombieProcess=Exception,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        monkeypatch.setattr(
            "hermes_cli.process_identity.ledger_entries",
            lambda: [
                {"pid": 101, "purpose": "serve", "create_time": 101.0},
                {"pid": 102, "purpose": "chat", "create_time": 102.0},
                {"pid": 103, "purpose": "serve", "create_time": 103.0},
                {"pid": 104, "purpose": "serve", "create_time": 104.0},
                {"pid": 105, "purpose": "chat", "create_time": 105.0},
                {"pid": 106, "purpose": "serve", "create_time": 1.0},
                {"pid": 107, "purpose": "serve", "create_time": None},
            ],
        )
        pids = process_stop._profile_bound_backend_pids("coder", profile_dir)
        assert pids == [(101, 101.0), (104, 104.0)]

    def test_backend_scan_uses_ledger_for_shebang_exec_shape(self, profile_env, monkeypatch):
        """A ledger-identified backend remains matchable across shim argv shapes."""
        create_profile("coder", no_alias=True)
        profile_dir = get_profile_dir("coder")

        class FakeProc:
            def __init__(self, pid, cmdline, username="me"):
                self.pid = pid
                self.info = {"pid": pid, "name": "python3", "username": username, "cmdline": cmdline}

            def parent(self):
                return None

            def username(self):
                return "me"

            def environ(self):
                return {}

        self_pid = os.getpid()
        procs = [
            # Shebang-exec'd shim bound to coder → matched despite argv[0]
            # being the python interpreter, not "hermes".
            FakeProc(201, ["/usr/bin/python3", "/Users/x/.local/bin/hermes", "--profile", "coder", "serve",
                            "--host", "127.0.0.1", "--port", "0"]),
            # Same shape but a different profile → skipped.
            FakeProc(202, ["/usr/bin/python3", "/Users/x/.local/bin/hermes", "--profile", "other", "serve"]),
            # Non-hermes script run by python3 → skipped.
            FakeProc(203, ["/usr/bin/python3", "/Users/x/some_script.py", "--profile", "coder", "serve"]),
        ]
        for proc in procs:
            proc.info["create_time"] = float(proc.pid)

        fake_psutil = types.SimpleNamespace(
            process_iter=lambda attrs=None: iter(procs),
            Process=lambda pid=None: FakeProc(self_pid, []),
            NoSuchProcess=Exception,
            AccessDenied=Exception,
            ZombieProcess=Exception,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        monkeypatch.setattr(
            "hermes_cli.process_identity.ledger_entries",
            lambda: [
                {"pid": 201, "purpose": "serve", "create_time": 201.0},
                {"pid": 202, "purpose": "serve", "create_time": 202.0},
            ],
        )
        pids = process_stop._profile_bound_backend_pids("coder", profile_dir)
        assert pids == [(201, 201.0)]

    def test_backend_scan_rejects_unrelated_hermes_prefixed_script(self, profile_env, monkeypatch):
        """A user's own script that happens to start with "hermes" (e.g.
        hermes-notes.py, hermes-unrelated-tool) must NOT be misidentified as
        the console-script shim just because argv[0] is a python interpreter
        and argv[1]'s basename starts with "hermes" -- only the actual known
        console-script entry points (hermes, hermes-agent, hermes-acp) count.
        """
        create_profile("coder", no_alias=True)
        profile_dir = get_profile_dir("coder")

        class FakeProc:
            def __init__(self, pid, cmdline, username="me"):
                self.pid = pid
                self.info = {"pid": pid, "name": "python3", "username": username, "cmdline": cmdline}

            def parent(self):
                return None

            def username(self):
                return "me"

            def environ(self):
                return {}

        self_pid = os.getpid()
        procs = [
            # Looks like the shim by prefix alone, but is the user's own
            # unrelated tool -- must be rejected, not killed by profile delete.
            FakeProc(301, ["/usr/bin/python3", "/Users/x/scripts/hermes-notes.py",
                            "--profile", "coder", "serve"]),
            FakeProc(302, ["/usr/bin/python3", "/Users/x/scripts/hermes-unrelated-tool",
                            "--profile", "coder", "serve"]),
        ]
        for proc in procs:
            proc.info["create_time"] = float(proc.pid)

        fake_psutil = types.SimpleNamespace(
            process_iter=lambda attrs=None: iter(procs),
            Process=lambda pid=None: FakeProc(self_pid, []),
            NoSuchProcess=Exception,
            AccessDenied=Exception,
            ZombieProcess=Exception,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        monkeypatch.setattr("hermes_cli.process_identity.ledger_entries", lambda: [])
        pids = process_stop._profile_bound_backend_pids("coder", profile_dir)
        assert pids == []

    def test_backend_scan_matches_ledger_identified_backend_shapes(self, profile_env, monkeypatch):
        """Once the ledger proves backend identity, argv shim shape is irrelevant."""
        create_profile("coder", no_alias=True)
        profile_dir = get_profile_dir("coder")

        class FakeProc:
            def __init__(self, pid, cmdline, username="me"):
                self.pid = pid
                self.info = {"pid": pid, "name": "python3", "username": username, "cmdline": cmdline}

            def parent(self):
                return None

            def username(self):
                return "me"

            def environ(self):
                return {}

        self_pid = os.getpid()
        procs = [
            FakeProc(401, ["/usr/bin/python3", "/Users/x/.local/bin/hermes-agent",
                            "--profile", "coder", "serve"]),
            FakeProc(402, ["/usr/bin/python3", "/Users/x/.local/bin/hermes-acp",
                            "--profile", "coder", "serve"]),
        ]
        for proc in procs:
            proc.info["create_time"] = float(proc.pid)

        fake_psutil = types.SimpleNamespace(
            process_iter=lambda attrs=None: iter(procs),
            Process=lambda pid=None: FakeProc(self_pid, []),
            NoSuchProcess=Exception,
            AccessDenied=Exception,
            ZombieProcess=Exception,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        monkeypatch.setattr(
            "hermes_cli.process_identity.ledger_entries",
            lambda: [
                {"pid": 401, "purpose": "serve", "create_time": 401.0},
                {"pid": 402, "purpose": "dashboard", "create_time": 402.0},
            ],
        )
        pids = process_stop._profile_bound_backend_pids("coder", profile_dir)
        assert set(pids) == {(401, 401.0), (402, 402.0)}

    def test_backend_stop_never_force_kills_reused_pid(self, profile_env, monkeypatch):
        profile_dir = create_profile("coder")
        create_time_calls = 0

        class Process:
            def __init__(self, _pid):
                pass

            def create_time(self):
                nonlocal create_time_calls
                create_time_calls += 1
                return 10.0 if create_time_calls == 1 else 11.0

        class NoSuchProcess(Exception):
            pass

        fake_psutil = types.SimpleNamespace(
            Process=Process,
            NoSuchProcess=NoSuchProcess,
            AccessDenied=PermissionError,
            ZombieProcess=ProcessLookupError,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        monkeypatch.setattr(
            process_stop,
            "_profile_bound_backend_pids",
            lambda _canon, _home: [(4242, 10.0)],
        )
        from gateway import status

        signals: list[tuple[int, bool]] = []
        monkeypatch.setattr(
            status,
            "terminate_pid",
            lambda pid, *, force=False: signals.append((pid, force)),
        )

        process_stop._stop_profile_backends("coder", profile_dir)

        assert signals == [(4242, False)]
