"""Desktop SSH ownership check must accept interpreter flags before -c/-m.

The published Windows launcher's --print-runtime-command yields ``python -I -c "...hermes_cli.main..."``;
process_state() only accepted ``argv[1] == "-c"``, so the Desktop app rejected its own spawned backend
("Remote Windows backend exited before announcing its port", reason "argv")."""
import psutil
import pytest

from hermes_cli import windows_ssh_runtime as wsr

NONCE = "2d9973c57339e6b6"


def _fake(monkeypatch, argv, created=1.5):
    class P:
        def __init__(self, pid):
            pass

        def create_time(self):
            return created

        def cmdline(self):
            return argv

        def is_running(self):
            return True

    monkeypatch.setattr(psutil, "Process", P)
    return int(created * 1_000_000_000)


SERVE = ["serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", NONCE]


@pytest.mark.parametrize("prefix", [
    ["-I", "-c", "import runpy; runpy.run_module('hermes_cli.main')"],
    ["-I", "-B", "-c", "import runpy; runpy.run_module('hermes_cli.main')"],
    ["-X", "utf8", "-c", "import runpy; runpy.run_module('hermes_cli.main')"],
    ["-c", "import runpy; runpy.run_module('hermes_cli.main')"],
    ["-m", "hermes_cli.main"],
])
def test_interpreter_flags_are_accepted(monkeypatch, prefix):
    created = _fake(monkeypatch, [r"C:\tools\python-3.14\python.exe", *prefix, *SERVE])
    state = wsr.process_state(1234, created, r"C:\hermes\venv\Scripts\hermes-launcher.exe", NONCE)
    assert state["owned"] is True


def test_foreign_module_is_still_rejected(monkeypatch):
    created = _fake(monkeypatch, [r"C:\tools\python.exe", "-I", "-c", "import other", *SERVE])
    assert wsr.process_state(1234, created, r"C:\hermes\hermes.exe", NONCE)["owned"] is False


def test_wrong_nonce_is_still_rejected(monkeypatch):
    argv = [r"C:\tools\python.exe", "-I", "-c", "runpy hermes_cli.main", *SERVE[:-1], "ffffffffffffffff"]
    created = _fake(monkeypatch, argv)
    assert wsr.process_state(1234, created, r"C:\hermes\hermes.exe", NONCE)["owned"] is False
