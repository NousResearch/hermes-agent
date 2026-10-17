"""``process_state`` must recognise a Desktop-over-SSH ``serve --isolated`` backend whose argv
carries interpreter flags before ``-c``.

The source launcher's ``--print-runtime-command`` reports ``python.exe -I -c <bootstrap>``, so the
spawned backend's argv is ``[python, "-I", "-c", ..., "serve", "--isolated", ...]``. The ownership
check used to require ``argv[1] == "-c"``, judged that process ``owned: false`` (``reason: argv``)
and Desktop reported "Remote Windows backend exited before announcing its port" for a live backend.
Identity stays anchored by creation time + owner nonce; the flag skip only widens the shape match.
"""

from __future__ import annotations

import os
import types

import pytest

from hermes_cli import windows_ssh_runtime as w

NONCE = "b3f728cb78ba540f"
CREATED_NS = 1_791_713_679_000_000_000
# Host-native paths: process_state() uses the host os.path, and this mocked test runs on every
# lane (a Windows-style literal would not split on POSIX and every positive row would fail).
HERMES = os.path.abspath(os.path.join("hermes", "ssh-runtime", "hermes.exe"))
PY = os.path.abspath(os.path.join("hermes", "tools", "python", "python.exe"))
NOT_PYTHON = os.path.abspath(os.path.join("evil", "tool.exe"))
BOOT = "import hermes_bootstrap; runpy.run_module('hermes_cli.main', run_name='__main__')"


def _fake_psutil(monkeypatch, argv):
    proc = types.SimpleNamespace(
        create_time=lambda: CREATED_NS / 1_000_000_000,
        cmdline=lambda: argv,
        is_running=lambda: True,
    )
    fake = types.SimpleNamespace(
        Process=lambda pid: proc,
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        AccessDenied=type("AccessDenied", (Exception,), {}),
    )
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake)


def _serve(nonce=NONCE):
    return ["serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", nonce]


@pytest.mark.parametrize(
    "prefix",
    [
        [PY, "-I", "-c", BOOT],          # source launcher runtime command (the regression)
        [PY, "-I", "-E", "-c", BOOT],    # several isolation flags
        [PY, "-I", "-X", "utf8", "-c", BOOT],   # value-taking option, separate token
        [PY, "-Xutf8", "-W", "ignore", "-c", BOOT],  # attached and separate value forms
        [PY, "-c", BOOT],                # previous shape keeps working
        [PY, "-I", "-m", "hermes_cli.main"],
        [PY, "-m", "hermes_cli.main"],
    ],
)
def test_python_bootstrap_with_interpreter_flags_is_owned(monkeypatch, prefix):
    _fake_psutil(monkeypatch, prefix + _serve())
    state = w.process_state(4242, CREATED_NS, HERMES, NONCE)
    assert state["owned"] is True
    assert state["reason"] == "owned"


def test_wrong_nonce_is_not_owned(monkeypatch):
    _fake_psutil(monkeypatch, [PY, "-I", "-c", BOOT] + _serve("0" * 16))
    assert w.process_state(4242, CREATED_NS, HERMES, NONCE)["owned"] is False


def test_python_running_something_else_is_not_owned(monkeypatch):
    _fake_psutil(monkeypatch, [PY, "-I", "-c", "import time; time.sleep(9)"] + _serve())
    assert w.process_state(4242, CREATED_NS, HERMES, NONCE)["owned"] is False


def test_flag_skip_does_not_accept_a_non_python_arg0(monkeypatch):
    _fake_psutil(monkeypatch, [NOT_PYTHON, "-I", "-c", BOOT] + _serve())
    assert w.process_state(4242, CREATED_NS, HERMES, NONCE)["owned"] is False
