"""Exact-int CLI status propagation (#62810).

``bool`` subclasses ``int``, so ``isinstance(rc, int) and rc != 0`` maps a
handler returning ``True`` to exit 1. Plugin CLI handlers are unconstrained
callables; some return booleans for success.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.main import (
    _handler_exit_code,
    _oneshot_exit_code,
    cmd_proxy,
    main as hermes_main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Status(IntEnum):
    FAIL = 2


class _IntSubclass(int):
    pass


@pytest.mark.parametrize(
    ("rc", "expected"),
    [
        (None, None),
        (False, None),
        (True, None),
        (0, None),
        (1, 1),
        (2, 2),
        (-1, -1),
        ("2", None),
        (2.0, None),
        (_Status.FAIL, None),
        (_IntSubclass(2), None),
    ],
)
def test_handler_exit_code_accepts_only_exact_nonzero_int(rc, expected):
    assert _handler_exit_code(rc) == expected


@pytest.mark.parametrize(
    ("rc", "expected"),
    [
        (None, 0),
        (False, 0),
        (True, 0),
        (0, 0),
        (1, 1),
        (2, 2),
        (-1, -1),
        ("2", 1),
        (2.0, 1),
        (_Status.FAIL, 1),
        (_IntSubclass(2), 1),
    ],
)
def test_oneshot_exit_code_bools_are_success(rc, expected):
    assert _oneshot_exit_code(rc) == expected


def test_cmd_proxy_true_does_not_raise_systemexit(monkeypatch):
    monkeypatch.setattr("hermes_cli.proxy.cli.cmd_proxy", lambda args: True)
    cmd_proxy(SimpleNamespace())


def test_cmd_proxy_false_does_not_raise_systemexit(monkeypatch):
    monkeypatch.setattr("hermes_cli.proxy.cli.cmd_proxy", lambda args: False)
    cmd_proxy(SimpleNamespace())


def test_cmd_proxy_none_does_not_raise_systemexit(monkeypatch):
    monkeypatch.setattr("hermes_cli.proxy.cli.cmd_proxy", lambda args: None)
    cmd_proxy(SimpleNamespace())


def test_cmd_proxy_nonzero_int_raises_systemexit(monkeypatch):
    monkeypatch.setattr("hermes_cli.proxy.cli.cmd_proxy", lambda args: 3)
    with pytest.raises(SystemExit) as exc:
        cmd_proxy(SimpleNamespace())
    assert exc.value.code == 3


def test_cmd_proxy_zero_does_not_raise_systemexit(monkeypatch):
    monkeypatch.setattr("hermes_cli.proxy.cli.cmd_proxy", lambda args: 0)
    cmd_proxy(SimpleNamespace())


def _stub_main_to_handler(monkeypatch, result):
    ns = argparse.Namespace(
        version=False,
        yolo=False,
        oneshot=None,
        command="d6-test",
        func=lambda args: result,
    )
    monkeypatch.setattr("hermes_cli.main._set_process_title", lambda: None)
    monkeypatch.setattr("hermes_cli.main._advertise_agent_env", lambda: None)
    monkeypatch.setattr("hermes_cli.main._cleanup_quarantined_exes", lambda: None)
    monkeypatch.setattr("hermes_cli.main._sweep_stale_bytecode_if_checkout_changed", lambda: None)
    monkeypatch.setattr("hermes_cli.main._recover_from_interrupted_install", lambda: None)
    monkeypatch.setattr("hermes_cli.main._try_termux_fast_tui_launch", lambda: False)
    monkeypatch.setattr("hermes_cli.main._try_termux_fast_cli_launch", lambda: False)
    monkeypatch.setattr("hermes_cli.main._try_fast_serve_launch", lambda: False)
    monkeypatch.setattr("hermes_cli.main._try_fast_chat_launch", lambda: False)
    monkeypatch.setattr("hermes_cli.main._build_cli_parser", lambda: (object(), object()))
    monkeypatch.setattr("hermes_cli.config.get_container_exec_info", lambda: None)
    monkeypatch.setattr("hermes_cli.main._parse_cli_args", lambda *a, **k: ns)
    monkeypatch.setattr("hermes_cli.main._prepare_agent_startup", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["hermes", "d6-test"])


@pytest.mark.parametrize("result", [None, False, True, 0, "2"])
def test_main_treats_non_status_returns_as_success(monkeypatch, result):
    _stub_main_to_handler(monkeypatch, result)
    hermes_main()


def test_main_propagates_exact_nonzero_int(monkeypatch):
    _stub_main_to_handler(monkeypatch, 7)
    with pytest.raises(SystemExit) as exc:
        hermes_main()
    assert exc.value.code == 7


def test_main_handler_systemexit_keeps_its_code(monkeypatch):
    def boom(args):
        raise SystemExit(4)

    _stub_main_to_handler(monkeypatch, None)
    ns = argparse.Namespace(
        version=False, yolo=False, oneshot=None, command="d6-test", func=boom,
    )
    monkeypatch.setattr("hermes_cli.main._parse_cli_args", lambda *a, **k: ns)
    with pytest.raises(SystemExit) as exc:
        hermes_main()
    assert exc.value.code == 4


_DRIVER = """
import argparse
import os
import sys
from hermes_cli import main as main_mod

values = {
    "none": None,
    "false": False,
    "true": True,
    "zero": 0,
    "two": 2,
    "text_two": "2",
}
value = values[os.environ["D6_TEST_RETURN_VALUE"]]
main_mod._set_process_title = lambda: None
main_mod._advertise_agent_env = lambda: None
main_mod._cleanup_quarantined_exes = lambda: None
main_mod._sweep_stale_bytecode_if_checkout_changed = lambda: None
main_mod._recover_from_interrupted_install = lambda: None
main_mod._try_termux_fast_tui_launch = lambda: False
main_mod._try_termux_fast_cli_launch = lambda: False
main_mod._try_fast_serve_launch = lambda: False
main_mod._try_fast_chat_launch = lambda: False
main_mod._build_cli_parser = lambda: (object(), object())
main_mod._parse_cli_args = lambda *a, **k: argparse.Namespace(
    version=False, yolo=False, oneshot=None, command="d6-test",
    func=lambda args: value,
)
main_mod._prepare_agent_startup = lambda args: None
import hermes_cli.config as config
config.get_container_exec_info = lambda: None
sys.argv = ["hermes", "d6-test"]
main_mod.main()
"""


def _run_main_process(tmp_path: Path, value_name: str) -> subprocess.CompletedProcess[str]:
    driver = tmp_path / "d6_main_driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path / "home"),
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "PYTHONPATH": str(REPO_ROOT),
        "D6_TEST_RETURN_VALUE": value_name,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if os.environ.get("SYSTEMROOT"):
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return subprocess.run(
        [sys.executable, str(driver)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    ("value_name", "expected_code"),
    [
        ("none", 0),
        ("false", 0),
        ("true", 0),
        ("zero", 0),
        ("two", 2),
        ("text_two", 0),
    ],
)
def test_actual_process_propagates_only_exact_integer_return_codes(
    tmp_path: Path, value_name: str, expected_code: int,
) -> None:
    result = _run_main_process(tmp_path, value_name)
    assert result.returncode == expected_code, result.stderr
