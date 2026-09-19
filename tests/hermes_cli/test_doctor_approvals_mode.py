"""Doctor must surface approvals.mode=off (#111661).

The check resolves the effective mode through the terminal guard's own reader
(``tools.approval_context._get_approval_mode``): YAML 1.1 parses a bare
``mode: off`` as boolean ``False`` and only the guard's normalizer maps that
back to ``"off"`` — a raw config read would miss the exact case the check
exists for. These tests drive the REAL config loader against a temp
``HERMES_HOME`` so the bool-from-YAML path is exercised for real.
"""

import io
import contextlib
from argparse import Namespace

import pytest

from hermes_cli import config as hc
from hermes_cli.doctor_config import _check_approvals_mode
from hermes_cli.doctor_report import Finding


@pytest.fixture
def approvals_home(tmp_path, monkeypatch):
    """Temp HERMES_HOME whose config.yaml carries the given approvals block verbatim."""

    def _make(raw_approvals: str):
        home = tmp_path / "hermes"
        home.mkdir(exist_ok=True)
        (home / "config.yaml").write_text(
            "model:\n  default: test-model\n" + raw_approvals, encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        hc._LOAD_CONFIG_CACHE.clear()
        return home

    yield _make
    hc._LOAD_CONFIG_CACHE.clear()


def _run_check():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        f = _check_approvals_mode(False)
    return buf.getvalue(), f


def test_bare_off_bool_from_yaml_warns(approvals_home):
    approvals_home("approvals:\n  mode: off\n")  # YAML 1.1 → mode is False here
    assert (
        hc.load_config_readonly()["approvals"]["mode"] is False
    )  # the trap, exercised for real
    out, f = _run_check()
    assert "Approvals are disabled (approvals.mode=off)" in out
    assert "/approvals manual" in out
    assert f.manual_issues and "approvals manual" in f.manual_issues[0].lower()


def test_quoted_off_warns(approvals_home):
    approvals_home('approvals:\n  mode: "off"\n')
    out, f = _run_check()
    assert "Approvals are disabled (approvals.mode=off)" in out
    assert f.manual_issues


def test_manual_is_ok(approvals_home):
    approvals_home("approvals:\n  mode: manual\n")
    out, f = _run_check()
    assert "Approval mode: manual" in out
    assert "disabled" not in out
    assert not f.manual_issues


def test_smart_is_ok(approvals_home):
    approvals_home("approvals:\n  mode: smart\n")
    out, f = _run_check()
    assert "Approval mode: smart" in out
    assert not f.manual_issues


def test_missing_block_defaults_to_ok(approvals_home):
    approvals_home("command_allowlist: []\n")  # no approvals block: merged defaults decide
    out, f = _run_check()
    assert "Approval mode:" in out  # visible either way (defaults merge "smart" today)
    assert "disabled" not in out
    assert not f.manual_issues


def test_full_doctor_run_prints_the_warning(approvals_home, monkeypatch, tmp_path):
    """The check is wired into DOCTOR_CHECKS: a full doctor run on mode=off surfaces the row."""
    import types, sys
    import hermes_cli.doctor as doctor_mod

    home = approvals_home("approvals:\n  mode: off\n")
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    (tmp_path / "project").mkdir(exist_ok=True)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)
    try:
        from hermes_cli import auth as _auth_mod

        monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False, ack=None))
    out = buf.getvalue()
    assert "Approvals Mode" in out
    assert "Approvals are disabled (approvals.mode=off)" in out
    assert "Approvals are off: run /approvals manual" in out
