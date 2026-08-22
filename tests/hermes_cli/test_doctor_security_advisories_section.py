"""Tests for the doctor "Supply-Chain Advisories" section wording.

hermes doctor's advisory check only scans a small hardcoded catalog of
known-compromised (supply-chain-worm) package versions — it is not a CVE
scanner. Before this fix it printed the all-clear as "No active security
advisories" under a section literally titled "Security Advisories", which
reads as a blanket clearance even though `hermes security audit` (an OSV.dev
CVE scan) can report active findings at the same time. See #91931.
"""

import contextlib
import io
import sys
import types
from argparse import Namespace
from pathlib import Path

import hermes_cli.doctor as doctor_mod


def _setup_doctor_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")

    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    venv_bin_dir = project / "venv" / "bin"
    venv_bin_dir.mkdir(parents=True, exist_ok=True)
    hermes_bin = venv_bin_dir / "hermes"
    hermes_bin.write_text("#!/usr/bin/env python\n# entry point\n")
    hermes_bin.chmod(0o755)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
    except Exception:
        pass

    try:
        import httpx
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: types.SimpleNamespace(status_code=200))
    except Exception:
        pass


def _run_doctor():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    return buf.getvalue()


class TestSecurityAdvisoriesSectionWording:
    def test_section_is_scoped_to_supply_chain_not_all_security(self, monkeypatch, tmp_path):
        _setup_doctor_env(monkeypatch, tmp_path)

        out = _run_doctor()

        assert "Supply-Chain Advisories" in out
        assert "No active security advisories" not in out
        assert "hermes security audit" in out

    def test_all_clear_message_names_compromised_packages_not_advisories(
        self, monkeypatch, tmp_path
    ):
        _setup_doctor_env(monkeypatch, tmp_path)

        out = _run_doctor()

        assert "No known-compromised packages detected" in out
