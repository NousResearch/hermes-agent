"""Invariant tests for the optional wavespeed skill.

Covers optional-skills/creative/wavespeed. Asserts contracts (frontmatter
hardline, the setup checker's exit codes with a stubbed binary), not
snapshots of skill prose. No network.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
SKILL_DIR = REPO / "optional-skills" / "creative" / "wavespeed"
SKILL_MD = SKILL_DIR / "SKILL.md"
CHECK_SETUP = SKILL_DIR / "scripts" / "check_setup.py"


def _frontmatter() -> dict:
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    end = text.index("\n---", 4)
    return yaml.safe_load(text[4:end])


def _load_check_setup():
    spec = importlib.util.spec_from_file_location("check_setup", CHECK_SETUP)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_wavespeed(tmp_path: Path, status_stdout: str, status_rc: int = 0) -> Path:
    """Write a fake `wavespeed` binary that answers --version and status."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI only
        script = bindir / "wavespeed.cmd"
        script.write_text(
            "@echo off\r\n"
            'if "%1"=="--version" (echo 0.4.8 & exit /b 0)\r\n'
            f'if "%1"=="status" (echo {status_stdout} & exit /b {status_rc})\r\n'
            "exit /b 1\r\n",
            encoding="utf-8",
        )
    else:
        script = bindir / "wavespeed"
        script.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then echo 0.4.8; exit 0; fi\n'
            f'if [ "$1" = "status" ]; then echo "{status_stdout}"; exit {status_rc}; fi\n'
            "exit 1\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return bindir


# --- frontmatter contracts ---------------------------------------------------


def test_frontmatter_hardline():
    fm = _frontmatter()
    assert fm["name"] == "wavespeed" == SKILL_DIR.name
    assert len(fm["description"]) <= 60 and fm["description"].endswith(".")
    assert fm["license"] == "MIT"
    assert set(fm["platforms"]) == {"linux", "macos", "windows"}
    assert fm["metadata"]["hermes"]["category"] == "creative"
    assert "wavespeed" in fm["prerequisites"]["commands"]


def test_api_key_is_optional_and_never_requested_in_chat():
    fm = _frontmatter()
    env = {e["name"]: e for e in fm["required_environment_variables"]}
    assert env["WAVESPEED_API_KEY"]["optional"] is True
    body = SKILL_MD.read_text(encoding="utf-8").lower()
    assert "never ask the user to paste a key" in body


def test_referenced_files_exist():
    body = SKILL_MD.read_text(encoding="utf-8")
    for rel in ("scripts/check_setup.py", "references/models.md"):
        assert rel in body
        assert (SKILL_DIR / rel).is_file(), rel


# --- scripts/check_setup.py --------------------------------------------------


def test_check_setup_reports_missing_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    mod = _load_check_setup()
    result = mod.check()
    assert result == {"cli": None, "signed_in": None, "hint": mod.INSTALL_HINT}


def test_check_setup_signed_out(tmp_path, monkeypatch):
    bindir = _stub_wavespeed(tmp_path, "Not signed in", status_rc=1)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    mod = _load_check_setup()
    result = mod.check()
    assert result["cli"] == "0.4.8"
    assert result["signed_in"] is False
    assert result["hint"] == mod.LOGIN_HINT


def test_check_setup_signed_in(tmp_path, monkeypatch):
    bindir = _stub_wavespeed(tmp_path, "Signed in as user@example.com")
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    mod = _load_check_setup()
    result = mod.check()
    assert result["signed_in"] is True and result["hint"] == "ready"


def test_check_setup_env_key_short_circuits_status(tmp_path, monkeypatch):
    bindir = _stub_wavespeed(tmp_path, "Not signed in", status_rc=1)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("WAVESPEED_API_KEY", "wsk_test")
    mod = _load_check_setup()
    assert mod.check()["signed_in"] is True


@pytest.mark.skipif(os.name == "nt", reason="exit-code contract exercised via the POSIX stub")
def test_check_setup_exit_codes(tmp_path, monkeypatch):
    env = dict(os.environ, PATH=str(tmp_path))
    env.pop("WAVESPEED_API_KEY", None)
    proc = subprocess.run([sys.executable, str(CHECK_SETUP)], capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["cli"] is None

    bindir = _stub_wavespeed(tmp_path, "Not signed in", status_rc=1)
    env["PATH"] = str(bindir)
    proc = subprocess.run([sys.executable, str(CHECK_SETUP)], capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 2
