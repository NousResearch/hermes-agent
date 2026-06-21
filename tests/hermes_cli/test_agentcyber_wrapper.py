from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "agentcyber"


def test_agentcyber_wrapper_exists_and_is_valid_shell() -> None:
    assert WRAPPER.exists(), "scripts/agentcyber wrapper is missing"

    result = subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_agentcyber_wrapper_uses_dedicated_home_and_repo_root(tmp_path: Path) -> None:
    agentcyber_home = tmp_path / "agentcyber-home"

    result = subprocess.run(
        [str(WRAPPER), "--print-runtime-env", "status", "--json"],
        capture_output=True,
        text=True,
        env={"AGENTCYBER_HOME": str(agentcyber_home), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["repo_root"] == str(REPO_ROOT)
    assert payload["hermes_home"] == str(agentcyber_home)
    assert payload["argv"] == ["agentcyber", "status", "--json"]


def test_agentcyber_wrapper_defaults_to_repo_local_home_and_chat() -> None:
    result = subprocess.run(
        [str(WRAPPER), "--print-runtime-env"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["hermes_home"] == str(REPO_ROOT / ".agentcyber-home")
    assert payload["argv"] == []


def test_agentcyber_wrapper_rejects_profile_flags(tmp_path: Path) -> None:
    result = subprocess.run(
        [str(WRAPPER), "status", "--profile", "default"],
        capture_output=True,
        text=True,
        env={"AGENTCYBER_HOME": str(tmp_path / "agentcyber-home"), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 2
    assert "does not accept --profile/-p" in result.stderr


def test_agentcyber_wrapper_rejects_default_hermes_home(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    default_hermes = fake_home / ".hermes"

    result = subprocess.run(
        [str(WRAPPER), "--print-runtime-env", "status"],
        capture_output=True,
        text=True,
        env={"AGENTCYBER_HOME": str(default_hermes), "HOME": str(fake_home), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 2
    assert "must not point at default ~/.hermes" in result.stderr


def test_agentcyber_wrapper_rejects_noncanonical_default_hermes_home(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    noncanonical_default = fake_home / "." / ".hermes" / "profiles" / "coder"

    result = subprocess.run(
        [str(WRAPPER), "--print-runtime-env", "status"],
        capture_output=True,
        text=True,
        env={
            "AGENTCYBER_HOME": str(noncanonical_default),
            "HOME": str(fake_home),
            "PATH": "/usr/bin:/bin",
        },
    )

    assert result.returncode == 2
    assert "must not point at default ~/.hermes" in result.stderr


def test_agentcyber_wrapper_ignores_sticky_active_profile(tmp_path: Path) -> None:
    agentcyber_home = tmp_path / "agentcyber-home"
    profile_home = agentcyber_home / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (agentcyber_home / "active_profile").write_text("coder\n", encoding="utf-8")

    result = subprocess.run(
        [str(WRAPPER), "hermes", "config", "path"],
        capture_output=True,
        text=True,
        env={
            "AGENTCYBER_HOME": str(agentcyber_home),
            "HOME": str(tmp_path / "home"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        },
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(agentcyber_home / "config.yaml")
