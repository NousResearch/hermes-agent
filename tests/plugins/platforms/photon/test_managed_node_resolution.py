"""Photon must resolve the Hermes-managed Node, not just PATH.

``$HERMES_HOME/node`` is never on an arbitrary process's PATH — only the
generated service units get it prepended — so on an install whose Node *is*
the managed one (the installer provisions it precisely when system Node/npm
is missing or below the engines floor) every bare ``shutil.which()`` in the
photon platform resolves nothing:

  * ``check_requirements()`` reports photon unavailable, so the gateway never
    creates the adapter at all,
  * ``_reinstall_sidecar_deps()`` gives up before running npm,
  * ``hermes photon install-sidecar`` refuses to run,
  * the sidecar spawn falls back to the literal ``"node"`` and dies with
    FileNotFoundError on every connect.

These tests pin the managed rung for each of those call sites. PATH is
emptied in every one, so they exercise the managed lookup specifically.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import hermes_constants
from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as adapter_mod
from plugins.platforms.photon import cli as cli_mod
from plugins.platforms.photon.adapter import PhotonAdapter


MANAGED_NODE = str(Path("/opt/hermes/node/bin/node"))
MANAGED_NPM = str(Path("/opt/hermes/node/bin/npm"))
MANAGED_DIR = str(Path("/opt/hermes/node/bin"))


@pytest.fixture
def managed_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """An install with a managed Node tree and nothing on PATH."""

    def _managed(command: str) -> str | None:
        return {"node": MANAGED_NODE, "npm": MANAGED_NPM}.get(Path(command).name)

    def _with_managed_path(env: dict[str, str] | None = None) -> dict[str, str]:
        merged = dict(os.environ if env is None else env)
        merged["PATH"] = os.pathsep.join([MANAGED_DIR, merged.get("PATH", "")])
        return merged

    def _which(command: str, *args: Any, **kwargs: Any) -> str | None:
        """No system Node/npm; an absolute path resolves only if it exists.

        Pinned rather than inherited from the host so the managed rung is what
        decides the outcome on a developer box that happens to have Node.
        """
        candidate = Path(command)
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        return None

    monkeypatch.setattr(hermes_constants, "find_node_executable", _managed)
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", _with_managed_path)
    monkeypatch.setattr(adapter_mod.shutil, "which", _which)
    monkeypatch.delenv("PHOTON_NODE_BIN", raising=False)


def _installed_sidecar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(adapter_mod, "_SIDECAR_DIR", tmp_path)
    (tmp_path / "node_modules" / "spectrum-ts").mkdir(parents=True)


# ---------------------------------------------------------------------------
# check_requirements() — the gate that decides photon exists at all
# ---------------------------------------------------------------------------


def test_check_requirements_accepts_a_managed_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A managed Node with an empty PATH must still report photon available."""
    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", True)
    _installed_sidecar(monkeypatch, tmp_path)

    assert adapter_mod.check_requirements() is True


def test_check_requirements_accepts_an_explicit_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``PHOTON_NODE_BIN`` stays authoritative over the managed tree."""
    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", True)
    _installed_sidecar(monkeypatch, tmp_path)
    override = tmp_path / "custom-node"
    override.write_text("", encoding="utf-8")
    monkeypatch.setenv("PHOTON_NODE_BIN", str(override))

    assert adapter_mod.check_requirements() is True


def test_check_requirements_rejects_a_broken_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An override that points nowhere is still a hard failure."""
    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", True)
    _installed_sidecar(monkeypatch, tmp_path)
    monkeypatch.setenv("PHOTON_NODE_BIN", str(tmp_path / "missing-node"))

    assert adapter_mod.check_requirements() is False


def test_check_requirements_false_without_any_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No managed tree and no PATH entry is still unavailable."""
    monkeypatch.setattr(adapter_mod, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda c: None)
    _installed_sidecar(monkeypatch, tmp_path)

    assert adapter_mod.check_requirements() is False


# ---------------------------------------------------------------------------
# Sidecar spawn
# ---------------------------------------------------------------------------


def test_sidecar_spawns_the_managed_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interpreter the sidecar is spawned with must be the managed one."""
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")

    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))

    assert adapter._node_bin == MANAGED_NODE


def test_explicit_override_still_wins_for_the_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    monkeypatch.setenv("PHOTON_NODE_BIN", "/custom/node")

    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))

    assert adapter._node_bin == "/custom/node"


# ---------------------------------------------------------------------------
# Dependency (re)install — adapter side
# ---------------------------------------------------------------------------


def test_reinstall_runs_managed_npm_with_managed_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """npm must be the managed binary, and its child node reachable by name."""
    _installed_sidecar(monkeypatch, tmp_path)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(cmd: list[str], **kwargs: Any) -> _Result:
        calls.append((cmd, kwargs))
        return _Result()

    monkeypatch.setattr(adapter_mod.subprocess, "run", _run)

    adapter_mod._reinstall_sidecar_deps()

    assert calls, "npm was never run"
    cmd, kwargs = calls[0]
    assert cmd[0] == MANAGED_NPM
    assert cmd[1] == "ci"
    assert MANAGED_DIR in kwargs["env"]["PATH"].split(os.pathsep)


def test_reinstall_gives_up_when_no_npm_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With neither a managed nor a PATH npm, the reinstall is still skipped."""
    _installed_sidecar(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda c: None)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        adapter_mod.subprocess, "run", lambda cmd, **k: calls.append(cmd)
    )

    adapter_mod._reinstall_sidecar_deps()

    assert calls == []


# ---------------------------------------------------------------------------
# `hermes photon install-sidecar` — CLI side
# ---------------------------------------------------------------------------


def test_install_sidecar_runs_managed_npm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI installer must not refuse an install that has a managed npm."""
    monkeypatch.setattr(cli_mod, "_SIDECAR_DIR", tmp_path)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class _Result:
        returncode = 0
        stderr = ""

    def _run(cmd: list[str], **kwargs: Any) -> _Result:
        calls.append((cmd, kwargs))
        return _Result()

    monkeypatch.setattr(cli_mod.subprocess, "run", _run)

    assert cli_mod._install_sidecar() == 0
    assert calls, "npm was never run"
    cmd, kwargs = calls[0]
    assert cmd[0] == MANAGED_NPM
    assert MANAGED_DIR in kwargs["env"]["PATH"].split(os.pathsep)


def test_install_sidecar_reports_when_no_npm_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no npm anywhere the installer still exits non-zero with advice."""
    monkeypatch.setattr(cli_mod, "_SIDECAR_DIR", tmp_path)
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda c: None)

    assert cli_mod._install_sidecar() == 1
    assert "Node.js" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Env helper
# ---------------------------------------------------------------------------


def test_sidecar_env_keeps_the_caller_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The managed PATH is an overlay — it must not drop existing variables."""
    monkeypatch.setenv("PHOTON_MARKER", "kept")

    env = adapter_mod.sidecar_node_env()

    assert env["PHOTON_MARKER"] == "kept"
    assert MANAGED_DIR in env["PATH"].split(os.pathsep)


pytestmark = pytest.mark.usefixtures("managed_only")
