"""Tests for ``hermes_cli.config.runtime_repair_capability`` (#79179).

The resolver answers "what can ``hermes update`` repair for this install?"
as a property of who owns the interpreter, not of the install-method label:

- ``docker`` / ``nix`` / ``nixos`` — method-owned runtimes.
- ``managed`` — the checkout owns a venv (official git clone, dev checkout),
  so ``hermes update`` can rebuild the interpreter and its linked SQLite.
- ``environment`` — system Python, pip --user, or an unknown layout; the
  environment owner must upgrade the interpreter.
"""
from pathlib import Path

import pytest

from hermes_cli.config import runtime_repair_capability


def _checkout(tmp_path: Path, with_venv: bool) -> Path:
    """A bare git-like checkout, optionally carrying a Hermes-owned venv."""
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = \"hermes-agent\"\n")
    if with_venv:
        (root / "venv" / "bin").mkdir(parents=True)
        (root / "venv" / "bin" / "python").touch()
    return root


def _patch_method(monkeypatch, method: str):
    monkeypatch.setattr(
        "hermes_cli.config.detect_install_method",
        lambda _root=None: method,
    )


class TestRuntimeRepairCapability:
    def test_docker_is_method_owned(self, monkeypatch, tmp_path):
        _patch_method(monkeypatch, "docker")
        assert runtime_repair_capability(tmp_path) == "docker"

    def test_nix_is_method_owned(self, monkeypatch, tmp_path):
        _patch_method(monkeypatch, "nix")
        assert runtime_repair_capability(tmp_path) == "nix"

    def test_nixos_is_method_owned(self, monkeypatch, tmp_path):
        _patch_method(monkeypatch, "nixos")
        assert runtime_repair_capability(tmp_path) == "nixos"

    def test_official_git_clone_with_managed_venv_is_managed(self, monkeypatch, tmp_path):
        """The curl installer git-clones AND creates <checkout>/venv — the
        stamp says ``git`` but the runtime is Hermes-owned and repairable."""
        _patch_method(monkeypatch, "git")
        assert runtime_repair_capability(_checkout(tmp_path, with_venv=True)) == "managed"

    def test_dev_checkout_with_dot_venv_is_managed(self, monkeypatch, tmp_path):
        """A dev checkout's .venv is also Hermes-owned (uv default layout)."""
        _patch_method(monkeypatch, "git")
        root = tmp_path / "dev"
        root.mkdir()
        (root / ".git").mkdir()
        (root / ".venv" / "bin").mkdir(parents=True)
        (root / ".venv" / "bin" / "python").touch()
        assert runtime_repair_capability(root) == "managed"

    def test_git_checkout_without_venv_is_environment_owned(self, monkeypatch, tmp_path):
        """A bare git checkout running system Python cannot be repaired by
        ``hermes update`` — the environment owner must upgrade."""
        _patch_method(monkeypatch, "git")
        assert (
            runtime_repair_capability(_checkout(tmp_path, with_venv=False))
            == "environment"
        )

    def test_unknown_layout_without_venv_is_environment_owned(self, monkeypatch, tmp_path):
        _patch_method(monkeypatch, "unknown")
        assert runtime_repair_capability(tmp_path) == "environment"

    def test_windows_scripts_python_layout_is_managed(self, monkeypatch, tmp_path):
        _patch_method(monkeypatch, "git")
        root = tmp_path / "win"
        root.mkdir()
        (root / ".git").mkdir()
        (root / "venv" / "Scripts").mkdir(parents=True)
        (root / "venv" / "Scripts" / "python.exe").touch()
        assert runtime_repair_capability(root) == "managed"
