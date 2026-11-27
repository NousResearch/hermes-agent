"""Unit tests for NixOS updater support (CA bundle autodetect & native toolchain wrapping)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import update_cmd
from hermes_cli import update_cmd_nixos
from hermes_cli.update_cmd_nixos import (
    autodetect_nixos_ca_bundle,
    check_nixos_toolchain,
    is_nixos,
    missing_native_toolchain,
    needs_nixos_toolchain,
    wrap_nixos_toolchain,
)


def _mock_path_exists(existing_paths: set[str]):
    """Return a path-exists mock that returns True only for paths in existing_paths."""
    def fake_exists(path):
        return str(path) in existing_paths
    return fake_exists


def _mock_which(found_tools: dict[str, str | None]):
    """Return a shutil.which mock matching found_tools."""
    def fake_which(cmd, path=None):
        return found_tools.get(cmd)
    return fake_which


# ---------------------------------------------------------------------------
# CA bundle autodetect tests
# ---------------------------------------------------------------------------


def test_nixos_ca_bundle_autodetect_sets_env_when_unset(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(
        Path, "exists",
        _mock_path_exists({"/etc/NIXOS", "/etc/ssl/certs/ca-bundle.crt"}),
    )

    assert autodetect_nixos_ca_bundle() is True
    assert os.environ["SSL_CERT_FILE"] == "/etc/ssl/certs/ca-bundle.crt"


def test_nixos_ca_bundle_autodetect_preserves_explicit_env(monkeypatch):
    custom_cert = "/custom/path/cert.pem"
    monkeypatch.setenv("SSL_CERT_FILE", custom_cert)
    monkeypatch.setattr(
        Path, "exists",
        _mock_path_exists({"/etc/NIXOS", "/etc/ssl/certs/ca-bundle.crt"}),
    )

    assert autodetect_nixos_ca_bundle() is False
    assert os.environ["SSL_CERT_FILE"] == custom_cert


def test_nixos_ca_bundle_autodetect_noop_on_non_nixos(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    # ca-bundle exists, but not /etc/NIXOS
    monkeypatch.setattr(
        Path, "exists",
        _mock_path_exists({"/etc/ssl/certs/ca-bundle.crt"}),
    )

    assert autodetect_nixos_ca_bundle() is False
    assert "SSL_CERT_FILE" not in os.environ


def test_nixos_ca_bundle_autodetect_noop_when_bundle_missing(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    # /etc/NIXOS exists, but ca-bundle does not
    monkeypatch.setattr(
        Path, "exists",
        _mock_path_exists({"/etc/NIXOS"}),
    )

    assert autodetect_nixos_ca_bundle() is False
    assert "SSL_CERT_FILE" not in os.environ


def test_map_ssl_cert_file_for_git_autodetects_on_nixos(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("GIT_SSL_CAINFO", raising=False)
    monkeypatch.setattr(
        Path, "exists",
        _mock_path_exists({"/etc/NIXOS", "/etc/ssl/certs/ca-bundle.crt"}),
    )
    monkeypatch.setattr(
        update_cmd, "_git_run",
        lambda cmd, args, **kwargs: subprocess.CompletedProcess(cmd, 1, "", ""),
    )

    update_cmd._map_ssl_cert_file_for_git(["git"])
    assert os.environ["SSL_CERT_FILE"] == "/etc/ssl/certs/ca-bundle.crt"
    assert os.environ["GIT_SSL_CAINFO"] == "/etc/ssl/certs/ca-bundle.crt"


def test_map_ssl_cert_file_for_git_preserves_explicit_git_ssl_cainfo(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv("GIT_SSL_CAINFO", "/my/git/ca.pem")
    monkeypatch.setattr(
        Path, "exists",
        _mock_path_exists({"/etc/NIXOS", "/etc/ssl/certs/ca-bundle.crt"}),
    )
    monkeypatch.setattr(
        update_cmd, "_git_run",
        lambda cmd, args, **kwargs: subprocess.CompletedProcess(cmd, 1, "", ""),
    )

    update_cmd._map_ssl_cert_file_for_git(["git"])
    # SSL_CERT_FILE is set for Python, but explicit GIT_SSL_CAINFO is preserved
    assert os.environ["SSL_CERT_FILE"] == "/etc/ssl/certs/ca-bundle.crt"
    assert os.environ["GIT_SSL_CAINFO"] == "/my/git/ca.pem"


# ---------------------------------------------------------------------------
# Native toolchain detection & wrapping tests
# ---------------------------------------------------------------------------


def test_needs_nixos_toolchain_when_missing_cc_or_make(monkeypatch):
    monkeypatch.setattr(update_cmd_nixos, "is_nixos", lambda: True)

    # Missing both
    monkeypatch.setattr(update_cmd_nixos.shutil, "which", _mock_which({"cc": None, "make": None}))
    assert missing_native_toolchain() is True
    assert needs_nixos_toolchain() is True

    # Missing cc only
    monkeypatch.setattr(update_cmd_nixos.shutil, "which", _mock_which({"cc": None, "make": "/usr/bin/make"}))
    assert missing_native_toolchain() is True
    assert needs_nixos_toolchain() is True

    # Missing make only
    monkeypatch.setattr(update_cmd_nixos.shutil, "which", _mock_which({"cc": "/usr/bin/cc", "make": None}))
    assert missing_native_toolchain() is True
    assert needs_nixos_toolchain() is True

    # Both present
    monkeypatch.setattr(update_cmd_nixos.shutil, "which", _mock_which({"cc": "/usr/bin/cc", "make": "/usr/bin/make"}))
    assert missing_native_toolchain() is False
    assert needs_nixos_toolchain() is False


def test_needs_nixos_toolchain_noop_on_non_nixos(monkeypatch):
    monkeypatch.setattr(update_cmd_nixos, "is_nixos", lambda: False)
    monkeypatch.setattr(update_cmd_nixos.shutil, "which", _mock_which({"cc": None, "make": None}))

    assert missing_native_toolchain() is True
    assert needs_nixos_toolchain() is False


def test_wrap_nixos_toolchain_wraps_when_needed_and_nix_shell_available(monkeypatch):
    monkeypatch.setattr(update_cmd_nixos, "is_nixos", lambda: True)
    monkeypatch.setattr(
        update_cmd_nixos.shutil, "which",
        _mock_which({"cc": None, "make": None, "nix-shell": "/run/current-system/sw/bin/nix-shell"}),
    )

    cmd = ["python", "-m", "build", "--out", "dist"]
    wrapped = wrap_nixos_toolchain(cmd)
    assert wrapped == ["nix-shell", "-p", "gnumake", "gcc", "--run", "python -m build --out dist"]


def test_wrap_nixos_toolchain_noop_when_toolchain_already_present(monkeypatch):
    monkeypatch.setattr(update_cmd_nixos, "is_nixos", lambda: True)
    monkeypatch.setattr(
        update_cmd_nixos.shutil, "which",
        _mock_which({
            "cc": "/nix/store/gcc/bin/cc",
            "make": "/nix/store/make/bin/make",
            "nix-shell": "/run/current-system/sw/bin/nix-shell",
        }),
    )

    cmd = ["python", "-m", "build"]
    assert wrap_nixos_toolchain(cmd) == cmd


def test_wrap_nixos_toolchain_noop_on_non_nixos(monkeypatch):
    monkeypatch.setattr(update_cmd_nixos, "is_nixos", lambda: False)
    monkeypatch.setattr(
        update_cmd_nixos.shutil, "which",
        _mock_which({"cc": None, "make": None, "nix-shell": "/bin/nix-shell"}),
    )

    cmd = ["python", "-m", "build"]
    assert wrap_nixos_toolchain(cmd) == cmd


def test_check_nixos_toolchain_passes_when_nix_shell_available(monkeypatch):
    monkeypatch.setattr(update_cmd_nixos, "is_nixos", lambda: True)
    monkeypatch.setattr(
        update_cmd_nixos.shutil, "which",
        _mock_which({"cc": None, "make": None, "nix-shell": "/run/current-system/sw/bin/nix-shell"}),
    )

    # Should not raise or exit
    check_nixos_toolchain()


def test_check_nixos_toolchain_fails_fast_when_nix_shell_missing(monkeypatch, capsys):
    monkeypatch.setattr(update_cmd_nixos, "is_nixos", lambda: True)
    monkeypatch.setattr(
        update_cmd_nixos.shutil, "which",
        _mock_which({"cc": None, "make": None, "nix-shell": None}),
    )

    stopped = []
    with pytest.raises(SystemExit) as exc_info:
        check_nixos_toolchain(stop_fn=lambda: stopped.append("stopped"))

    assert exc_info.value.code == 1
    assert stopped == ["stopped"]

    captured = capsys.readouterr()
    assert "nix profile install nixpkgs#gcc nixpkgs#gnumake" in captured.out
    assert "nix-shell -p gnumake gcc --run hermes update" in captured.out
