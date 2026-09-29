from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gateway import run as gateway_run


@pytest.mark.parametrize("config_key", ["version", "version_info"])
def test_windows_gateway_venv_imports_skips_cross_minor_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_key: str,
) -> None:
    """Both stdlib and uv pyvenv.cfg formats must prevent cross-minor adoption."""
    venv_dir = tmp_path / "cross-minor"
    site_packages = venv_dir / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (venv_dir / "pyvenv.cfg").write_text(
        f"home = fixture\n{config_key} = 3.11.9\n", encoding="utf-8",
    )

    original_path = list(sys.path)
    monkeypatch.setattr(gateway_run.sys, "platform", "win32")
    monkeypatch.setattr(gateway_run.sys, "version_info", (3, 14, 0))
    monkeypatch.setattr(gateway_run.sys, "path", original_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_dir))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    assert str(site_packages) not in gateway_run.sys.path
    assert gateway_run.os.environ["VIRTUAL_ENV"] == str(venv_dir)


def test_windows_gateway_venv_imports_adopts_matching_stdlib_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal stdlib ``version = X.Y.Z`` environment remains usable."""
    venv_dir = tmp_path / "matching-minor"
    site_packages = venv_dir / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (venv_dir / "pyvenv.cfg").write_text(
        "home = fixture\nversion = 3.14.7\n", encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run.sys, "platform", "win32")
    monkeypatch.setattr(gateway_run.sys, "version_info", (3, 14, 0))
    monkeypatch.setattr(gateway_run.sys, "path", [])
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_dir))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    assert str(site_packages) in gateway_run.sys.path
    assert gateway_run.os.environ["VIRTUAL_ENV"] == str(venv_dir.resolve())
