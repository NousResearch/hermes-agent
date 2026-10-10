"""Windows gateway boot must trust only a venv proven to match this interpreter.

Regression: a leftover CPython 3.11 dev ``venv`` beside the checkout won
``_ensure_windows_gateway_venv_imports()`` on every boot of the store-Python
3.14 gateway -- ``Lib/site-packages`` exists in both trees, so existence alone
decided -- and the first lazy ``import pydantic`` resolved
``_pydantic_core.cp311-win_amd64.pyd`` under 3.14. Every turn died at client
init (``No module named 'pydantic_core._pydantic_core'`` -> ``Failed to
initialize OpenAI client``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gateway.run as gateway_run

RUNNING = "%d.%d.%d" % sys.version_info[:3]
FOREIGN = "%d.%d" % (sys.version_info.major, sys.version_info.minor + 1)


def _make_venv(parent: Path, name: str = "venv", *, cfg: str | None = None) -> Path:
    venv = parent / name
    (venv / "Lib" / "site-packages").mkdir(parents=True)
    if cfg is not None:
        (venv / "pyvenv.cfg").write_text(cfg, encoding="utf-8")
    return venv


def _site_packages(venv: Path) -> str:
    return str(venv / "Lib" / "site-packages")


def _windows_boot(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(gateway_run, "__file__", str(tmp_path / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return tmp_path


def test_foreign_abi_in_tree_venv_is_skipped(monkeypatch, tmp_path) -> None:
    root = _windows_boot(monkeypatch, tmp_path)
    stale = _make_venv(root, cfg="version_info = %s\n" % FOREIGN)

    gateway_run._ensure_windows_gateway_venv_imports()

    assert _site_packages(stale) not in sys.path
    assert "VIRTUAL_ENV" not in os.environ
    assert "PYTHONPATH" not in os.environ


def test_matching_abi_venv_is_still_injected(monkeypatch, tmp_path) -> None:
    root = _windows_boot(monkeypatch, tmp_path)
    venv = _make_venv(root, cfg="version_info = %s.final.0\n" % RUNNING)

    gateway_run._ensure_windows_gateway_venv_imports()

    assert _site_packages(venv) in sys.path
    assert os.path.normcase(os.environ.get("VIRTUAL_ENV", "")) == os.path.normcase(str(venv))


def test_foreign_or_unproven_virtual_env_falls_through(monkeypatch, tmp_path) -> None:
    root = _windows_boot(monkeypatch, tmp_path)
    stale = _make_venv(root, name="stale", cfg="version_info = %s\n" % FOREIGN)
    unproven = _make_venv(root, name="unproven", cfg="home = C:\\nowhere\n")
    good = _make_venv(root, cfg="version = %s\n" % RUNNING)

    monkeypatch.setenv("VIRTUAL_ENV", str(stale))
    gateway_run._ensure_windows_gateway_venv_imports()
    assert _site_packages(good) in sys.path
    assert _site_packages(stale) not in sys.path

    monkeypatch.setattr(sys, "path", list(sys.path))  # fresh look at the candidates
    monkeypatch.setenv("VIRTUAL_ENV", str(unproven))
    gateway_run._ensure_windows_gateway_venv_imports()
    assert _site_packages(unproven) not in sys.path, "an unprovable ABI must not be trusted"
    assert _site_packages(good) in sys.path
