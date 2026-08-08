"""Regression tests for Python dependency post-setup hooks."""

import builtins
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_cli import setup, tools_config


@pytest.mark.parametrize(
    ("post_setup_key", "module_name"),
    [
        ("faster_whisper", "faster_whisper"),
        ("kittentts", "kittentts"),
        ("piper", "piper"),
        ("ddgs", "ddgs"),
    ],
)
def test_missing_python_backend_install_does_not_force_dependency_upgrades(
    monkeypatch: pytest.MonkeyPatch,
    post_setup_key: str,
    module_name: str,
) -> None:
    """Post-setup can run beside a live Gateway using the same venv."""
    real_import = builtins.__import__

    def import_with_backend_missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == module_name:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    install_calls: list[list[str]] = []

    def record_install(args: list[str], **_kwargs: Any) -> SimpleNamespace:
        install_calls.append(args)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(builtins, "__import__", import_with_backend_missing)
    monkeypatch.setattr(tools_config, "_pip_install", record_install)

    tools_config._run_post_setup(post_setup_key)

    assert len(install_calls) == 1
    assert "-U" not in install_calls[0]
    assert "--upgrade" not in install_calls[0]


@pytest.mark.parametrize(
    "installer",
    [setup._install_neutts_deps, setup._install_kittentts_deps],
)
def test_setup_backend_install_does_not_force_dependency_upgrades(
    monkeypatch: pytest.MonkeyPatch,
    installer: Callable[[], bool],
) -> None:
    """Re-running setup must not eagerly upgrade a live Gateway's venv."""
    install_calls: list[list[str]] = []

    def record_install(args: list[str], **_kwargs: Any) -> SimpleNamespace:
        install_calls.append(args)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(setup, "_check_espeak_ng", lambda: True)
    monkeypatch.setattr(tools_config, "_pip_install", record_install)

    assert installer() is True
    assert len(install_calls) == 1
    assert "-U" not in install_calls[0]
    assert "--upgrade" not in install_calls[0]
