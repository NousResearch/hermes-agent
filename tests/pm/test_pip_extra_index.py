"""Regression for #144691: pip file extra indexes must reach uv build isolation."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from pm import index_config
from pm.environment import PythonEnvironment, _base_environment


@pytest.fixture
def pip_file(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith(("UV_", "PIP_")):
            monkeypatch.delenv(key)
    # Never read the developer's real pip files, including site/global configuration.
    paths = []
    monkeypatch.setattr(index_config, "pip_config_candidates", lambda env: paths)
    return tmp_path, paths


@pytest.mark.parametrize("name", ["pip.ini", "pip.conf"])
@pytest.mark.parametrize("no_config", [False, True])
@pytest.mark.parametrize("extra", [
    "https://user:p%40ss@mirror.example/simple https://pypi.org/simple",
    "\n    https://user:p%40ss@mirror.example/simple\n    https://pypi.org/simple",
])
def test_file_extra_indexes_reach_uv_child(pip_file, monkeypatch, name, no_config, extra):
    root, paths = pip_file
    config = root / name
    config.write_text("[global]\nindex-url = https://download.pytorch.org/whl/cu134\n"
                      f"extra-index-url = {extra}\n", encoding="utf-8")
    paths.append(config)
    seen = []

    def record(command, **kwargs):
        seen.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", record)
    PythonEnvironment(uv=root / "uv", python=root / "python", destination=root / "venv",
                      cache=root / "cache", env=_base_environment(),
                      no_config=no_config)._run(["sync"], cwd=root, timeout=5)

    (child,) = seen
    if no_config:
        assert "UV_INDEX_URL" not in child
        assert "UV_EXTRA_INDEX_URL" not in child
        return
    assert child["UV_INDEX_URL"] == "https://download.pytorch.org/whl/cu134"
    assert child["UV_EXTRA_INDEX_URL"] == (
        "https://user:p%40ss@mirror.example/simple https://pypi.org/simple"
    )


@pytest.mark.parametrize("overrides,expected", [
    ({}, "https://pypi.org/simple"),
    ({"PIP_EXTRA_INDEX_URL": " https://pip-env.example/simple "}, "https://pip-env.example/simple"),
    ({"UV_EXTRA_INDEX_URL": "https://uv.example/simple",
      "PIP_EXTRA_INDEX_URL": "https://pip-env.example/simple"}, "https://uv.example/simple"),
    ({"UV_DEFAULT_INDEX": "https://uv.example/simple"}, "https://pypi.org/simple"),
])
def test_extra_index_precedence_and_file_overlay(pip_file, overrides, expected):
    root, paths = pip_file
    for name, value in [("global.ini", "https://old.example/simple"),
                        ("pip.ini", "https://pypi.org/simple")]:
        config = root / name
        config.write_text(f"[global]\nextra-index-url = {value}\n", encoding="utf-8")
        paths.append(config)

    settings = index_config.bridged_index_settings(overrides)

    assert settings["UV_EXTRA_INDEX_URL"] == expected
    assert "UV_INDEX_URL" not in settings


@pytest.mark.parametrize("text", [
    "[global]\nindex-url = https://mirror.example/simple\n",
    "[global]\nextra-index-url = \n",
    "[install]\nextra-index-url = https://command.example/simple\n",
    "not an ini file\n",
])
def test_missing_or_unusable_file_extra_index_is_not_injected(pip_file, text):
    root, paths = pip_file
    config = root / "pip.ini"
    config.write_text(text, encoding="utf-8")
    paths.append(config)

    assert "UV_EXTRA_INDEX_URL" not in index_config.bridged_index_settings({})


@pytest.mark.platforms("windows")
def test_windows_user_pip_ini_discovery_bridges_extra_indexes(tmp_path, monkeypatch):
    # Exercise native discovery without opening any real user, site or global files.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(index_config.sys, "prefix", str(tmp_path / "site"))
    roaming = tmp_path / "roaming"
    config = roaming / "pip" / "pip.ini"
    config.parent.mkdir(parents=True)
    config.write_text("[global]\nindex-url = https://download.pytorch.org/whl/cu134\n"
                      "extra-index-url =\n    https://pypi.org/simple\n", encoding="utf-8")
    ambient = {"APPDATA": str(roaming), "ProgramData": str(tmp_path / "global")}

    settings = index_config.bridged_index_settings(ambient)

    assert settings["UV_INDEX_URL"] == "https://download.pytorch.org/whl/cu134"
    assert settings["UV_EXTRA_INDEX_URL"] == "https://pypi.org/simple"
    assert "UV_EXTRA_INDEX_URL" not in index_config.bridged_index_settings(
        {**ambient, "PIP_CONFIG_FILE": os.devnull}
    )
