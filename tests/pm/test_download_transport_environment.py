"""Download tuning survives PM isolation without changing lock resolution (#148262)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from pm.environment import PythonEnvironment, _base_environment


@pytest.mark.parametrize("no_config", [False, True])
def test_download_tuning_reaches_child_without_overriding_pm_paths(tmp_path, monkeypatch, no_config):
    for key in list(os.environ):
        if key.startswith(("UV_", "PIP_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    settings = {
        "UV_CONCURRENT_DOWNLOADS": "2",
        "UV_HTTP_RETRIES": "7",
        "UV_HTTP_TIMEOUT": "900",
        "UV_INDEX_URL": "https://mirror.example/simple",
        "UV_PYTHON": "/unwanted/python",
        "UV_PROJECT_ENVIRONMENT": "/unwanted/venv",
        "UV_CACHE_DIR": "/unwanted/cache",
        "UV_NO_SYNC": "1",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    probe = tmp_path / "child_environment.py"
    probe.write_text(
        "import json, os\n"
        "print(json.dumps({k: v for k, v in os.environ.items() if k.startswith('UV_')}))\n",
        encoding="utf-8",
    )
    environment = PythonEnvironment(
        uv=Path(sys.executable),
        python=Path(sys.executable),
        destination=tmp_path / "venv",
        cache=tmp_path / "cache",
        env=_base_environment(),
        no_config=no_config,
    )

    # Use a real child to inspect the environment handed to the uv boundary,
    # without downloading packages or relying on the host's uv installation.
    result = environment._run([str(probe)], cwd=tmp_path, timeout=10)

    assert result.returncode == 0, result.stderr
    child = json.loads(result.stdout)
    assert child["UV_CONCURRENT_DOWNLOADS"] == "2"
    assert child["UV_HTTP_RETRIES"] == "7"
    assert child["UV_HTTP_TIMEOUT"] == "900"
    assert child["UV_PYTHON"] == str(environment.python)
    assert child["UV_PROJECT_ENVIRONMENT"] == str(environment.destination)
    assert child["UV_CACHE_DIR"] == str(environment.cache)
    assert "UV_NO_SYNC" not in child
    if no_config:
        assert "UV_INDEX_URL" not in child
    else:
        assert child["UV_INDEX_URL"] == settings["UV_INDEX_URL"]
