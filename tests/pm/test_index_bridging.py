"""Mirrored and air-gapped networks configure indexes through pip or uv; PM forwards
exactly that into uv while still refusing every other ambient uv setting."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pm.environment import PythonEnvironment, _base_environment, classify_uv_failure
from pm.package import InstallError


@pytest.fixture
def clean_index_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith(("UV_", "PIP_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    return tmp_path


def test_pip_index_reaches_uv_but_ambient_uv_selection_does_not(clean_index_env, monkeypatch):
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.example/simple")
    monkeypatch.setenv("PIP_TRUSTED_HOST", "mirror.example")
    monkeypatch.setenv("UV_HTTP_TIMEOUT", "300")
    monkeypatch.setenv("UV_INDEX_CORP_PASSWORD", "s3cret")
    monkeypatch.setenv("UV_PYTHON", "/poison/python")
    monkeypatch.setenv("UV_CACHE_DIR", "/poison/cache")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/poison/venv")

    env = _base_environment()

    assert env["UV_INDEX_URL"] == "https://mirror.example/simple"
    assert env["UV_INSECURE_HOST"] == "mirror.example"
    assert env["UV_HTTP_TIMEOUT"] == "300"
    assert env["UV_INDEX_CORP_PASSWORD"] == "s3cret"
    assert not {"UV_PYTHON", "UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT"} & env.keys()


@pytest.mark.parametrize("no_config", [False, True])
def test_download_tuning_reaches_child_without_overriding_pm_paths(clean_index_env, monkeypatch, no_config):
    tmp_path = clean_index_env
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

    # A real child inspects the environment handed to the uv boundary (#148262),
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


def test_pip_conf_is_bridged_only_when_uv_has_no_index(clean_index_env, monkeypatch):
    pip_conf = clean_index_env / "pip.conf"
    # Percent-encoded credentials: pip reads its config raw, so must the bridge.
    pip_conf.write_text("[global]\nindex-url = https://user:p%40ss@mirror.example/simple\n", encoding="utf-8")
    monkeypatch.setenv("PIP_CONFIG_FILE", str(pip_conf))

    assert _base_environment()["UV_INDEX_URL"] == "https://user:p%40ss@mirror.example/simple"

    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://explicit.example/simple")
    env = _base_environment()
    assert env["UV_DEFAULT_INDEX"] == "https://explicit.example/simple"
    assert "UV_INDEX_URL" not in env


def test_streamed_runs_do_not_request_uv_debug_output(tmp_path, monkeypatch):
    import io
    from pm import environment

    monkeypatch.setenv("HERMES_VERBOSE", "1")

    seen: list[list[str]] = []
    kwargs_seen: list[dict] = []

    def record(command, **kwargs):
        seen.append(command)
        kwargs_seen.append(kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(environment, "_run_streaming", record)
    PythonEnvironment(uv=tmp_path / "uv", python=tmp_path / "python", destination=tmp_path / "venv",
                      cache=tmp_path / "cache", env={}, output=io.StringIO())._run(["sync"], cwd=tmp_path, timeout=5)
    (command,), (kwargs,) = seen, kwargs_seen
    # Verbose only where the build backend speaks; uv's own DEBUG stays silent.
    assert "--verbose" in command and kwargs["env"]["RUST_LOG"] == "uv_build_frontend=debug"


def test_uv_timeout_names_the_mirror_knobs(tmp_path, monkeypatch):
    def stall(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", stall)
    environment = PythonEnvironment(uv=tmp_path / "uv", python=tmp_path / "python",
                                    destination=tmp_path / "venv", cache=tmp_path / "cache", env={})
    with pytest.raises(InstallError, match="UV_INDEX_URL") as info:
        environment._run(["sync"], cwd=tmp_path, timeout=7)
    assert "timed out after 7s" in str(info.value)


def test_uv_own_download_timeout_names_the_concurrency_knob():
    # uv's real tail when its HTTP retries run out on a slow link (#148262).
    output = ("error: Failed to download `pydantic==2.14.1`\n"
              "  Caused by: Request failed after 5 retries in 127.8s\n"
              "  Caused by: operation timed out\n")
    assert "UV_CONCURRENT_DOWNLOADS" in classify_uv_failure("sync", 1, output).remedy
    local_write = "error: Failed to write to the cache\n  Caused by: operation timed out\n"
    assert "UV_CONCURRENT_DOWNLOADS" not in classify_uv_failure("sync", 1, local_write).remedy
