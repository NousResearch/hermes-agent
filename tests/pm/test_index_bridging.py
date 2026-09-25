"""Mirrored and air-gapped networks configure indexes through pip or uv; PM forwards
exactly that into uv while still refusing every other ambient uv setting."""
from __future__ import annotations

import os
import subprocess

import pytest

from pm.environment import PythonEnvironment, _base_environment
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


def test_default_index_override_detects_mirrors_only(clean_index_env):
    from pm.index_config import default_index_override

    assert default_index_override({}) is None
    assert default_index_override({"UV_INDEX_URL": "https://pypi.org/simple"}) is None
    # Trailing slash is the same registry; extra indexes never move the default.
    assert default_index_override({"UV_DEFAULT_INDEX": "https://pypi.org/simple/"}) is None
    assert default_index_override({"UV_INDEX": "https://mirror.example/simple"}) is None
    assert default_index_override({"UV_INDEX_URL": "https://mirror.example/simple"}) == \
        "https://mirror.example/simple"
    assert default_index_override({"UV_DEFAULT_INDEX": "https://mirror.example/simple/"}) == \
        "https://mirror.example/simple"


def test_bridged_pip_mirror_counts_as_default_index_override(clean_index_env, monkeypatch):
    from pm.index_config import default_index_override

    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.example/simple")

    env = _base_environment()
    assert default_index_override(env) == "https://mirror.example/simple"


def test_stage_runtime_relocks_snapshot_when_index_is_mirrored(clean_index_env, monkeypatch, tmp_path):
    """The pip.conf bridge is what breaks ``hermes pm doctor`` on mirrored hosts (#122112).

    uv --locked rejects the committed lock once resolution goes through a mirror,
    so the staged snapshot — a caller-owned copy — must re-resolve instead.
    """
    import pm.runtime_stage as runtime_stage
    from pm.environment import PythonEnvironment

    sync_calls: list[bool] = []

    def record_sync(self, source, *, locked, **kwargs):
        sync_calls.append(locked)

    monkeypatch.setattr(PythonEnvironment, "create", lambda self: None)
    monkeypatch.setattr(PythonEnvironment, "sync", record_sync)
    monkeypatch.setattr(runtime_stage.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.example/simple")
    runtime_stage.stage_runtime(tmp_path / "uv", tmp_path / "python", tmp_path / "runtime")
    assert sync_calls == [False]

    sync_calls.clear()
    monkeypatch.delenv("PIP_INDEX_URL")
    runtime_stage.stage_runtime(tmp_path / "uv", tmp_path / "python", tmp_path / "runtime2")
    assert sync_calls == [True]
