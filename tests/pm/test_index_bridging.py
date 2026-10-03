"""Mirrored and air-gapped networks configure indexes through pip or uv; PM forwards
exactly that into uv while still refusing every other ambient uv setting."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

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


class _RecordingEnvironment:
    """Records the uv steps a staged runtime asks for; no uv, no network, no home."""

    instances: list["_RecordingEnvironment"] = []

    def __init__(self, **kwargs):
        self.env = kwargs["env"]
        self.executable = Path("/nonexistent/python")
        self.steps: list[str] = []
        self.sources: dict[str, Path] = {}
        _RecordingEnvironment.instances.append(self)

    def create(self) -> None:
        self.steps.append("create")

    def lock(self, source, *, upgrade: bool = False, **kwargs) -> None:
        assert upgrade is False, "re-recording must not move pins"
        self.steps.append("lock")
        self.sources["lock"] = Path(source)

    def sync(self, source, **kwargs) -> None:
        assert kwargs.get("locked") is True, "staging must verify the lock, not bypass it"
        self.steps.append("sync")
        self.sources["sync"] = Path(source)


#: The shape of pm/uv.lock: one virtual root, every dependency from one registry.
_STAGED_LOCK = """version = 1
revision = 3
requires-python = "==3.14.*"

[[package]]
name = "hermes-pm-runtime"
version = "0.0.0"
source = { virtual = "." }

[[package]]
name = "packaging"
version = "26.0"
source = { registry = "%s" }
sdist = { url = "https://mirror.example/packages/packaging-26.0.tar.gz", hash = "sha256:00" }
"""


def _stage_runtime(monkeypatch, tmp_path, env, *, registry: str | None = "https://pypi.org/simple"):
    from pm.runtime_stage import stage_runtime

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "hermes-pm-runtime"\nversion = "0"\n',
                                            encoding="utf-8")
    (project / "uv.lock").write_text("version = 1\n" if registry is None else _STAGED_LOCK % registry,
                                     encoding="utf-8")
    _RecordingEnvironment.instances.clear()
    monkeypatch.setattr("pm.environment.PythonEnvironment", _RecordingEnvironment)
    monkeypatch.setattr("pm.runtime.runtime_environment", lambda: dict(env))
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "", ""))
    stage_runtime(tmp_path / "uv", tmp_path / "python", tmp_path / "runtime",
                  project=project, cache=tmp_path / "cache")
    (environment,) = _RecordingEnvironment.instances
    return environment, project


def test_a_mirrored_index_relocks_the_snapshot_before_the_locked_sync(monkeypatch, tmp_path):
    """uv reads a lock's registry as part of its identity: the upstream lock records
    pypi.org, so against a bridged mirror `--locked` rejects it before installing
    anything, and `--frozen` fetches the recorded files.pythonhosted.org URLs."""
    environment, project = _stage_runtime(monkeypatch, tmp_path,
                                          {"UV_INDEX_URL": "https://mirror.example/simple"})

    assert environment.steps == ["create", "lock", "sync"]
    # Re-recorded in the scratch snapshot (never the repository project), then still
    # verified, and the mirror stays in force: only where the files come from changes.
    assert environment.sources["lock"] == environment.sources["sync"] != project
    assert environment.env["UV_INDEX_URL"] == "https://mirror.example/simple"


def test_an_unmirrored_build_keeps_the_repository_lock_authoritative(monkeypatch, tmp_path):
    environment, _ = _stage_runtime(monkeypatch, tmp_path, {})

    assert environment.steps == ["create", "sync"]


@pytest.mark.parametrize("env", [
    {"UV_INDEX_URL": "https://pypi.org/simple"},
    {"UV_DEFAULT_INDEX": "https://pypi.org/simple/"},
    {"UV_DEFAULT_INDEX": "https://pypi.org:443/simple"},
    {"UV_INDEX_URL": "https://__token__:@pypi.org/simple"},
    {"UV_INDEX_URL": "  "},
    # These add an index next to the default; neither can desync the lock.
    {"UV_INDEX": "https://mirror.example/simple"},
    {"UV_EXTRA_INDEX_URL": "https://mirror.example/simple"},
])
def test_indexes_that_are_not_the_locked_registry_are_left_alone(monkeypatch, tmp_path, env):
    environment, _ = _stage_runtime(monkeypatch, tmp_path, env)

    assert environment.steps == ["create", "sync"]


def test_a_mirror_on_a_non_default_port_is_a_different_registry(monkeypatch, tmp_path):
    environment, _ = _stage_runtime(monkeypatch, tmp_path,
                                    {"UV_DEFAULT_INDEX": "https://pypi.org:8443/simple"})

    assert environment.steps == ["create", "lock", "sync"]


def test_a_lock_that_records_no_single_registry_is_left_alone(monkeypatch, tmp_path):
    """Nothing to compare against: keep today's behaviour instead of re-resolving."""
    environment, _ = _stage_runtime(monkeypatch, tmp_path,
                                    {"UV_INDEX_URL": "https://mirror.example/simple"},
                                    registry=None)

    assert environment.steps == ["create", "sync"]


def test_index_identity_ignores_credentials_and_trailing_slashes():
    from pm.runtime_stage import _index_identity

    pypi = _index_identity("https://pypi.org/simple")
    assert _index_identity("https://pypi.org/simple/") == pypi
    assert _index_identity("https://pypi.org:443/simple") == pypi
    assert _index_identity("HTTPS://PyPI.ORG/simple") == pypi
    assert _index_identity("https://user:p%40ss@pypi.org/simple") == pypi
    assert _index_identity("https://pypi.org:8443/simple") != pypi
    assert _index_identity("https://mirror.example/simple") != pypi