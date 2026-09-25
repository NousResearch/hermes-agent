"""Mirrored and air-gapped networks configure indexes through pip or uv; PM forwards
exactly that into uv while still refusing every other ambient uv setting."""
from __future__ import annotations

import json
import os
import subprocess
import sys

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


def test_lock_verification_keeps_the_recorded_registry(clean_index_env, tmp_path, monkeypatch):
    """Regression for #122740: a bridged default index must not make a valid lock look stale.

    The installer hands uv ``_base_environment()``. A pip mirror becomes
    ``UV_INDEX_URL``, and ``uv sync --locked`` / ``uv lock --check`` then
    re-resolve against that registry and reject a lock that still records
    the one it was built from. Verification drops the override and still
    passes ``--locked``; a resolving sync keeps the mirror.
    """
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.example/simple")
    monkeypatch.setenv("UV_HTTP_TIMEOUT", "120")
    record = tmp_path / "calls.jsonl"
    uv = tmp_path / "uv"
    uv.write_text(
        "\n".join([
            f"#!{sys.executable}",
            "import json, os, sys",
            "from pathlib import Path",
            "row = {",
            "    'args': sys.argv[1:],",
            "    'UV_INDEX_URL': os.environ.get('UV_INDEX_URL'),",
            "    'UV_DEFAULT_INDEX': os.environ.get('UV_DEFAULT_INDEX'),",
            "    'UV_HTTP_TIMEOUT': os.environ.get('UV_HTTP_TIMEOUT'),",
            "    'UV_INDEX': os.environ.get('UV_INDEX'),",
            "}",
            f"Path({str(record)!r}).open('a', encoding='utf-8').write(json.dumps(row) + '\\n')",
            "",
        ]),
        encoding="utf-8",
    )
    uv.chmod(0o755)
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "lock-index"\nversion = "0"\n', encoding="utf-8")
    lock = project / "uv.lock"
    lock.write_text(
        'version = 1\n[[package]]\nname = "demo"\n'
        'source = { registry = "https://pypi.org/simple" }\n',
        encoding="utf-8",
    )
    lock_bytes = lock.read_bytes()

    env = _base_environment()
    assert env["UV_INDEX_URL"] == "https://mirror.example/simple"
    # An explicit extra index does not replace the default, so it must survive
    # verification. UV_DEFAULT_INDEX does replace it and must not.
    env["UV_INDEX"] = "https://extra.example/simple"
    env["UV_DEFAULT_INDEX"] = "https://other-mirror.example/simple"
    environment = PythonEnvironment(
        uv=uv, python=tmp_path / "python", destination=tmp_path / "venv",
        cache=tmp_path / "cache", env=env, no_config=True,
    )
    environment.sync(project, locked=True, no_default_groups=True,
                     no_install_project=True, timeout=30)
    environment.check_lock(project)
    environment.sync(project, timeout=30)

    rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
    locked = next(row for row in rows if "--locked" in row["args"])
    checked = next(row for row in rows if row["args"][:1] == ["lock"] and "--check" in row["args"])
    frozen = next(row for row in rows if "--frozen" in row["args"])
    for row in (locked, checked):
        assert row["UV_INDEX_URL"] is None
        assert row["UV_DEFAULT_INDEX"] is None
        assert row["UV_HTTP_TIMEOUT"] == "120"
        assert row["UV_INDEX"] == "https://extra.example/simple"
        assert "--no-config" in row["args"]
    assert "--frozen" not in locked["args"]
    assert frozen["UV_INDEX_URL"] == "https://mirror.example/simple"
    assert frozen["UV_DEFAULT_INDEX"] == "https://other-mirror.example/simple"
    assert frozen["UV_HTTP_TIMEOUT"] == "120"
    assert lock.read_bytes() == lock_bytes
