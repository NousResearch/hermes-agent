"""A configured package mirror must carry the downloads.

``uv sync --frozen`` installs the absolute artifact URLs recorded in
``uv.lock`` and never consults the configured index, so on a mirrored network
every artifact still comes from ``files.pythonhosted.org`` (#122888). PM now
exports the same lock to a hash-pinned requirements file and installs it
through whichever index is in force.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from pm.environment import (
    PythonEnvironment,
    configured_indexes,
    locked_registries,
    mirror_diverges_from_lock,
)

LOCK = """version = 1
revision = 5
requires-python = ">=3.11"

[[package]]
name = "demo"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
"""


@pytest.fixture
def clean_index_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("UV_", "PIP_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "1.0.0"\n', encoding="utf-8")
    (tmp_path / "uv.lock").write_text(LOCK, encoding="utf-8")
    return tmp_path


def recording_environment(project: Path, calls: list[list[str]],
                          export_body: str | None = "demo==1.0.0 \\\n    --hash=sha256:abc\n") -> PythonEnvironment:
    """A PythonEnvironment whose uv invocations are recorded instead of run.

    ``PythonEnvironment`` is frozen, so the recorder is a subclass rather than a
    patched attribute. ``export_body`` is what a real ``uv export`` would write;
    ``None`` leaves the file absent.
    """
    class Recording(PythonEnvironment):
        def _run(self, args, *, cwd, timeout):
            calls.append(list(args))
            if export_body is not None and args[0] == "export":
                Path(args[args.index("--output-file") + 1]).write_text(export_body, encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, "", "")

    return Recording(uv=Path("uv"), python=Path("python"), destination=project / "venv",
                     cache=project / "cache", env=dict(os.environ))


def test_locked_registries_reads_the_lock_sources(project):
    assert locked_registries(project / "uv.lock") == {"https://pypi.org/simple"}


def test_configured_indexes_include_every_uv_knob(clean_index_env, monkeypatch):
    monkeypatch.setenv("UV_INDEX_URL", "https://mirror.example/simple/")
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://extra.example/simple")

    assert configured_indexes(os.environ) == {"https://mirror.example/simple"}

    monkeypatch.delenv("UV_INDEX_URL")
    monkeypatch.setenv("UV_INDEX", "corp=https://corp.example/simple")
    assert "https://corp.example/simple" in configured_indexes(os.environ)


def test_mirror_diverges_only_when_the_index_is_not_the_locks(clean_index_env, monkeypatch, project):
    lock = project / "uv.lock"

    assert not mirror_diverges_from_lock(lock, os.environ)

    monkeypatch.setenv("UV_INDEX_URL", "https://pypi.org/simple")
    assert not mirror_diverges_from_lock(lock, os.environ)

    monkeypatch.setenv("UV_INDEX_URL", "https://mirror.example/simple")
    assert mirror_diverges_from_lock(lock, os.environ)


def test_sync_installs_the_locked_graph_through_the_mirror(clean_index_env, monkeypatch, project):
    monkeypatch.setenv("UV_INDEX_URL", "https://mirror.example/simple")
    calls: list[list[str]] = []
    recording_environment(project, calls).sync(project)

    assert calls, "sync must run uv"
    assert calls[0][:3] == ["export", "--frozen", "--all-packages"]
    assert "--no-emit-project" in calls[0]
    install = calls[-1]
    assert install[:2] == ["pip", "install"]
    assert "--require-hashes" in install
    assert "--compile-bytecode" in install
    assert "--no-deps" in install, "the export is the whole closure; re-resolving it fights the lock's overrides"
    assert "requirements-flat.txt" in " ".join(install)
    assert not any(call[0] == "sync" for call in calls), "a mirror must not be routed through --frozen"


def test_sync_keeps_frozen_when_no_mirror_is_configured(clean_index_env, monkeypatch, project):
    calls: list[list[str]] = []
    recording_environment(project, calls).sync(project)

    assert calls[0][0] == "sync"
    assert "--frozen" in calls[0]


def test_locked_sync_never_takes_the_mirror_path(clean_index_env, monkeypatch, project):
    """`--locked` already validates the lock's own registry; the mirror path
    would bypass that check."""
    monkeypatch.setenv("UV_INDEX_URL", "https://mirror.example/simple")
    calls: list[list[str]] = []
    recording_environment(project, calls).sync(project, locked=True)

    assert calls[0][0] == "sync"
    assert "--locked" in calls[0]


def test_sync_keeps_frozen_when_an_applicable_requirement_has_no_hash(clean_index_env, monkeypatch, project):
    """A path or git requirement exports without a hash and `--require-hashes`
    would reject the whole file, so the mirror path declines."""
    monkeypatch.setenv("UV_INDEX_URL", "https://mirror.example/simple")
    calls: list[list[str]] = []

    recording_environment(project, calls, export_body="demo @ file:///tmp/demo\n").sync(project)

    assert calls[-1][0] == "sync"
    assert "--frozen" in calls[-1]


def test_requirements_hash_complete_skips_markers_that_do_not_apply(tmp_path):
    from pm.environment import requirements_hash_complete

    applicable = tmp_path / "a.txt"
    applicable.write_text("demo==1.0 \\\n    --hash=sha256:abc\n"
                          "other==2.0 ; sys_platform == 'android'\n", encoding="utf-8")
    assert requirements_hash_complete(applicable)

    missing = tmp_path / "b.txt"
    missing.write_text("demo @ file:///tmp/demo\n", encoding="utf-8")
    assert not requirements_hash_complete(missing)
