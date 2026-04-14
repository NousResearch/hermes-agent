import json
import subprocess
from pathlib import Path

import pytest

from plugins.memory import load_memory_provider
from plugins.memory.qmd_gbrain import QmdGbrainMemoryProvider


class _FakeCompleted:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_provider_loads_via_plugin_registry():
    provider = load_memory_provider("qmd_gbrain")
    assert provider is not None
    assert provider.name == "qmd_gbrain"


def test_initialize_uses_profile_home_subdir_when_present(tmp_path):
    hermes_home = tmp_path / "profiles" / "mutx"
    backend_home = hermes_home / "home"
    backend_home.mkdir(parents=True)

    provider = QmdGbrainMemoryProvider()
    provider.initialize("session-1", hermes_home=str(hermes_home), agent_identity="mutx", platform="cli")

    assert provider._backend_home == backend_home
    assert provider._profile_name == "mutx"


def test_prefetch_merges_qmd_and_gbrain_results(monkeypatch, tmp_path):
    hermes_home = tmp_path / "profiles" / "mutx"
    backend_home = hermes_home / "home"
    backend_home.mkdir(parents=True)

    provider = QmdGbrainMemoryProvider()
    provider.initialize("session-1", hermes_home=str(hermes_home), agent_identity="mutx", platform="cli")

    qmd_rows = [
        {
            "title": "Repo Index",
            "file": "qmd://mutx-lane/repo-index.md",
            "score": 0.91,
            "snippet": "Canonical MUTX repo map.",
        }
    ]
    gbrain_rows = [
        {
            "slug": "projects/mutx/repo-index",
            "score": 0.77,
            "chunk_text": "Repo topology and operator runbooks.",
        }
    ]

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["qmd", "search", "repo map"]:
            return _FakeCompleted(stdout=json.dumps(qmd_rows))
        if cmd[:4] == ["/Users/fortune/.bun/bin/gbrain", "call", "search", '{"query":"repo map","limit":4}']:
            return _FakeCompleted(stdout=json.dumps(gbrain_rows))
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = provider.prefetch("repo map")

    assert "QMD+GBrain Memory" in result
    assert "Repo Index" in result
    assert "Canonical MUTX repo map." in result
    assert "projects/mutx/repo-index" in result
    assert "Repo topology and operator runbooks." in result


def test_prefetch_uses_gbrain_search_when_qmd_is_empty(monkeypatch, tmp_path):
    hermes_home = tmp_path / "profiles" / "mutx"
    backend_home = hermes_home / "home"
    backend_home.mkdir(parents=True)

    provider = QmdGbrainMemoryProvider()
    provider.initialize("session-1", hermes_home=str(hermes_home), platform="cli")

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["qmd", "search", "deploy"]:
            return _FakeCompleted(stdout="[]")
        if cmd[:4] == ["/Users/fortune/.bun/bin/gbrain", "call", "search", '{"query":"deploy","limit":4}']:
            return _FakeCompleted(stdout=json.dumps([
                {"slug": "projects/mutx/deploy", "score": 0.66, "chunk_text": "Deployment quickstart."}
            ]))
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = provider.prefetch("deploy")

    assert "projects/mutx/deploy" in result
    assert "Deployment quickstart." in result
