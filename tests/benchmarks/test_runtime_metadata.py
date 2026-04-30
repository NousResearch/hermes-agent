import os

from benchmarks.interface import BenchmarkConfig
from benchmarks.runner import capture_runtime_metadata, credential_presence


def test_credential_presence_reports_only_set_or_missing(monkeypatch):
    monkeypatch.setenv("MEM0_API_KEY", "super-secret-value")

    presence = credential_presence("mem0")

    assert presence == {"MEM0_API_KEY": "set"}
    assert "super-secret-value" not in repr(presence)


def test_runtime_metadata_contains_versions_and_no_secret_values(monkeypatch):
    monkeypatch.setenv("MEM0_API_KEY", "super-secret-value")

    metadata = capture_runtime_metadata("mem0")

    assert "python" in metadata
    assert "platform" in metadata
    assert "packages" in metadata
    assert "git" in metadata
    assert metadata["credentials"] == {"MEM0_API_KEY": "set"}
    assert "super-secret-value" not in repr(metadata)
