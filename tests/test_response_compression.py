"""Behavior contracts for selective dashboard response compression."""

import gzip
import json

from hermes_cli.response_compression import (
    _accepts_gzip,
    _is_compressible_content_type,
    _is_excluded_path,
)
from agent.trajectory import save_trajectory


def test_gzip_acceptance_honors_explicit_quality_values():
    assert _accepts_gzip(["br, gzip;q=0.8"]) is True
    assert _accepts_gzip(["gzip;q=0"]) is False
    assert _accepts_gzip(["*;q=0.5"]) is True


def test_compression_allowlist_accepts_json_variants_only():
    assert _is_compressible_content_type(["application/json; charset=utf-8"]) is True
    assert _is_compressible_content_type(["application/problem+json"]) is True
    assert _is_compressible_content_type(["text/html"]) is False
    assert _is_compressible_content_type(["application/json", "text/plain"]) is False


def test_sensitive_dashboard_routes_are_never_compressed():
    assert _is_excluded_path("/api/config/raw") is True
    assert _is_excluded_path("/api/auth/session") is True
    assert _is_excluded_path("/api/configuration") is False
    assert _is_excluded_path("/api/status") is False


def test_trajectory_defaults_to_readable_gzip_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_trajectory([{"from": "human", "value": "hello"}], "test-model", True)
    output = tmp_path / "trajectory_samples.jsonl.gz"
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["model"] == "test-model"
