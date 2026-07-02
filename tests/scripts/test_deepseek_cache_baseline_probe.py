from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "deepseek_cache_baseline_probe.py"


def run_probe(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_dry_run_outputs_cache_schema(tmp_path):
    output = tmp_path / "probe.json"
    result = run_probe("--output", str(output))

    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["probe"] == "deepseek_cache_baseline"
    assert data["mode"] == "dry_run"
    assert data["model"] == "deepseek-v4-flash"
    assert data["summary"]["prompt_cache_hit_tokens"] == 1500
    assert data["summary"]["prompt_cache_miss_tokens"] == 3552
    assert 0 < data["summary"]["cache_hit_rate"] < 1
    assert data["summary"]["all_usage_fields_present"] is True
    assert len(data["prefix_sha256"]) == 64
    for row in data["rows"]:
        assert "prompt_cache_hit_tokens" in row
        assert "prompt_cache_miss_tokens" in row
        assert "cache_hit_rate" in row
        assert "estimated_usd" in row


def test_live_missing_credential_reports_only_env_name(monkeypatch):
    monkeypatch.delenv("TEST_DEEPSEEK_API_KEY", raising=False)
    result = run_probe("--live", "--api-key-env", "TEST_DEEPSEEK_API_KEY")

    assert result.returncode == 2
    data = json.loads(result.stdout)
    assert data["credential_available"] is False
    assert data["credential_env"] == "TEST_DEEPSEEK_API_KEY"
    assert "TEST_DEEPSEEK_API_KEY" in data["error"]
    assert "sk-" not in result.stdout
