"""Tests for local Decision Trace publish/deploy scripts."""

import importlib.util
import os
import subprocess
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path.home() / ".hermes" / "scripts"
PUBLISH_SCRIPT = SCRIPT_DIR / "decision_trace_publish.py"
DEPLOY_SCRIPT = SCRIPT_DIR / "decision_trace_deploy_pages.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decision_trace_publish_writes_new_domain_and_public_dir(tmp_path):
    md = tmp_path / "input.md"
    md.write_text("# 测试\n\n## 结论\n独立发布。", encoding="utf-8")

    proc = subprocess.run(
        [
            "python",
            str(PUBLISH_SCRIPT),
            "--title",
            "脚本发布验证",
            "--slug",
            "script-publish-check",
            "--input",
            str(md),
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "https://taoge-decision-traces.pages.dev/" in proc.stdout
    assert "/llm-wikis/decision-traces/public/" in proc.stdout
    assert "/daily-brief-archive/html/decision-traces/" not in proc.stdout


def test_decision_trace_publish_rejects_legacy_morning_brief_base_url(tmp_path):
    md = tmp_path / "input.md"
    md.write_text("# 测试\n\n## 结论\n禁止旧域名。", encoding="utf-8")

    proc = subprocess.run(
        [
            "python",
            str(PUBLISH_SCRIPT),
            "--title",
            "旧域名拒绝验证",
            "--slug",
            "legacy-base-url-check",
            "--input",
            str(md),
            "--base-url",
            "https://taoge-morning-brief.pages.dev/decision-traces",
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert proc.returncode == 2
    assert "invalid_decision_trace_base_url" in proc.stdout


def test_decision_trace_deploy_rejects_morning_brief_target():
    module = _load_module(DEPLOY_SCRIPT, "decision_trace_deploy_pages_test")

    ok, error = module.validate_pages_target(
        "taoge-morning-brief",
        "https://taoge-morning-brief.pages.dev",
    )

    assert not ok
    assert "invalid_decision_trace_project" in error


def test_decision_trace_deploy_accepts_decision_trace_target():
    module = _load_module(DEPLOY_SCRIPT, "decision_trace_deploy_pages_test_ok")

    ok, error = module.validate_pages_target(
        "taoge-decision-traces",
        "https://taoge-decision-traces.pages.dev",
    )

    assert ok
    assert error == ""
