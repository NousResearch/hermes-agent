from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HARNESS = Path(
    "/home/curioctylab/.claude/deployment-evidence/"
    "p0f-r5-h-controlled-codex-smoke-20260723/run_r5h_codex_smoke_r3.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("r5h_smoke_harness", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_argv_validation_does_not_match_claude_in_workspace_path():
    harness = _load_harness()
    argv = [
        "codex",
        "exec",
        "--sandbox",
        "workspace-write",
        "--cd",
        "/tmp/.claude/evidence/workspace",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
    ]
    assert harness._validate_codex_argv_summary(json.dumps(argv)) == []


def test_argv_validation_rejects_actual_forbidden_argument():
    harness = _load_harness()
    argv = [
        "codex",
        "exec",
        "--sandbox",
        "workspace-write",
        "--cd",
        "/tmp/workspace",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    failures = harness._validate_codex_argv_summary(json.dumps(argv))
    assert failures == [
        "argv_summary contained forbidden argument "
        "'--dangerously-bypass-approvals-and-sandbox'"
    ]
