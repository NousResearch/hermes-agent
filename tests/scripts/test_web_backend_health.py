"""Focused contract tests for the DDGS health probe."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "web-backend-health.py"


@pytest.fixture
def health_module(monkeypatch):
    monkeypatch.setattr("sys.argv", [str(SCRIPT)])
    spec = importlib.util.spec_from_file_location("web_backend_health", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completed(payload: dict, *, stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["python", "-c", "probe"],
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr=stderr,
    )


def test_ddgs_success_uses_current_interpreter_and_reports_result(health_module, monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return _completed({"ok": True, "count": 1})

    monkeypatch.setattr(health_module.subprocess, "run", fake_run)

    ok, detail = health_module.check_ddgs()

    assert ok is True
    assert detail == "working (1 result)"
    assert observed["command"][0] == health_module.sys.executable
    assert observed["kwargs"]["timeout"] == 15
    assert observed["kwargs"]["capture_output"] is True


def test_missing_ddgs_returns_actionable_untruncated_error(health_module, monkeypatch):
    monkeypatch.setattr(
        health_module.subprocess,
        "run",
        lambda *_a, **_kw: _completed(
            {
                "ok": False,
                "kind": "missing_dependency",
                "message": "No module named 'ddgs'",
            },
            returncode=0,
        ),
    )

    ok, detail = health_module.check_ddgs()

    assert ok is False
    assert "dependency unavailable" in detail
    assert health_module.sys.executable in detail
    assert "-m pip install ddgs" in detail
    assert "Traceback" not in detail


def test_search_failure_preserves_actionable_message_without_traceback(health_module, monkeypatch):
    message = "DuckDuckGo rate limit exceeded; retry later"
    monkeypatch.setattr(
        health_module.subprocess,
        "run",
        lambda *_a, **_kw: _completed(
            {"ok": False, "kind": "search_error", "message": message}
        ),
    )

    ok, detail = health_module.check_ddgs()

    assert ok is False
    assert detail == f"search error: {message}"
    assert "Traceback" not in detail


def test_malformed_worker_output_reports_protocol_error(health_module, monkeypatch):
    monkeypatch.setattr(
        health_module.subprocess,
        "run",
        lambda *_a, **_kw: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="worker failed\nfull reason"
        ),
    )

    ok, detail = health_module.check_ddgs()

    assert ok is False
    assert detail == "probe protocol error: worker failed full reason"


def test_ddgs_timeout_is_bounded(health_module, monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="ddgs", timeout=15)

    monkeypatch.setattr(health_module.subprocess, "run", timeout)

    assert health_module.check_ddgs() == (False, "search timed out after 15s")
