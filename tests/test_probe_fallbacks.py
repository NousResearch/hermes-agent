from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_fallbacks.py"
SPEC = importlib.util.spec_from_file_location("probe_fallbacks", MODULE_PATH)
probe_fallbacks = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = probe_fallbacks
SPEC.loader.exec_module(probe_fallbacks)


def test_load_routes_from_config_normalizes_google_provider(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  default: gpt-5.4
  provider: openai-codex
fallback_providers:
  - provider: openrouter
    model: openai/gpt-oss-20b:free
  - provider: google
    model: gemini-2.0-flash
""".strip()
    )

    routes = probe_fallbacks.load_routes_from_config(config_path)

    assert [(route.provider, route.cli_provider, route.model) for route in routes] == [
        ("openrouter", "openrouter", "openai/gpt-oss-20b:free"),
        ("google", "gemini", "gemini-2.0-flash"),
    ]


def test_build_probe_command_uses_low_lift_pong_probe():
    route = probe_fallbacks.ProbeRoute(provider="google", cli_provider="gemini", model="gemini-2.0-flash")

    command = probe_fallbacks.build_probe_command(route)

    assert command.startswith("bash -lc ")
    assert "Reply with exactly: PONG" in command
    assert "--provider gemini" in command
    assert "-m gemini-2.0-flash" in command
    assert "--max-turns 2" in command


def test_run_probe_reports_success(monkeypatch, tmp_path):
    route = probe_fallbacks.ProbeRoute(provider="openrouter", cli_provider="openrouter", model="openai/gpt-oss-20b:free")

    def fake_run(command, *, shell, cwd, capture_output, text, timeout):
        assert shell is True
        assert cwd == str(tmp_path)
        assert timeout == 42
        return subprocess.CompletedProcess(command, 0, stdout="PONG\n\nsession_id: abc", stderr="")

    monkeypatch.setattr(probe_fallbacks.subprocess, "run", fake_run)

    result = probe_fallbacks.run_probe(route, repo_root=tmp_path, timeout=42)

    assert result["ok"] is True
    assert result["provider"] == "openrouter"
    assert result["model"] == "openai/gpt-oss-20b:free"
    assert result["returncode"] == 0
    assert result["classification"] == "ok"


def test_run_probe_classifies_rate_limit(monkeypatch, tmp_path):
    route = probe_fallbacks.ProbeRoute(provider="google", cli_provider="gemini", model="gemini-2.0-flash")

    def fake_run(command, *, shell, cwd, capture_output, text, timeout):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="Error code: 429 quota exceeded")

    monkeypatch.setattr(probe_fallbacks.subprocess, "run", fake_run)

    result = probe_fallbacks.run_probe(route, repo_root=tmp_path, timeout=42)

    assert result["ok"] is False
    assert result["classification"] == "rate_limit"


def test_probe_routes_runs_all_routes_and_sets_failure_exit_code(monkeypatch, tmp_path):
    routes = [
        probe_fallbacks.ProbeRoute(provider="openrouter", cli_provider="openrouter", model="one"),
        probe_fallbacks.ProbeRoute(provider="google", cli_provider="gemini", model="two"),
    ]

    def fake_run_probe(route, *, repo_root, timeout):
        return {
            "provider": route.provider,
            "cli_provider": route.cli_provider,
            "model": route.model,
            "ok": route.provider == "openrouter",
            "classification": "ok" if route.provider == "openrouter" else "rate_limit",
            "returncode": 0 if route.provider == "openrouter" else 1,
            "stdout": "PONG" if route.provider == "openrouter" else "",
            "stderr": "" if route.provider == "openrouter" else "429",
            "command": "cmd",
        }

    monkeypatch.setattr(probe_fallbacks, "run_probe", fake_run_probe)

    summary = probe_fallbacks.probe_routes(routes, repo_root=tmp_path, timeout=30, max_workers=2)

    assert summary["ok"] is False
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert [item["provider"] for item in summary["results"]] == ["openrouter", "google"]


def test_main_json_output(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "fallback_providers:\n  - provider: openrouter\n    model: openai/gpt-oss-20b:free\n"
    )

    monkeypatch.setattr(
        probe_fallbacks,
        "probe_routes",
        lambda routes, *, repo_root, timeout, max_workers: {
            "ok": True,
            "passed": 1,
            "failed": 0,
            "results": [
                {
                    "provider": routes[0].provider,
                    "cli_provider": routes[0].cli_provider,
                    "model": routes[0].model,
                    "ok": True,
                    "classification": "ok",
                    "returncode": 0,
                    "stdout": "PONG",
                    "stderr": "",
                    "command": "cmd",
                }
            ],
        },
    )

    exit_code = probe_fallbacks.main(
        [
            "--config",
            str(config_path),
            "--repo-root",
            str(tmp_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["passed"] == 1
