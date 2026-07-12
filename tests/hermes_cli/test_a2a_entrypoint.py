"""Clean-process regressions for the deferred A2A CLI and plugin skill."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(home: Path, *, poison_sdk: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    if poison_sdk is not None:
        poison_sdk.mkdir()
        (poison_sdk / "a2a.py").write_text(
            "raise RuntimeError('optional A2A SDK was imported')\n", encoding="utf-8"
        )
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(poison_sdk) + (os.pathsep + existing if existing else "")
    return env


def _run(home: Path, *args: str, poison_sdk: Path | None = None):
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "a2a", *args],
        cwd=REPO_ROOT,
        env=_env(home, poison_sdk=poison_sdk),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_clean_entrypoint_discovers_deferred_cli_without_optional_sdk(tmp_path):
    result = _run(tmp_path / "home", "status", poison_sdk=tmp_path / "poison")

    assert result.returncode == 0, result.stderr
    assert "enabled: no" in result.stdout
    assert "principals: 0" in result.stdout
    assert "optional A2A SDK was imported" not in result.stderr


def test_clean_entrypoint_propagates_success_usage_and_failure_codes(tmp_path):
    success = _run(tmp_path / "success", "status")
    usage = _run(tmp_path / "usage", "ask", "peer")
    failure = _run(
        tmp_path / "failure",
        "setup",
        "--public-url",
        "http://public.example/a2a",
    )

    assert success.returncode == 0
    assert usage.returncode == 2
    assert usage.stdout == ""
    assert usage.stderr == "hermes a2a: MESSAGE is required (or pass --stdin)\n"
    assert failure.returncode == 1
    assert failure.stdout == ""
    assert "require HTTPS" in failure.stderr
    assert "Traceback" not in failure.stderr


def test_clean_process_resolves_qualified_deferred_skill_only(tmp_path):
    probe = """
import json
from hermes_cli.plugins import discover_plugins, get_plugin_manager
from tools.skills_tool import skill_view

discover_plugins()
manager = get_plugin_manager()
before = manager.find_plugin_skill('a2a-platform:a2a-peer')
qualified = json.loads(skill_view('a2a-platform:a2a-peer', preprocess=False))
bare = json.loads(skill_view('a2a-peer', preprocess=False))
print(json.dumps({
    'path': str(before) if before else None,
    'qualified': qualified,
    'bare_success': bare.get('success'),
    'registered': manager.list_plugin_skills('a2a-platform'),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=_env(tmp_path / "home", poison_sdk=tmp_path / "poison"),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["path"].endswith("plugins/platforms/a2a/skills/a2a-peer/SKILL.md")
    assert payload["qualified"]["success"] is True
    assert payload["qualified"]["name"] == "a2a-platform:a2a-peer"
    assert payload["bare_success"] is False
    assert payload["registered"] == ["a2a-peer"]
