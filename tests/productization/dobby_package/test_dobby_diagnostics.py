"""Golden safety/status diagnostics for the local Dobby package renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dobby_package import load_env_file, render_diagnostics


GOLDEN_DIR = Path(__file__).with_name("golden") / "safety_status"


def _fixtures():
    return sorted(GOLDEN_DIR.glob("*.json"))


@pytest.mark.parametrize("fixture_path", _fixtures(), ids=lambda path: path.stem)
def test_golden_safety_status_diagnostics(fixture_path):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    result = render_diagnostics(
        fixture.get("package_root", "/tmp/dobby-package"),
        env=fixture.get("env"),
        config=fixture.get("config"),
        tool_policy=fixture.get("tool_policy"),
        tool_output=fixture.get("tool_output"),
    )

    assert result["status"] == fixture["expected_status"]
    for check_id, expected_status in fixture.get("expected_checks", {}).items():
        assert result["checks"][check_id]["status"] == expected_status

    rendered = json.dumps(result, sort_keys=True)
    for raw_value in fixture.get("absent_raw_values", []):
        assert raw_value not in rendered
    for expected_text in fixture.get("report_contains", []):
        assert expected_text in result["report"]
    for expected_text in fixture.get("redacted_tool_output_contains", []):
        assert expected_text in result["redacted_tool_output"]


def test_diagnostics_ignores_process_env_without_explicit_input(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + ("P" * 32))

    result = render_diagnostics(
        "/tmp/dobby-package",
        config={},
        tool_policy={},
    )

    assert result["env_source"] == "empty"
    assert result["checks"]["model"]["status"] == "fail"
    assert "sk-" + ("P" * 32) not in json.dumps(result)


def test_diagnostics_redacts_runtime_secret_shaped_values():
    openai_key = "sk-" + ("R" * 32)
    discord_token = "M" + ("D" * 23) + "." + ("E" * 6) + "." + ("F" * 32)
    webhook_secret = "whsec_" + ("W" * 32)
    tool_output = (
        f"OPENAI_API_KEY={openai_key} "
        f"DISCORD_BOT_TOKEN={discord_token} "
        f"WEBHOOK_SECRET={webhook_secret}"
    )

    result = render_diagnostics(
        "/tmp/dobby-package",
        env={
            "OPENAI_API_KEY": openai_key,
            "DISCORD_BOT_TOKEN": discord_token,
            "WEBHOOK_SECRET": webhook_secret,
            "HERMES_REDACT_SECRETS": "true",
        },
        tool_output=tool_output,
    )

    rendered = json.dumps(result, sort_keys=True)
    for raw_value in (openai_key, discord_token, webhook_secret):
        assert raw_value not in rendered
    assert "OPENAI_API_KEY=[REDACTED]" in result["redacted_tool_output"]
    assert "DISCORD_BOT_TOKEN=[REDACTED]" in result["redacted_tool_output"]
    assert "WEBHOOK_SECRET=[REDACTED]" in result["redacted_tool_output"]


def test_env_file_is_read_only_when_caller_supplies_path(tmp_path):
    env_file = tmp_path / "dobby.env"
    env_file.write_text("OPENAI_API_KEY='sk-" + ("Q" * 32) + "'\n", encoding="utf-8")

    assert load_env_file(env_file) == {"OPENAI_API_KEY": "sk-" + ("Q" * 32)}
