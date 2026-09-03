"""Tests for the composio optional skill (optional-skills/productivity/composio)."""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "composio"
)
SCRIPT = SKILL_DIR / "scripts" / "composio_cli.py"


@pytest.fixture
def cli(monkeypatch):
    """Import the helper script as a module with a stubbed composio SDK."""
    fake_sdk = types.ModuleType("composio")
    fake_sdk.Composio = MagicMock(name="Composio")
    monkeypatch.setitem(sys.modules, "composio", fake_sdk)
    monkeypatch.setenv("COMPOSIO_API_KEY", "test-key")

    import importlib.util

    spec = importlib.util.spec_from_file_location("composio_cli_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_skill_md_frontmatter():
    text = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^description: \"?(.*?)\"?$", text, re.MULTILINE)
    assert m, "SKILL.md must have a description"
    assert len(m.group(1)) <= 60, f"description too long: {len(m.group(1))}"
    assert m.group(1).endswith(".")
    for section in ("## When to Use", "## Prerequisites", "## How to Run",
                    "## Quick Reference", "## Procedure", "## Pitfalls",
                    "## Verification"):
        assert section in text, f"missing section: {section}"


def test_missing_api_key_fails_closed(cli, monkeypatch, capsys):
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        cli._get_client()
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["successful"] is False
    assert "COMPOSIO_API_KEY" in out["error"]


def test_tool_summaries_shape(cli):
    raw = [
        {
            "function": {
                "name": "GMAIL_SEND_EMAIL",
                "description": "Send an email via Gmail",
                "parameters": {"required": ["recipient_email"]},
            }
        },
        "not-a-dict",
        {"no_function_key": True},
    ]
    tools = cli._tool_summaries(raw)
    assert len(tools) == 2  # junk string skipped, dict-without-function kept as empty
    assert tools[0] == {
        "name": "GMAIL_SEND_EMAIL",
        "description": "Send an email via Gmail",
        "required_params": ["recipient_email"],
    }
    assert cli._tool_summaries("garbage") == []


def test_execute_dict_result(cli):
    client = MagicMock()
    client.tools.execute.return_value = {
        "successful": True,
        "error": None,
        "data": {"id": "msg_1"},
    }
    args = types.SimpleNamespace(slug="GMAIL_SEND_EMAIL", args='{"x": 1}', user="hermes")
    result = cli.cmd_execute(client, args)
    assert result == {"successful": True, "error": None, "data": {"id": "msg_1"}}
    client.tools.execute.assert_called_once()
    kwargs = client.tools.execute.call_args.kwargs
    assert kwargs["user_id"] == "hermes"
    assert kwargs["arguments"] == {"x": 1}


def test_execute_rejects_bad_json(cli):
    client = MagicMock()
    args = types.SimpleNamespace(slug="X", args="{not json", user="hermes")
    result = cli.cmd_execute(client, args)
    assert result["successful"] is False
    assert "JSON" in result["error"]
    client.tools.execute.assert_not_called()

    args2 = types.SimpleNamespace(slug="X", args='["list"]', user="hermes")
    result2 = cli.cmd_execute(client, args2)
    assert result2["successful"] is False
    client.tools.execute.assert_not_called()


def test_schema_finds_exact_slug(cli):
    client = MagicMock()
    client.tools.get.return_value = [
        {"function": {"name": "OTHER_TOOL", "parameters": {}}},
        {
            "function": {
                "name": "NOTION_SEARCH",
                "description": "Search Notion",
                "parameters": {"properties": {"q": {"type": "string"}}},
            }
        },
    ]
    args = types.SimpleNamespace(slug="NOTION_SEARCH", user="hermes")
    result = cli.cmd_schema(client, args)
    assert result["successful"] is True
    assert result["name"] == "NOTION_SEARCH"
    assert "properties" in result["parameters"]

    args_missing = types.SimpleNamespace(slug="NOPE", user="hermes")
    assert cli.cmd_schema(client, args_missing)["successful"] is False


def test_main_wire_error_surfaced(cli, monkeypatch, capsys):
    """Exceptions from the SDK surface as the wire error, never swallowed."""
    client = MagicMock()
    client.tools.get.side_effect = RuntimeError("boom from composio")
    monkeypatch.setattr(cli, "_get_client", lambda: client)
    rc = cli.main(["search", "anything"])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["successful"] is False
    assert "boom from composio" in out["error"]


def test_env_var_registered_in_config_defaults():
    """COMPOSIO_API_KEY must be a documented .env secret."""
    from hermes_cli.config_defaults import OPTIONAL_ENV_VARS

    entry = OPTIONAL_ENV_VARS.get("COMPOSIO_API_KEY")
    assert entry is not None
    assert entry["password"] is True
    assert entry["category"] == "tool"
    assert entry.get("url")
