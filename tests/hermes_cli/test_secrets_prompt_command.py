from argparse import Namespace
from unittest.mock import patch

from hermes_cli import secrets_cli
from tools import skills_tool


def test_prompt_command_stores_value_without_printing_it(capsys):
    args = Namespace(
        env_var="PENPOT_MCP_TOKEN",
        prompt="Penpot token",
    )

    with patch.object(secrets_cli, "masked_secret_prompt", return_value="super-secret"), \
         patch.object(
             secrets_cli,
             "save_env_value_secure",
             return_value={"success": True, "stored_as": "PENPOT_MCP_TOKEN"},
         ), \
         patch.object(secrets_cli, "get_env_path", return_value="/tmp/hermes/.env"):
        assert secrets_cli.cmd_prompt(args) == 0

    output = capsys.readouterr()
    assert "PENPOT_MCP_TOKEN" in output.out
    assert "super-secret" not in output.out
    assert "super-secret" not in output.err


def test_prompt_command_rejects_invalid_secret_name():
    args = Namespace(env_var="PENPOT-TOKEN", prompt="Token")
    assert secrets_cli.cmd_prompt(args) == 2


def test_prompt_command_does_not_store_empty_value():
    args = Namespace(env_var="PENPOT_MCP_TOKEN", prompt="Token")
    with patch.object(secrets_cli, "masked_secret_prompt", return_value=""), \
         patch.object(secrets_cli, "save_env_value_secure") as save_secret:
        assert secrets_cli.cmd_prompt(args) == 1
    save_secret.assert_not_called()


def test_capture_secret_uses_interactive_callback_without_exposing_value():
    callback = lambda env_var, prompt, metadata: {
        "success": True,
        "stored_as": env_var,
        "validated": True,
        "skipped": False,
        "metadata": metadata,
    }
    original = skills_tool._secret_capture_callback
    try:
        skills_tool.set_secret_capture_callback(callback)
        result = skills_tool.capture_secret(
            "PENPOT_MCP_TOKEN",
            "Enter the Penpot token",
            {"source": "test"},
        )
    finally:
        skills_tool.set_secret_capture_callback(original)

    assert result["success"] is True
    assert result["stored_as"] == "PENPOT_MCP_TOKEN"
    assert result["metadata"] == {"source": "test"}


def test_capture_secret_skips_without_interactive_callback():
    original = skills_tool._secret_capture_callback
    try:
        skills_tool.set_secret_capture_callback(None)
        result = skills_tool.capture_secret("PENPOT_MCP_TOKEN", "Token")
    finally:
        skills_tool.set_secret_capture_callback(original)

    assert result["success"] is False
    assert result["skipped"] is True
    assert "secret" not in str(result.get("value", "")).lower()
