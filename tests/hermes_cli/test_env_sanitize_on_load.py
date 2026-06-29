"""Tests for .env sanitization during load to prevent token duplication (#8908)."""

import tempfile
from pathlib import Path
from unittest.mock import patch


def test_load_env_sanitizes_concatenated_lines():
    """Verify load_env() splits concatenated KEY=VALUE pairs.

    Reproduces the scenario from #8908 where a corrupted .env file
    contained multiple tokens on a single line, causing the bot token
    to be duplicated 8 times.
    """
    from hermes_cli.config import load_env

    token = "0123456789:test"
    # Simulate concatenated line: TOKEN=xxx followed immediately by another key
    corrupted = f"TELEGRAM_BOT_TOKEN={token}ANTHROPIC_API_KEY=sk-ant-test123\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, encoding="utf-8"
    ) as f:
        f.write(corrupted)
        env_path = Path(f.name)

    try:
        with patch("hermes_cli.config.get_env_path", return_value=env_path):
            result = load_env()
        assert result.get("TELEGRAM_BOT_TOKEN") == token, (
            f"Token should be exactly '{token}', got '{result.get('TELEGRAM_BOT_TOKEN')}'"
        )
        assert result.get("ANTHROPIC_API_KEY") == "sk-ant-test123"
    finally:
        env_path.unlink(missing_ok=True)


def test_load_env_normal_file_unchanged():
    """A well-formed .env file should be parsed identically."""
    from hermes_cli.config import load_env

    content = (
        "TELEGRAM_BOT_TOKEN=mytoken123\n"
        "ANTHROPIC_API_KEY=sk-ant-key\n"
        "# comment\n"
        "\n"
        "OPENAI_API_KEY=sk-openai\n"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        env_path = Path(f.name)

    try:
        with patch("hermes_cli.config.get_env_path", return_value=env_path):
            result = load_env()
        assert result["TELEGRAM_BOT_TOKEN"] == "mytoken123"
        assert result["ANTHROPIC_API_KEY"] == "sk-ant-key"
        assert result["OPENAI_API_KEY"] == "sk-openai"
    finally:
        env_path.unlink(missing_ok=True)


def test_env_loader_sanitizes_before_dotenv():
    """Verify env_loader._sanitize_env_file_if_needed fixes corrupted files."""
    from hermes_cli.env_loader import _sanitize_env_file_if_needed

    token = "0123456789:test"
    corrupted = f"TELEGRAM_BOT_TOKEN={token}ANTHROPIC_API_KEY=sk-ant-test\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, encoding="utf-8"
    ) as f:
        f.write(corrupted)
        env_path = Path(f.name)

    try:
        _sanitize_env_file_if_needed(env_path)
        with open(env_path, encoding="utf-8") as f:
            lines = f.readlines()
        # Should be split into two separate lines
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}: {lines}"
        assert lines[0].startswith("TELEGRAM_BOT_TOKEN=")
        assert lines[1].startswith("ANTHROPIC_API_KEY=")
        # Token should not contain the second key
        parsed_token = lines[0].strip().split("=", 1)[1]
        assert parsed_token == token
    finally:
        env_path.unlink(missing_ok=True)


def test_sanitize_env_file_if_needed_preserves_permissions():
    """_sanitize_env_file_if_needed preserves the original file mode.

    Mirrors the sanitize_env_file permission-preservation tests in
    test_config.py.  When a .env file has an operator-set mode (e.g.
    0o640 for Docker volume mounts), the sanitize rewrite must not
    clobber it to 0o600 (mkstemp default).
    """
    import os
    import stat

    from hermes_cli.env_loader import _sanitize_env_file_if_needed

    token = "0123456789:test"
    # Concatenated line that triggers sanitization
    corrupted = f"TELEGRAM_BOT_TOKEN={token}ANTHROPIC_API_KEY=sk-ant-test\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, encoding="utf-8"
    ) as f:
        f.write(corrupted)
        env_path = Path(f.name)

    try:
        # Set a non-default mode (simulates Docker volume mount at 0640)
        target_mode = 0o640
        os.chmod(env_path, target_mode)

        _sanitize_env_file_if_needed(env_path)

        actual_mode = stat.S_IMODE(env_path.stat().st_mode)
        assert actual_mode == target_mode, (
            f"Expected mode {oct(target_mode)}, got {oct(actual_mode)} — "
            "permissions were clobbered by sanitize"
        )
    finally:
        env_path.unlink(missing_ok=True)
