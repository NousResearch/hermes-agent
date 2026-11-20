"""Tests for .env sanitization during load to prevent token duplication (#8908)."""

import tempfile
from pathlib import Path
from unittest.mock import patch


def test_load_env_preserves_concatenated_text_as_value_data():
    """Verify load_env() does not infer assignments within a physical line.

    A missing newline is ambiguous: text resembling a second assignment may
    instead be part of the first value, so it must remain opaque value data.
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
        assert result.get("TELEGRAM_BOT_TOKEN") == (
            f"{token}ANTHROPIC_API_KEY=sk-ant-test123"
        )
        assert "ANTHROPIC_API_KEY" not in result
    finally:
        env_path.unlink(missing_ok=True)




def test_env_loader_does_not_split_concatenated_text():
    """Verify sanitization preserves one assignment per physical line."""
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
        assert lines == [corrupted]
        parsed_token = lines[0].strip().split("=", 1)[1]
        assert parsed_token == f"{token}ANTHROPIC_API_KEY=sk-ant-test"
    finally:
        env_path.unlink(missing_ok=True)


def test_a_quoted_multiline_value_keeps_its_lines_on_load(tmp_path, monkeypatch):
    """A quoted value that spans lines is value data: its continuation lines keep their whitespace,
    in the process and in the file that the load writes back."""
    from hermes_cli.env_loader import load_hermes_dotenv

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("ZZ_MULTI", "ZZ_PLAIN"):  # set, then delete: the test restores both afterwards
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    env_path = tmp_path / ".env"
    env_path.write_text('  ZZ_PLAIN=1  \nZZ_MULTI="line1  \n    line2"\n', encoding="utf-8")

    load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=False)

    import os
    assert os.environ["ZZ_MULTI"] == "line1  \n    line2"
    assert env_path.read_text(encoding="utf-8") == 'ZZ_PLAIN=1\nZZ_MULTI="line1  \n    line2"\n'
