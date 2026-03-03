"""Tests for shared .env loader helpers."""

from pathlib import Path

import pytest

from agent.env_loader import read_env_text_with_fallback


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_read_env_text_with_fallback_accepts_utf8(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    _write_bytes(
        env_file,
        b"FIRECRAWL_API_KEY=fc_valid_token_1234567890\n",
    )
    text, encoding = read_env_text_with_fallback(env_file)
    assert "FIRECRAWL_API_KEY=fc_valid_token_1234567890" in text
    assert encoding in ("utf-8", "utf-8-sig")


def test_read_env_text_with_fallback_rejects_corrupted_sensitive_value(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    # cp1252-encoded non-ASCII byte in a sensitive token value.
    _write_bytes(env_file, b"DISCORD_BOT_TOKEN=\xe0bad_token\n")

    with pytest.raises(ValueError) as exc:
        read_env_text_with_fallback(env_file)

    assert "non-ASCII bytes" in str(exc.value)
    assert "DISCORD_BOT_TOKEN" in str(exc.value)


def test_read_env_text_with_fallback_allows_non_ascii_comments(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    # Non-ASCII bytes in comments are tolerated as long as sensitive values are valid.
    _write_bytes(
        env_file,
        b"# comment with em dash \xe2\x80\x94 allowed\nFIRECRAWL_API_KEY=fc_valid_token_1234567890\n",
    )

    text, _ = read_env_text_with_fallback(env_file)
    assert "FIRECRAWL_API_KEY=fc_valid_token_1234567890" in text
