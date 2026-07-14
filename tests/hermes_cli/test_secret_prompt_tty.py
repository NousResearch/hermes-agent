"""Tests for masked_secret_prompt TTY detection and error handling."""

import pytest
import sys
from unittest.mock import patch, MagicMock

from hermes_cli.secret_prompt import masked_secret_prompt


def test_non_tty_raises_runtime_error():
    """masked_secret_prompt raises RuntimeError on non-TTY stdin/stdout."""
    with patch("hermes_cli.secret_prompt._stream_is_tty", return_value=False):
        with pytest.raises(RuntimeError) as exc_info:
            masked_secret_prompt("Test: ")

        error_msg = str(exc_info.value)
        assert "interactive terminal" in error_msg
        assert "hermes config set" in error_msg or ".env" in error_msg


def test_tty_stream_isatty_exception():
    """_stream_is_tty handles exceptions gracefully."""
    # Simulate isatty() raising an exception
    fake_stream = MagicMock()
    fake_stream.isatty.side_effect = OSError("Mock TTY error")

    from hermes_cli.secret_prompt import _stream_is_tty

    assert _stream_is_tty(fake_stream) is False