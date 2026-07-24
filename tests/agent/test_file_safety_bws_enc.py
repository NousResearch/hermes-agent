"""Regression tests verifying bws_cache.enc.json is denied in get_read_block_error."""

import os
from pathlib import Path
from unittest.mock import patch

from agent.file_safety import get_read_block_error, is_write_denied, _hermes_home_path


class TestFileSafetyBwsEnc:
    """Verify bws_cache.enc.json read protection parity with write protection."""

    def test_bws_cache_enc_read_blocked(self, tmp_path):
        home = tmp_path / "hermes_home"
        home.mkdir()
        cache_dir = home / "cache"
        cache_dir.mkdir()
        bws_enc = cache_dir / "bws_cache.enc.json"
        bws_enc.write_text("{}")

        with patch("agent.file_safety._hermes_home_path", return_value=home), \
             patch("agent.file_safety._hermes_root_path", return_value=home):
            err = get_read_block_error(str(bws_enc))
            assert err is not None
            assert "Access denied" in err
            assert "Hermes credential store" in err

    def test_bws_cache_plain_and_enc_parity(self, tmp_path):
        home = tmp_path / "hermes_home"
        home.mkdir()
        cache_dir = home / "cache"
        cache_dir.mkdir()

        plain = cache_dir / "bws_cache.json"
        enc = cache_dir / "bws_cache.enc.json"
        other = cache_dir / "image.png"

        plain.write_text("{}")
        enc.write_text("{}")
        other.write_text("data")

        with patch("agent.file_safety._hermes_home_path", return_value=home), \
             patch("agent.file_safety._hermes_root_path", return_value=home):
            assert get_read_block_error(str(plain)) is not None
            assert get_read_block_error(str(enc)) is not None
            assert get_read_block_error(str(other)) is None
