"""Tests for utils.atomic_yaml_write — crash-safe YAML file writes."""

import os
import stat
import sys
from unittest.mock import patch

import pytest
import yaml

from utils import atomic_yaml_write


class TestAtomicYamlWrite:
    def test_writes_valid_yaml(self, tmp_path):
        target = tmp_path / "data.yaml"
        data = {"key": "value", "nested": {"a": 1}}

        atomic_yaml_write(target, data)

        assert yaml.safe_load(target.read_text(encoding="utf-8")) == data

    def test_cleans_up_temp_file_on_baseexception(self, tmp_path):
        class SimulatedAbort(BaseException):
            pass

        target = tmp_path / "data.yaml"
        original = {"preserved": True}
        target.write_text(yaml.safe_dump(original), encoding="utf-8")

        with patch("utils.yaml.dump", side_effect=SimulatedAbort):
            with pytest.raises(SimulatedAbort):
                atomic_yaml_write(target, {"new": True})

        tmp_files = [f for f in tmp_path.iterdir() if ".tmp" in f.name]
        assert len(tmp_files) == 0
        assert yaml.safe_load(target.read_text(encoding="utf-8")) == original

    def test_appends_extra_content(self, tmp_path):
        target = tmp_path / "data.yaml"

        atomic_yaml_write(target, {"key": "value"}, extra_content="\n# comment\n")

        text = target.read_text(encoding="utf-8")
        assert "key: value" in text
        assert "# comment" in text

    def test_writes_unicode_unescaped_and_round_trips(self, tmp_path):
        """Emoji/kaomoji are written as real UTF-8, not fragile escape sequences.

        Regression for GitHub #51356: without allow_unicode=True, PyYAML emitted
        astral-plane chars (emoji) as 8-digit `\\UXXXXXXXX` escapes inside
        multi-line double-quoted strings wrapped with `\\` continuations, which
        stricter/non-PyYAML parsers and hand-edits broke into unclosed quotes,
        corrupting the entire config.
        """
        target = tmp_path / "config.yaml"
        # Mirrors the default personalities + skin cursor shipped in cli.py.
        data = {
            "personalities": {
                "kawaii": "kawaii desu~! (◕‿◕) ★ ♪ ヽ(>∀<☆)ノ",
                "catgirl": "nya~! (=^･ω･^=) ฅ^•ﻌ•^ฅ",
                "surfer": "Cowabunga! 🤙 totally rad bro",
                "hype": "LET'S GOOOO!!! 🔥 LEGENDARY!",
            },
            "display": {"cursor": " ▉"},
        }

        atomic_yaml_write(target, data)

        text = target.read_text(encoding="utf-8")
        # No escape artifacts of any kind — real characters on disk.
        assert "\\U" not in text
        assert "\\u" not in text
        # Real glyphs are present verbatim.
        assert "🔥" in text
        assert "(=^･ω･^=)" in text
        # And it reloads to exactly what was written.
        assert yaml.safe_load(text) == data


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX mode bits not enforced on Windows",
)
class TestConfigYamlModeCap:
    """config.yaml must never round-trip wider than 0600 through this
    function, even for call sites that write it directly without following
    up with hermes_cli.config._secure_file() (slash commands, auth, doctor
    migrations, onboarding all do this today).

    Without the cap, a config.yaml that was ever wide (pre-hardening
    install, a HERMES_HOME_MODE window, a manual chmod) stays wide forever
    through those call sites, because _restore_file_mode() otherwise
    round-trips whatever mode the file had before the write.
    """

    def _mode(self, path) -> int:
        return stat.S_IMODE(os.stat(path).st_mode)

    def test_wide_preexisting_config_yaml_is_capped_to_0600(self, tmp_path):
        target = tmp_path / "config.yaml"
        target.write_text("agent:\n  system_prompt: old\n", encoding="utf-8")
        os.chmod(target, 0o666)

        atomic_yaml_write(target, {"agent": {"system_prompt": "new"}})

        assert self._mode(target) == 0o600

    def test_new_config_yaml_is_0600(self, tmp_path):
        target = tmp_path / "config.yaml"

        atomic_yaml_write(target, {"agent": {"system_prompt": "hello"}})

        assert self._mode(target) == 0o600

    def test_non_config_yaml_file_mode_still_preserved(self, tmp_path):
        """Sanity check: the cap is name-scoped to config.yaml -- it must not
        regress the Docker/NAS mode-preservation behavior for other files."""
        target = tmp_path / "other.yaml"
        target.write_text("key: old\n", encoding="utf-8")
        os.chmod(target, 0o666)

        atomic_yaml_write(target, {"key": "new"})

        assert self._mode(target) == 0o666
