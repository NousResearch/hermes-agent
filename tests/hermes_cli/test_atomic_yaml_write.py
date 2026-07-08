"""Tests for utils.atomic_yaml_write — crash-safe YAML file writes."""

import os
import stat
from unittest.mock import patch

import pytest
import yaml

from utils import atomic_roundtrip_yaml_update, atomic_yaml_write


class TestAtomicYamlWrite:
    def test_writes_valid_yaml(self, tmp_path):
        target = tmp_path / "data.yaml"
        data = {"key": "value", "nested": {"a": 1}}

        atomic_yaml_write(target, data)

        assert yaml.safe_load(target.read_text(encoding="utf-8")) == data

    def test_config_yaml_write_forces_owner_only_permissions(self, tmp_path):
        if os.name != "posix":
            pytest.skip("POSIX-only")

        target = tmp_path / "config.yaml"
        target.write_text("old: true\n", encoding="utf-8")
        target.chmod(0o666)

        atomic_yaml_write(target, {"new": True})

        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_config_yaml_write_secures_symlink_target(self, tmp_path):
        if os.name != "posix":
            pytest.skip("POSIX-only")

        real = tmp_path / "real.yaml"
        link = tmp_path / "config.yaml"
        real.write_text("old: true\n", encoding="utf-8")
        real.chmod(0o666)
        link.symlink_to(real)

        atomic_yaml_write(link, {"new": True})

        assert link.is_symlink()
        assert stat.S_IMODE(real.stat().st_mode) == 0o600

    def test_roundtrip_config_yaml_update_forces_owner_only_permissions(self, tmp_path):
        if os.name != "posix":
            pytest.skip("POSIX-only")

        target = tmp_path / "config.yaml"
        target.write_text("model:\n  provider: openrouter\n", encoding="utf-8")
        target.chmod(0o666)

        atomic_roundtrip_yaml_update(target, "model.provider", "nvidia")

        assert yaml.safe_load(target.read_text(encoding="utf-8")) == {
            "model": {"provider": "nvidia"}
        }
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_non_config_yaml_preserves_existing_permissions(self, tmp_path):
        if os.name != "posix":
            pytest.skip("POSIX-only")

        target = tmp_path / "state.yaml"
        target.write_text("old: true\n", encoding="utf-8")
        target.chmod(0o664)

        atomic_yaml_write(target, {"new": True})

        assert stat.S_IMODE(target.stat().st_mode) == 0o664

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
