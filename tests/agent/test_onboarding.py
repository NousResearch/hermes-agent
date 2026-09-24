"""Tests for agent/onboarding.py — contextual first-touch hint helpers."""

from __future__ import annotations

import yaml

from agent.onboarding import (
    BUSY_INPUT_FLAG,
    TOOL_PROGRESS_FLAG,
    detect_openclaw_residue,
    is_seen,
    mark_seen,
)


class TestIsSeen:
    def test_empty_config_unseen(self):
        assert is_seen({}, BUSY_INPUT_FLAG) is False




    def test_seen_flag_true(self):
        cfg = {"onboarding": {"seen": {BUSY_INPUT_FLAG: True}}}
        assert is_seen(cfg, BUSY_INPUT_FLAG) is True

    def test_seen_flag_falsy(self):
        cfg = {"onboarding": {"seen": {BUSY_INPUT_FLAG: False}}}
        assert is_seen(cfg, BUSY_INPUT_FLAG) is False



class TestMarkSeen:

    def test_preserves_other_config(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "model": {"default": "claude-sonnet-4.6"},
            "display": {"skin": "default"},
        }))

        assert mark_seen(cfg_path, BUSY_INPUT_FLAG) is True
        loaded = yaml.safe_load(cfg_path.read_text())

        assert loaded["model"]["default"] == "claude-sonnet-4.6"
        assert loaded["display"]["skin"] == "default"
        assert loaded["onboarding"]["seen"][BUSY_INPUT_FLAG] is True


    def test_idempotent(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        mark_seen(cfg_path, BUSY_INPUT_FLAG)
        first = cfg_path.read_text()

        # Second call must be a no-op on-disk content (file may be touched,
        # but the YAML contents should be identical).
        mark_seen(cfg_path, BUSY_INPUT_FLAG)
        second = cfg_path.read_text()

        assert yaml.safe_load(first) == yaml.safe_load(second)






class TestRoundTrip:
    """After mark_seen, is_seen on the re-loaded config must return True."""

    def test_mark_then_is_seen(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"

        assert mark_seen(cfg_path, BUSY_INPUT_FLAG) is True
        loaded = yaml.safe_load(cfg_path.read_text())

        assert is_seen(loaded, BUSY_INPUT_FLAG) is True
        assert is_seen(loaded, TOOL_PROGRESS_FLAG) is False

    def test_mark_both_flags_independently(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"

        mark_seen(cfg_path, BUSY_INPUT_FLAG)
        mark_seen(cfg_path, TOOL_PROGRESS_FLAG)
        loaded = yaml.safe_load(cfg_path.read_text())

        assert is_seen(loaded, BUSY_INPUT_FLAG) is True
        assert is_seen(loaded, TOOL_PROGRESS_FLAG) is True


# ---------------------------------------------------------------------------
# OpenClaw residue banner
# ---------------------------------------------------------------------------


class TestDetectOpenclawResidue:
    def test_returns_true_when_openclaw_dir_present(self, tmp_path):
        (tmp_path / ".openclaw").mkdir()
        assert detect_openclaw_residue(home=tmp_path) is True


    def test_returns_false_when_path_is_a_file(self, tmp_path):
        # A stray file named ``.openclaw`` is NOT a workspace — skip the banner.
        (tmp_path / ".openclaw").write_text("oops")
        assert detect_openclaw_residue(home=tmp_path) is False







class TestProfileBuildMode:



    def test_non_mapping_config_safe(self):
        from agent.onboarding import profile_build_mode

        assert profile_build_mode("not a dict") == "ask"  # type: ignore[arg-type]
        assert profile_build_mode({"onboarding": "nope"}) == "ask"






