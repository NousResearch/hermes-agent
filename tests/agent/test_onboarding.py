"""Tests for agent/onboarding.py — contextual first-touch hint helpers."""

from __future__ import annotations

import yaml
import pytest

from agent.onboarding import (
    BUSY_INPUT_FLAG,
    TOOL_PROGRESS_FLAG,
    busy_input_hint_cli,
    busy_input_hint_gateway,
    is_seen,
    mark_seen,
    tool_progress_hint_cli,
    tool_progress_hint_gateway,
)


class TestIsSeen:
    def test_empty_config_unseen(self):
        assert is_seen({}, BUSY_INPUT_FLAG) is False

    def test_missing_onboarding_unseen(self):
        assert is_seen({"display": {}}, BUSY_INPUT_FLAG) is False

    def test_onboarding_not_dict_unseen(self):
        assert is_seen({"onboarding": "nope"}, BUSY_INPUT_FLAG) is False

    def test_seen_dict_missing_flag(self):
        assert is_seen({"onboarding": {"seen": {}}}, BUSY_INPUT_FLAG) is False

    def test_seen_flag_true(self):
        cfg = {"onboarding": {"seen": {BUSY_INPUT_FLAG: True}}}
        assert is_seen(cfg, BUSY_INPUT_FLAG) is True

    def test_seen_flag_falsy(self):
        cfg = {"onboarding": {"seen": {BUSY_INPUT_FLAG: False}}}
        assert is_seen(cfg, BUSY_INPUT_FLAG) is False

    def test_other_flags_isolated(self):
        cfg = {"onboarding": {"seen": {BUSY_INPUT_FLAG: True}}}
        assert is_seen(cfg, TOOL_PROGRESS_FLAG) is False


class TestMarkSeen:
    def test_creates_missing_file_and_sets_flag(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        assert mark_seen(cfg_path, BUSY_INPUT_FLAG) is True

        loaded = yaml.safe_load(cfg_path.read_text())
        assert loaded["onboarding"]["seen"][BUSY_INPUT_FLAG] is True

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

    def test_preserves_other_seen_flags(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "onboarding": {"seen": {TOOL_PROGRESS_FLAG: True}},
        }))

        assert mark_seen(cfg_path, BUSY_INPUT_FLAG) is True
        loaded = yaml.safe_load(cfg_path.read_text())

        assert loaded["onboarding"]["seen"][TOOL_PROGRESS_FLAG] is True
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

    def test_handles_non_dict_onboarding(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"onboarding": "corrupted"}))

        assert mark_seen(cfg_path, BUSY_INPUT_FLAG) is True
        loaded = yaml.safe_load(cfg_path.read_text())
        assert loaded["onboarding"]["seen"][BUSY_INPUT_FLAG] is True

    def test_handles_non_dict_seen(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"onboarding": {"seen": "corrupted"}}))

        assert mark_seen(cfg_path, BUSY_INPUT_FLAG) is True
        loaded = yaml.safe_load(cfg_path.read_text())
        assert loaded["onboarding"]["seen"][BUSY_INPUT_FLAG] is True


class TestHintMessages:
    def test_busy_input_hint_gateway_interrupt(self):
        msg = busy_input_hint_gateway("interrupt")
        assert "/busy queue" in msg
        assert "interrupted" in msg.lower()

    def test_busy_input_hint_gateway_queue(self):
        msg = busy_input_hint_gateway("queue")
        assert "/busy interrupt" in msg
        assert "queued" in msg.lower()

    def test_busy_input_hint_gateway_steer(self):
        msg = busy_input_hint_gateway("steer")
        assert "/busy interrupt" in msg
        assert "/busy queue" in msg
        assert "steer" in msg.lower()

    def test_busy_input_hint_cli_interrupt(self):
        msg = busy_input_hint_cli("interrupt")
        assert "/busy queue" in msg

    def test_busy_input_hint_cli_queue(self):
        msg = busy_input_hint_cli("queue")
        assert "/busy interrupt" in msg

    def test_busy_input_hint_cli_steer(self):
        msg = busy_input_hint_cli("steer")
        assert "/busy interrupt" in msg
        assert "/busy queue" in msg
        assert "steer" in msg.lower()

    def test_tool_progress_hints_mention_verbose(self):
        assert "/verbose" in tool_progress_hint_gateway()
        assert "/verbose" in tool_progress_hint_cli()

    def test_hints_are_not_empty(self):
        for hint in (
            busy_input_hint_gateway("queue"),
            busy_input_hint_gateway("interrupt"),
            busy_input_hint_gateway("steer"),
            busy_input_hint_cli("queue"),
            busy_input_hint_cli("interrupt"),
            busy_input_hint_cli("steer"),
            tool_progress_hint_gateway(),
            tool_progress_hint_cli(),
        ):
            assert hint.strip()


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
