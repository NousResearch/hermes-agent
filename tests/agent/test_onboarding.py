"""Tests for agent/onboarding.py — contextual first-touch hint helpers."""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

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


class TestFirstContactTurnNote:
    @pytest.mark.parametrize("guest", [False, True])
    def test_first_contact_offer_follows_the_onboarding_gate(self, tmp_path, monkeypatch, guest):
        # The /initiate-setup offer ships with guest onboarding; everyone else keeps the memory profile offer.
        from agent.onboarding import (
            PROFILE_BUILD_FLAG,
            SETUP_OFFER_NOTE,
            first_contact_turn_note,
            profile_build_directive,
        )

        if guest:
            monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
        else:
            monkeypatch.delenv("HERMES_GUEST_ONBOARDING", raising=False)
        cfg_path = tmp_path / "config.yaml"
        cfg = {"onboarding": {"profile_build": "ask"}}
        note = first_contact_turn_note(
            cfg,
            cfg_path,
            session_history_empty=True,
            install_has_prior_sessions=False,
            message="hello",
        )
        expected = SETUP_OFFER_NOTE.format(command="/initiate-setup") if guest else profile_build_directive().strip()
        assert note == expected
        loaded = yaml.safe_load(cfg_path.read_text())
        assert loaded["onboarding"]["seen"][PROFILE_BUILD_FLAG] is True

    def test_every_first_contact_note_puts_a_real_task_first(self, tmp_path):
        # Default "ask" (offer) and "off" (plain intro) must both
        # tell the model to do a first-message task before the intro/offer.
        from agent.onboarding import TASK_FIRST_CLAUSE, first_contact_turn_note

        for mode in ("ask", "off"):
            note = first_contact_turn_note(
                {"onboarding": {"profile_build": mode}},
                tmp_path / f"{mode}.yaml",
                session_history_empty=True,
                install_has_prior_sessions=False,
                message="hello",
            )
            assert TASK_FIRST_CLAUSE in note, mode

    def test_no_note_in_the_task_chat_setup_hands_off_to(self, tmp_path, monkeypatch):
        # The first-task chat is the first chat in the user's own profile, but setup already ran:
        # no setup offer, no intro, and the offer latch stays unset.
        from agent.first_task_prompt import MARKER
        from agent.onboarding import PROFILE_BUILD_FLAG, first_contact_turn_note

        monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
        cfg_path = tmp_path / "config.yaml"
        note = first_contact_turn_note(
            {"onboarding": {"profile_build": "ask"}},
            cfg_path,
            session_history_empty=True,
            install_has_prior_sessions=False,
            message=f"Set up Blender for me{MARKER}skill text",
            setup_handoff=True,
        )
        assert note is None
        assert not cfg_path.exists() or PROFILE_BUILD_FLAG not in cfg_path.read_text()

    def test_a_first_message_quoting_the_handoff_marker_still_gets_the_offer(self, tmp_path, monkeypatch):
        # Only start_chat marks a handoff; a user who types the marker text is still a first contact.
        from agent.first_task_prompt import MARKER
        from agent.onboarding import SETUP_OFFER_NOTE, first_contact_turn_note

        monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
        note = first_contact_turn_note(
            {"onboarding": {"profile_build": "ask"}},
            tmp_path / "config.yaml",
            session_history_empty=True,
            install_has_prior_sessions=False,
            message=f"what does this mean?{MARKER}",
        )
        assert note == SETUP_OFFER_NOTE.format(command="/initiate-setup")

    def test_returns_none_when_not_first_contact(self, tmp_path):
        from agent.onboarding import first_contact_turn_note

        cfg_path = tmp_path / "config.yaml"
        assert (
            first_contact_turn_note(
                {},
                cfg_path,
                session_history_empty=False,
                install_has_prior_sessions=False,
                message="hello",
            )
            is None
        )
        assert (
            first_contact_turn_note(
                {},
                cfg_path,
                session_history_empty=True,
                install_has_prior_sessions=True,
                message="hello",
            )
            is None
        )
        assert not cfg_path.exists()
