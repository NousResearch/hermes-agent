"""Tests for agent/onboarding.py — contextual first-touch hint helpers."""

from __future__ import annotations

import yaml

from agent.onboarding import (
    BUSY_INPUT_FLAG,
    OPENCLAW_RESIDUE_FLAG,
    TOOL_PROGRESS_FLAG,
    busy_input_hint_cli,
    busy_input_hint_gateway,
    detect_openclaw_residue,
    is_seen,
    mark_seen,
    openclaw_residue_hint_cli,
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


# ---------------------------------------------------------------------------
# OpenClaw residue banner
# ---------------------------------------------------------------------------


class TestDetectOpenclawResidue:
    def test_returns_true_when_openclaw_dir_present(self, tmp_path):
        (tmp_path / ".openclaw").mkdir()
        assert detect_openclaw_residue(home=tmp_path) is True

    def test_returns_false_when_absent(self, tmp_path):
        assert detect_openclaw_residue(home=tmp_path) is False

    def test_returns_false_when_path_is_a_file(self, tmp_path):
        # A stray file named ``.openclaw`` is NOT a workspace — skip the banner.
        (tmp_path / ".openclaw").write_text("oops")
        assert detect_openclaw_residue(home=tmp_path) is False

    def test_default_home_does_not_crash(self):
        # Smoke: real $HOME lookup must not raise regardless of state.
        assert isinstance(detect_openclaw_residue(), bool)


class TestOpenclawResidueHint:
    def test_hint_mentions_migrate_command(self):
        # `migrate` is the non-destructive path — should lead the banner.
        msg = openclaw_residue_hint_cli()
        assert "hermes claw migrate" in msg
        assert "~/.openclaw" in msg

    def test_hint_mentions_cleanup_command(self):
        # `cleanup` is mentioned as the follow-up archive step.
        assert "hermes claw cleanup" in openclaw_residue_hint_cli()

    def test_hint_warns_cleanup_breaks_openclaw(self):
        # Archiving the directory breaks OpenClaw for users still running it —
        # the banner must flag that side effect.
        msg = openclaw_residue_hint_cli().lower()
        assert "openclaw will stop working" in msg or "stop working" in msg

    def test_hint_not_empty(self):
        assert openclaw_residue_hint_cli().strip()


class TestOpenclawResidueSeenFlag:
    def test_flag_independent_of_other_flags(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        mark_seen(cfg_path, BUSY_INPUT_FLAG)
        loaded = yaml.safe_load(cfg_path.read_text())
        assert is_seen(loaded, OPENCLAW_RESIDUE_FLAG) is False

    def test_flag_round_trips(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        assert mark_seen(cfg_path, OPENCLAW_RESIDUE_FLAG) is True
        loaded = yaml.safe_load(cfg_path.read_text())
        assert is_seen(loaded, OPENCLAW_RESIDUE_FLAG) is True


class TestProfileBuildMode:
    def test_default_is_ask(self):
        from agent.onboarding import profile_build_mode

        assert profile_build_mode({}) == "ask"
        assert profile_build_mode({"onboarding": {}}) == "ask"
        assert profile_build_mode({"onboarding": {"profile_build": "ask"}}) == "ask"

    def test_off_disables(self):
        from agent.onboarding import profile_build_mode

        assert profile_build_mode({"onboarding": {"profile_build": "off"}}) == "off"
        assert profile_build_mode({"onboarding": {"profile_build": "OFF"}}) == "off"

    def test_unknown_value_falls_back_to_ask(self):
        from agent.onboarding import profile_build_mode

        assert profile_build_mode({"onboarding": {"profile_build": "banana"}}) == "ask"

    def test_non_mapping_config_safe(self):
        from agent.onboarding import profile_build_mode

        assert profile_build_mode("not a dict") == "ask"  # type: ignore[arg-type]
        assert profile_build_mode({"onboarding": "nope"}) == "ask"


class TestProfileBuildDirective:
    def test_directive_is_opt_in_and_consent_gated(self):
        from agent.onboarding import profile_build_directive

        d = profile_build_directive()
        # Must OFFER, not assume.
        assert "OFFER" in d
        # Must require consent before external lookups.
        assert "consent" in d.lower()
        # Must forbid silently reading connected accounts.
        assert "silently" in d.lower()
        # Must persist via the user-profile memory store.
        assert 'target="user"' in d
        # Must allow declining.
        assert "decline" in d.lower()

    def test_directive_mentions_first_message(self):
        from agent.onboarding import profile_build_directive

        assert "first message ever" in profile_build_directive()


class TestProfileBuildSeenFlag:
    def test_flag_round_trips(self, tmp_path):
        from agent.onboarding import PROFILE_BUILD_FLAG

        cfg_path = tmp_path / "config.yaml"
        assert mark_seen(cfg_path, PROFILE_BUILD_FLAG) is True
        loaded = yaml.safe_load(cfg_path.read_text())
        assert is_seen(loaded, PROFILE_BUILD_FLAG) is True

    def test_flag_independent_of_busy_input(self, tmp_path):
        from agent.onboarding import PROFILE_BUILD_FLAG

        cfg_path = tmp_path / "config.yaml"
        mark_seen(cfg_path, BUSY_INPUT_FLAG)
        loaded = yaml.safe_load(cfg_path.read_text())
        assert is_seen(loaded, PROFILE_BUILD_FLAG) is False


class TestProfileBuildConfigDefault:
    def test_default_config_carries_ask(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["onboarding"]["profile_build"] == "ask"


class TestSethomeNoticeGate:
    """Gate for the '📬 No home channel is set…' onboarding notice.

    Incident 2026-07-07 (multi-user Feishu deployment): the notice fired in
    every chat whose session history was empty, so end users/customers kept
    seeing a staff-facing setup prompt. Default mode is now "once" — at most
    one showing per platform per install.
    """

    def test_default_mode_is_once(self):
        from agent.onboarding import sethome_notice_mode

        assert sethome_notice_mode({}) == "once"
        assert sethome_notice_mode({"onboarding": {}}) == "once"
        assert sethome_notice_mode({"onboarding": {"sethome_notice": "once"}}) == "once"

    def test_off_and_always_modes(self):
        from agent.onboarding import sethome_notice_mode

        assert sethome_notice_mode({"onboarding": {"sethome_notice": "off"}}) == "off"
        assert sethome_notice_mode({"onboarding": {"sethome_notice": "OFF"}}) == "off"
        assert sethome_notice_mode({"onboarding": {"sethome_notice": "always"}}) == "always"

    def test_unknown_value_falls_back_to_once(self):
        from agent.onboarding import sethome_notice_mode

        assert sethome_notice_mode({"onboarding": {"sethome_notice": "banana"}}) == "once"
        assert sethome_notice_mode("not a dict") == "once"  # type: ignore[arg-type]

    def test_gate_once_shows_then_marks(self):
        from agent.onboarding import sethome_notice_gate

        show, flag = sethome_notice_gate({}, "feishu")
        assert show is True
        assert flag == "sethome_notice_feishu"

    def test_gate_once_suppresses_after_seen(self):
        from agent.onboarding import sethome_notice_gate

        cfg = {"onboarding": {"seen": {"sethome_notice_feishu": True}}}
        show, flag = sethome_notice_gate(cfg, "feishu")
        assert show is False
        assert flag is None
        # Other platforms are independent.
        show, flag = sethome_notice_gate(cfg, "telegram")
        assert show is True
        assert flag == "sethome_notice_telegram"

    def test_gate_off_never_shows(self):
        from agent.onboarding import sethome_notice_gate

        show, flag = sethome_notice_gate(
            {"onboarding": {"sethome_notice": "off"}}, "feishu"
        )
        assert show is False
        assert flag is None

    def test_gate_always_shows_without_marking(self):
        from agent.onboarding import sethome_notice_gate

        cfg = {
            "onboarding": {
                "sethome_notice": "always",
                "seen": {"sethome_notice_feishu": True},
            }
        }
        show, flag = sethome_notice_gate(cfg, "feishu")
        assert show is True
        assert flag is None  # legacy mode: never persist, never suppress

    def test_gate_roundtrip_with_mark_seen(self, tmp_path):
        import yaml

        from agent.onboarding import mark_seen, sethome_notice_gate

        cfg_path = tmp_path / "config.yaml"
        show, flag = sethome_notice_gate({}, "feishu")
        assert show and flag
        assert mark_seen(cfg_path, flag) is True
        loaded = yaml.safe_load(cfg_path.read_text())
        show, flag = sethome_notice_gate(loaded, "feishu")
        assert show is False and flag is None


class TestSethomeNoticeConfigDefault:
    def test_default_config_carries_once(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["onboarding"]["sethome_notice"] == "once"
