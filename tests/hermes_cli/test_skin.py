"""Tests for the Hermes CLI skin helpers."""

from pathlib import Path
from unittest.mock import Mock, patch

import cli as cli_module
from cli import HermesCLI
from hermes_cli.skin import (
    HermesLoreState,
    build_orbit_line,
    build_progress_meter,
    build_relay_telemetry,
    get_banner_title,
    get_caduceus_frame,
    get_mod_brand_name,
    get_mod_hero_animation_interval,
    get_mod_system_prompt,
    is_ares_skin,
    is_hermes_skin,
    is_mod_skin,
    is_posideon_skin,
    load_lore_state,
    normalize_skin_name,
    parse_dice_spec,
    record_published_skill,
    resolve_skin_request,
    set_active_skin_globals,
)
from hermes_state import SessionDB


class TestNormalizeSkinName:
    def test_aliases_resolve(self):
        assert normalize_skin_name("winged") == "ares"
        assert normalize_skin_name("poseidon") == "posideon"
        assert normalize_skin_name("stone") == "sisyphus"
        assert normalize_skin_name("classic") == "hermes"
        assert normalize_skin_name("ares") == "ares"
        assert normalize_skin_name("posideon") == "posideon"
        assert normalize_skin_name("sisyphus") == "sisyphus"
        assert normalize_skin_name("unknown-theme") == "hermes"

    def test_skin_request_resolution_accepts_title_case(self):
        assert resolve_skin_request("Hermes") == "hermes"
        assert resolve_skin_request("Ares") == "ares"
        assert resolve_skin_request("Posideon") == "posideon"
        assert resolve_skin_request("Sisyphus") == "sisyphus"
        assert resolve_skin_request("Poseidon") == "posideon"
        assert resolve_skin_request("unknown-theme") is None

    def test_skin_predicates_split_hermes_and_mod_skins(self):
        assert is_hermes_skin("hermes")
        assert not is_hermes_skin("ares")
        assert is_ares_skin("ares")
        assert not is_ares_skin("hermes")
        assert is_posideon_skin("posideon")
        assert not is_posideon_skin("ares")
        assert is_mod_skin("ares")
        assert is_mod_skin("posideon")
        assert not is_mod_skin("hermes")


class TestLoadLoreState:
    def test_counts_cli_sessions_and_clever_replies(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("cli_one", "cli", model="test")
        db.create_session("cli_two", "cli", model="test")
        db.create_session("discord_one", "discord", model="test")

        db.append_message("cli_one", "user", "Quick ping")
        db.append_message(
            "cli_two",
            "user",
            "Please design a more expressive terminal layout,\ncompare two options,\nand explain the tradeoffs.",
        )
        db.append_message("discord_one", "user", "This should not count")

        lore = load_lore_state(db)

        assert isinstance(lore, HermesLoreState)
        assert lore.sessions == 2
        assert lore.user_messages == 2
        assert lore.clever_replies == 1

        db.close()

    def test_published_skills_are_loaded_from_lore_file(self, tmp_path):
        with patch.dict("os.environ", {"HERMES_HOME": str(tmp_path)}):
            record_published_skill("skill-a")
            record_published_skill("skill-b")
            record_published_skill("skill-a")

            lore = load_lore_state()

        assert lore.published_skills[:2] == ["skill-a", "skill-b"]


class TestParseDiceSpec:
    def test_parses_common_dice_specs(self):
        assert parse_dice_spec("d20") == 20
        assert parse_dice_spec("12") == 12
        assert parse_dice_spec("garbage") == 6


class TestTelemetryHelpers:
    def test_progress_meter_shows_partial_progress(self):
        meter = build_progress_meter("Wing ascent", 1, 50, width=8)
        assert "Wing ascent" in meter
        assert "1/50" in meter
        assert "■" in meter

    def test_relay_telemetry_and_orbit_line_include_lore(self):
        lore = HermesLoreState(sessions=4, clever_replies=2, published_skills=["alpha", "beta"])
        telemetry = build_relay_telemetry(lore, phase=2, width=34, active=True)
        orbit = build_orbit_line(lore, phase=1, width=28)

        assert "warpath active" in telemetry
        assert "orbit 2" in telemetry
        assert "alpha" in orbit or "beta" in orbit

    def test_banner_title_uses_ares_branding(self):
        title = get_banner_title(HermesLoreState())
        assert "Ares Agent" in title

    def test_posideon_payload_overrides_branding_and_telemetry(self):
        with patch.dict("os.environ", {"HERMES_CLI_SKIN": "posideon"}):
            set_active_skin_globals("posideon")
            lore = HermesLoreState(sessions=5, clever_replies=2, published_skills=["undertow"])
            brand = get_mod_brand_name()
            system_prompt = get_mod_system_prompt()
            title = get_banner_title(lore)
            telemetry = build_relay_telemetry(lore, phase=2, width=40, active=True)
            hero = get_caduceus_frame(lore, width=24, height=16)
            assert brand == "Posideon Agent"
            assert "You are Posideon Agent" in system_prompt
            assert "Posideon Agent" in title
            assert "tidewatch active" in telemetry
            assert hero

        set_active_skin_globals("hermes")

    def test_ares_payload_exposes_skin_system_prompt(self):
        with patch.dict("os.environ", {"HERMES_CLI_SKIN": "ares"}):
            set_active_skin_globals("ares")
            system_prompt = get_mod_system_prompt()
            assert "You are Ares Agent" in system_prompt

        set_active_skin_globals("hermes")

    def test_sisyphus_payload_exposes_animated_hero_and_system_prompt(self):
        with patch.dict("os.environ", {"HERMES_CLI_SKIN": "sisyphus"}):
            set_active_skin_globals("sisyphus")
            lore = HermesLoreState(sessions=2, clever_replies=1, published_skills=["stone"])
            system_prompt = get_mod_system_prompt()
            hero = get_caduceus_frame(lore, phase=1, width=28, height=14)
            interval = get_mod_hero_animation_interval("sisyphus")

            assert "You are Sisyphus Agent" in system_prompt
            assert hero
            assert interval >= 0.12

        set_active_skin_globals("hermes")


class TestCliSkinSwitching:
    def test_process_command_reloads_ui_for_title_case_skin(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli.skin = "hermes"
        events = []

        cli._set_skin = lambda skin, persist=False: events.append(("set", skin, persist))
        cli._reload_skin_ui = lambda: events.append(("reload",))

        with patch("builtins.print"):
            assert cli.process_command("/skin Posideon") is True

        assert events == [("set", "posideon", True), ("reload",)]

    def test_clear_command_uses_safe_reload_ui_path(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli.agent = Mock()
        cli.conversation_history = [{"role": "user", "content": "hi"}]
        cli._reload_skin_ui = Mock()

        with patch("builtins.print"):
            assert cli.process_command("/clear") is True

        cli.agent.flush_memories.assert_called_once_with([{"role": "user", "content": "hi"}])
        cli._reload_skin_ui.assert_called_once_with()
        assert cli.conversation_history == []

    def test_reload_skin_ui_uses_prompt_toolkit_safe_redraw_when_app_running(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli._app = Mock(is_running=True)
        cli._build_banner_ansi = lambda: "banner"
        cli.console = Mock()
        cli.show_banner = Mock()

        with patch("cli._cprint") as cprint:
            cli._reload_skin_ui()

        cprint.assert_called_once_with("\033[2J\033[Hbanner")
        cli._app.invalidate.assert_called_once()
        cli.show_banner.assert_not_called()

    def test_build_banner_ansi_does_not_advance_phase(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli.compact = True
        cli.skin = "sisyphus"
        cli._banner_phase = 7
        cli.console = Mock(width=120)
        cli._show_status = Mock()
        cli._show_tool_availability_warnings = Mock()

        ansi = cli._build_banner_ansi()

        assert ansi
        assert cli._banner_phase == 7

    def test_advance_live_banner_frame_updates_phase_and_snapshot(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli._banner_phase = 2
        cli._banner_snapshot_lines = ["frame-2"]
        cli._banner_snapshot_line_count = 1
        cli._build_banner_ansi = lambda phase=None: f"frame-{phase}"
        cli._rewrite_banner_lines_absolute = Mock()

        cli._advance_live_banner_frame()

        cli._rewrite_banner_lines_absolute.assert_called_once_with(["frame-2"], ["frame-3"])
        assert cli._banner_phase == 3
        assert cli._banner_snapshot_lines == ["frame-3"]

    def test_compose_system_prompt_layers_skin_persona_and_user_prompt(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli.skin = "ares"
        cli.user_system_prompt = "Answer like a terse staff officer."

        composed = cli._compose_system_prompt()

        assert "You are Ares Agent" in composed
        assert "Answer like a terse staff officer." in composed

    def test_freeze_managed_banner_is_permanent_for_session(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli._managed_banner_frozen = False
        cli._ui_phase = 12
        cli._app = Mock(is_running=True)
        cli._uses_managed_banner = lambda: True

        cli._freeze_managed_banner()

        assert cli._managed_banner_frozen is True
        assert cli._ui_phase == 12
        cli._app.invalidate.assert_called_once_with()

    def test_mod_scroll_body_uses_skin_specific_response_colors(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli.skin = "posideon"
        cli._ui_phase = 0
        cli._ares_skin_active = lambda: True

        with patch.object(cli_module, "ARES_SAND", "#EAF7FF"), patch.object(cli_module, "ARES_ASH", "#6FA6C8"):
            body = cli._format_hermes_scroll_body("hold the line")

        assert "\033[38;2;234;247;255mhold the line\033[0m" in body
        assert "\033[38;2;111;166;200m╎\033[0m" in body

    def test_mod_response_frame_color_uses_skin_accent(self):
        cli = HermesCLI.__new__(HermesCLI)
        with patch.object(cli_module, "ARES_EMBER", "#5DB8F5"):
            assert cli._mod_response_frame_color() == "\033[38;2;93;184;245m"
