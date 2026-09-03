import time
from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cli as cli_mod
from cli import HermesCLI


def _make_cli(model: str = "anthropic/claude-sonnet-4-20250514"):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = model
    cli_obj.session_start = datetime.now() - timedelta(minutes=14, seconds=32)
    cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
    cli_obj.agent = None
    return cli_obj


def _attach_agent(
    cli_obj,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    api_calls: int,
    context_tokens: int,
    context_length: int,
    compressions: int = 0,
):
    cli_obj.agent = SimpleNamespace(
        model=cli_obj.model,
        provider="anthropic" if cli_obj.model.startswith("anthropic/") else None,
        base_url="",
        session_input_tokens=input_tokens if input_tokens is not None else prompt_tokens,
        session_output_tokens=output_tokens if output_tokens is not None else completion_tokens,
        session_cache_read_tokens=cache_read_tokens,
        session_cache_write_tokens=cache_write_tokens,
        session_prompt_tokens=prompt_tokens,
        session_completion_tokens=completion_tokens,
        session_total_tokens=total_tokens,
        session_api_calls=api_calls,
        get_rate_limit_state=lambda: None,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=context_tokens,
            context_length=context_length,
            compression_count=compressions,
        ),
    )
    return cli_obj


class TestCLIStatusBar:
    def test_session_title_is_right_aligned_after_it_is_queued(self):
        cli_obj = _make_cli()
        cli_obj._pending_title = "weekly-digest"

        text = cli_obj._build_status_bar_text(width=80)

        assert text.endswith(" weekly-digest ")
        assert cli_obj._status_bar_display_width(text) == 80

    def test_snapshot_refreshes_persisted_session_title(self):
        cli_obj = _make_cli()
        cli_obj.session_id = "session-1"
        cli_obj._session_db = SimpleNamespace(  # type: ignore[assignment]
            get_session_title=lambda sid: "user-profiles" if sid == "session-1" else None
        )

        snapshot = cli_obj._get_status_bar_snapshot()

        assert snapshot["session_title"] == "user-profiles"

    def test_status_bar_config_helper_treats_persisted_off_as_hidden(self):
        for value in (False, "off", "false", "hidden", "no", "0"):
            assert cli_mod._status_bar_visible_from_display_config({"tui_statusbar": value}) is False

        for value in (True, "top", "bottom", "on", None):
            assert cli_mod._status_bar_visible_from_display_config({"tui_statusbar": value}) is True

    def test_status_bar_initial_visibility_honors_tui_statusbar_config(self, monkeypatch):
        config = deepcopy(cli_mod.CLI_CONFIG)
        config.setdefault("display", {})["tui_statusbar"] = False
        config["display"].pop("statusbar", None)
        monkeypatch.setattr(cli_mod, "CLI_CONFIG", config)

        cli_obj = HermesCLI(model="test-model", toolsets=[], provider="auto")

        assert cli_obj._status_bar_visible is False

    def test_context_style_thresholds(self):
        cli_obj = _make_cli()

        assert cli_obj._status_bar_context_style(None) == "class:status-bar-dim"
        assert cli_obj._status_bar_context_style(10) == "class:status-bar-good"
        assert cli_obj._status_bar_context_style(50) == "class:status-bar-warn"
        assert cli_obj._status_bar_context_style(81) == "class:status-bar-bad"
        assert cli_obj._status_bar_context_style(95) == "class:status-bar-critical"

    def test_build_status_bar_text_for_wide_terminal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "claude-sonnet-4-20250514" in text
        assert "12.4K/200K" in text
        assert "6%" in text
        assert "$0.06" not in text  # cost hidden by default
        assert "15m" in text

    def test_status_bar_shows_session_suffix_by_default(self):
        cli_obj = _make_cli()
        cli_obj.session_id = "20260901_120000_abc123"

        text = cli_obj._build_status_bar_text(width=120)

        assert "sid abc123" in text

    def test_status_bar_session_field_can_be_hidden(self):
        cli_obj = _make_cli()
        cli_obj.session_id = "20260901_120000_abc123"

        with patch.object(cli_mod, "CLI_CONFIG", {"display": {"status_bar": {"fields": ["model", "duration"]}}}):
            text = cli_obj._build_status_bar_text(width=120)

        assert "sid abc123" not in text

    def test_status_bar_session_field_can_be_explicitly_shown(self):
        cli_obj = _make_cli()
        cli_obj.session_id = "20260901_120000_abc123"

        with patch.object(cli_mod, "CLI_CONFIG", {"display": {"status_bar": {"fields": ["model", "session"]}}}):
            text = cli_obj._build_status_bar_text(width=120)

        assert "sid abc123" in text


    def test_input_height_counts_prompt_only_on_first_wrapped_row(self):
        # Regression for prompt_toolkit classic CLI resize glitches: the prompt
        # is inserted by BeforeInput only on logical line 0. At three terminal
        # cells, "⚔ " leaves one cell for the first input character, but
        # wrapped continuation rows use the full three cells. Estimating every
        # wrapped row as one-cell wide over-allocates the TextArea and can leave
        # stale prompt/input cells visible after resize.
        assert cli_mod._estimate_tui_input_height(["abcdef"], "⚔ ", 3) == 3






    def test_compression_count_shown_in_wide_status_bar(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=3,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "🗜️ 3" in text







    def test_minimal_tui_chrome_threshold(self):
        cli_obj = _make_cli()

        assert cli_obj._use_minimal_tui_chrome(width=63) is True
        assert cli_obj._use_minimal_tui_chrome(width=64) is False




    def test_scheduled_unsuppress_debounces_resize_storm(self):
        """A fresh resize cancels the pending unsuppress and restarts it."""
        cli_obj = _make_cli()
        cli_obj._status_bar_unsuppress_timer = None
        cli_obj._status_bar_suppressed_after_resize = True
        app = MagicMock()
        app.loop = None

        # First schedule (long delay) then a second should cancel the first.
        cli_obj._schedule_status_bar_unsuppress(app, delay=5.0)
        first_timer = cli_obj._status_bar_unsuppress_timer
        assert first_timer is not None
        cli_obj._schedule_status_bar_unsuppress(app, delay=0.01)
        assert first_timer is not cli_obj._status_bar_unsuppress_timer
        assert not first_timer.is_alive() or first_timer.finished.is_set()
        time.sleep(0.1)
        assert cli_obj._status_bar_suppressed_after_resize is False




    def test_spinner_height_uses_display_width_for_wide_characters(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "你" * 40
        cli_obj._tool_start_time = 0

        assert cli_obj._spinner_widget_height(width=64) == 2


    def test_voice_status_bar_compacts_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = False
        cli_obj._voice_tts = True
        cli_obj._voice_continuous = True

        fragments = cli_obj._get_voice_status_fragments(width=50)

        assert fragments == [("class:voice-status", " 🎤 Ctrl+B ")]


    # Round-13 Copilot review regressions on #19835. The label in voice
    # status bar / recording hint / placeholder must render the
    # configured ``voice.record_key`` — not hardcoded Ctrl+B. Pinning
    # the cache (``set_voice_record_key_cache``) keeps display in sync
    # with the prompt_toolkit binding without re-reading config on
    # every render.
    def test_voice_status_bar_renders_configured_ctrl_letter(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = False
        cli_obj._voice_tts = False
        cli_obj._voice_continuous = False
        cli_obj.set_voice_record_key_cache("ctrl+o")

        wide = cli_obj._get_voice_status_fragments(width=120)
        assert any("Ctrl+O to record" in text for _cls, text in wide)

        compact = cli_obj._get_voice_status_fragments(width=50)
        assert compact == [("class:voice-status", " 🎤 Ctrl+O ")]





class TestCLIUsageReport:
    def test_show_usage_omits_cost_reporting(self, capsys):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=1,
        )
        cli_obj.verbose = False

        cli_obj._show_usage()
        output = capsys.readouterr().out

        # Token counts and session metadata still shown.
        assert "Model:" in output
        assert "Input tokens:" in output
        assert "Output tokens:" in output
        assert "Total tokens:" in output
        assert "Session duration:" in output
        assert "Compressions:" in output
        # Cost and cache-hit reporting is removed everywhere.
        assert "Total cost:" not in output
        assert "Cost status:" not in output
        assert "Cost source:" not in output
        assert "Cache read tokens:" not in output
        assert "Cache write tokens:" not in output


class TestStatusBarWidthSource:
    """Ensure status bar fragments don't overflow the terminal width."""

    def _make_wide_cli(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=100_000,
            completion_tokens=5_000,
            total_tokens=105_000,
            api_calls=20,
            context_tokens=100_000,
            context_length=200_000,
        )
        cli_obj._status_bar_visible = True
        return cli_obj

    def test_fragments_fit_within_announced_width(self):
        """Total fragment text length must not exceed the width used to build them."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        for width in (40, 52, 76, 80, 120, 200):
            mock_app = MagicMock()
            mock_app.output.get_size.return_value = MagicMock(columns=width)

            with patch("prompt_toolkit.application.get_app", return_value=mock_app):
                frags = cli_obj._get_status_bar_fragments()

            total_text = "".join(text for _, text in frags)
            display_width = cli_obj._status_bar_display_width(total_text)
            assert display_width <= width + 4, (  # +4 for minor padding chars
                f"At width={width}, fragment total {display_width} cells overflows "
                f"({total_text!r})"
            )

    def test_fragments_put_session_title_at_far_right(self):
        cli_obj = self._make_wide_cli()
        cli_obj._pending_title = "weekly-digest"
        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=100)

        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            frags = cli_obj._get_status_bar_fragments()

        text = "".join(value for _, value in frags)
        assert text.endswith(" weekly-digest ")
        assert cli_obj._status_bar_display_width(text) == 100

    def test_fragments_use_pt_width_over_shutil(self):
        """When prompt_toolkit reports a width, shutil.get_terminal_size must not be used."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)

        with patch("prompt_toolkit.application.get_app", return_value=mock_app) as mock_get_app, \
             patch("shutil.get_terminal_size") as mock_shutil:
            cli_obj._get_status_bar_fragments()

        mock_shutil.assert_not_called()


    def test_build_status_bar_text_uses_pt_width(self):
        """_build_status_bar_text() must also prefer prompt_toolkit width."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=80)

        with patch("prompt_toolkit.application.get_app", return_value=mock_app), \
             patch("shutil.get_terminal_size") as mock_shutil:
            text = cli_obj._build_status_bar_text()  # no explicit width

        mock_shutil.assert_not_called()
        assert isinstance(text, str)
        assert len(text) > 0



class TestIdleSinceLastTurn:
    """Time-since-last-final-agent-response read-out on the status bar."""

    def test_hidden_before_first_turn(self):
        assert HermesCLI._format_idle_since(None, turn_live=False) == ""

    def test_hidden_while_turn_is_live(self):
        assert HermesCLI._format_idle_since(time.time() - 30, turn_live=True) == ""

    def test_shows_compact_idle_time_after_turn(self):
        label = HermesCLI._format_idle_since(time.time() - 42, turn_live=False)
        assert label.startswith("✓ ")
        assert label == "✓ 42s"


    def test_snapshot_carries_idle_since(self):
        cli_obj = _make_cli()
        cli_obj._last_turn_finished_at = time.time() - 10
        cli_obj._prompt_start_time = None
        cli_obj._prompt_duration = 5.0
        snapshot = cli_obj._get_status_bar_snapshot()
        assert snapshot["idle_since"].startswith("✓ ")




class TestStatusBarFieldConfig:
    """Tests for display.status_bar.fields config customization (#41909)."""

    def _cli_with_fields(self, fields, width=120):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=7,
        )
        with patch.object(cli_mod, "CLI_CONFIG", {"display": {"status_bar": {"fields": fields}}}):
            text = cli_obj._build_status_bar_text(width=width)
        return text

    def test_default_fields_show_all(self):
        """With no config, all default fields appear."""
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=7,
        )
        with patch.object(cli_mod, "CLI_CONFIG", {}):
            text = cli_obj._build_status_bar_text(width=120)
        assert "claude-sonnet-4-20250514" in text
        assert "12.4K/200K" in text
        assert "🗜️" in text
        assert "15m" in text

    def test_only_model_and_duration(self):
        text = self._cli_with_fields(["model", "duration"])
        assert "claude-sonnet-4-20250514" in text
        assert "15m" in text
        assert "12.4K/200K" not in text
        assert "🗜️" not in text
        assert "%" not in text

    def test_only_model(self):
        text = self._cli_with_fields(["model"])
        assert "claude-sonnet-4-20250514" in text
        assert "15m" not in text
        assert "12.4K/200K" not in text

    def test_context_pct_only(self):
        text = self._cli_with_fields(["context_pct"])
        assert "%" in text
        assert "claude-sonnet-4-20250514" not in text

    def test_compressions_only(self):
        text = self._cli_with_fields(["compressions"])
        assert "🗜️ 7" in text
        assert "claude-sonnet-4-20250514" not in text

    def test_total_tokens_when_explicitly_requested(self):
        text = self._cli_with_fields(["model", "total_tokens"])
        assert "Σ12.4K" in text
        assert "claude-sonnet-4-20250514" in text

    def test_total_tokens_hidden_by_default(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=7,
        )
        with patch.object(cli_mod, "CLI_CONFIG", {}):
            text = cli_obj._build_status_bar_text(width=120)
        assert "Σ" not in text

    def test_narrow_terminal_drops_context_detail(self):
        """Narrow terminal (<76) ignores context_detail even if configured."""
        text = self._cli_with_fields(["model", "context_detail", "duration"], width=60)
        assert "claude-sonnet-4-20250514" in text
        assert "15m" in text
        assert "12.4K/200K" not in text

    def test_field_config_never_empties_the_bar(self):
        """A fields list matching nothing still anchors on the model name."""
        text = self._cli_with_fields(["nonexistent_field"])
        assert "claude-sonnet-4-20250514" in text

    def test_fragments_respect_field_config(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=7,
        )
        cli_obj._status_bar_visible = True
        with patch.object(cli_mod, "CLI_CONFIG", {"display": {"status_bar": {"fields": ["model", "duration"]}}}), \
                patch.object(cli_obj, "_get_tui_terminal_width", return_value=120):
            frags = cli_obj._get_status_bar_fragments()
        frag_texts = [text for _, text in frags]
        assert any("claude-sonnet-4-20250514" in t for t in frag_texts)
        assert any("15m" in t for t in frag_texts)
        assert not any("🗜️" in t for t in frag_texts)
        assert not any("12.4K" in t for t in frag_texts)

    def test_field_order_is_fixed(self):
        """Config controls visibility, not ordering — model stays first."""
        text = self._cli_with_fields(["duration", "model", "compressions"])
        model_pos = text.find("claude-sonnet-4-20250514")
        comp_pos = text.find("🗜️")
        dur_pos = text.find("15m")
        assert 0 <= model_pos < comp_pos < dur_pos

    def test_empty_fields_list_uses_defaults(self):
        text = self._cli_with_fields([])
        assert "claude-sonnet-4-20250514" in text
        assert "12.4K/200K" in text
        assert "🗜️" in text

    def test_field_set_is_cached_per_instance(self):
        cli_obj = _make_cli()
        with patch.object(cli_mod, "CLI_CONFIG", {"display": {"status_bar": {"fields": ["model"]}}}):
            first = cli_obj._get_status_bar_field_set()
        # Cache holds even if config object changes afterwards (per-session semantics).
        with patch.object(cli_mod, "CLI_CONFIG", {"display": {"status_bar": {"fields": ["duration"]}}}):
            second = cli_obj._get_status_bar_field_set()
        assert first == second == frozenset({"model"})


class TestCacheHitRate:
    def test_cache_hit_rate_shown_in_wide_terminal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000,
            completion_tokens=2_000,
            total_tokens=12_000,
            api_calls=5,
            context_tokens=12_000,
            context_length=200_000,
            cache_read_tokens=7600,
            cache_write_tokens=0,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "◎ 76.0%" in text

    def test_cache_hit_rate_shown_in_narrow_terminal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000,
            completion_tokens=2_000,
            total_tokens=12_000,
            api_calls=5,
            context_tokens=12_000,
            context_length=200_000,
            cache_read_tokens=5000,
            cache_write_tokens=0,
        )

        text = cli_obj._build_status_bar_text(width=60)

        assert "◎ 50%" in text

    def test_cache_hit_rate_hidden_when_zero(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000,
            completion_tokens=2_000,
            total_tokens=12_000,
            api_calls=5,
            context_tokens=12_000,
            context_length=200_000,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "◎" not in text

    def test_cache_hit_rate_hidden_when_no_data(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000,
            completion_tokens=2_000,
            total_tokens=12_000,
            api_calls=5,
            context_tokens=12_000,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "◎" not in text

    def test_cache_hit_rate_one_decimal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000,
            completion_tokens=2_000,
            total_tokens=12_000,
            api_calls=5,
            context_tokens=12_000,
            context_length=200_000,
            cache_read_tokens=7620,
            cache_write_tokens=0,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "◎ 76.2%" in text

    def test_cache_hit_rate_with_anthropic_style_cache(self):
        """Anthropic has both cache_read and cache_write"""
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000,
            completion_tokens=2_000,
            total_tokens=12_000,
            api_calls=5,
            context_tokens=12_000,
            context_length=200_000,
            cache_read_tokens=5000,
            cache_write_tokens=2000,
        )

        text = cli_obj._build_status_bar_text(width=120)

        # cache_read / prompt_tokens = 5000 / 10000 = 50%
        assert "◎ 50.0%" in text


class TestRollingLatencyVelocity:
    def _with_history(self, cli_obj, latencies, outputs):
        from collections import deque
        cli_obj.agent._api_latency_history = deque(latencies, maxlen=10)
        cli_obj.agent._api_output_history = deque(outputs, maxlen=10)
        return cli_obj

    def test_latency_and_tps_shown_in_wide_terminal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000, completion_tokens=2_000, total_tokens=12_000,
            api_calls=5, context_tokens=12_000, context_length=200_000,
        )
        self._with_history(cli_obj, [2.0, 4.0], [120, 180])

        text = cli_obj._build_status_bar_text(width=140)

        assert "\u25f7 3.0s" in text           # mean latency (2+4)/2
        assert "\u2191 50 t/s" in text          # true throughput 300/6.0

    def test_latency_hidden_without_history(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000, completion_tokens=2_000, total_tokens=12_000,
            api_calls=5, context_tokens=12_000, context_length=200_000,
        )
        text = cli_obj._build_status_bar_text(width=140)
        assert "\u25f7" not in text
        assert "t/s" not in text

    def test_latency_and_tps_respect_field_filter(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000, completion_tokens=2_000, total_tokens=12_000,
            api_calls=5, context_tokens=12_000, context_length=200_000,
        )
        self._with_history(cli_obj, [2.0], [100])
        with patch.object(cli_mod, "CLI_CONFIG", {"display": {"status_bar": {"fields": ["model", "duration"]}}}):
            text = cli_obj._build_status_bar_text(width=140)
        assert "\u25f7" not in text
        assert "t/s" not in text

    def test_negative_latency_guard(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000, completion_tokens=2_000, total_tokens=12_000,
            api_calls=5, context_tokens=12_000, context_length=200_000,
        )
        self._with_history(cli_obj, [-0.8], [100])
        snapshot = cli_obj._get_status_bar_snapshot()
        assert snapshot["avg_latency"] is None
        assert snapshot["avg_velocity"] is None


class TestCacheHitBaselineReset:
    def test_baseline_resets_on_model_switch(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000, completion_tokens=2_000, total_tokens=12_000,
            api_calls=5, context_tokens=12_000, context_length=200_000,
            cache_read_tokens=9_000,
        )
        first = cli_obj._get_status_bar_snapshot()
        assert first["cache_hit_pct"] == 90.0

        # Switch model. The bar repaints every frame, so the switch is
        # observed (and the baseline reset) before new tokens accrue.
        cli_obj.model = "openai/gpt-5"
        cli_obj.agent.model = "openai/gpt-5"
        reset_snap = cli_obj._get_status_bar_snapshot()
        assert reset_snap["cache_hit_pct"] is None  # new regime, no data yet

        cli_obj.agent.session_prompt_tokens = 12_000
        cli_obj.agent.session_cache_read_tokens = 9_500
        second = cli_obj._get_status_bar_snapshot()
        # Delta since switch: 500/2000 = 25%, not the lifetime 79%.
        assert second["cache_hit_pct"] == 25.0

    def test_baseline_resets_on_compression(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000, completion_tokens=2_000, total_tokens=12_000,
            api_calls=5, context_tokens=12_000, context_length=200_000,
            cache_read_tokens=8_000,
        )
        cli_obj._get_status_bar_snapshot()

        cli_obj.agent.context_compressor.compression_count = 1
        cli_obj._get_status_bar_snapshot()  # repaint observes the compression

        cli_obj.agent.session_prompt_tokens = 14_000
        cli_obj.agent.session_cache_read_tokens = 8_400
        snap = cli_obj._get_status_bar_snapshot()
        assert snap["cache_hit_pct"] == 10.0  # 400/4000 post-compression

    def test_title_field_filter_hides_session_badge(self):
        cli_obj = _make_cli()
        cli_obj._pending_title = "weekly-digest"
        with patch.object(cli_mod, "CLI_CONFIG", {"display": {"status_bar": {"fields": ["model", "duration"]}}}):
            text = cli_obj._build_status_bar_text(width=80)
        assert "weekly-digest" not in text
