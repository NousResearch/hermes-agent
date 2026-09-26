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

        assert "🗜 3" in text















    def test_spinner_height_uses_display_width_for_wide_characters(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "你" * 40
        cli_obj._tool_start_time = 0

        assert cli_obj._spinner_widget_height(width=64) == 2




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







class TestCLIUsageNoAgentAccountLimits:
    """#42904: `/usage` without a live agent (TUI/Desktop slash-worker) still renders account limits."""

    def _no_agent_cli(self, monkeypatch, snapshot):
        from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
        cli_obj = _make_cli()
        cli_obj.provider, cli_obj.base_url, cli_obj.api_key = "openai-codex", None, None
        cli_obj._print_nous_credits_block = lambda: False
        seen = {}

        def _fetch(provider, **kwargs):
            seen["provider"] = provider
            if snapshot is None:
                return None
            return AccountUsageSnapshot(
                provider=provider, source="usage_api", fetched_at=datetime.now(), plan="Pro",
                windows=(AccountUsageWindow(label="Weekly", used_percent=1.0),),
            )
        monkeypatch.setattr("agent.account_usage.fetch_account_usage", _fetch)
        return cli_obj, seen

    def test_no_agent_prints_codex_account_limits(self, capsys, monkeypatch):
        cli_obj, seen = self._no_agent_cli(monkeypatch, snapshot=True)
        cli_obj._show_usage()
        out = capsys.readouterr().out
        assert seen["provider"] == "openai-codex"
        assert "Account limits" in out and "Weekly: 99% remaining (1% used)" in out
        assert "No active agent" not in out

    def test_no_agent_keeps_fallback_message_when_limits_unavailable(self, capsys, monkeypatch):
        cli_obj, _ = self._no_agent_cli(monkeypatch, snapshot=None)
        cli_obj._show_usage()
        out = capsys.readouterr().out
        assert "No active agent" in out and "Account limits" not in out


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


    def test_only_model_and_duration(self):
        text = self._cli_with_fields(["model", "duration"])
        assert "claude-sonnet-4-20250514" in text
        assert "15m" in text
        assert "12.4K/200K" not in text
        assert "🗜" not in text
        assert "%" not in text




    def test_total_tokens_when_explicitly_requested(self):
        text = self._cli_with_fields(["model", "total_tokens"])
        assert "Σ12.4K" in text
        assert "claude-sonnet-4-20250514" in text



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
        assert not any("🗜" in t for t in frag_texts)
        assert not any("12.4K" in t for t in frag_texts)


    def test_empty_fields_list_uses_defaults(self):
        text = self._cli_with_fields([])
        assert "claude-sonnet-4-20250514" in text
        assert "12.4K/200K" in text
        assert "🗜" in text



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
