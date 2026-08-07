import time
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


    def test_wide_status_bar_text_includes_idle(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
        )
        cli_obj._last_turn_finished_at = time.time() - 42
        cli_obj._prompt_start_time = None
        cli_obj._prompt_duration = 7.0
        text = cli_obj._build_status_bar_text(width=160)
        assert "✓ 42s" in text


class TestBedrockPrefixStripping:
    """Regression tests for Bedrock cross-region inference-profile prefix stripping."""

    def _f(self, model: str) -> str:
        from hermes_cli.model_switch import format_model_for_display
        return format_model_for_display(model)

    # --- All eleven regional prefixes strip to <model> slug ---

    def test_global_prefix_stripped(self):
        assert self._f("global.anthropic.claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_us_prefix_stripped(self):
        assert self._f("us.amazon.nova-pro-v1:0") == "nova-pro-v1:0"

    def test_eu_prefix_stripped(self):
        assert self._f("eu.meta.llama3-70b") == "llama3-70b"

    def test_apac_prefix_stripped(self):
        assert self._f("apac.anthropic.claude-3-5") == "claude-3-5"

    def test_ap_prefix_stripped(self):
        assert self._f("ap.anthropic.claude-3") == "claude-3"

    def test_au_prefix_stripped(self):
        assert self._f("au.meta.llama3") == "llama3"

    def test_jp_prefix_stripped(self):
        assert self._f("jp.x.y") == "y"

    def test_ca_prefix_stripped(self):
        assert self._f("ca.x.y") == "y"

    def test_sa_prefix_stripped(self):
        assert self._f("sa.x.y") == "y"

    def test_me_prefix_stripped(self):
        assert self._f("me.x.y") == "y"

    def test_af_prefix_stripped(self):
        assert self._f("af.x.y") == "y"

    # --- apac. vs ap. ordering trap ---

    def test_apac_does_not_match_ap_leaving_residue(self):
        # "ap." must NOT match the front of "apac.anthropic.claude-3-5"
        # because that would produce "ac.anthropic.claude-3-5" (corruption).
        result = self._f("apac.anthropic.claude-3-5")
        assert result == "claude-3-5"
        assert result != "ac.anthropic.claude-3-5"

    # --- Guard / unchanged cases ---

    def test_bare_foundation_model_id_unchanged(self):
        # No region prefix — bare Bedrock foundation model ID.
        assert self._f("anthropic.claude-sonnet-4") == "anthropic.claude-sonnet-4"

    def test_us_gguf_unchanged(self):
        # After stripping "us." the remainder "gguf" has no dot → not stripped.
        assert self._f("us.gguf") == "us.gguf"

    def test_useful_model_unchanged(self):
        # "us" is not a dot-delimited segment here.
        assert self._f("useful-model-v2") == "useful-model-v2"

    def test_slash_model_unchanged(self):
        assert self._f("meta-llama/Llama-3.3-70B") == "meta-llama/Llama-3.3-70B"

    def test_plain_model_unchanged(self):
        assert self._f("claude-4-7-opus-20260101") == "claude-4-7-opus-20260101"

    # --- Palantir RID regression ---

    def test_palantir_rid_still_stripped(self):
        assert self._f(
            "ri.language-model-service..language-model.anthropic-claude-4-7-opus"
        ) == "anthropic-claude-4-7-opus"

    # --- End-to-end: status bar snapshot uses the helper ---

    def test_status_bar_snapshot_strips_bedrock_prefix(self):
        cli_obj = _make_cli(model="global.anthropic.claude-sonnet-4-6")
        snapshot = cli_obj._get_status_bar_snapshot()
        assert snapshot["model_short"] == "claude-sonnet-4-6"

    def test_status_bar_snapshot_strips_us_prefix(self):
        cli_obj = _make_cli(model="us.amazon.nova-pro-v1:0")
        snapshot = cli_obj._get_status_bar_snapshot()
        assert snapshot["model_short"] == "nova-pro-v1:0"
