from datetime import datetime, timedelta
from types import SimpleNamespace

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

    def test_build_status_bar_text_no_cost_in_status_bar(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10000,
            completion_tokens=5000,
            total_tokens=15000,
            api_calls=7,
            context_tokens=50000,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=120)
        assert "$" not in text  # cost is never shown in status bar

    def test_build_status_bar_text_collapses_for_narrow_terminal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10000,
            completion_tokens=2400,
            total_tokens=12400,
            api_calls=7,
            context_tokens=12400,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=60)

        assert "⚕" in text
        assert "$0.06" not in text  # cost hidden by default
        assert "15m" in text
        assert "200K" not in text

    def test_build_status_bar_text_handles_missing_agent(self):
        cli_obj = _make_cli()

        text = cli_obj._build_status_bar_text(width=100)

        assert "⚕" in text
        assert "claude-sonnet-4-20250514" in text


class TestCLIUsageReport:
    def test_show_usage_includes_estimated_cost(self, capsys):
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

        assert "Model:" in output
        assert "Cost status:" in output
        assert "Cost source:" in output
        assert "Total cost:" in output
        assert "$" in output
        assert "0.064" in output
        assert "Session duration:" in output
        assert "Compressions:" in output

    def test_show_usage_marks_unknown_pricing(self, capsys):
        cli_obj = _attach_agent(
            _make_cli(model="local/my-custom-model"),
            prompt_tokens=1_000,
            completion_tokens=500,
            total_tokens=1_500,
            api_calls=1,
            context_tokens=1_000,
            context_length=32_000,
        )
        cli_obj.verbose = False

        cli_obj._show_usage()
        output = capsys.readouterr().out

        assert "Total cost:" in output
        assert "n/a" in output
        assert "Pricing unknown for local/my-custom-model" in output

    def test_zero_priced_provider_models_stay_unknown(self, capsys):
        cli_obj = _attach_agent(
            _make_cli(model="glm-5"),
            prompt_tokens=1_000,
            completion_tokens=500,
            total_tokens=1_500,
            api_calls=1,
            context_tokens=1_000,
            context_length=32_000,
        )
        cli_obj.verbose = False

        cli_obj._show_usage()
        output = capsys.readouterr().out

        assert "Total cost:" in output
        assert "n/a" in output
        assert "Pricing unknown for glm-5" in output


class TestStatusBarWidthSource:
    """Ensure status bar fragments don't overflow the terminal width."""

    def _make_wide_cli(self):
        from datetime import datetime, timedelta
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
            assert len(total_text) <= width + 4, (  # +4 for minor padding chars
                f"At width={width}, fragment total {len(total_text)} chars overflows "
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

    def test_fragments_fall_back_to_shutil_when_no_app(self):
        """Outside a TUI context (no running app), shutil must be used as fallback."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        with patch("prompt_toolkit.application.get_app", side_effect=Exception("no app")), \
             patch("shutil.get_terminal_size", return_value=MagicMock(columns=100)) as mock_shutil:
            frags = cli_obj._get_status_bar_fragments()

        mock_shutil.assert_called()
        assert len(frags) > 0

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

    def test_explicit_width_skips_pt_lookup(self):
        """An explicit width= argument must bypass both PT and shutil lookups."""
        from unittest.mock import patch
        cli_obj = self._make_wide_cli()

        with patch("prompt_toolkit.application.get_app") as mock_get_app, \
             patch("shutil.get_terminal_size") as mock_shutil:
            text = cli_obj._build_status_bar_text(width=100)

        mock_get_app.assert_not_called()
        mock_shutil.assert_not_called()
        assert len(text) > 0
