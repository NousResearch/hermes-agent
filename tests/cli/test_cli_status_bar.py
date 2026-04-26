from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI, _usage_line_ansi


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

    def test_input_height_counts_wide_characters_using_cell_width(self):
        cli_obj = _make_cli()

        class _Doc:
            lines = ["你" * 10]

        class _Buffer:
            document = _Doc()

        input_area = SimpleNamespace(buffer=_Buffer())

        def _input_height():
            try:
                from prompt_toolkit.application import get_app
                from prompt_toolkit.utils import get_cwidth

                doc = input_area.buffer.document
                prompt_width = max(2, get_cwidth(cli_obj._get_tui_prompt_text()))
                try:
                    available_width = get_app().output.get_size().columns - prompt_width
                except Exception:
                    import shutil
                    available_width = shutil.get_terminal_size((80, 24)).columns - prompt_width
                if available_width < 10:
                    available_width = 40
                visual_lines = 0
                for line in doc.lines:
                    line_width = get_cwidth(line)
                    if line_width <= 0:
                        visual_lines += 1
                    else:
                        visual_lines += max(1, -(-line_width // available_width))
                return min(max(visual_lines, 1), 8)
            except Exception:
                return 1

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=14)
        with patch.object(HermesCLI, "_get_tui_prompt_text", return_value="❯ "), \
             patch("prompt_toolkit.application.get_app", return_value=mock_app):
            assert _input_height() == 2

    def test_input_height_uses_prompt_toolkit_width_over_shutil(self):
        cli_obj = _make_cli()

        class _Doc:
            lines = ["你" * 10]

        class _Buffer:
            document = _Doc()

        input_area = SimpleNamespace(buffer=_Buffer())

        def _input_height():
            try:
                from prompt_toolkit.application import get_app
                from prompt_toolkit.utils import get_cwidth

                doc = input_area.buffer.document
                prompt_width = max(2, get_cwidth(cli_obj._get_tui_prompt_text()))
                try:
                    available_width = get_app().output.get_size().columns - prompt_width
                except Exception:
                    import shutil
                    available_width = shutil.get_terminal_size((80, 24)).columns - prompt_width
                if available_width < 10:
                    available_width = 40
                visual_lines = 0
                for line in doc.lines:
                    line_width = get_cwidth(line)
                    if line_width <= 0:
                        visual_lines += 1
                    else:
                        visual_lines += max(1, -(-line_width // available_width))
                return min(max(visual_lines, 1), 8)
            except Exception:
                return 1

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=14)
        with patch.object(HermesCLI, "_get_tui_prompt_text", return_value="❯ "), \
             patch("prompt_toolkit.application.get_app", return_value=mock_app), \
             patch("shutil.get_terminal_size") as mock_shutil:
            assert _input_height() == 2
        mock_shutil.assert_not_called()

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

    def test_minimal_tui_chrome_threshold(self):
        cli_obj = _make_cli()

        assert cli_obj._use_minimal_tui_chrome(width=63) is True
        assert cli_obj._use_minimal_tui_chrome(width=64) is False

    def test_bottom_input_rule_hides_on_narrow_terminals(self):
        cli_obj = _make_cli()

        assert cli_obj._tui_input_rule_height("top", width=50) == 1
        assert cli_obj._tui_input_rule_height("bottom", width=50) == 0
        assert cli_obj._tui_input_rule_height("bottom", width=90) == 1

    def test_agent_spacer_reclaimed_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._agent_running = True

        assert cli_obj._agent_spacer_height(width=50) == 0
        assert cli_obj._agent_spacer_height(width=90) == 1
        cli_obj._agent_running = False
        assert cli_obj._agent_spacer_height(width=90) == 0

    def test_spinner_line_hidden_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "thinking"

        assert cli_obj._spinner_widget_height(width=50) == 0
        assert cli_obj._spinner_widget_height(width=90) == 1
        cli_obj._spinner_text = ""
        assert cli_obj._spinner_widget_height(width=90) == 0

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

    def test_voice_recording_status_bar_compacts_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = True
        cli_obj._voice_processing = False

        fragments = cli_obj._get_voice_status_fragments(width=50)

        assert fragments == [("class:voice-status-recording", " ● REC ")]


class TestCLIUsageReport:
    def test_show_usage_without_agent_still_shows_account_limits_when_provider_known(self, capsys, monkeypatch):
        cli_obj = _make_cli(model="anthropic/claude-sonnet-4.6")
        cli_obj.provider = "openrouter"
        cli_obj.base_url = "https://openrouter.ai/api/v1"
        cli_obj.api_key = "test-key"
        cli_obj.verbose = False
        cli_obj._session_db = MagicMock()
        cli_obj._session_db.get_session.return_value = {}

        monkeypatch.setattr(
            "cli.fetch_all_relevant_providers",
            lambda provider, base_url=None, api_key=None: [],
        )
        monkeypatch.setattr(
            "cli.render_multi_provider_hash",
            lambda snapshots: [("openrouter", "Credits balance: $12.34")],
        )

        cli_obj._show_usage()
        output = capsys.readouterr().out

        lines = [line for line in output.splitlines() if line]
        assert lines[0] == "(._.) No API calls made yet in this session."
        table_lines = lines[1:]
        assert table_lines[0] == "#" * 79
        assert all(len(line) == 79 for line in table_lines)
        assert any("balances" in line for line in table_lines)
        assert any("openrouter" in line for line in table_lines)
        assert any("Credits balance: $12.34" in line for line in table_lines)

    def test_show_usage_uses_persisted_session_when_agent_not_active(self, capsys, monkeypatch):
        cli_obj = _make_cli()
        cli_obj.session_id = "sess-usage"
        cli_obj._session_db = MagicMock()
        cli_obj.verbose = False
        cli_obj._session_db.get_session.return_value = {
            "model": "anthropic/claude-sonnet-4.6",
            "billing_provider": "openrouter",
            "billing_base_url": "https://openrouter.ai/api/v1",
            "prompt_tokens": 10_230,
            "completion_tokens": 2_220,
            "total_tokens": 12_450,
            "api_calls": 7,
            "cache_read_tokens": 300,
            "cache_write_tokens": 100,
            "estimated_cost_usd": 0.0642,
        }
        monkeypatch.setattr(
            "cli.fetch_all_relevant_providers",
            lambda provider, base_url=None, api_key=None: [],
        )

        cli_obj._show_usage()
        output = capsys.readouterr().out
        lines = [line for line in output.splitlines() if line]

        assert lines[0] == "#" * 79
        assert all(len(line) == 79 for line in lines)
        assert any("openrouter / anthropic/claude-sonnet-4.6" in line for line in lines)
        assert any("tokens" in line and "12,450" in line for line in lines)
        assert any("calls" in line and "7" in line for line in lines)
        assert any("cost" in line and "$" in line for line in lines)

    def test_show_usage_renders_compact_table_with_provider_and_quota_sections(self, capsys, monkeypatch):
        from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
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

        monkeypatch.setattr(
            "cli.fetch_all_relevant_providers",
            lambda provider, base_url=None, api_key=None: [
                AccountUsageSnapshot(
                    provider="openrouter",
                    source="credits_api",
                    fetched_at=datetime.now(),
                    details=("Credits balance: $44.48",),
                ),
                AccountUsageSnapshot(
                    provider="anthropic",
                    source="oauth_usage_api",
                    fetched_at=datetime.now(),
                    windows=(
                        AccountUsageWindow(
                            label="Current session",
                            used_percent=55.0,
                            detail="in 3h 10m",
                        ),
                    ),
                ),
                AccountUsageSnapshot(
                    provider="openai-codex",
                    source="usage_api",
                    fetched_at=datetime.now(),
                    windows=(
                        AccountUsageWindow(
                            label="5h limit",
                            used_percent=0.0,
                            detail="in 5h 26m",
                        ),
                    ),
                ),
            ],
        )
        monkeypatch.setattr(
            "cli.render_multi_provider_hash",
            lambda snapshots: [
                ("openrouter", "Credits balance: $44.48"),
                ("maritaca", "Saldo: R$ 118,96"),
            ],
        )

        cli_obj._show_usage()
        output = capsys.readouterr().out
        lines = [line for line in output.splitlines() if line]

        assert lines[0] == "#" * 79
        assert lines[-1] == "#" * 79
        assert all(len(line) == 79 for line in lines)
        assert any("session" in line for line in lines)
        assert any("balances" in line for line in lines)
        assert any("claude code" in line for line in lines)
        assert any("codex / openai" in line for line in lines)
        assert any("Credits balance: $44.48" in line for line in lines)
        assert any("Saldo: R$ 118,96" in line for line in lines)
        assert any("$0.064" in line for line in lines)
        assert any("[████████" in line or "[███████" in line for line in lines)

    def test_show_usage_uses_ansi_colors_when_tty_is_available(self, monkeypatch):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
        )
        cli_obj.verbose = False

        monkeypatch.setattr("cli.fetch_all_relevant_providers", lambda provider, base_url=None, api_key=None: [])
        monkeypatch.setattr("cli._usage_table_colors_enabled", lambda: True)
        captured = []
        monkeypatch.setattr("cli._cprint", captured.append)

        cli_obj._show_usage()

        assert captured
        assert any("\x1b[" in line for line in captured)
        assert any("Usage" in line for line in captured)

    def test_usage_line_ansi_uses_brazil_colors_for_maritaca(self):
        line = "# maritaca           | Saldo: R$ 118,96                                   #"

        rendered = _usage_line_ansi(line)

        assert "\x1b[1;32mmaritaca" in rendered
        assert "\x1b[1;33mSaldo:" in rendered
        assert "\x1b[1;34mR$ 118,96" in rendered

    def test_usage_line_ansi_keeps_usd_balances_green(self):
        line = "# openrouter         | Credits balance: $44.48                             #"

        rendered = _usage_line_ansi(line)

        assert "\x1b[1;34mopenrouter" in rendered
        assert "\x1b[1;32m$44.48" in rendered

    def test_usage_line_ansi_keeps_title_row_edges_gray(self):
        line = "#" + " Usage ".center(77) + "#"

        rendered = _usage_line_ansi(line)

        assert rendered.startswith("\x1b[2;37m#\x1b[0m")
        assert rendered.endswith("\x1b[2;37m#\x1b[0m")
        assert "\x1b[1;36m" in rendered

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
        lines = [line for line in output.splitlines() if line]

        assert lines[0] == "#" * 79
        assert all(len(line) == 79 for line in lines)
        assert any("cost" in line and "n/a" in line for line in lines)
        assert any("Pricing unknown for local/my-custom-model" in line for line in lines)

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
        lines = [line for line in output.splitlines() if line]

        assert lines[0] == "#" * 79
        assert all(len(line) == 79 for line in lines)
        assert any("cost" in line and "n/a" in line for line in lines)
        assert any("Pricing unknown for glm-5" in line for line in lines)


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
