import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cli as cli_mod
from cli import HermesCLI


def _make_cli(model: str = "anthropic/claude-sonnet-4-20250514"):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = model
    cli_obj.provider = "anthropic"
    cli_obj.requested_provider = "anthropic"
    cli_obj.session_start = datetime.now() - timedelta(minutes=14, seconds=32)
    cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
    cli_obj.agent = None
    cli_obj._codex_usage_snapshot = None
    cli_obj._codex_usage_last_attempt = 0.0
    cli_obj._codex_usage_refreshing = False
    cli_obj._codex_usage_lock = threading.Lock()
    cli_obj._codex_usage_scope = None
    return cli_obj


def _codex_usage_snapshot(
    used_percent: object,
    *,
    reset_at: datetime | None = None,
    label: str = "Session",
):
    return SimpleNamespace(
        windows=(SimpleNamespace(label=label, used_percent=used_percent, reset_at=reset_at),)
    )


def _attach_codex_runtime(
    cli_obj,
    *,
    api_key: str = "codex-test-token",
    base_url: str = "https://chatgpt.com/backend-api/codex",
) -> None:
    cli_obj.provider = "openai-codex"
    cli_obj.requested_provider = "openai-codex"
    cli_obj.agent = SimpleNamespace(
        model=cli_obj.model,
        provider="openai-codex",
        api_key=api_key,
        base_url=base_url,
        client=SimpleNamespace(api_key=api_key, base_url=base_url),
    )


def _seed_codex_usage(cli_obj, used_percent: object, *, reset_at: datetime | None = None) -> None:
    _attach_codex_runtime(cli_obj)
    cli_obj._codex_usage_snapshot = _codex_usage_snapshot(used_percent, reset_at=reset_at)
    cli_obj._codex_usage_scope = cli_obj._codex_usage_scope_for_active_credentials()
    cli_obj._codex_usage_last_attempt = time.monotonic()


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

    def test_codex_session_limit_formats_remaining_percent(self):
        result = HermesCLI._format_codex_session_limit(_codex_usage_snapshot(23.4))

        assert result == ("Codex 77%", 77)

    def test_codex_session_limit_formats_reset_countdown_from_injected_time(self):
        now = datetime(2026, 7, 15, 12, 0, 0)
        reset_at = now + timedelta(hours=2, minutes=10, seconds=30)

        result = HermesCLI._format_codex_session_limit(
            _codex_usage_snapshot(23.4, reset_at=reset_at),
            now=now,
        )

        assert result == ("Codex 77% reset 2h 10m", 77)

    def test_codex_session_limit_rejects_missing_or_invalid_session_data(self):
        assert HermesCLI._format_codex_session_limit(None) is None
        assert HermesCLI._format_codex_session_limit(_codex_usage_snapshot(20, label="Weekly")) is None
        assert HermesCLI._format_codex_session_limit(_codex_usage_snapshot("unknown")) is None
        assert HermesCLI._format_codex_session_limit(_codex_usage_snapshot(float("nan"))) is None
        assert HermesCLI._format_codex_session_limit(_codex_usage_snapshot(float("inf"))) is None
        assert HermesCLI._format_codex_session_limit(_codex_usage_snapshot(float("-inf"))) is None

    def test_codex_session_limit_clamps_remaining_percent(self):
        assert HermesCLI._format_codex_session_limit(_codex_usage_snapshot(-10)) == ("Codex 100%", 100)
        assert HermesCLI._format_codex_session_limit(_codex_usage_snapshot(125)) == ("Codex 0%", 0)

    def test_codex_session_limit_style_thresholds(self):
        cli_obj = _make_cli()

        assert cli_obj._codex_session_limit_style(None) == "class:status-bar-dim"
        assert cli_obj._codex_session_limit_style(40) == "class:status-bar-good"
        assert cli_obj._codex_session_limit_style(35) == "class:status-bar-warn"
        assert cli_obj._codex_session_limit_style(15) == "class:status-bar-bad"
        assert cli_obj._codex_session_limit_style(5) == "class:status-bar-critical"

    def test_snapshot_includes_cached_codex_limit_for_active_codex_provider(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        _seed_codex_usage(cli_obj, 12)

        snapshot = cli_obj._get_status_bar_snapshot()

        assert snapshot["codex_session_limit"] == "Codex 88%"
        assert snapshot["codex_session_remaining_percent"] == 88

    def test_snapshot_hides_cached_codex_limit_after_provider_switch(self):
        cli_obj = _make_cli()
        cli_obj._codex_usage_snapshot = _codex_usage_snapshot(12)
        cli_obj._codex_usage_last_attempt = time.monotonic()

        snapshot = cli_obj._get_status_bar_snapshot()

        assert snapshot["codex_session_limit"] is None
        assert snapshot["codex_session_remaining_percent"] is None

    def test_active_agent_provider_overrides_stale_cli_provider(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        _seed_codex_usage(cli_obj, 12)
        cli_obj.agent = SimpleNamespace(model="anthropic/claude-opus-4.6", provider="anthropic")

        snapshot = cli_obj._get_status_bar_snapshot()

        assert snapshot["codex_session_limit"] is None

    def test_codex_limit_shows_only_in_wide_plain_text_status(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        _seed_codex_usage(cli_obj, 66)

        assert "Codex 34%" in cli_obj._build_status_bar_text(width=120)
        assert "Codex" not in cli_obj._build_status_bar_text(width=75)
        assert "Codex" not in cli_obj._build_status_bar_text(width=51)

    def test_wide_fragments_style_cached_codex_limit(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        _seed_codex_usage(cli_obj, 94)
        cli_obj._status_bar_visible = True
        cli_obj._get_tui_terminal_width = lambda: 120

        fragments = cli_obj._get_status_bar_fragments()

        assert ("class:status-bar-bad", "Codex 6%") in fragments

    def test_codex_refresh_is_skipped_for_other_providers(self):
        cli_obj = _make_cli()

        with patch("cli.threading.Thread") as thread_cls:
            cli_obj._maybe_refresh_codex_usage_snapshot()

        thread_cls.assert_not_called()

    def test_codex_refresh_rejects_mixed_provider_or_non_codex_client(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        cli_obj.provider = "openai-codex"

        class SwitchingProviderAgent:
            model = "openai/gpt-5.4"
            client = SimpleNamespace(
                api_key="anthropic-token",
                base_url="https://api.anthropic.com",
            )

            def __init__(self):
                self._reads = 0

            @property
            def provider(self):
                self._reads += 1
                return "openai-codex" if self._reads == 1 else "anthropic"

        cli_obj.agent = SwitchingProviderAgent()
        cli_obj._codex_usage_snapshot = _codex_usage_snapshot(20)

        with patch("cli.threading.Thread") as thread_cls:
            cli_obj._maybe_refresh_codex_usage_snapshot()

        thread_cls.assert_not_called()
        assert cli_obj._codex_usage_snapshot is None
        assert cli_obj._codex_usage_scope is None

    def test_codex_refresh_is_skipped_while_cache_interval_is_fresh(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        _attach_codex_runtime(cli_obj)
        cli_obj._codex_usage_scope = cli_obj._codex_usage_scope_for_active_credentials()
        cli_obj._codex_usage_last_attempt = 900.0

        with patch("cli.time.monotonic", return_value=1000.0), patch("cli.threading.Thread") as thread_cls:
            cli_obj._maybe_refresh_codex_usage_snapshot(min_interval=300.0)

        thread_cls.assert_not_called()

    def test_codex_refresh_is_single_flight(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        _attach_codex_runtime(cli_obj)
        deferred_targets = []
        thread_options = []

        class DeferredThread:
            def __init__(self, *, target, **kwargs):
                deferred_targets.append(target)
                thread_options.append(kwargs)

            def start(self):
                return None

        with patch("cli.time.monotonic", return_value=1000.0), patch(
            "cli.threading.Thread", DeferredThread
        ):
            cli_obj._maybe_refresh_codex_usage_snapshot()
            cli_obj._maybe_refresh_codex_usage_snapshot()

        assert len(deferred_targets) == 1
        assert thread_options == [{"name": "codex-usage-status-refresh", "daemon": True}]
        assert cli_obj._codex_usage_refreshing is True

    def test_codex_refresh_discards_in_flight_result_after_credential_failover(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        cli_obj.provider = "openai-codex"
        cli_obj.agent = SimpleNamespace(
            model="openai/gpt-5.4",
            provider="openai-codex",
            api_key="pooled-token-a",
            base_url="https://chatgpt.com/backend-api/codex",
            client=SimpleNamespace(
                api_key="pooled-token-a",
                base_url="https://chatgpt.com/backend-api/codex",
            ),
        )
        cli_obj._invalidate = MagicMock()
        deferred_targets = []
        account_a_snapshot = _codex_usage_snapshot(80)

        class DeferredThread:
            def __init__(self, *, target, **_kwargs):
                deferred_targets.append(target)

            def start(self):
                return None

        with patch("cli.time.monotonic", return_value=1000.0), patch(
            "cli.threading.Thread", DeferredThread
        ), patch("agent.account_usage.fetch_account_usage", return_value=account_a_snapshot):
            cli_obj._maybe_refresh_codex_usage_snapshot()
            cli_obj.agent.api_key = "pooled-token-b"
            cli_obj.agent.client = SimpleNamespace(
                api_key="pooled-token-b",
                base_url="https://chatgpt.com/backend-api/codex",
            )
            cli_obj._maybe_refresh_codex_usage_snapshot()
            deferred_targets[0]()

        assert cli_obj._codex_usage_snapshot is None
        assert cli_obj._codex_usage_refreshing is False

    def test_codex_credential_failover_clears_cache_and_bypasses_rate_limit(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        cli_obj.provider = "openai-codex"
        cli_obj.agent = SimpleNamespace(
            model="openai/gpt-5.4",
            provider="openai-codex",
            api_key="pooled-token-a",
            base_url="https://chatgpt.com/backend-api/codex",
            client=SimpleNamespace(
                api_key="pooled-token-a",
                base_url="https://chatgpt.com/backend-api/codex",
            ),
        )
        cli_obj._codex_usage_snapshot = _codex_usage_snapshot(80)
        cli_obj._codex_usage_scope = cli_obj._codex_usage_scope_for_active_credentials()
        cli_obj._codex_usage_last_attempt = 900.0
        cli_obj._invalidate = MagicMock()
        account_b_snapshot = _codex_usage_snapshot(20)

        class ImmediateThread:
            def __init__(self, *, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()

        cli_obj.agent.api_key = "pooled-token-b"
        cli_obj.agent.client = SimpleNamespace(
            api_key="pooled-token-b",
            base_url="https://chatgpt.com/backend-api/codex",
        )
        with patch("cli.time.monotonic", return_value=1000.0), patch(
            "cli.threading.Thread", ImmediateThread
        ), patch("agent.account_usage.fetch_account_usage", return_value=account_b_snapshot) as fetch:
            cli_obj._maybe_refresh_codex_usage_snapshot()

        fetch.assert_called_once()
        assert cli_obj._codex_usage_snapshot is account_b_snapshot
        assert cli_obj._codex_usage_last_attempt == 1000.0

    def test_codex_refresh_uses_same_credential_capture_for_scope_and_fetch(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        cli_obj.provider = "openai-codex"
        cli_obj._invalidate = MagicMock()
        account_a_snapshot = _codex_usage_snapshot(30)
        codex_url = "https://chatgpt.com/backend-api/codex"

        class RotatingAgent:
            model = "openai/gpt-5.4"
            provider = "openai-codex"

            def __init__(self):
                self.api_key = "account-a-token"
                self.client = SimpleNamespace(api_key="account-a-token", base_url=codex_url)

            @property
            def base_url(self):
                # Reproduce rotation between separate live-field reads. The
                # already-captured client object remains an immutable A snapshot.
                self.api_key = "account-b-token"
                return codex_url

        cli_obj.agent = RotatingAgent()

        class ImmediateThread:
            def __init__(self, *, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()

        with patch("cli.time.monotonic", return_value=1000.0), patch(
            "cli.threading.Thread", ImmediateThread
        ), patch("agent.account_usage.fetch_account_usage", return_value=account_a_snapshot) as fetch:
            cli_obj._maybe_refresh_codex_usage_snapshot()

        fetch.assert_called_once_with(
            "openai-codex",
            base_url=codex_url,
            api_key="account-a-token",
        )
        assert cli_obj._codex_usage_snapshot is account_a_snapshot

    def test_codex_refresh_uses_active_agent_credentials_and_repaints(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        cli_obj.provider = "openai-codex"
        cli_obj.api_key = "stale-cli-token"
        cli_obj.base_url = "https://stale.example"
        cli_obj.agent = SimpleNamespace(
            model="openai/gpt-5.4",
            provider="openai-codex",
            api_key="active-agent-token",
            base_url="https://chatgpt.com/backend-api/codex",
            client=SimpleNamespace(
                api_key="active-agent-token",
                base_url="https://chatgpt.com/backend-api/codex",
            ),
        )
        cli_obj._invalidate = MagicMock()
        refreshed = _codex_usage_snapshot(42)

        class ImmediateThread:
            def __init__(self, *, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()

        with patch("cli.time.monotonic", return_value=1000.0), patch(
            "cli.threading.Thread", ImmediateThread
        ), patch("agent.account_usage.fetch_account_usage", return_value=refreshed) as fetch:
            cli_obj._maybe_refresh_codex_usage_snapshot()

        fetch.assert_called_once_with(
            "openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="active-agent-token",
        )
        assert cli_obj._codex_usage_snapshot is refreshed
        assert cli_obj._codex_usage_refreshing is False
        cli_obj._invalidate.assert_called_once_with(min_interval=0.0)

    def test_codex_refresh_failure_keeps_stale_cache_and_is_rate_limited(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        _attach_codex_runtime(cli_obj)
        stale = _codex_usage_snapshot(73)
        cli_obj._codex_usage_snapshot = stale
        cli_obj._codex_usage_scope = cli_obj._codex_usage_scope_for_active_credentials()
        cli_obj._invalidate = MagicMock()
        starts = []

        class ImmediateThread:
            def __init__(self, *, target, **_kwargs):
                self.target = target

            def start(self):
                starts.append(1)
                self.target()

        with patch("cli.threading.Thread", ImmediateThread), patch(
            "agent.account_usage.fetch_account_usage", side_effect=RuntimeError("temporary failure")
        ):
            with patch("cli.time.monotonic", return_value=1000.0):
                cli_obj._maybe_refresh_codex_usage_snapshot()
            with patch("cli.time.monotonic", return_value=1100.0):
                cli_obj._maybe_refresh_codex_usage_snapshot()

        assert len(starts) == 1
        assert cli_obj._codex_usage_snapshot is stale
        assert cli_obj._codex_usage_refreshing is False

    def test_codex_refresh_thread_start_failure_allows_immediate_retry(self):
        cli_obj = _make_cli("openai/gpt-5.4")
        _attach_codex_runtime(cli_obj)

        with patch("cli.time.monotonic", return_value=1000.0), patch(
            "cli.threading.Thread"
        ) as thread_cls:
            thread_cls.return_value.start.side_effect = RuntimeError("thread unavailable")
            cli_obj._maybe_refresh_codex_usage_snapshot()

        assert cli_obj._codex_usage_refreshing is False
        assert cli_obj._codex_usage_last_attempt == 0.0

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

    def test_post_compression_sentinel_does_not_render_negative(self):
        """Right after a compression, last_prompt_tokens is parked at the -1
        sentinel until the next API call reports real usage. The status bar
        must clamp it to 0 instead of rendering "-1/200K" / "-1%".
        """
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=-1,
            context_length=200_000,
        )

        snapshot = cli_obj._get_status_bar_snapshot()
        assert snapshot["context_tokens"] == 0
        assert snapshot["context_percent"] == 0

        text = cli_obj._build_status_bar_text(width=120)
        assert "-1" not in text
        assert "0/200K" in text

    def test_input_height_counts_prompt_only_on_first_wrapped_row(self):
        # Regression for prompt_toolkit classic CLI resize glitches: the prompt
        # is inserted by BeforeInput only on logical line 0. At three terminal
        # cells, "⚔ " leaves one cell for the first input character, but
        # wrapped continuation rows use the full three cells. Estimating every
        # wrapped row as one-cell wide over-allocates the TextArea and can leave
        # stale prompt/input cells visible after resize.
        assert cli_mod._estimate_tui_input_height(["abcdef"], "⚔ ", 3) == 3

    def test_input_height_counts_wide_characters_using_cell_width(self):
        # Prompt width (2 cells) + ten CJK chars (20 cells) = 22 display cells,
        # which wraps to two rows at 14 terminal columns.
        assert cli_mod._estimate_tui_input_height(["你" * 10], "❯ ", 14) == 2

    def test_input_height_clamps_zero_width_to_one_cell(self):
        # Some terminals briefly report zero columns during resize. Treat that
        # as a one-cell terminal rather than falling back to a fake wide width.
        assert cli_mod._estimate_tui_input_height(["abcd"], "", 0) == 4

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

    def test_compression_count_hidden_when_zero(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=0,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "🗜️" not in text

    def test_compression_count_shown_in_medium_status_bar(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000,
            completion_tokens=2_400,
            total_tokens=12_400,
            api_calls=7,
            context_tokens=12_400,
            context_length=200_000,
            compressions=2,
        )

        text = cli_obj._build_status_bar_text(width=60)

        assert "🗜️ 2" in text

    def test_compression_count_hidden_in_narrow_status_bar(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000,
            completion_tokens=2_400,
            total_tokens=12_400,
            api_calls=7,
            context_tokens=12_400,
            context_length=200_000,
            compressions=5,
        )

        text = cli_obj._build_status_bar_text(width=50)

        assert "🗜️" not in text

    def test_compression_count_style_thresholds(self):
        cli_obj = _make_cli()

        assert cli_obj._compression_count_style(1) == "class:status-bar-dim"
        assert cli_obj._compression_count_style(4) == "class:status-bar-dim"
        assert cli_obj._compression_count_style(5) == "class:status-bar-warn"
        assert cli_obj._compression_count_style(9) == "class:status-bar-warn"
        assert cli_obj._compression_count_style(10) == "class:status-bar-bad"
        assert cli_obj._compression_count_style(25) == "class:status-bar-bad"

    def test_compression_count_in_wide_fragments(self):
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

        frags = cli_obj._get_status_bar_fragments()
        frag_texts = [text for _, text in frags]

        assert "🗜️ 7" in frag_texts
        frag_styles = {text: style for style, text in frags}
        assert frag_styles["🗜️ 7"] == "class:status-bar-warn"

    def test_compression_count_absent_from_fragments_when_zero(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=0,
        )
        cli_obj._status_bar_visible = True

        frags = cli_obj._get_status_bar_fragments()
        frag_texts = [text for _, text in frags]

        assert not any("🗜️" in t for t in frag_texts)

    def test_minimal_tui_chrome_threshold(self):
        cli_obj = _make_cli()

        assert cli_obj._use_minimal_tui_chrome(width=63) is True
        assert cli_obj._use_minimal_tui_chrome(width=64) is False

    def test_bottom_input_rule_hides_on_narrow_terminals(self):
        cli_obj = _make_cli()

        assert cli_obj._tui_input_rule_height("top", width=50) == 1
        assert cli_obj._tui_input_rule_height("bottom", width=50) == 0
        assert cli_obj._tui_input_rule_height("bottom", width=90) == 1

    def test_input_rules_hide_after_resize_until_next_input(self):
        """When _status_bar_suppressed_after_resize is set, both rules hide.

        See _recover_after_resize — column shrink reflows already-rendered
        bars into scrollback, so we hide the separators while the reflow
        settles, then clear the flag (either via the scheduled unsuppress
        timer or the next submitted input).
        """
        cli_obj = _make_cli()
        cli_obj._status_bar_suppressed_after_resize = True

        assert cli_obj._tui_input_rule_height("top", width=90) == 0
        assert cli_obj._tui_input_rule_height("bottom", width=90) == 0

        cli_obj._status_bar_suppressed_after_resize = False
        assert cli_obj._tui_input_rule_height("top", width=90) == 1
        assert cli_obj._tui_input_rule_height("bottom", width=90) == 1

    def test_scheduled_unsuppress_clears_flag_and_repaints_without_input(self):
        """The status bar returns during idle after a resize, without a keypress.

        Regression: the suppression flag was only cleared on the next
        *submitted* input, so a resize/reflow followed by idle left the bar
        hidden indefinitely even while the refresh clock kept ticking. The
        scheduled unsuppress timer must clear the flag and invalidate the app
        on its own.
        """
        cli_obj = _make_cli()
        cli_obj._status_bar_unsuppress_timer = None
        cli_obj._status_bar_suppressed_after_resize = True
        app = MagicMock()
        app.loop = None  # force the synchronous _clear path

        # Schedule with ~0 delay so the timer fires promptly under test.
        cli_obj._schedule_status_bar_unsuppress(app, delay=0.01)
        time.sleep(0.1)

        assert cli_obj._status_bar_suppressed_after_resize is False
        app.invalidate.assert_called()
        # Bar chrome is visible again with no submitted input.
        assert cli_obj._tui_input_rule_height("top", width=90) == 1

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

    def test_scrollback_box_width_returns_viewport_width(self):
        """Decorative scrollback boxes use the full viewport width.

        The previous clamp (max 56 cols) was reverted in favour of the
        prompt_toolkit ``_output_screen_diff`` monkey-patch landed in
        #26137, which keeps chrome out of scrollback at the source.
        We accept that an aggressive column-shrink may visually reflow
        already printed Panel borders — that's a cosmetic artifact of
        stamped scrollback history, not a live-render bug.
        """
        from cli import HermesCLI

        # Floor at 32 — narrow terminals still get something usable
        # (avoids negative ``'─' * (w - 2)`` math).
        assert HermesCLI._scrollback_box_width(20) == 32
        assert HermesCLI._scrollback_box_width(32) == 32
        # Above the floor, return the actual viewport width — no cap.
        assert HermesCLI._scrollback_box_width(48) == 48
        assert HermesCLI._scrollback_box_width(80) == 80
        assert HermesCLI._scrollback_box_width(120) == 120
        assert HermesCLI._scrollback_box_width(200) == 200

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

    def test_spinner_elapsed_format_is_fixed_width_to_reduce_wrap_jitter(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "running tool"

        # Pin the clock: time.monotonic()'s epoch is arbitrary (often near
        # boot), so deriving _tool_start_time from the real monotonic clock
        # made the test fail on hosts where monotonic() < 65.2 — the start
        # time went negative, the (t0 > 0) guard in _render_spinner_text
        # dropped the "(elapsed)" suffix entirely, and the split below hit an
        # IndexError. A fixed clock keeps both elapsed paths deterministic.
        with patch.object(cli_mod.time, "monotonic", return_value=1000.0):
            # <60s path
            cli_obj._tool_start_time = 1000.0 - 9.2
            short = cli_obj._render_spinner_text()

            # >=60s path
            cli_obj._tool_start_time = 1000.0 - 65.2
            long = cli_obj._render_spinner_text()

        short_elapsed = short.split("(", 1)[1].rstrip(")")
        long_elapsed = long.split("(", 1)[1].rstrip(")")

        assert len(short_elapsed) == len(long_elapsed)
        assert "m" in long_elapsed and "s" in long_elapsed

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

    def test_voice_recording_status_bar_renders_configured_named_key(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = True
        cli_obj._voice_processing = False
        cli_obj.set_voice_record_key_cache("ctrl+space")

        fragments = cli_obj._get_voice_status_fragments(width=120)

        assert fragments == [("class:voice-status-recording", " ● REC  Ctrl+Space to stop ")]

    def test_voice_status_bar_falls_back_to_ctrl_b_without_cache(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = False
        cli_obj._voice_tts = False
        cli_obj._voice_continuous = False
        # No cache set — mirrors pre-startup state; fall back to
        # documented Ctrl+B default (Copilot round-13 review).

        compact = cli_obj._get_voice_status_fragments(width=50)

        assert compact == [("class:voice-status", " 🎤 Ctrl+B ")]

    def test_voice_status_bar_renders_malformed_config_as_default(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = False
        cli_obj._voice_tts = False
        cli_obj._voice_continuous = False
        # Non-string / typoed configs fall through the formatter to the
        # documented default so the status bar never advertises an
        # invalid shortcut.
        cli_obj.set_voice_record_key_cache(True)

        compact = cli_obj._get_voice_status_fragments(width=50)

        assert compact == [("class:voice-status", " 🎤 Ctrl+B ")]


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

    def test_scales_to_minutes(self):
        label = HermesCLI._format_idle_since(time.time() - 3 * 60, turn_live=False)
        assert label == "✓ 3m"

    def test_snapshot_carries_idle_since(self):
        cli_obj = _make_cli()
        cli_obj._last_turn_finished_at = time.time() - 10
        cli_obj._prompt_start_time = None
        cli_obj._prompt_duration = 5.0
        snapshot = cli_obj._get_status_bar_snapshot()
        assert snapshot["idle_since"].startswith("✓ ")

    def test_snapshot_idle_empty_during_live_turn(self):
        cli_obj = _make_cli()
        cli_obj._last_turn_finished_at = time.time() - 10
        cli_obj._prompt_start_time = time.time()
        cli_obj._prompt_duration = 0.0
        snapshot = cli_obj._get_status_bar_snapshot()
        assert snapshot["idle_since"] == ""

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
