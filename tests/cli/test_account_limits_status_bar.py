from __future__ import annotations

import threading
import time
from datetime import datetime
from types import SimpleNamespace

import cli as cli_mod
from cli import HermesCLI


def _shell() -> HermesCLI:
    shell = HermesCLI.__new__(HermesCLI)
    shell.model = "openai-codex/gpt-5.5-codex"
    shell.provider = "openai-codex"
    shell.base_url = None
    shell.api_key = None
    shell.session_start = datetime.now()
    shell._prompt_start_time = None
    shell._prompt_duration = 0.0
    shell._background_tasks = {}
    shell._account_limits_lock = threading.Lock()
    shell._account_limits_label = None
    shell._account_limits_style = "class:status-bar-good"
    shell._account_limits_provider = None
    shell._account_limits_checked_at = 0.0
    shell._account_limits_refreshing = False
    shell.agent = SimpleNamespace(
        model=shell.model,
        provider="openai-codex",
        base_url=None,
        api_key=None,
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_api_calls=0,
        context_compressor=None,
    )
    return shell


def test_account_limits_badge_marks_depleted_session_or_weekly_critical() -> None:
    assert HermesCLI._account_limits_badge_style("Acct S0% W42%") == "class:status-bar-critical"
    assert HermesCLI._account_limits_badge_style("Acct S42% W0%") == "class:status-bar-critical"
    assert HermesCLI._account_limits_badge_style("Acct S42% W99%") == "class:status-bar-good"


def test_status_bar_snapshot_styles_current_depleted_label(monkeypatch) -> None:
    shell = _shell()
    monkeypatch.setattr(shell, "_maybe_refresh_account_limits_badge", lambda agent: "Acct S0% W42%")

    snapshot = shell._get_status_bar_snapshot()

    assert snapshot["account_limits"] == "Acct S0% W42%"
    assert snapshot["account_limits_style"] == "class:status-bar-critical"


def test_non_codex_provider_clears_cached_codex_badge_without_starting_worker(monkeypatch) -> None:
    shell = _shell()
    shell._account_limits_label = "Acct S0% W42%"
    shell._account_limits_provider = "openai-codex"
    assert shell.agent is not None
    shell.agent.provider = "anthropic"
    started = False

    class ThreadMustNotStart:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            nonlocal started
            started = True

    monkeypatch.setattr(cli_mod.threading, "Thread", ThreadMustNotStart)

    assert shell._maybe_refresh_account_limits_badge(shell.agent) == ""
    assert shell._account_limits_label is None
    assert shell._account_limits_provider is None
    assert started is False


def test_stale_codex_badge_schedules_worker_without_fetching_in_render(monkeypatch) -> None:
    shell = _shell()
    started = []

    class DeferredThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            started.append(self)

    monkeypatch.setattr(cli_mod.threading, "Thread", DeferredThread)

    assert shell._maybe_refresh_account_limits_badge(shell.agent) == ""
    assert len(started) == 1
    assert started[0].name == "hermes-account-limits"
    assert shell._account_limits_refreshing is True


def test_failed_refresh_keeps_cached_badge_and_retries_soon(monkeypatch) -> None:
    shell = _shell()
    shell._account_limits_label = "Acct S42% W99%"
    shell._account_limits_provider = "openai-codex"
    shell._account_limits_checked_at = time.monotonic() - 301
    monkeypatch.setattr(shell, "_invalidate", lambda **kwargs: None)

    class ImmediateThread:
        def __init__(self, *, target, name, daemon):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(cli_mod.threading, "Thread", ImmediateThread)
    monkeypatch.setitem(__import__("sys").modules, "agent.account_usage", SimpleNamespace(fetch_account_usage=lambda **kwargs: None))

    assert shell._maybe_refresh_account_limits_badge(shell.agent) == "Acct S42% W99%"
    assert shell._account_limits_label == "Acct S42% W99%"
    assert shell._account_limits_checked_at < time.monotonic() - 269


def test_overflow_keeps_depleted_badge_critical(monkeypatch) -> None:
    shell = _shell()
    shell._status_bar_visible = True
    shell._model_picker_state = None
    monkeypatch.setattr(shell, "_get_tui_terminal_width", lambda: 76)
    monkeypatch.setattr(
        shell,
        "_get_status_bar_snapshot",
        lambda: {
            "model_short": "gpt-5.5-codex",
            "context_percent": 12,
            "context_length": 200_000,
            "context_tokens": 24_000,
            "account_limits": "Acct S0% W42%",
            "account_limits_style": "class:status-bar-critical",
            "compressions": 0,
            "active_background_tasks": 0,
            "active_background_processes": 0,
            "active_background_subagents": 0,
            "duration": "123m",
            "prompt_elapsed": "⏲ 99s",
            "idle_since": "",
        },
    )

    fragments = shell._get_status_bar_fragments()

    assert len(fragments) == 1
    assert fragments[0][0] == "class:status-bar-critical"
    assert "Acct S0%" in fragments[0][1]
