"""Watchdog and auxiliary errors share the main fallback eligibility policy."""
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import auxiliary_client as aux
from agent.context_compressor import take_pinned_summary_route
from agent.conversation_compression import (
    CompressionCommitFence,
    resolve_compression_fallback_route,
    run_compress_context_with_progress_timeout,
)


@pytest.fixture
def routes(monkeypatch):
    entry = {"provider": "custom:backup", "model": "backup-model",
             "base_url": "https://backup.invalid/v1", "api_key": "fixture-key",
             "api_mode": "chat_completions", "timeout": 19}
    config = {"provider": "auto", "model": ""}
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda: {"fallback_providers": [entry]})
    monkeypatch.setattr(aux, "_read_main_provider", lambda: "primary")
    monkeypatch.setattr(aux, "_read_main_model", lambda: "primary-model")
    monkeypatch.setattr(aux, "_custom_health_base_url", lambda provider, base_url=None: base_url or "")
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda *a, **kw: False)
    monkeypatch.setattr(aux, "_task_minimum_context_length", lambda task: 10000)
    monkeypatch.setattr(aux, "_candidate_context_window", lambda *a, **kw: 20000)
    monkeypatch.setattr(aux, "_resolve_fallback_entry",
                        lambda candidate: (SimpleNamespace(), candidate["model"]))
    return config, entry


def test_auto_inherits_main_fallback(routes):
    _, entry = routes
    route = resolve_compression_fallback_route()
    assert route is not None
    for key, value in entry.items():
        assert route[key] == value


def test_inherited_route_pins_inferred_transport(routes, monkeypatch):
    config, entry = routes
    config["api_mode"] = "responses"
    entry.pop("api_mode")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {"api_mode": "chat_completions"},
    )
    route = resolve_compression_fallback_route()
    assert route is not None
    assert route["api_mode"] == "chat_completions"


def test_inherited_route_preserves_resolved_credentials(routes, monkeypatch):
    config, entry = routes
    config["api_key"] = "primary-only-key"
    entry.pop("api_key")
    monkeypatch.setattr(
        aux, "_resolve_fallback_entry",
        lambda candidate: (SimpleNamespace(api_key="resolved-backup-key"), candidate["model"]),
    )
    route = resolve_compression_fallback_route()
    assert route is not None
    assert route["api_key"] == "resolved-backup-key"


@pytest.mark.parametrize("same_model", [True, False])
def test_inherited_exclusions_use_primary_backend_identity(routes, monkeypatch, same_model):
    _, entry = routes
    monkeypatch.setattr(aux, "_read_main_provider", lambda: entry["provider"])
    monkeypatch.setattr(aux, "_read_main_model", lambda: "primary-model")
    monkeypatch.setattr(aux, "_custom_health_base_url", lambda *a, **kw: entry["base_url"])
    if same_model:
        entry["model"] = "primary-model"
    route = resolve_compression_fallback_route()
    assert (route is None) == same_model


def test_explicit_chain_precedes_main(routes):
    config, _ = routes
    config["fallback_chain"] = [{"provider": "other", "model": "task-backup"}]
    route = resolve_compression_fallback_route()
    assert route is not None
    assert route["model"] == "task-backup"


def test_explicit_provider_does_not_inherit(routes):
    config, _ = routes
    config["provider"] = "explicit"
    assert resolve_compression_fallback_route() is None


@pytest.mark.parametrize("ineligible", ["context", "unhealthy", "unavailable", "excluded"])
def test_no_eligible_inherited_route(routes, monkeypatch, ineligible):
    _, entry = routes
    if ineligible == "context":
        monkeypatch.setattr(aux, "_candidate_context_window", lambda *a, **kw: 100)
    elif ineligible == "unhealthy":
        monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda *a, **kw: True)
    elif ineligible == "unavailable":
        monkeypatch.setattr(aux, "_resolve_fallback_entry", lambda entry: (None, None))
    else:
        entry["provider"] = "auto"
    assert resolve_compression_fallback_route() is None


@pytest.mark.parametrize("stopped", [False, True])
def test_real_watchdog_inherits_once_and_fences_late_primary(routes, stopped):
    original = [{"role": "user", "content": "original"}]
    compressed = [{"role": "user", "content": "summary"}]
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    stop = threading.Event()
    if stopped:
        stop.set()
    calls = []
    committed = []
    primary = CompressionCommitFence()

    def worker(fence):
        calls.append(take_pinned_summary_route())
        if fence is primary:
            started.set()
            assert release.wait(5)
            try:
                if fence.begin_commit():
                    committed.append("stale")
                    fence.finish_commit()
                return original, "stale"
            finally:
                finished.set()
        assert fence.begin_commit()
        committed.append("fallback")
        fence.finish_commit()
        return compressed, "compressed-prompt"

    # Advance the primary's observed idle age only once its worker is blocked.
    # No stopwatch thresholds or scheduling-dependent sleeps.
    def expired():
        assert started.wait(5)
        return 1000.0

    try:
        with patch.object(primary, "seconds_since_progress", side_effect=expired):
            result = run_compress_context_with_progress_timeout(
                worker=worker, messages=original, system_prompt_fallback="original-prompt",
                idle_timeout_seconds=1, total_ceiling_seconds=10, fence=primary,
                telemetry_agent=SimpleNamespace(_hard_interrupt_requested=stop),
                new_fence=CompressionCommitFence,
            )
    finally:
        release.set()
        assert finished.wait(5)
    assert calls[0] is None
    if stopped:
        assert result == (original, "original-prompt")
        assert len(calls) == 1
        assert committed == []
    else:
        assert result == (compressed, "compressed-prompt")
        assert len(calls) == 2
        assert calls[1]["provider"] == "custom:backup"
        assert calls[1]["model"] == "backup-model"
        assert committed == ["fallback"]
    assert primary.is_cancelled
