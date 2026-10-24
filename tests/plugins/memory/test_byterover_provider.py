"""Tests for the ByteRover memory provider config gates."""

from types import SimpleNamespace

from plugins.memory import byterover
from plugins.memory.byterover import ByteRoverMemoryProvider


def test_auto_extract_false_skips_sync_turn(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": False})
    provider.initialize("session-1")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", lambda *args, **kwargs: calls.append((args, kwargs)))

    provider.sync_turn("please remember this detail", "acknowledged")

    assert calls == []
    assert provider._sync_thread is None


def test_query_budget_clears_slow_platform_cold_starts():
    """brv's Node/oclif cold start alone takes ~22s on low-power platforms (Termux); the old 10s
    desktop budget timed every query out before the CLI even booted (#139954)."""
    assert ByteRoverMemoryProvider({})._query_timeout_s >= 90


def test_query_budget_config_tuning():
    """``memory.byterover.query_timeout_s`` retunes the budget; only positive ints are honoured."""
    assert ByteRoverMemoryProvider({"query_timeout_s": 45})._query_timeout_s == 45
    assert ByteRoverMemoryProvider({"query_timeout_s": "not-a-number"})._query_timeout_s == 90
    assert ByteRoverMemoryProvider({"query_timeout_s": 0})._query_timeout_s == 90


def test_brv_runs_use_the_query_budget(monkeypatch):
    """brv query/status share brv's cold-start cost, so both must run under the configured query
    budget — the status call previously hardcoded a 15s desktop budget (#139954)."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(byterover, "_resolve_brv_path", lambda: "/fake/brv")
    monkeypatch.setattr(byterover, "_brv_child_env", lambda brv_path: {})
    monkeypatch.setattr(byterover.subprocess, "run", fake_run)

    provider = ByteRoverMemoryProvider({"query_timeout_s": 77})
    provider._cwd = "."
    provider._query("anything at all")
    assert seen["timeout"] == 77
    byterover._TOOLS["brv_status"][2](provider, None)
    assert seen["timeout"] == 77
