"""Tests for acp_adapter.entry startup wiring."""

import sys

import acp
import pytest

from acp_adapter import entry


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {}

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_main_skips_configured_mcp_discovery_when_requested(monkeypatch):
    discovery_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "tools.mcp_tool.discover_mcp_tools",
        lambda: discovery_calls.append(True),
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert discovery_calls == []


def test_main_eagerly_imports_memory_provider_before_other_threads(monkeypatch):
    """Regression test for #58083's real root cause on Windows.

    `session/new` deadlocked indefinitely because the configured memory
    provider's first import (e.g. mnemosyne -> numpy) ran inside
    `asyncio.to_thread()`'s worker thread at the same time another thread was
    starting up (MCP discovery, ACP's stdin-reader thread). Confirmed via
    `py-spy dump`: the worker thread got stuck forever at the exact same
    `import numpy` frame; pre-importing the provider on the main thread
    before any other thread exists made the hang disappear (a subsequent
    `import` from a worker thread is then just a cheap `sys.modules` cache
    hit, no fresh native-extension load, so there's nothing left to race).

    This test only pins the *ordering* contract that fixes the race: the
    warm-up import must run before the MCP discovery thread is started and
    before `HermesACPAgent()`/`acp.run_agent()` do anything that could spawn
    the next thread. It intentionally does not attempt to reproduce the
    underlying OS-level threading deadlock itself (not reliably
    reproducible in a fast, deterministic unit test).
    """
    call_order = []

    async def fake_run_agent(agent, **kwargs):
        call_order.append(("acp.run_agent", kwargs.get("use_unstable_protocol")))

    def fake_load_config():
        return {"memory": {"provider": "mnemosyne"}}

    def fake_warmup(name):
        call_order.append(("warmup_import_memory_provider_module", name))
        return None

    def fake_start_background_mcp_discovery(*, logger, thread_name):
        call_order.append(("start_background_mcp_discovery",))

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(entry.sys, "platform", "win32")
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "")
    monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)
    monkeypatch.setattr(
        "plugins.memory.warmup_import_memory_provider_module", fake_warmup
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        fake_start_background_mcp_discovery,
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert call_order == [
        ("warmup_import_memory_provider_module", "mnemosyne"),
        ("start_background_mcp_discovery",),
        ("acp.run_agent", True),
    ], (
        "The memory provider warm-up import must run BEFORE MCP discovery "
        "starts its background thread — reordering this reopens the "
        "Windows import-lock/loader-lock deadlock from #58083."
    )


def test_main_skips_memory_provider_warmup_when_no_provider_configured(monkeypatch):
    """No warm-up import (and no crash) when memory.provider is unset."""
    warmup_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(entry.sys, "platform", "win32")
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(
        "plugins.memory.warmup_import_memory_provider_module",
        lambda name: warmup_calls.append(name),
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert warmup_calls == []


def test_main_skips_memory_provider_warmup_on_non_windows(monkeypatch):
    """Warm-up is Windows-only; other platforms stay lazy."""
    warmup_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(entry.sys, "platform", "linux")
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "mnemosyne"}},
    )
    monkeypatch.setattr(
        "plugins.memory.warmup_import_memory_provider_module",
        lambda name: warmup_calls.append(name),
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert warmup_calls == []


def test_main_continues_when_warmup_import_fails(monkeypatch):
    """Warm-up errors are swallowed so startup proceeds."""
    call_order = []

    async def fake_run_agent(agent, **kwargs):
        call_order.append(("acp.run_agent", kwargs.get("use_unstable_protocol")))

    def fake_load_config():
        return {"memory": {"provider": "mnemosyne"}}

    def failing_warmup(name):
        call_order.append(("warmup_import_memory_provider_module", name))
        raise RuntimeError("boom")

    def fake_start_background_mcp_discovery(*, logger, thread_name):
        call_order.append(("start_background_mcp_discovery",))

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(entry.sys, "platform", "win32")
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "")
    monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)
    monkeypatch.setattr(
        "plugins.memory.warmup_import_memory_provider_module", failing_warmup
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        fake_start_background_mcp_discovery,
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert call_order == [
        ("warmup_import_memory_provider_module", "mnemosyne"),
        ("start_background_mcp_discovery",),
        ("acp.run_agent", True),
    ]










def test_main_setup_offers_browser_install_when_tty(monkeypatch):
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    bootstrap_calls = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: bootstrap_calls.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert bootstrap_calls == [False]










def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    def fake_ensure(dep, interactive=True):
        return dep != "browser"  # browser fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1
