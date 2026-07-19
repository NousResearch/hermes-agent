"""Lifecycle coverage for ``hermes -z`` resource teardown."""

import sys
import types


def test_oneshot_cleanup_releases_global_and_agent_resources(monkeypatch):
    import hermes_cli.oneshot as oneshot_mod

    calls = []
    messages = [{"role": "user", "content": "hello"}]
    agent = types.SimpleNamespace(
        session_id="oneshot-session",
        platform="cli",
        _session_messages=messages,
        _memory_manager=types.SimpleNamespace(
            flush_pending=lambda timeout: calls.append(("memory_flush", timeout))
        ),
        shutdown_memory_provider=lambda received: calls.append(("memory", received)),
        close=lambda: calls.append(("agent", None)),
    )
    session_db = types.SimpleNamespace(close=lambda: calls.append(("session_db", None)))

    def module(name, **attrs):
        fake = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(fake, key, value)
        return fake

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        module(
            "hermes_cli.plugins",
            invoke_hook=lambda name, **kwargs: calls.append((name, kwargs)),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.terminal_tool",
        module(
            "tools.terminal_tool",
            cleanup_all_environments=lambda: calls.append(("terminal", None)),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.async_delegation",
        module(
            "tools.async_delegation",
            interrupt_all=lambda **kwargs: calls.append(("delegation", kwargs)),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.browser_tool",
        module(
            "tools.browser_tool",
            _emergency_cleanup_all_sessions=lambda: calls.append(("browser", None)),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        module(
            "tools.mcp_tool", shutdown_mcp_servers=lambda: calls.append(("mcp", None))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.auxiliary_client",
        module(
            "agent.auxiliary_client",
            shutdown_cached_clients=lambda: calls.append(("auxiliary", None)),
        ),
    )

    oneshot_mod._cleanup_oneshot_resources(agent, session_db)

    assert calls == [
        (
            "on_session_finalize",
            {
                "session_id": "oneshot-session",
                "platform": "cli",
                "reason": "shutdown",
            },
        ),
        ("terminal", None),
        ("delegation", {"reason": "oneshot shutdown"}),
        ("browser", None),
        ("mcp", None),
        ("auxiliary", None),
        ("memory_flush", 10),
        ("memory", messages),
        ("agent", None),
        ("session_db", None),
    ]
