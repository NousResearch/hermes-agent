"""End-to-end blackbox tests through the REAL plugin loader.

The unit/seam tests call the plugin's functions directly. That hid a wiring
bug: the package-level ``register()`` only registered hooks and never
delegated to ``commands.register()``, so ``/cost`` would not exist in a live
gateway despite every unit test passing.

These tests drive the actual ``PluginManager.discover_and_load()`` →
``invoke_hook()`` → registered-command-handler path, with a real SQLite store
on a temp ``HERMES_HOME``. No direct calls into ``plugins.blackbox`` internals
and no mock store. This is the closest thing to "boot a gateway and run a
turn" we can do in-process, and it is the layer that catches registration and
contract drift.
"""

from __future__ import annotations

import importlib
import sys

import pytest
import yaml


@pytest.fixture
def home(tmp_path, monkeypatch):
    hh = tmp_path / ".hermes"
    hh.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hh))
    # Reload modules whose per-profile paths are derived from HERMES_HOME.
    import hermes_constants
    importlib.reload(hermes_constants)
    import plugins.blackbox.store as store
    importlib.reload(store)
    # The package itself caches nothing env-dependent, but reload for parity
    # with how a fresh process would import it.
    import plugins.blackbox as bb
    importlib.reload(bb)
    return hh, store


def _enable(hh, **over):
    cfg = {
        "plugins": {"enabled": ["blackbox"]},
        "blackbox": {
            "enabled": True,
            "cost_alert_threshold_usd": 999.0,  # high → no alert send during E2E
            "store_text": True,
            "record_subagents": True,
        },
    }
    cfg["blackbox"].update(over)
    (hh / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _fresh_manager():
    from hermes_cli import plugins as pmod
    return pmod.PluginManager()


# ---------------------------------------------------------------------------
# Registration: the bug the unit tests missed.
# ---------------------------------------------------------------------------

def test_blackbox_registers_hooks_and_cost_command(home):
    hh, _store = home
    _enable(hh)
    mgr = _fresh_manager()
    mgr.discover_and_load()

    assert "blackbox" in mgr._plugins, "blackbox not discovered"
    loaded = mgr._plugins["blackbox"]
    assert loaded.enabled, f"blackbox failed to enable: {loaded.error!r}"

    # All three lifecycle hooks must be wired.
    for hook in ("on_session_start", "post_tool_call", "on_session_end"):
        assert hook in loaded.hooks_registered, f"{hook} not registered"

    # The regression guard: /cost MUST be registered through the loader.
    assert "cost" in loaded.commands_registered, "/cost command not registered"
    assert "cost" in mgr._plugin_commands
    assert callable(mgr._plugin_commands["cost"]["handler"])


def test_blackbox_not_loaded_without_opt_in(home):
    hh, _store = home
    # blackbox table present + enabled, but NOT in plugins.enabled allow-list.
    (hh / "config.yaml").write_text(
        yaml.safe_dump({"blackbox": {"enabled": True}}), encoding="utf-8"
    )
    mgr = _fresh_manager()
    mgr.discover_and_load()
    loaded = mgr._plugins.get("blackbox")
    assert loaded is not None, "blackbox should still be discovered"
    assert not loaded.enabled, "blackbox must not load without plugins.enabled opt-in"


# ---------------------------------------------------------------------------
# Full lifecycle through the manager: hooks fire → store persists → /cost reads
# ---------------------------------------------------------------------------

def _usage():
    return {
        "api_calls": 2,
        "input_tokens": 500_000, "output_tokens": 559,
        "cache_read_tokens": 499_000, "cache_write_tokens": 1000,
        "reasoning_tokens": 10, "latency_s": 23.0,
        "calls": [
            {"input_tokens": 250_000, "output_tokens": 300},
            {"input_tokens": 250_000, "output_tokens": 259},
        ],
        "context_used": 500_000, "context_length": 1_050_000,
        "is_subagent": False,
    }


def test_full_turn_lifecycle_through_loader(home, monkeypatch):
    hh, store = home
    _enable(hh)
    mgr = _fresh_manager()
    mgr.discover_and_load()

    # The loader imports the plugin as ``hermes_plugins.blackbox`` — a DIFFERENT
    # module object than ``plugins.blackbox``. Patch the instance the loader
    # actually wired the hooks from, or the patches no-op.
    bb = sys.modules["hermes_plugins.blackbox"]
    monkeypatch.setattr(bb, "compute_turn_cost", lambda *a, **k: (1.26, "estimated", {"uncached":0.5,"cache_read":0.5,"cache_write":0.13,"output":0.13}))
    monkeypatch.setattr(bb, "_turn_id", lambda: "turn_e2e")

    # Drive the lifecycle entirely through the manager's hook dispatch — the
    # same call path the core agent loop uses.
    mgr.invoke_hook("on_session_start", session_id="sess-1")
    mgr.invoke_hook(
        "post_tool_call", session_id="sess-1", tool_name="terminal",
        args={"command": "ls"}, result="a\nb\n",
    )
    mgr.invoke_hook(
        "post_tool_call", session_id="sess-1", tool_name="terminal",
        args={"command": "pwd"}, result="/tmp",
    )
    mgr.invoke_hook(
        "post_tool_call", session_id="sess-1", tool_name="read_file",
        args={"path": "/etc/hosts"}, result="127.0.0.1",
    )
    mgr.invoke_hook(
        "on_session_end", session_id="sess-1", completed=True, interrupted=False,
        model="openai-codex/gpt-5.4", platform="discord", provider="openai-codex",
        chat_id="C9", chat_name="ops",
        user_message="run ls and pwd", final_response="done",
        turn_usage=_usage(),
    )

    # A real row landed in the real DB.
    row = store.get_turn("turn_e2e")
    assert row is not None, "turn not persisted through the loader path"
    assert row["api_calls"] == 2
    assert row["input_tokens"] == 500_000
    assert row["cost_usd"] == pytest.approx(1.26)
    assert row["tools"] == ["terminal", "terminal", "read_file"]

    # Tool args/results landed in the side table (the dig-in source).
    calls = store.get_tool_calls("turn_e2e")
    assert [c["name"] for c in calls] == ["terminal", "terminal", "read_file"]
    assert "ls" in calls[0]["args_preview"]

    # /cost <id> dispatched through the REGISTERED command handler renders the
    # card + dig-in (no direct import of commands).
    handler = mgr._plugin_commands["cost"]["handler"]
    # The command resolves the channel from gateway.session_context; pin it so
    # the same-channel guard passes.
    import gateway.session_context as sc
    monkeypatch.setattr(sc, "get_session_env", lambda name, default="": {
        "HERMES_SESSION_PLATFORM": "discord",
        "HERMES_SESSION_CHAT_ID": "C9",
    }.get(name, default))

    out = handler("turn_e2e")
    assert "💸 Spending Alert" in out
    assert "Turn Cost: $1.26" in out
    assert "/cost turn_e2e" in out
    # Dig-in shows the tool args/results.
    assert "Tools:" in out
    assert "terminal" in out
    assert "ls" in out

    # /cost debug surfaces operational health through the registered handler.
    dbg = handler("debug")
    assert "🩺 blackbox debug" in dbg
    assert "Config: ENABLED" in dbg
    assert "Turns: 1" in dbg
    assert "tool_calls=3" in dbg
    assert "turns.db" in dbg


def test_disabled_config_makes_hooks_noop_through_loader(home, monkeypatch):
    hh, store = home
    # Loaded into the manager (plugins.enabled) but feature gate OFF.
    (hh / "config.yaml").write_text(
        yaml.safe_dump({
            "plugins": {"enabled": ["blackbox"]},
            "blackbox": {"enabled": False},
        }),
        encoding="utf-8",
    )
    mgr = _fresh_manager()
    mgr.discover_and_load()

    import plugins.blackbox as bb
    monkeypatch.setattr(bb, "_turn_id", lambda: "turn_off")

    mgr.invoke_hook("on_session_start", session_id="s")
    mgr.invoke_hook("post_tool_call", session_id="s", tool_name="terminal", args={}, result="x")
    mgr.invoke_hook(
        "on_session_end", session_id="s", model="m", platform="discord",
        provider="p", turn_usage=_usage(),
    )

    assert store.get_turn("turn_off") is None, "disabled gate must not persist"
    assert store.top_turns(5, 30) == []
