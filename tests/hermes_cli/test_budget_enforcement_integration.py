"""Integration tests for the ``on_budget_check`` enforcement contract.

Unlike ``test_budget_enforcement.py`` (which mocks ``invoke_hook`` to unit-test
the aggregation rules of ``get_budget_check_verdict`` in isolation), these tests
exercise the **real production extension path** end to end:

    plugin on disk under <HERMES_HOME>/plugins/<name>
      -> PluginManager.discover_and_load()   (real discovery)
      -> register(ctx) / ctx.register_hook("on_budget_check", cb)  (real wiring)
      -> get_budget_check_verdict(**context)  (real invoke_hook dispatch)

Driving the full agent loop from a unit test would be prohibitively heavy
(see ``test_transform_llm_output_hook.py`` for the same rationale), so the tests
call ``get_budget_check_verdict`` with the exact kwargs the core supplies at the
production call site (``agent/turn_context.py`` — ``session_id``, ``task_id``,
``turn_id``, ``platform``, ``sender_id``, ``model``).
"""

import json
import textwrap
from pathlib import Path

import yaml

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import (
    OBSERVER_SCHEMA_VERSION,
    PluginManager,
    budget_enforcement_bootstrap_notice,
    get_budget_check_verdict,
    has_hook,
)

# The kwargs the core passes to on_budget_check at the production call site
# (agent/turn_context.py). Kept here so the "API-only" assertion below is
# anchored to the real contract, not a guess.
_PRODUCTION_CONTEXT_KEYS = {
    "session_id",
    "task_id",
    "turn_id",
    "platform",
    "sender_id",
    "model",
}

# PluginManager.invoke_hook stamps every hook dispatch with the observer schema
# version (plugins.py — kwargs.setdefault("telemetry_schema_version", ...)), so a
# budget plugin sees this scalar in addition to the production context keys. It is
# still API-only (a plain version string), which the serialization check enforces.
_INJECTED_KEYS = {"telemetry_schema_version"}


def _write_plugin(hermes_home: Path, name: str, init_src: str) -> None:
    """Create an opted-in plugin under ``<hermes_home>/plugins/<name>``."""
    plugin_dir = hermes_home / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "0.1.0"}), encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        textwrap.dedent(init_src), encoding="utf-8",
    )
    cfg_path = hermes_home / "config.yaml"
    cfg = (yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}) or {}
    cfg.setdefault("plugins", {}).setdefault("enabled", []).append(name)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _reload_from(hermes_home: Path, monkeypatch) -> None:
    """Point HERMES_HOME at ``hermes_home`` and rebuild the global manager
    by real discovery, so module-level ``get_budget_check_verdict`` / ``has_hook``
    (which use the singleton) see the freshly discovered plugins."""
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    mgr = PluginManager()
    mgr.discover_and_load(force=True)
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)


def _production_context(**overrides):
    ctx = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "1",
        "platform": "cli",
        "sender_id": "u1",
        "model": "anthropic/claude-sonnet-4.6",
    }
    ctx.update(overrides)
    return ctx


def test_registered_plugin_verdict_flows_through_real_discovery(tmp_path, monkeypatch):
    """A real plugin's verdict reaches get_budget_check_verdict via disk
    discovery + ctx.register_hook — no mocks on the extension path."""
    home = tmp_path / "hermes_test"
    home.mkdir(exist_ok=True)
    _write_plugin(
        home,
        "budget_soft",
        """
        def register(ctx):
            ctx.register_hook(
                "on_budget_check",
                lambda **kw: {"status": "soft", "message": "over soft cap"},
            )
        """,
    )
    _reload_from(home, monkeypatch)

    assert has_hook("on_budget_check") is True
    verdict = get_budget_check_verdict(**_production_context())
    assert verdict == {"status": "soft", "message": "over soft cap"}


def test_context_passed_to_plugin_is_api_only(tmp_path, monkeypatch):
    """The context the core hands a budget plugin must be API-only: the six
    documented scalar kwargs, all JSON-serializable, with no internal AIAgent
    object leaking through.

    The plugin captures its received kwargs by JSON-serializing them to disk.
    If any non-serializable internal object were passed, ``json.dumps`` would
    raise inside the callback, ``invoke_hook`` would swallow it, and the capture
    file would never be written — so the file's existence + exact contents prove
    the context is API-only.
    """
    home = tmp_path / "hermes_test"
    home.mkdir(exist_ok=True)
    capture = home / "captured_context.json"
    _write_plugin(
        home,
        "budget_capture",
        f"""
        import json
        from pathlib import Path

        def register(ctx):
            def _cb(**kwargs):
                Path(r"{capture}").write_text(
                    json.dumps(kwargs, sort_keys=True), encoding="utf-8"
                )
                return {{"status": "soft", "message": "over"}}

            ctx.register_hook("on_budget_check", _cb)
        """,
    )
    _reload_from(home, monkeypatch)

    ctx = _production_context(session_id="cron_42_1700000000", turn_id="7")
    verdict = get_budget_check_verdict(**ctx)
    assert verdict == {"status": "soft", "message": "over"}

    assert capture.exists(), "hook was not invoked or context was not serializable"
    received = json.loads(capture.read_text(encoding="utf-8"))
    # Everything the core passed survives a JSON round-trip => API-only. The
    # plugin sees exactly the production context keys plus the auto-injected
    # telemetry schema version, and nothing else (no internal AIAgent object).
    assert set(received) == _PRODUCTION_CONTEXT_KEYS | _INJECTED_KEYS
    assert {k: received[k] for k in ctx} == ctx
    assert received["telemetry_schema_version"] == OBSERVER_SCHEMA_VERSION


def test_most_severe_verdict_wins_across_real_plugins(tmp_path, monkeypatch):
    """With two real plugins registered, the most-severe verdict wins through
    the real invoke_hook dispatch (not the mocked aggregation unit test)."""
    home = tmp_path / "hermes_test"
    home.mkdir(exist_ok=True)
    _write_plugin(
        home,
        "budget_soft",
        """
        def register(ctx):
            ctx.register_hook(
                "on_budget_check",
                lambda **kw: {"status": "soft", "message": "soft"},
            )
        """,
    )
    _write_plugin(
        home,
        "budget_hard",
        """
        def register(ctx):
            ctx.register_hook(
                "on_budget_check",
                lambda **kw: {"status": "hard", "message": "hard cap hit"},
            )
        """,
    )
    _reload_from(home, monkeypatch)

    verdict = get_budget_check_verdict(**_production_context())
    assert verdict == {"status": "hard", "message": "hard cap hit"}


def test_bootstrap_notice_suppressed_when_real_plugin_registered(tmp_path, monkeypatch):
    """The discoverability nudge must go silent once a real budget plugin
    registers on_budget_check."""
    home = tmp_path / "hermes_test"
    home.mkdir(exist_ok=True)
    _write_plugin(
        home,
        "budget_soft",
        """
        def register(ctx):
            ctx.register_hook(
                "on_budget_check",
                lambda **kw: {"status": "ok"},
            )
        """,
    )
    _reload_from(home, monkeypatch)

    assert has_hook("on_budget_check") is True
    assert budget_enforcement_bootstrap_notice() is None


def test_bootstrap_notice_present_when_no_budget_plugin(tmp_path, monkeypatch):
    """With no budget plugin on disk, the nudge is advertised."""
    home = tmp_path / "hermes_test"
    home.mkdir(exist_ok=True)
    _reload_from(home, monkeypatch)

    assert has_hook("on_budget_check") is False
    notice = budget_enforcement_bootstrap_notice()
    assert notice is not None
    assert "budget" in notice.lower()
