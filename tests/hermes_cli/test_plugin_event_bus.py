"""Tests for the inter-plugin event bus (PluginContext.emit / subscribe).

Covers:
  - Two plugins communicate via emit/subscribe; emit returns listener count
  - Namespace is FORCED to the emitting plugin's own key
  - Namespace spoofing (hermes:, foreign, already-colon'd) is rejected
  - Per-callback isolation: one raising subscriber does not break the rest
  - Recursion cap: mutually-emitting plugins terminate + warn
  - Manifest emits/listens parsed as optional advisory fields
  - `hermes plugins show` output includes emits/listens
"""

from __future__ import annotations

import logging

import pytest

from hermes_cli.plugins import (
    _EVENT_EMIT_DEPTH_CAP,
    PluginContext,
    PluginManager,
    PluginManifest,
    get_plugin_subscriptions,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_ctx(manager: PluginManager, name: str, key: str = "") -> PluginContext:
    """Build a PluginContext for *name* wired to *manager*."""
    manifest = PluginManifest(name=name, key=key)
    return PluginContext(manifest, manager)


def _fresh_manager() -> PluginManager:
    manager = PluginManager()
    manager._discovered = True  # skip auto-discovery
    return manager


# ── 1. Two plugins communicate ───────────────────────────────────────────────


def test_two_plugins_communicate():
    manager = _fresh_manager()
    ctx_a = _make_ctx(manager, "plugin_a", key="a")
    ctx_b = _make_ctx(manager, "plugin_b", key="b")

    received = []

    def on_ping(**payload):
        received.append(payload)

    # A subscribes to b:ping; B emits the bare name "ping".
    ctx_a.subscribe("b:ping", on_ping)
    count = ctx_b.emit("ping", {"n": 42})

    assert count == 1  # one listener invoked
    assert received == [{"n": 42}]


def test_emit_with_no_subscribers_returns_zero():
    manager = _fresh_manager()
    ctx_b = _make_ctx(manager, "plugin_b", key="b")
    assert ctx_b.emit("ping", {"x": 1}) == 0


def test_emit_none_payload_delivers_empty_kwargs():
    manager = _fresh_manager()
    ctx_a = _make_ctx(manager, "plugin_a", key="a")
    ctx_b = _make_ctx(manager, "plugin_b", key="b")

    seen = []
    ctx_a.subscribe("b:ping", lambda **p: seen.append(p))
    count = ctx_b.emit("ping")  # payload omitted

    assert count == 1
    assert seen == [{}]


# ── 2. Namespace is forced to the emitter's own key ──────────────────────────


def test_namespace_forced_to_emitter_key():
    manager = _fresh_manager()
    ctx_a = _make_ctx(manager, "plugin_a", key="a")
    ctx_b = _make_ctx(manager, "plugin_b", key="b")

    delivered_events = []

    # Subscribe to what we expect the fully-qualified name to be.
    ctx_a.subscribe("b:ping", lambda **p: delivered_events.append("b:ping"))
    # A wrong-namespace subscription must NOT fire.
    ctx_a.subscribe("a:ping", lambda **p: delivered_events.append("a:ping"))

    ctx_b.emit("ping")

    # Delivered under the emitter's own key ("b"), never "a".
    assert delivered_events == ["b:ping"]
    # Registry stores it under the forced full name.
    assert "b:ping" in manager._subscriptions


def test_namespace_falls_back_to_name_when_key_empty():
    manager = _fresh_manager()
    # No key → namespace derives from name.
    ctx = _make_ctx(manager, "plugin_named", key="")
    got = []
    ctx.subscribe("plugin_named:evt", lambda **p: got.append(p))
    count = ctx.emit("evt", {"v": 1})
    assert count == 1
    assert got == [{"v": 1}]


# ── 3. Namespace spoofing is rejected (fail-closed) ──────────────────────────


@pytest.mark.parametrize(
    "bad_event",
    [
        "hermes:x",   # reserved core prefix
        "a:x",        # foreign namespace
        "b:x",        # even the plugin's own colon'd name — must pass bare only
        "other:evt",
        ":x",
        "x:",
    ],
)
def test_emit_rejects_namespaced_names(bad_event):
    manager = _fresh_manager()
    ctx_b = _make_ctx(manager, "plugin_b", key="b")

    fired = []
    # Subscribe to every plausible delivery target so we can prove no delivery.
    for name in (bad_event, f"b:{bad_event}", "hermes:x", "a:x", "b:x"):
        ctx_b.subscribe(name, lambda **p: fired.append(name))

    with pytest.raises(ValueError):
        ctx_b.emit(bad_event)

    assert fired == []  # nothing delivered


def test_emit_rejects_empty_event():
    manager = _fresh_manager()
    ctx_b = _make_ctx(manager, "plugin_b", key="b")
    with pytest.raises(ValueError):
        ctx_b.emit("")


def test_subscribe_is_unrestricted():
    """Any plugin may subscribe to any event, including hermes: and foreign."""
    manager = _fresh_manager()
    ctx_a = _make_ctx(manager, "plugin_a", key="a")
    got = []
    # None of these raise — only emit is namespace-gated.
    ctx_a.subscribe("hermes:core_event", lambda **p: got.append("hermes"))
    ctx_a.subscribe("b:ping", lambda **p: got.append("b"))
    assert "hermes:core_event" in manager._subscriptions
    assert "b:ping" in manager._subscriptions


# ── 4. Per-callback isolation ────────────────────────────────────────────────


def test_per_callback_isolation(caplog):
    manager = _fresh_manager()
    ctx_a = _make_ctx(manager, "plugin_a", key="a")
    ctx_b = _make_ctx(manager, "plugin_b", key="b")

    received = []

    def boom(**payload):
        raise RuntimeError("subscriber exploded")

    def good(**payload):
        received.append(payload)

    # Registration order: raising subscriber first, healthy one second.
    ctx_a.subscribe("b:ping", boom)
    ctx_a.subscribe("b:ping", good)

    with caplog.at_level(logging.WARNING):
        count = ctx_b.emit("ping", {"ok": True})

    # Both listeners were invoked despite the first raising.
    assert count == 2
    assert received == [{"ok": True}]
    assert any("subscriber exploded" in r.message or "raised" in r.message
               for r in caplog.records)


# ── 5. Recursion cap ─────────────────────────────────────────────────────────


def test_recursion_cap_terminates(caplog):
    manager = _fresh_manager()
    ctx_a = _make_ctx(manager, "plugin_a", key="a")
    ctx_b = _make_ctx(manager, "plugin_b", key="b")

    calls = {"a": 0, "b": 0}

    # A hears b:ping and re-emits a:ping; B hears a:ping and re-emits b:ping.
    def a_on_bping(**payload):
        calls["a"] += 1
        ctx_a.emit("ping")

    def b_on_aping(**payload):
        calls["b"] += 1
        ctx_b.emit("ping")

    ctx_a.subscribe("b:ping", a_on_bping)
    ctx_b.subscribe("a:ping", b_on_aping)

    with caplog.at_level(logging.WARNING):
        # Kick off the loop — must terminate, not hang or RecursionError.
        result = ctx_b.emit("ping")

    # Returned cleanly.
    assert result == 1
    # Bounded by the depth cap — nowhere near unbounded.
    assert calls["a"] + calls["b"] <= _EVENT_EMIT_DEPTH_CAP + 1
    # Exactly the recursion-cap warning fired.
    assert any("recursion cap" in r.message.lower() for r in caplog.records)


# ── 6. Manifest emits/listens parsed as optional ─────────────────────────────


def test_manifest_emits_listens_default_empty():
    m = PluginManifest(name="plain")
    assert m.emits == []
    assert m.listens == []


def test_manifest_emits_listens_present():
    m = PluginManifest(
        name="declar",
        key="declar",
        emits=["ping", "pong"],
        listens=["other:ready"],
    )
    assert m.emits == ["ping", "pong"]
    assert m.listens == ["other:ready"]


def test_manifest_parse_reads_emits_listens(tmp_path):
    """_parse_manifest picks up optional emits/listens from plugin.yaml."""
    import yaml

    plugin_dir = tmp_path / "myplug"
    plugin_dir.mkdir()
    manifest_file = plugin_dir / "plugin.yaml"
    manifest_file.write_text(
        yaml.safe_dump(
            {
                "name": "myplug",
                "emits": ["ping"],
                "listens": ["other:evt"],
            }
        ),
        encoding="utf-8",
    )

    manager = _fresh_manager()
    manifest = manager._parse_manifest(manifest_file, plugin_dir, "user", "")
    assert manifest is not None
    assert manifest.emits == ["ping"]
    assert manifest.listens == ["other:evt"]


def test_manifest_parse_absent_emits_listens(tmp_path):
    import yaml

    plugin_dir = tmp_path / "bare"
    plugin_dir.mkdir()
    manifest_file = plugin_dir / "plugin.yaml"
    manifest_file.write_text(
        yaml.safe_dump({"name": "bare"}), encoding="utf-8"
    )

    manager = _fresh_manager()
    manifest = manager._parse_manifest(manifest_file, plugin_dir, "user", "")
    assert manifest is not None
    assert manifest.emits == []
    assert manifest.listens == []


# ── Module-level accessor ────────────────────────────────────────────────────


def test_get_plugin_subscriptions_accessor(monkeypatch):
    from hermes_cli import plugins as plugins_mod

    fresh = _fresh_manager()
    monkeypatch.setattr(plugins_mod, "_ensure_plugins_discovered", lambda force=False: fresh)

    ctx = _make_ctx(fresh, "plugin_a", key="a")
    ctx.subscribe("b:ping", lambda **p: None)

    subs = get_plugin_subscriptions()
    assert "b:ping" in subs
    assert len(subs["b:ping"]) == 1


# ── 7. plugins show output includes emits/listens ────────────────────────────


def test_plugins_show_includes_emits_listens(tmp_path, monkeypatch, capsys):
    import yaml
    from hermes_cli import plugins_cmd

    plugin_dir = tmp_path / "showplug"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "showplug",
                "version": "1.2.3",
                "description": "a demo plugin",
                "emits": ["ping", "pong"],
                "listens": ["other:ready"],
            }
        ),
        encoding="utf-8",
    )

    # entry = (name, version, description, source, dir_path, key)
    entry = ("showplug", "1.2.3", "a demo plugin", "user", str(plugin_dir), "showplug")
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: [entry])
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())

    plugins_cmd.cmd_show("showplug")

    out = capsys.readouterr().out
    assert "showplug" in out
    assert "Emits:" in out
    assert "ping" in out
    assert "pong" in out
    assert "Listens:" in out
    assert "other:ready" in out


def test_plugins_show_not_found_exits(monkeypatch, capsys):
    from hermes_cli import plugins_cmd

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: [])
    with pytest.raises(SystemExit):
        plugins_cmd.cmd_show("nope")
    out = capsys.readouterr().out
    assert "not found" in out.lower()
