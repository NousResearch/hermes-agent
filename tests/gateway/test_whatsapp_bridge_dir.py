"""WhatsApp's part of the sidecar resolver.

The rungs are covered by tests/gateway/test_sidecar_runtime.py. What matters
here: the bridge resolves through the shared resolver, under its own name, so
it gains the staleness check and the refresh that its own resolver never had.
"""
from __future__ import annotations

from pathlib import Path

from gateway.platforms.whatsapp_common import (
    SOURCE_BRIDGE_DIR,
    resolve_whatsapp_bridge_dir,
)


def test_override_is_honoured(tmp_path, monkeypatch) -> None:
    """An operator can point the gateway at a bridge checkout of their own."""
    override = tmp_path / "bridge"
    override.mkdir()
    monkeypatch.setenv("HERMES_WHATSAPP_SIDECAR_DIR", str(override))
    assert resolve_whatsapp_bridge_dir() == override


def test_it_resolves_under_the_whatsapp_name(tmp_path, monkeypatch) -> None:
    """The name picks the mirror directory and the override variable.

    A collision with another sidecar's name would put two sidecars in one
    directory.
    """
    seen = {}

    def _record(name, source):
        seen["name"] = name
        seen["source"] = source
        return source

    monkeypatch.setattr(
        "gateway.sidecar_runtime.resolve_sidecar", _record, raising=True
    )
    resolve_whatsapp_bridge_dir()
    assert seen["name"] == "whatsapp"
    assert seen["source"] == SOURCE_BRIDGE_DIR


def test_the_source_dir_holds_the_bridge(monkeypatch) -> None:
    """SOURCE_BRIDGE_DIR must point at the shipped bridge, not near it."""
    assert SOURCE_BRIDGE_DIR.name == "whatsapp-bridge"
    assert (SOURCE_BRIDGE_DIR / "package.json").is_file()
    assert (SOURCE_BRIDGE_DIR / "bridge.js").is_file()


def test_a_writable_checkout_runs_in_place(monkeypatch) -> None:
    """A source install must keep running the bridge where it ships."""
    monkeypatch.delenv("HERMES_WHATSAPP_SIDECAR_DIR", raising=False)
    monkeypatch.setattr(
        "gateway.sidecar_runtime.dir_writable", lambda p: True
    )
    assert resolve_whatsapp_bridge_dir() == SOURCE_BRIDGE_DIR


def test_a_readonly_tree_moves_to_hermes_home(tmp_path, monkeypatch) -> None:
    """A read-only install tree must not be where npm runs (#49561).

    In the container image /opt/hermes/scripts/whatsapp-bridge is read-only,
    so an install there fails with EACCES. The bridge moves to HERMES_HOME,
    which is writable.

    The bridge also gains a staleness check that its own resolver lacked:
    the old code returned any existing mirror without comparing it against
    the lockfile, so a bridge upgrade kept running the old node_modules.
    """
    monkeypatch.delenv("HERMES_WHATSAPP_SIDECAR_DIR", raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "gateway.sidecar_runtime.dir_writable", lambda p: False
    )
    resolved = resolve_whatsapp_bridge_dir()
    assert resolved == home / "sidecars" / "whatsapp"
    assert (resolved / "bridge.js").is_file()
    assert not (resolved / "node_modules").exists()


def test_the_bridge_source_is_copied_whole(tmp_path, monkeypatch) -> None:
    """Every file the bridge ships must reach the mirror.

    bridge.js imports allowlist.js and other siblings, and Node's ESM
    resolver reads them from beside the entry file. A partial copy fails at
    import, on read-only installs only.
    """
    monkeypatch.delenv("HERMES_WHATSAPP_SIDECAR_DIR", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "gateway.sidecar_runtime.dir_writable", lambda p: False
    )
    mirror = resolve_whatsapp_bridge_dir()

    want = {
        p.relative_to(SOURCE_BRIDGE_DIR)
        for p in SOURCE_BRIDGE_DIR.rglob("*")
        if p.is_file() and "node_modules" not in p.parts
    }
    got = {
        p.relative_to(mirror)
        for p in mirror.rglob("*")
        if p.is_file() and "node_modules" not in p.parts
    }
    assert want, "no bridge sources found — the fixture is wrong, not the code"
    assert want <= got, f"missing from the mirror: {sorted(want - got)}"
