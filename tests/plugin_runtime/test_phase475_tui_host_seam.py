"""Phase 4.7 step 5: TUI refresh crosses the runtime-owned host seam."""

from __future__ import annotations

import ast
import threading
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_activation_has_no_tui_gateway_backedge() -> None:
    path = ROOT / "plugin_runtime" / "activation.py"
    source = path.read_text(encoding="utf-8")
    imports = _imports(path)

    assert all(not module.startswith("tui_gateway") for module in imports)
    assert "tui_gateway.server" not in source
    assert "sys.modules" not in source
    assert "refresh_tui_plugin_sessions" in source


def test_lifecycle_host_seam_has_no_tui_gateway_dependency() -> None:
    imports = _imports(ROOT / "plugin_runtime" / "lifecycle.py")
    assert all(not module.startswith("tui_gateway") for module in imports)


def test_tui_host_publishes_through_runtime_lifecycle() -> None:
    path = ROOT / "tui_gateway" / "plugin_inject.py"
    source = path.read_text(encoding="utf-8")
    assert "from plugin_runtime.lifecycle import get_plugin_manager, publish_tui_message_host" in source
    assert "from plugin_runtime.lifecycle import clear_published_tui_message_host, get_plugin_manager" in source
    assert "from hermes_cli.plugins import" not in source


def test_published_tui_host_refreshes_active_profile(monkeypatch, tmp_path) -> None:
    import plugin_runtime.lifecycle as lifecycle

    owner = object()
    calls = []
    monkeypatch.setattr(lifecycle, "get_hermes_home", lambda: tmp_path)

    lifecycle.publish_tui_message_host(
        owner,
        lambda **_kwargs: True,
        lambda home, note: calls.append((home, note)),
    )
    try:
        assert lifecycle.refresh_tui_plugin_sessions("plugin live") is True
        assert calls == [(Path(tmp_path), "plugin live")]
    finally:
        lifecycle.clear_published_tui_message_host(owner)

    assert lifecycle.refresh_tui_plugin_sessions("after clear") is False


def test_activation_refreshes_through_lifecycle_host(monkeypatch) -> None:
    import plugin_runtime.activation as activation
    import plugin_runtime.activation_live as activation_live
    import plugin_runtime.lifecycle as lifecycle

    loaded = SimpleNamespace(
        manifest=SimpleNamespace(name="demo", provides_tools=[]),
        deferred=False,
        error=None,
        tools_registered=[],
    )

    class Manager:
        _discovery_lock = threading.RLock()
        _plugins = {"demo": loaded}
        _ownership_ledger = {"demo": []}
        _platform_handler_factories = {}
        _portable_mcp_server_plugins = {}

        def discover_and_load(self, force: bool = False) -> None:
            assert force is True

        def get_portable_mcp_servers(self) -> dict:
            return {}

    refreshed = []
    monkeypatch.setattr(lifecycle, "join_background_discovery", lambda: None)
    monkeypatch.setattr(lifecycle, "get_plugin_manager", lambda: Manager())
    monkeypatch.setattr(lifecycle, "refresh_tui_plugin_sessions", refreshed.append)
    monkeypatch.setattr(activation_live, "connect_plugin_mcp", lambda _activation, _portable: [])
    monkeypatch.setattr(
        activation_live,
        "plugin_skills",
        lambda _key: [{"name": "demo:skill", "description": "Demo"}],
    )
    monkeypatch.setattr(activation_live, "live_notice", lambda _activation: "plugin live")

    result = activation._go_live("demo")

    assert result is not None
    assert result["live_now"]["skills"] == [{"name": "demo:skill", "description": "Demo"}]
    assert refreshed == ["plugin live"]
