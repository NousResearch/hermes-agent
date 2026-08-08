"""Shell hooks must register on dashboard + TUI slash-worker surfaces.

CLI chat and ``gateway run`` already call ``register_from_config``. Dashboard
chat sessions (``hermes_cli/web_server._lifespan``) and TUI slash workers
(``tui_gateway/slash_worker._prepare_slash_worker_runtime``) run full agent
turns without those entry points, so declarative hooks were silently dead
on those surfaces (#64178 / #50776).

These tests assert the two call sites actually invoke registration (with
``accept_hooks=False``, matching gateway/run.py).
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import hermes_cli.web_server as web_server_mod


def test_dashboard_lifespan_registers_shell_hooks():
    """Dashboard/serve lifespan must call register_from_config(accept_hooks=False)."""
    from fastapi.testclient import TestClient

    cfg = {"hooks": {}}
    register = MagicMock(return_value=[])
    outbound = MagicMock(return_value=[])

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.shell_hooks.register_from_config", register),
        patch("agent.outbound_webhooks.register_from_config", outbound),
        patch.object(web_server_mod, "_warm_gateway_module", lambda: None),
    ):
        with TestClient(web_server_mod.app, raise_server_exceptions=False):
            pass

    register.assert_called_once_with(cfg, accept_hooks=False)
    outbound.assert_called_once_with(cfg)


def test_slash_worker_runtime_registers_shell_hooks():
    """Slash-worker runtime prep must register hooks before HermesCLI turns."""
    # slash_worker imports cli (prompt_toolkit-heavy) at module load; stub it
    # so this unit test stays dependency-light.
    stub_cli = types.ModuleType("cli")
    stub_cli.HermesCLI = object  # type: ignore[attr-defined]
    stub_hb = types.ModuleType("hermes_bootstrap")
    stub_hb.harden_import_path = lambda: None  # type: ignore[attr-defined]
    stub_rec = types.ModuleType("tui_gateway._stdin_recovery")
    stub_rec.handle_spurious_eof = lambda *a, **k: False  # type: ignore[attr-defined]
    stub_rich = types.ModuleType("rich")
    stub_console = types.ModuleType("rich.console")
    stub_console.Console = object  # type: ignore[attr-defined]

    cfg = {"hooks": {}}
    register = MagicMock(return_value=[])
    outbound = MagicMock(return_value=[])

    with patch.dict(
        sys.modules,
        {
            "cli": stub_cli,
            "hermes_bootstrap": stub_hb,
            "tui_gateway._stdin_recovery": stub_rec,
            "rich": stub_rich,
            "rich.console": stub_console,
        },
    ):
        # Force a clean import under stubs
        sys.modules.pop("tui_gateway.slash_worker", None)
        slash_worker = importlib.import_module("tui_gateway.slash_worker")

        with (
            patch(
                "hermes_cli.mcp_startup.start_background_mcp_discovery",
                MagicMock(),
            ),
            patch("hermes_cli.mcp_startup.wait_for_mcp_discovery", MagicMock()),
            patch("hermes_cli.config.load_config", return_value=cfg),
            patch("agent.shell_hooks.register_from_config", register),
            patch("agent.outbound_webhooks.register_from_config", outbound),
        ):
            slash_worker._prepare_slash_worker_runtime()

    register.assert_called_once_with(cfg, accept_hooks=False)
    outbound.assert_called_once_with(cfg)
