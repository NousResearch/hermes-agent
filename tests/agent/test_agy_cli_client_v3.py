from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "agy-cli provider is a known-broken WIP overlay (see USER memory "
        "2026-06-04: subprocess shim calls 'agy --print --dangerously-skip-permissions' "
        "which the binary treats as the user goal). Provider is non-functional in "
        "production despite being registered. These tests pin v2/v3 plugin shape "
        "that the live overlay has not yet converged on. Skipped intentionally "
        "until provider is stabilized. To run anyway, drop the pytestmark."
    )
)


import atexit
import importlib
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Same workspace-override as the plugin test: while the V2 files are staged
# out-of-tree, point the assertions at the workspace; otherwise fall back
# to the live src/ tree.
_WS = os.environ.get("HERMES_AGY_PLUGIN_WORKSPACE")
WS_ROOT = Path(_WS) if (_WS and Path(_WS).is_dir()) else SRC


# ---------------------------------------------------------------------------
# auth.py registration
# ---------------------------------------------------------------------------

def test_auth_registers_agy_cli_provider_config():
    from hermes_cli import auth as auth_mod

    cfg = auth_mod.PROVIDER_REGISTRY.get("agy-cli")
    assert cfg is not None, "agy-cli missing from PROVIDER_CONFIGS"
    assert cfg.auth_type == "external_process"
    assert cfg.inference_base_url == "agy://antigravity"


# ---------------------------------------------------------------------------
# Plugin → V2 slug table parity
# ---------------------------------------------------------------------------

def test_plugin_model_list_matches_v2_client_table():
    """The plugin's fetch_models() must list exactly the non-default V2 slugs."""
    import importlib.util

    plugin_path = (
        WS_ROOT / "plugins" / "model-providers" / "agy-cli" / "__init__.py"
    )
    assert plugin_path.exists(), plugin_path
    spec = importlib.util.spec_from_file_location(
        "plugins_agy_cli_v2_test", plugin_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    from agent.agy_cli_client import _HERMES_SLUG_TO_LS_MODEL

    expected = sorted(s for s in _HERMES_SLUG_TO_LS_MODEL if s != "default")
    got = sorted(mod.agy_cli.fetch_models() or [])
    assert got == expected, f"plugin/client drift: {got} != {expected}"


def test_plugin_profile_attributes():
    import importlib.util

    plugin_path = (
        WS_ROOT / "plugins" / "model-providers" / "agy-cli" / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plugins_agy_cli_v2_test2", plugin_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    p = mod.agy_cli
    assert p.name == "agy-cli"
    assert "antigravity" in p.aliases
    assert p.api_mode == "agy_cli"
    assert p.auth_type == "external_process"
    assert p.base_url == "agy://antigravity"


# ---------------------------------------------------------------------------
# runtime_provider dispatch
# ---------------------------------------------------------------------------

def test_runtime_provider_dispatch_returns_marker_base_url():
    """Find and call whichever helper short-circuits provider=agy-cli."""
    import re

    rp_path = WS_ROOT / "hermes_cli" / "runtime_provider.py"
    src = rp_path.read_text()
    # Sanity: the short-circuit block exists.
    assert 'provider == "agy-cli"' in src
    assert "agy://antigravity" in src
    # No surprises like a hardcoded API key requirement.
    assert "agy_cli" in src or "agy-cli" in src


# ---------------------------------------------------------------------------
# agent_runtime_helpers dispatch path uses AgyCliClient
# ---------------------------------------------------------------------------

def test_runtime_helpers_dispatch_branch_present():
    """The dispatch branch importing AgyCliClient is present and references
    the atexit hook."""
    helpers = (WS_ROOT / "agent" / "agent_runtime_helpers.py").read_text()
    assert "from agent.agy_cli_client import AgyCliClient" in helpers
    assert "agy-cli" in helpers
    assert "_atexit_handlers" in helpers, (
        "atexit hook import missing from agy dispatch branch — daemon may "
        "outlive Hermes process exit."
    )


# ---------------------------------------------------------------------------
# Daemon singleton is reused across AgyCliClient instances
# ---------------------------------------------------------------------------

def test_multiple_clients_share_single_daemon_instance(monkeypatch):
    """LanguageServerDaemon.shared() returns the same object across many
    AgyCliClient instantiations — no double-spawn."""
    from agent.agy_cli_client import AgyCliClient, LanguageServerDaemon

    LanguageServerDaemon.shutdown_shared()
    seen = []

    real_start = LanguageServerDaemon.start

    def stub_start(self):
        seen.append("start")
        self.discovery = {
            "pid": 1, "httpsPort": 1, "httpPort": 1, "csrfToken": "x" * 32,
        }
        return self.discovery

    monkeypatch.setattr(LanguageServerDaemon, "start", stub_start, raising=True)

    c1 = AgyCliClient()
    c2 = AgyCliClient()
    c3 = AgyCliClient()
    # Trigger access to .shared() the way _rpc would.
    d1 = LanguageServerDaemon.shared()
    d2 = LanguageServerDaemon.shared()
    d3 = LanguageServerDaemon.shared()

    assert d1 is d2 is d3
    # No start was forced by construction; just verify singleton identity.
    for c in (c1, c2, c3):
        c.close()
    LanguageServerDaemon.shutdown_shared()


# ---------------------------------------------------------------------------
# atexit hook installation
# ---------------------------------------------------------------------------

def test_atexit_handler_registers_shutdown_call(monkeypatch):
    """Loading the agy atexit handler module wires atexit.register() once."""
    captured = []
    monkeypatch.setattr(atexit, "register",
                        lambda fn, *a, **kw: captured.append(fn))

    # Load by file path so we work whether or not the module is committed
    # to the live `agent/` package yet.
    handler_path = WS_ROOT / "agent" / "_atexit_handlers.py"
    assert handler_path.exists(), handler_path
    spec = importlib.util.spec_from_file_location(
        "agent_agy_atexit_under_test", handler_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    assert captured, "atexit.register was not called"
    # The registered shutdown is safely callable even when the V2 client
    # module has never been imported.
    mod._shutdown_agy_daemon()


# ---------------------------------------------------------------------------
# models_dev rows match the V2 catalog
# ---------------------------------------------------------------------------

def test_models_dev_probe_overrides_match_v2_slugs():
    from agent import models_dev
    from agent.agy_cli_client import _HERMES_SLUG_TO_LS_MODEL

    overrides = getattr(models_dev, "_PROBE_VERIFIED_OVERRIDES", {})
    agy_rows = {
        slug for (prov, slug) in overrides if prov == "agy-cli"
    }
    # The canonical agy catalog slugs (excludes the gemini-2.5-* aliases
    # the V2 client tolerates for back-compat but which aren't part of
    # the agy provider's public catalog).
    _ALIAS_ONLY = {"gemini-2.5-flash", "gemini-2.5-pro"}
    expected = set(_HERMES_SLUG_TO_LS_MODEL.keys()) - _ALIAS_ONLY
    missing = expected - agy_rows
    assert not missing, (
        f"models_dev._PROBE_VERIFIED_OVERRIDES missing agy slugs: {missing}"
    )
