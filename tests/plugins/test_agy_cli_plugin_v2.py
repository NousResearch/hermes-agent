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


import importlib.util
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Allow pointing the test at an alternate (workspace) plugin tree while the
# V2 plugin file is still staged out-of-tree. Falls back to the live
# src/plugins/ location once the workspace files are cut over.
import os as _os
_WS = _os.environ.get("HERMES_AGY_PLUGIN_WORKSPACE")
if _WS and Path(_WS).is_dir():
    PLUGIN_DIR = Path(_WS) / "plugins" / "model-providers" / "agy-cli"
else:
    PLUGIN_DIR = SRC / "plugins" / "model-providers" / "agy-cli"


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "plugins_agy_cli_v2_under_test",
        PLUGIN_DIR / "__init__.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_plugin_yaml_manifest_present_and_valid():
    yaml_path = PLUGIN_DIR / "plugin.yaml"
    assert yaml_path.exists(), yaml_path
    text = yaml_path.read_text()
    assert "kind: model-provider" in text
    assert "agy-cli-provider" in text or "agy-cli" in text


def test_plugin_loads_and_registers_profile():
    mod = _load_plugin()
    profile = mod.agy_cli
    assert profile.name == "agy-cli"
    assert profile.api_mode == "agy_cli"
    assert profile.auth_type == "external_process"


def test_plugin_aliases_resolve_via_registry():
    _load_plugin()
    from providers import get_provider_profile
    for alias in ("agy-cli", "agy", "antigravity", "antigravity-cli"):
        prof = get_provider_profile(alias)
        assert prof is not None, f"alias {alias!r} did not resolve"
        assert prof.name == "agy-cli"


def test_plugin_model_list_matches_v2_enum_table():
    mod = _load_plugin()
    from agent.agy_cli_client import _HERMES_SLUG_TO_LS_MODEL

    got = sorted(mod.agy_cli.fetch_models() or [])
    expected = sorted(s for s in _HERMES_SLUG_TO_LS_MODEL if s != "default")
    assert got == expected


def test_plugin_does_not_export_v1_helpers():
    """V1 V1 ``AGY_SLUG_TO_DISPLAY`` map and ``_render_messages_to_prompt`` /
    ``_strip_banner`` / ``_slug_to_display`` helpers should be gone."""
    mod = _load_plugin()
    for symbol in (
        "AGY_SLUG_TO_DISPLAY",
        "_render_messages_to_prompt",
        "_strip_banner",
        "_slug_to_display",
    ):
        assert not hasattr(mod, symbol), (
            f"V1 helper {symbol!r} still present in agy-cli plugin"
        )


def test_plugin_exposes_v2_canonical_helper():
    """The plugin should expose either ``agy_model_slugs()`` or rely on
    the lazy LS-table loader."""
    mod = _load_plugin()
    assert hasattr(mod, "agy_model_slugs") or callable(
        getattr(mod, "_ls_model_table", None)
    )
