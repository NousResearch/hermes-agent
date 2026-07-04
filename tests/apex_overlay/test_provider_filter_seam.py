"""Seam-test + behavior test for the apex_overlay hc-392 provider denylist.

This is the PILOT seam-test (see ``apex_overlay/README.md``). It pins the two
upstream symbols that ``apex_overlay.provider_filter`` monkey-patches so an
upstream rename/move turns a *silently disarmed guard* into a *loud CI failure*
— the prerequisite for trusting monkey-patch.

It also proves the behavior the original +34 in-place hc-392 lines provided:

* With the denylist active, GitHub Copilot is skipped **before** the live
  catalog fetch — ``fetch_github_model_catalog`` is called **zero** times and
  no ``copilot`` row appears.
* With an **empty** denylist (control), the exact same setup *does* fetch and
  surface Copilot — proving the suppression is the denylist's doing, not some
  unrelated reason.

Run via ``scripts/run_tests_parallel.py`` (per-file fresh interpreter), not a
single in-process pytest — R3 regressed precisely because a process-wide
monkey-patch behaved differently under single-process isolation.
"""

from __future__ import annotations

import inspect
import os
from unittest.mock import MagicMock, patch

import pytest

from apex_overlay import provider_filter


# ---------------------------------------------------------------------------
# Seam assertions — pin the patched symbols' existence + signature
# ---------------------------------------------------------------------------

def test_seam_target_cached_provider_model_ids_exists_with_compatible_signature():
    """apex_overlay patches hermes_cli.models.cached_provider_model_ids.

    If upstream renames/moves it or changes the first positional param away
    from ``provider``, our short-circuit (the 'no copilot fetch' guarantee)
    silently stops working. Fail loudly here instead.
    """
    from hermes_cli import models

    fn = getattr(models, provider_filter._TARGET_CACHED_FN, None)
    assert fn is not None, (
        "hermes_cli.models.cached_provider_model_ids is gone — apex_overlay "
        "provider denylist can no longer cut the copilot fetch. Update "
        "apex_overlay.provider_filter._TARGET_CACHED_FN and the wrapper."
    )
    params = list(inspect.signature(fn).parameters)
    assert params and params[0] == "provider", (
        f"cached_provider_model_ids first param changed to {params!r}; the "
        f"apex_overlay wrapper passes provider positionally."
    )


def test_seam_target_list_authenticated_providers_exists_with_compatible_signature():
    """apex_overlay patches hermes_cli.model_switch.list_authenticated_providers.

    The wrapper relies on the result being a list of row-dicts with a 'slug'
    key and on the call being forwardable via (*args, **kwargs). Pin the
    public keyword params the picker/prewarm depend on.
    """
    from hermes_cli import model_switch

    fn = getattr(model_switch, provider_filter._TARGET_LIST_FN, None)
    assert fn is not None, (
        "list_authenticated_providers is gone — apex_overlay can no longer "
        "drop disabled-provider rows. Update _TARGET_LIST_FN and the wrapper."
    )
    params = inspect.signature(fn).parameters
    for expected in ("current_provider", "max_models"):
        assert expected in params, (
            f"list_authenticated_providers lost the {expected!r} param the "
            f"picker/prewarm call with; apex_overlay forwarding assumptions "
            f"are now stale."
        )


def test_apply_binds_both_targets_and_is_idempotent():
    """apply() must succeed (both targets bound) and be a safe no-op on repeat.

    A False return means a target was missing — which the seam asserts above
    would also catch, but apply() must surface it too so the plugin can warn.
    """
    from hermes_cli import model_switch, models

    # Reset the module guard so this test exercises a real (re)apply even if
    # an earlier test in this file already applied.
    provider_filter._APPLIED = False
    assert provider_filter.apply() is True
    # Patched callables carry our marker.
    assert getattr(models.cached_provider_model_ids, provider_filter._MARK, False)
    assert getattr(model_switch.list_authenticated_providers, provider_filter._MARK, False)
    # Idempotent: second apply is a no-op (and must not double-wrap).
    assert provider_filter.apply() is True
    assert getattr(models.cached_provider_model_ids, provider_filter._MARK, False)


# ---------------------------------------------------------------------------
# Behavior — denylist active: copilot skipped BEFORE fetch
# ---------------------------------------------------------------------------

@pytest.fixture()
def _denylist(monkeypatch):
    """Return a setter that injects model.disabled_providers for this test.

    The hermetic test config has no denylist (so the existing
    test_copilot_in_model_list passes). We patch apex_overlay's config reader
    directly — clean, and decoupled from on-disk config layout.
    """
    def _set(*providers: str):
        wanted = {p.strip().lower() for p in providers}
        monkeypatch.setattr(
            provider_filter, "disabled_provider_set", lambda: set(wanted)
        )
    return _set


def _apply_fresh():
    """(Re)install the seam onto the current (possibly re-imported) modules."""
    provider_filter._APPLIED = False
    assert provider_filter.apply() is True


@patch.dict(os.environ, {"GH_TOKEN": "test-key"}, clear=False)
def test_copilot_denylisted_makes_no_github_fetch_and_no_row(_denylist):
    """copilot in denylist → 0 calls to fetch_github_model_catalog, no row.

    GH_TOKEN is present so copilot *would* be credentialed and fetched without
    the guard. We assert the live GitHub catalog call never happens (the
    hc-392 'no startup network call' contract) and copilot is absent.
    """
    from hermes_cli import model_switch

    _denylist("copilot")
    _apply_fresh()

    fetch_mock = MagicMock(return_value=[{"id": "gpt-5.4"}])
    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models._resolve_copilot_catalog_api_key", return_value="gh-token"), \
         patch("hermes_cli.models.fetch_github_model_catalog", fetch_mock):
        providers = model_switch.list_authenticated_providers(
            current_provider="openrouter", max_models=50,
        )

    assert fetch_mock.call_count == 0, (
        "copilot is denylisted but its GitHub model catalog was still fetched "
        "— the apex_overlay short-circuit on cached_provider_model_ids didn't "
        "fire before the live fetch."
    )
    slugs = {p["slug"] for p in providers}
    assert "copilot" not in slugs, f"denylisted copilot still in picker: {slugs}"


@patch.dict(os.environ, {"GH_TOKEN": "test-key"}, clear=False)
def test_empty_denylist_control_copilot_appears(_denylist):
    """Control: empty denylist + same setup → copilot DOES surface.

    Proves the suppression in the test above is caused by the denylist (the
    apex_overlay seam), not by an unrelated reason. Mirrors the upstream
    test_copilot_in_model_list contract.
    """
    from hermes_cli import model_switch

    _denylist()  # empty denylist
    _apply_fresh()

    live_models = ["gpt-5.4", "claude-sonnet-4.6"]
    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models._resolve_copilot_catalog_api_key", return_value="gh-token"), \
         patch("hermes_cli.models._fetch_github_models", return_value=live_models):
        providers = model_switch.list_authenticated_providers(
            current_provider="openrouter", max_models=50,
        )

    copilot = next((p for p in providers if p["slug"] == "copilot"), None)
    assert copilot is not None, (
        "With an empty denylist copilot must appear (GH_TOKEN is set) — if it "
        "doesn't, the test is no longer proving the denylist is the cause."
    )
    assert copilot["models"] == live_models


def test_cached_provider_model_ids_short_circuits_disabled_only(_denylist):
    """The patched cached_provider_model_ids returns [] for disabled providers
    and otherwise delegates untouched to upstream."""
    from hermes_cli import models

    _denylist("copilot")
    _apply_fresh()

    # disabled → [] without calling through (no fetch)
    with patch("hermes_cli.models.fetch_github_model_catalog") as fetch_mock:
        assert models.cached_provider_model_ids("copilot") == []
        assert fetch_mock.call_count == 0

    # a non-disabled provider still flows through to the real implementation.
    # Patch the underlying live fetch so we assert delegation without network.
    with patch("hermes_cli.models.provider_model_ids", return_value=["m1", "m2"]) as live:
        out = models.cached_provider_model_ids("deepseek")
    assert out == ["m1", "m2"]
    assert live.called, "non-disabled provider must delegate to upstream"


# ---------------------------------------------------------------------------
# Wiring / load-timing — the seam loads via the plugin, before picker prewarm
# ---------------------------------------------------------------------------

def test_plugin_register_applies_seam():
    """The bundled apex-overlay plugin's register() applies this seam too."""
    from tests.apex_overlay.conftest import run_plugin_register_with_stubbed_seams

    called = run_plugin_register_with_stubbed_seams("_apex_overlay_plugin_under_test")
    assert "provider_filter" in called, (
        "plugin.register() must call provider_filter.apply()"
    )


def test_apex_overlay_enabled_in_config_and_discovered_before_prewarm():
    """Belt-and-suspenders on the load contract:

    1. cli-config.yaml.example enables the apex-overlay plugin (config tier).
    2. In cli.py, plugin discovery is ordered before the picker prewarm call,
       so the seam is installed before any background catalog fetch.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]

    cfg = (repo / "cli-config.yaml.example").read_text(encoding="utf-8")
    assert "apex-overlay" in cfg, (
        "cli-config.yaml.example must list apex-overlay under plugins.enabled "
        "or the seam never loads in production."
    )

    cli_src = (repo / "cli.py").read_text(encoding="utf-8")
    discover_idx = cli_src.find("discover_plugins()")
    prewarm_idx = cli_src.find("prewarm_picker_cache_async()")
    assert discover_idx != -1 and prewarm_idx != -1
    assert discover_idx < prewarm_idx, (
        "plugin discovery must run before the picker cache prewarm so the "
        "apex_overlay denylist suppresses the copilot fetch in time."
    )
