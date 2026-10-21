"""Regression: runtime_provider's config helpers must be late-bound.

Test-pollution class found Aug 2026: ``hermes_cli.runtime_provider`` is
frequently imported lazily (inside functions like ``switch_model``'s
resolution path), so its first import in a process can happen while a test
has ``hermes_cli.config.load_config`` patched. With a module-level
from-import, that bound the MagicMock into ``runtime_provider.load_config``
for the life of the process: after the patch exited, every later caller
(e.g. MoA aggregator context-length resolution reading ``providers:``)
silently got the long-dead test's config. Live pair that exposed it:

  tests/hermes_cli/test_models.py::TestLocalOllamaModelDiscovery::
      test_switch_model_on_current_ollama_custom_endpoint_keeps_base_url
  tests/agent/test_model_metadata.py::TestMoAContextLength::
      test_moa_custom_context_configures_compressor_threshold

The fix: runtime_provider exposes late-bound delegates that resolve
``hermes_cli.config`` attributes at call time. These tests pin exactly
that property — a patch on ``hermes_cli.config.<fn>`` must be visible
through ``runtime_provider.<fn>`` while active, and must fully release
when it exits (no permanent capture). They fail if anyone reverts the
delegates back to module-level from-imports.

NOTE: deliberately no ``sys.modules`` pop/re-import here — a fresh
re-import under an active patch is itself a polluter (duplicate module
objects with diverging state).
"""

from unittest.mock import patch


def test_load_config_patch_is_visible_and_releases():
    import hermes_cli.runtime_provider as rp

    sentinel = {"providers": {"only-the-mock-has-this": {"api": "http://x/v1"}}}
    with patch("hermes_cli.config.load_config", return_value=sentinel):
        # While the patch is active, the delegate must see the mock…
        assert rp.load_config() is sentinel

    # …and once it exits, the same module must resolve the real function
    # again instead of replaying the mock's config (the permanent-capture
    # failure mode of a from-import binding).
    result = rp.load_config()
    assert result is not sentinel
    assert "only-the-mock-has-this" not in (result.get("providers") or {})


def test_sibling_delegates_are_late_bound_too():
    """get_compatible_custom_providers / normalize_extra_headers share the
    same import shape and must share the same late-binding behavior."""
    import hermes_cli.runtime_provider as rp

    marker = [{"name": "marker", "base_url": "http://m/v1"}]
    with patch(
        "hermes_cli.config.get_compatible_custom_providers", return_value=marker
    ):
        assert rp.get_compatible_custom_providers({}) is marker
    assert rp.get_compatible_custom_providers({}) is not marker

    hdrs = {"X-Marker": "1"}
    with patch("hermes_cli.config.normalize_extra_headers", return_value=hdrs):
        assert rp.normalize_extra_headers(None) is hdrs
    assert rp.normalize_extra_headers(None) is not hdrs
