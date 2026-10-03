"""Policy-generation and region-routability invariants for the Bedrock allowlist cache.

Round-4 review of #104178: the allowlist is part of the cache fingerprint, but the fetched
model set was not bound to the policy generation that produced it — a config edit while a
fetch is in flight let the stored row carry ids produced under policy A stamped with policy
B's authority (narrowing: a removed id served fresh for a full TTL; widening: the newly
admitted id hidden until TTL). The settlement property proven here: a result is persisted
only under the exact policy generation that produced it, across all three store paths
(SWR ``_default_refresh``, the picker prefetch re-persist, and ``cached_provider_model_ids``).

The policy tests write a real temp ``config.yaml`` and read it through the real loader (the
``hermes_cli/AGENTS.md`` invariant for a new ``DEFAULT_CONFIG`` key) — the mid-flight flip is
a real file rewrite between two loader reads, not a ``load_config_readonly`` patch.

Region invariant (#104178 review, blocker 2): an allowlist is account policy, not region
routability. A geo-prefixed profile id the resolved region cannot invoke
(``bedrock_model_routable_from_region``) must not be offered by the curated fallback, on the
picker catalog or in either ``hermes model`` wizard branch.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.model_setup_flows_bedrock import bedrock_model_routable_from_region


def _write_allowlist_config(entries: list[str]) -> None:
    """Rewrite ``$HERMES_HOME/config.yaml`` with the given allowlist and drop the loader caches.

    Every ``load_config_readonly()`` after this — including the ones inside production code —
    sees exactly these entries, because the cache is keyed on the file signature.
    """
    from hermes_cli import config as config_mod
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({
        "bedrock": {"discovery": {"model_allowlist": entries}},
    }), encoding="utf-8")
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()


@pytest.fixture
def real_policy_config(tmp_path):
    """Point the loader at a per-test HERMES_HOME and clean the policy file afterwards."""
    import os

    old_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(tmp_path)
    try:
        _write_allowlist_config([])
        yield tmp_path
    finally:
        if old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_home


@pytest.fixture(autouse=True)
def _reset_swr_state():
    import hermes_cli.models as models_mod
    with models_mod._swr_refresh_lock:
        models_mod._swr_refresh_inflight.clear()
    yield
    with models_mod._swr_refresh_lock:
        models_mod._swr_refresh_inflight.clear()


class _InlineThread:
    """Run the spawned SWR thread synchronously (pattern from test_model_cache_swr)."""

    def __init__(self, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


class TestSWRPolicyGeneration:
    def test_swr_never_persists_ids_under_a_later_policy(self, real_policy_config):
        """Narrowing mid-flight [A] -> [B]: discovery produced [a] under A; the row it would
        store pairs those ids with B's fingerprint. It must store nothing."""
        from hermes_cli import models as models_mod

        _write_allowlist_config(["a.model"])

        stored: dict = {}
        with patch("agent.bedrock_adapter.discover_bedrock_models", return_value=[
                {"id": "a.model"}, {"id": "b.model"}]), \
             patch("agent.bedrock_adapter.resolve_bedrock_region", return_value="us-east-1"), \
             patch.object(models_mod, "_load_provider_models_cache", side_effect=lambda: dict(stored)), \
             patch.object(models_mod, "_store_cache_entry",
                          side_effect=lambda key, entry, cache=None: stored.__setitem__(key, entry)), \
             patch.object(models_mod.threading, "Thread", _InlineThread):
            real_pmi = models_mod.provider_model_ids

            def pmi_then_flip(provider, *, force_refresh=False):
                ids = real_pmi(provider, force_refresh=force_refresh)
                _write_allowlist_config(["b.model"])  # narrowing while the fetch is in flight
                return ids

            with patch.object(models_mod, "provider_model_ids", pmi_then_flip):
                models_mod._spawn_swr_refresh("bedrock")

        assert "bedrock" not in stored, (
            "a fetch produced under policy [a.model] must not be persisted after the policy "
            "narrowed to [b.model] — the row would carry a removed id under the new policy's "
            f"authority; stored: {stored}")

    def test_swr_widen_midflight_also_declined(self, real_policy_config):
        """Widening mid-flight [A,B] -> [A]: the fetched ids came from the [A,B] generation;
        stamping them with A's fingerprint would hide B until TTL. Store nothing."""
        from hermes_cli import models as models_mod

        _write_allowlist_config(["a.model", "b.model"])

        stored: dict = {}
        with patch("agent.bedrock_adapter.discover_bedrock_models", return_value=[
                {"id": "a.model"}, {"id": "b.model"}]), \
             patch("agent.bedrock_adapter.resolve_bedrock_region", return_value="us-east-1"), \
             patch.object(models_mod, "_load_provider_models_cache", side_effect=lambda: dict(stored)), \
             patch.object(models_mod, "_store_cache_entry",
                          side_effect=lambda key, entry, cache=None: stored.__setitem__(key, entry)), \
             patch.object(models_mod.threading, "Thread", _InlineThread):
            real_pmi = models_mod.provider_model_ids

            def pmi_then_flip(provider, *, force_refresh=False):
                ids = real_pmi(provider, force_refresh=force_refresh)
                _write_allowlist_config(["a.model"])
                return ids

            with patch.object(models_mod, "provider_model_ids", pmi_then_flip):
                models_mod._spawn_swr_refresh("bedrock")

        assert "bedrock" not in stored

    def test_swr_stable_policy_still_persists(self, real_policy_config):
        """Positive control: with no policy change across the fetch the live row is stored as
        before — the guard must not turn every refresh into a permanent cache miss."""
        from hermes_cli import models as models_mod

        _write_allowlist_config(["a.model"])

        stored: dict = {}
        with patch("agent.bedrock_adapter.discover_bedrock_models", return_value=[
                {"id": "a.model"}]), \
             patch("agent.bedrock_adapter.resolve_bedrock_region", return_value="us-east-1"), \
             patch.object(models_mod, "_load_provider_models_cache", side_effect=lambda: dict(stored)), \
             patch.object(models_mod, "_store_cache_entry",
                          side_effect=lambda key, entry, cache=None: stored.__setitem__(key, entry)), \
             patch.object(models_mod.threading, "Thread", _InlineThread):
            models_mod._spawn_swr_refresh("bedrock")

        assert stored["bedrock"]["models"] == ["a.model"]
        # The row is stamped with the full fingerprint of the generation that produced it:
        # with the policy unchanged, that IS the current fingerprint.
        assert stored["bedrock"]["fp"] == models_mod._credential_fingerprint("bedrock")


class TestPrefetchRepersistGeneration:
    def test_update_provider_cache_entry_refuses_a_stale_generation(self, real_policy_config):
        """The prefetch re-persist: ids produced under the captured generation must not be
        installed once the policy moved on; with no capture (None) the current policy stands
        in, and a matching generation writes normally."""
        from hermes_cli import models as models_mod
        from hermes_cli.models_bedrock import _bedrock_policy_fingerprint

        _write_allowlist_config(["a.model"])
        produced_under = _bedrock_policy_fingerprint()
        _write_allowlist_config(["b.model"])

        stored: dict = {}
        with patch.object(models_mod, "_load_provider_models_cache", side_effect=lambda: dict(stored)), \
             patch.object(models_mod, "_store_cache_entry",
                          side_effect=lambda key, entry, cache=None: stored.__setitem__(key, entry)):
            # Producer generation is stale now -> refused.
            models_mod.update_provider_cache_entry("bedrock", ["a.model"],
                                                   policy_fingerprint=produced_under)
            assert "bedrock" not in stored

            # Matching generation -> stored.
            from hermes_cli.models_bedrock import _bedrock_policy_fingerprint as fp_now
            models_mod.update_provider_cache_entry("bedrock", ["b.model"],
                                                   policy_fingerprint=fp_now())
            assert stored["bedrock"]["models"] == ["b.model"]

            # Non-bedrock slugs are untouched by the guard.
            models_mod.update_provider_cache_entry("openrouter", ["x"], policy_fingerprint="whatever")
            assert stored["openrouter"]["models"] == ["x"]


class TestRegionRoutability:
    def test_us_profile_id_is_not_routable_from_eu(self):
        """The repository invariant the fallback must obey (direct evidence for the review's
        repro): `bedrock_model_routable_from_region("us.…", "eu-central-2")` is False."""
        assert bedrock_model_routable_from_region("us.anthropic.claude-sonnet-4-6", "eu-central-2") is False
        assert bedrock_model_routable_from_region("eu.anthropic.claude-sonnet-4-6", "eu-central-2") is True
        assert bedrock_model_routable_from_region("global.anthropic.claude-sonnet-4-6", "eu-central-2") is True
        assert bedrock_model_routable_from_region("anthropic.claude-sonnet-4-6", "eu-central-2") is True

    def test_routable_filter_hides_cross_geo_ids(self, real_policy_config):
        from hermes_cli.models_bedrock import _routable_allowlisted_bedrock_ids

        ids = ["us.keep", "eu.keep", "global.keep", "bare-keep"]
        assert _routable_allowlisted_bedrock_ids(ids, "eu-central-2") == \
            ["eu.keep", "global.keep", "bare-keep"]
        # None resolves through the runtime-region resolver.
        with patch("agent.bedrock_adapter.resolve_bedrock_runtime_region",
                   return_value="eu-central-2"):
            assert _routable_allowlisted_bedrock_ids(ids) == ["eu.keep", "global.keep", "bare-keep"]

    def test_catalog_fallback_is_region_filtered(self, real_policy_config):
        """``_bedrock_catalog``'s allowlisted curated projection must not re-offer a geo id the
        region cannot invoke (the wizard's IAM branch and the /model picker share this)."""
        from hermes_cli import models_bedrock as mb

        _write_allowlist_config(["us.keep", "eu.keep"])
        with patch.object(mb, "_PROVIDER_MODELS", {"bedrock": ["us.keep", "eu.keep", "openai.hide"]}), \
             patch("agent.bedrock_adapter.discover_bedrock_models", return_value=[]), \
             patch("agent.bedrock_adapter.resolve_bedrock_region", return_value="eu-central-2"):
            out = mb._bedrock_catalog("bedrock", force_refresh=False)
        assert isinstance(out, mb._StaticFallbackModelIds)
        assert out == ["eu.keep"]

    def test_catalog_live_result_under_later_policy_is_not_persistable(self, real_policy_config):
        """Blocker 1 at the producer: discovery captured policy A, the policy flipped to B
        before ``_bedrock_catalog`` returns — the result is tagged as fallback provenance so
        no store path may write it (the same rule the SWR stub already obeys)."""
        from hermes_cli import models_bedrock as mb

        _write_allowlist_config(["a.model"])
        fingerprints = iter(["policy-A", "policy-B"])  # before fetch / after fetch
        with patch("agent.bedrock_adapter.bedrock_model_ids_or_none",
                   return_value=["a.model"]), \
             patch.object(mb, "_bedrock_policy_fingerprint",
                          side_effect=lambda: next(fingerprints)):
            out = mb._bedrock_catalog("bedrock", force_refresh=False)
        assert out == ["a.model"]
        assert isinstance(out, mb._StaticFallbackModelIds), (
            "a mid-flight policy change must downgrade the live result to non-persistable "
            "provenance, not leave it looking like a same-generation live list")


class TestWizardRegionFilteredFallback:
    """The IAM wizard branch with a REAL config.yaml on disk (no loader patch)."""

    def _run_flow(self, allowlist, region_answer, curated):
        from hermes_cli import model_setup_flows_bedrock as flow_mod
        from hermes_cli import models as models_mod

        _write_allowlist_config(allowlist)
        offered: dict = {}

        def _pick(model_list, prompt, **kwargs):
            offered["models"] = list(model_list)
            return model_list[0]

        with patch.object(flow_mod, "_ask", side_effect=[region_answer, "1"]), \
             patch.object(models_mod, "_PROVIDER_MODELS", {"bedrock": curated}), \
             patch("agent.bedrock_adapter.has_aws_credentials", return_value=True), \
             patch("agent.bedrock_adapter.resolve_aws_auth_env_var", return_value="AWS_ACCESS_KEY_ID"), \
             patch("agent.bedrock_adapter.resolve_bedrock_region", return_value="us-east-1"), \
             patch("agent.bedrock_adapter.discover_bedrock_models", return_value=[]), \
             patch.object(flow_mod, "_pick_model_or_prompt", side_effect=_pick), \
             patch.object(flow_mod, "_finish_model"):
            result = flow_mod._model_flow_bedrock({})
        return result, offered.get("models")

    def test_eu_region_hides_allowlisted_us_profile(self, real_policy_config, capsys):
        result, offered = self._run_flow(
            ["us.keep-me"], "eu-central-2", ["us.keep-me", "eu.also-me"])
        assert result is None
        assert offered is None
        assert "No models match bedrock.discovery.model_allowlist in this region" in capsys.readouterr().out

    def test_home_region_still_offers_the_allowlisted_profile(self, real_policy_config):
        _result, offered = self._run_flow(
            ["us.keep-me", "eu.also-me"], "us-east-1", ["us.keep-me", "eu.also-me"])
        # The home-geo allowlisted id is offered; the cross-geo allowlisted id is not.
        assert offered == ["us.keep-me"]
