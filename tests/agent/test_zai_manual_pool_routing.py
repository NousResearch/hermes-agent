"""Regression tests for Z.AI manual-pool base_url routing fix.

Validates:
- Layer 1: zai_openai_urls in vision auto-detect now lists Coding Plan
  endpoints first (improves vision routing for Coding Plan keys).
- Layer 2: `_resolve_zai_base_url` is now invoked on the runtime pool
  path so manual-pool entries honor cached detected_endpoint state
  from auth.json and the GLM_BASE_URL env override.

Layer 2 verification: the auxiliary client iterates PROVIDER_REGISTRY
(36 providers). For unit testing, we shrink the iteration to only "zai"
so the new branch is reached deterministically.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest


# ────────────────────────────────────────────────────────────────────────────
# Layer 1: vision helper list ordering
# ────────────────────────────────────────────────────────────────────────────


class TestZaiCodingEndpointInVisionHelper:
    """The vision auto-detect list must include Coding Plan endpoints first."""

    def test_coding_endpoints_present_in_zai_openai_urls(self):
        from agent import auxiliary_client as ac
        with open(ac.__file__, "r", encoding="utf-8") as fh:
            src = fh.read()
        assert "https://api.z.ai/api/coding/paas/v4" in src, (
            "Coding Plan endpoint global is missing from vision helper"
        )
        assert "https://open.bigmodel.cn/api/coding/paas/v4" in src, (
            "Coding Plan endpoint CN is missing from vision helper"
        )

    def test_coding_endpoints_listed_before_metered(self):
        from agent import auxiliary_client as ac
        with open(ac.__file__, "r", encoding="utf-8") as fh:
            src = fh.read()
        coding_idx = src.index("api.z.ai/api/coding/paas/v4")
        metered_idx = src.index("api.z.ai/api/paas/v4", coding_idx)
        assert coding_idx < metered_idx, (
            "Coding Plan endpoint must be probed before metered "
            "so Coding Plan keys authenticate on first vision request"
        )


# ────────────────────────────────────────────────────────────────────────────
# Layer 2: runtime pool re-resolution
# ────────────────────────────────────────────────────────────────────────────


def _fake_zai_entry():
    """Build a fake Z.AI pool entry as it would look for a manual key."""
    from agent.credential_pool import PooledCredential
    entry = mock.Mock(spec=PooledCredential)
    entry.provider = "zai"
    entry.access_token = "sk-zai-fake-token"
    entry.runtime_api_key = "sk-zai-fake-token"
    entry.runtime_base_url = "https://api.z.ai/api/paas/v4"
    entry.base_url = "https://api.z.ai/api/paas/v4"
    entry.inference_base_url = None
    return entry


def _fake_zai_pconfig():
    pconfig = mock.Mock()
    pconfig.inference_base_url = "https://api.z.ai/api/paas/v4"
    pconfig.name = "Z.AI / GLM"
    pconfig.auth_type = "api_key"
    return pconfig


class TestRuntimePoolBaseUrlReresolution:
    """Verify _resolve_zai_base_url is called on the runtime pool path."""

    def _setup_mocks(self, glm_base_url=""):
        """Return a contextmanager stack that forces the iteration to zai."""
        from contextlib import ExitStack

        fake_entry = _fake_zai_entry()
        fake_pconfig = _fake_zai_pconfig()

        class FakeRegistry:
            def items(self_inner):
                yield ("zai", fake_pconfig)

        from agent import auxiliary_client as ac
        import hermes_cli.auth as auth_mod

        stack = ExitStack()
        # 1. Shrink PROVIDER_REGISTRY (imported locally inside the function
        # via `from hermes_cli.auth import PROVIDER_REGISTRY`) to just zai
        stack.enter_context(
            mock.patch.object(auth_mod, "PROVIDER_REGISTRY", FakeRegistry())
        )
        # 2. Pool selection returns our fake zai entry
        stack.enter_context(
            mock.patch.object(ac, "_select_pool_entry", return_value=(True, fake_entry))
        )
        # 3. _resolve_zai_base_url is our spy (caller injects return_value)
        resolve_mock = mock.MagicMock(return_value="https://api.z.ai/api/coding/paas/v4")
        stack.enter_context(
            mock.patch.object(auth_mod, "_resolve_zai_base_url", resolve_mock)
        )
        # 4. Pool runtime helpers
        stack.enter_context(
            mock.patch.object(ac, "_pool_runtime_api_key", return_value="sk-zai-fake-token")
        )
        stack.enter_context(
            mock.patch.object(ac, "_pool_runtime_base_url", return_value="https://api.z.ai/api/paas/v4")
        )
        stack.enter_context(
            mock.patch.object(ac, "_get_aux_model_for_provider", return_value="glm-5")
        )
        # 5. Client construction
        captured = {"base_url": None}

        def fake_create(*, api_key, base_url, **kwargs):
            captured["base_url"] = base_url
            return mock.Mock()

        stack.enter_context(
            mock.patch.object(ac, "_create_openai_client", side_effect=fake_create)
        )
        stack.enter_context(
            mock.patch.object(ac, "_maybe_wrap_anthropic", side_effect=lambda c, m, k, u: c)
        )
        # 6. GLM_BASE_URL
        env = {k: v for k, v in os.environ.items() if k != "GLM_BASE_URL"}
        if glm_base_url:
            env["GLM_BASE_URL"] = glm_base_url
        stack.enter_context(mock.patch.dict(os.environ, env, clear=True))

        return stack, resolve_mock, captured

    def test_zai_provider_triggers_resolve_zai_base_url(self):
        """When pool_select returns a zai entry, _resolve_zai_base_url is invoked."""
        from agent import auxiliary_client as ac

        stack, mock_resolve, captured = self._setup_mocks()
        with stack:
            result = ac._resolve_api_key_provider()

        assert mock_resolve.called, (
            "_resolve_zai_base_url was not called for zai manual-pool entry"
        )
        # The OpenAI client should have been built with the coding endpoint
        assert captured["base_url"] is not None, "OpenAI client was never constructed"
        assert "coding/paas/v4" in captured["base_url"], (
            f"Expected coding endpoint, got {captured['base_url']}"
        )

    def test_resolve_failure_does_not_break_pool(self):
        """Probe failure must not break pool selection."""
        from agent import auxiliary_client as ac
        import hermes_cli.auth as auth_mod

        fake_entry = _fake_zai_entry()
        fake_pconfig = _fake_zai_pconfig()

        class FakeRegistry:
            def items(self_inner):
                yield ("zai", fake_pconfig)

        with mock.patch.object(auth_mod, "PROVIDER_REGISTRY", FakeRegistry()), \
             mock.patch.object(ac, "_select_pool_entry", return_value=(True, fake_entry)), \
             mock.patch.object(auth_mod, "_resolve_zai_base_url", side_effect=Exception("probe fail")), \
             mock.patch.object(ac, "_pool_runtime_api_key", return_value="sk-zai-fake-token"), \
             mock.patch.object(ac, "_pool_runtime_base_url", return_value="https://api.z.ai/api/paas/v4"), \
             mock.patch.object(ac, "_get_aux_model_for_provider", return_value="glm-5"), \
             mock.patch.object(ac, "_create_openai_client", return_value=mock.Mock()), \
             mock.patch.object(ac, "_maybe_wrap_anthropic", side_effect=lambda c, m, k, u: c), \
             mock.patch.dict(os.environ, {}, clear=True):
            # Must not raise despite probe failure
            result = ac._resolve_api_key_provider()
        assert result is not None
        client, model = result
        assert model == "glm-5"

    def test_glm_base_url_forwarded_to_resolve(self):
        """GLM_BASE_URL is forwarded as the env_override argument."""
        from agent import auxiliary_client as ac

        stack, mock_resolve, _ = self._setup_mocks(
            glm_base_url="https://api.z.ai/api/coding/paas/v4"
        )
        with stack:
            ac._resolve_api_key_provider()

        assert mock_resolve.called
        env_arg = mock_resolve.call_args[0][2]
        assert env_arg == "https://api.z.ai/api/coding/paas/v4", (
            f"GLM_BASE_URL was not forwarded: got {env_arg!r}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Sanity: existing ZAI_ENDPOINT detection still works
# ────────────────────────────────────────────────────────────────────────────


class TestZaiEndpointsRegistryIntact:
    """The ZAI_ENDPOINTS list in hermes_cli.auth remains intact after the fix."""

    def test_zai_endpoints_still_has_four_candidates(self):
        from hermes_cli.auth import ZAI_ENDPOINTS
        assert len(ZAI_ENDPOINTS) == 4
        labels = [e[3] for e in ZAI_ENDPOINTS]
        assert "Global (Coding Plan)" in labels
        assert "China (Coding Plan)" in labels

    def test_resolve_zai_base_url_signature_intact(self):
        """The signature is (api_key, default_url, env_override)."""
        import inspect
        from hermes_cli.auth import _resolve_zai_base_url
        sig = inspect.signature(_resolve_zai_base_url)
        params = list(sig.parameters.keys())
        assert params == ["api_key", "default_url", "env_override"]