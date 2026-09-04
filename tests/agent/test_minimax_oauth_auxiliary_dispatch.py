"""Regression guard: ``resolve_provider_client`` must dispatch ``minimax-oauth``
and any ``auth_type == "oauth_minimax"`` provider to the MiniMax OAuth
credential resolver, not fall through to the generic
``unhandled auth_type %s for %s`` warning.

Background: ``hermes_cli.auth.ProviderConfig`` defines ``minimax-oauth`` with
``auth_type="oauth_minimax"`` (a third OAuth variant alongside
``oauth_device_code`` and ``oauth_external``). When the auxiliary-client
text-aux path resolves a provider, it dispatches on
``pconfig.auth_type in {"oauth_device_code", "oauth_external"}`` and only
hard-codes ``nous``, ``openai-codex``, ``xai-oauth`` as named branches.
Everything else — including ``minimax-oauth`` — falls through to a
``unhandled auth_type oauth_minimax for minimax-oauth`` warning and
returns ``(None, None)``.

User-visible impact: every auxiliary call (vision, session_search,
compression summarisation, curator review) is forced onto the configured
``fallback_providers`` (default: openrouter). For a minimax-oauth user
who does NOT have openrouter configured, the warnings pile up:
``Auxiliary: marking openrouter unhealthy for 60s (payment / credit
error)`` and the auxiliary tasks silently degrade or fail.

The fix: add a third named branch for ``minimax-oauth`` (mirroring the
existing xai-oauth branch) so the resolver uses
``resolve_minimax_oauth_runtime_credentials`` and builds a real client
against ``https://api.minimax.io/v1``.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ── ProviderConfig smoke test ────────────────────────────────────────────────


class TestProviderConfigMiniMaxOauthAuthType:
    """Pin the contract: ``minimax-oauth`` is registered with
    ``auth_type == "oauth_minimax"`` and has the right base URL. The fix
    for the bug is in the resolver, not the registry, but if a future
    refactor ever renames the auth_type, the dispatcher's `is` check
    needs to track that rename.
    """

    def test_minimax_oauth_registered_with_oauth_minimax_auth_type(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        cfg = PROVIDER_REGISTRY["minimax-oauth"]
        assert cfg.auth_type == "oauth_minimax", (
            f"minimax-oauth auth_type changed to {cfg.auth_type!r}; "
            f"update the dispatch in agent/auxiliary_client.py to match."
        )

    def test_minimax_oauth_inference_base_url_is_anthropic_path(self):
        """The inference URL ends in /anthropic, which the OpenAI SDK
        path needs to be rewritten to /v1 by ``_to_openai_base_url``."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        cfg = PROVIDER_REGISTRY["minimax-oauth"]
        assert cfg.inference_base_url.endswith("/anthropic"), (
            f"unexpected inference_base_url: {cfg.inference_base_url!r}"
        )


# ── Resolver dispatch test ───────────────────────────────────────────────────


class TestResolveProviderClientMiniMaxOauth:
    """The fix: ``resolve_provider_client("minimax-oauth", ...)`` must
    NOT hit the ``unhandled auth_type oauth_minimax`` warning path. It
    must either succeed (returning a real client + model) or fail with
    a specific, actionable warning about missing credentials — not the
    generic unhandled warning.
    """

    def _import_resolve(self):
        from agent.auxiliary_client import resolve_provider_client
        return resolve_provider_client

    def _import_get_pconfig(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        return PROVIDER_REGISTRY.__getitem__

    def test_minimax_oauth_no_credentials_returns_none_without_unhandled_warning(
        self, caplog
    ):
        """When no MiniMax OAuth credentials are stored, the resolver
        must return ``(None, None)`` with a specific 'no credentials'
        warning — NOT the generic 'unhandled auth_type' warning that
        hides the real cause.
        """
        import logging

        from hermes_cli import auth as auth_mod

        resolve_provider_client = self._import_resolve()
        get_provider_config = self._import_get_pconfig()
        pconfig = get_provider_config("minimax-oauth")

        # Force the "no token" path: clear the stored state AND make
        # the resolver raise as it does for un-authenticated users.
        with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"), \
             patch_resolve_raises(RuntimeError("no minimax-oauth token stored")), \
             patch_get_minimax_status({"logged_in": False}):
            client, model = resolve_provider_client("minimax-oauth", None)

        assert client is None
        assert model is None

        # The bug surfaces as this exact warning. After the fix, this
        # message must not appear — instead a "no credentials found"
        # or a successful client build should be the only path.
        unhandled_warnings = [
            r for r in caplog.records
            if "unhandled auth_type" in r.getMessage()
            and "minimax" in r.getMessage()
        ]
        assert not unhandled_warnings, (
            f"resolver fell through to the unhandled-auth-type path: "
            f"{[r.getMessage() for r in unhandled_warnings]}. "
            f"minimax-oauth needs its own named branch in "
            f"agent/auxiliary_client.py::resolve_provider_client."
        )

    def test_minimax_oauth_with_credentials_returns_real_client(self):
        """When MiniMax OAuth credentials ARE available, the resolver
        must build a real OpenAI client pointed at the minimax API URL.
        """
        resolve_provider_client = self._import_resolve()

        with patch_resolve_returns({
            "api_key": "fake-minimax-bearer",
            "base_url": "https://api.minimax.io/anthropic",
        }), patch_get_minimax_status({"logged_in": True}):
            client, model = resolve_provider_client("minimax-oauth", None)

        # The resolver should hand back SOMETHING (not None) when
        # credentials are present. Whether it returns a wrapped aux
        # client or a raw OpenAI client, both have a usable surface.
        assert client is not None, (
            "resolve_provider_client returned None even with valid "
            "minimax-oauth credentials — the dispatch is not reaching "
            "the build path."
        )
        # The base URL must be the rewritten /v1 form (the OpenAI SDK
        # doesn't understand the /anthropic path).
        base_url = str(getattr(client, "base_url", ""))
        assert "minimax.io" in base_url, (
            f"unexpected base_url {base_url!r}; expected the resolver "
            f"to use the minimax endpoint"
        )
        assert "/v1" in base_url, (
            f"base_url {base_url!r} should be rewritten to /v1 form "
            f"for the OpenAI SDK"
        )


# ── Test helpers (module-level so pytest can find them) ──────────────────────


def patch_resolve_returns(creds):
    """Patch ``resolve_minimax_oauth_runtime_credentials`` to return creds."""
    from unittest.mock import patch as _patch

    return _patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        return_value=creds,
    )


def patch_resolve_raises(exc):
    from unittest.mock import patch as _patch

    return _patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        side_effect=exc,
    )


def patch_get_minimax_status(status):
    from unittest.mock import patch as _patch

    return _patch(
        "hermes_cli.auth.get_minimax_oauth_auth_status",
        return_value=status,
    )
