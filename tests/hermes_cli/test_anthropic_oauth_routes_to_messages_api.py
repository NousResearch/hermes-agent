"""Regression coverage for issue #32243.

OAuth Pro/Max credentials must always reach Anthropic via the native
``/v1/messages`` endpoint, never the OpenAI-compat ``/chat/completions``
shim — the latter bills against a separate "extra usage" pool that
Pro/Max subscriptions don't fund, so any request that lands on it 400s
with "You're out of extra usage" the moment the gateway starts.

The root cause was an inconsistency between two URL→api_mode helpers:

* ``hermes_cli.providers.determine_api_mode`` correctly mapped
  ``api.anthropic.com`` to ``anthropic_messages``.
* ``hermes_cli.runtime_provider._detect_api_mode_for_url`` did NOT, so
  every code path that fell back to URL-only detection (named custom
  providers, direct-alias resolution, the api-key fallback inside
  ``resolve_runtime_provider``) returned ``None`` for that host and
  defaulted to ``chat_completions``.

Exhaustive host-shape coverage for the helper itself lives in
``test_detect_api_mode_for_url.py::TestDirectAnthropicHost``.  The
tests below pin the **integration contract**: every runtime branch
that resolves an Anthropic endpoint must return
``api_mode == "anthropic_messages"``, so a future refactor of any
single branch cannot silently revert #32243.
"""

from __future__ import annotations

from hermes_cli import runtime_provider as rp


class TestExplicitRuntimeForAnthropic:
    """``_resolve_explicit_runtime`` with provider='anthropic' must
    always return ``api_mode='anthropic_messages'`` regardless of
    base_url shape or stale persisted ``model.api_mode`` values.

    Exercised whenever the user (or a Hermes subcommand) passes an
    explicit ``--api-key`` / ``--base-url`` override to the runtime
    resolver.
    """

    def test_explicit_args_route_to_messages_api(self):
        result = rp._resolve_explicit_runtime(
            provider="anthropic",
            requested_provider="anthropic",
            model_cfg={},
            explicit_api_key="sk-ant-oat01-foo",
            explicit_base_url="https://api.anthropic.com",
        )
        assert result is not None
        assert result["api_mode"] == "anthropic_messages"
        assert result["provider"] == "anthropic"
        assert result["base_url"] == "https://api.anthropic.com"

    def test_stale_chat_completions_api_mode_in_config_is_ignored(self):
        # A user who previously had ``provider: openai`` and switched to
        # anthropic might still have ``model.api_mode: chat_completions``
        # in their config.yaml.  The anthropic branch must hard-pin
        # the mode — Anthropic's chat_completions shim is the bug
        # locus of #32243 and must never be reachable from this path.
        result = rp._resolve_explicit_runtime(
            provider="anthropic",
            requested_provider="anthropic",
            model_cfg={"provider": "anthropic", "api_mode": "chat_completions"},
            explicit_api_key="sk-ant-oat01-foo",
            explicit_base_url="https://api.anthropic.com",
        )
        assert result is not None
        assert result["api_mode"] == "anthropic_messages"

    def test_no_explicit_args_returns_none(self):
        # Guard the gating contract — _resolve_explicit_runtime only
        # fires when an explicit override is present; without one it
        # must return None so the caller falls through to the pool /
        # top-level anthropic branch.
        assert (
            rp._resolve_explicit_runtime(
                provider="anthropic",
                requested_provider="anthropic",
                model_cfg={"provider": "anthropic"},
            )
            is None
        )


class TestPoolEntryForAnthropic:
    """``_resolve_runtime_from_pool_entry`` is what runs when a user
    has added an OAuth credential via ``hermes auth add anthropic
    --type oauth`` (the exact flow from #32243).  Pin the contract
    alongside the URL-detector test so all three runtime branches
    stay aligned and a future refactor of one cannot diverge from
    the others.
    """

    def test_oauth_pool_entry_routes_to_messages_api(self):
        class _Entry:
            access_token = "sk-ant-oat01-pool"
            runtime_api_key = "sk-ant-oat01-pool"
            source = "manual:hermes_pkce"
            base_url = "https://api.anthropic.com"

        resolved = rp._resolve_runtime_from_pool_entry(
            provider="anthropic",
            entry=_Entry(),
            requested_provider="anthropic",
            model_cfg={"provider": "anthropic"},
        )

        assert resolved["provider"] == "anthropic"
        assert resolved["api_mode"] == "anthropic_messages"
        assert resolved["base_url"] == "https://api.anthropic.com"

    def test_stale_chat_completions_api_mode_in_config_is_ignored(self):
        # Same regression as the explicit-runtime test above, but on
        # the pool path: a stale persisted chat_completions api_mode
        # must NOT override the provider-pin.
        class _Entry:
            access_token = "sk-ant-oat01-pool"
            runtime_api_key = "sk-ant-oat01-pool"
            source = "manual:hermes_pkce"
            base_url = "https://api.anthropic.com"

        resolved = rp._resolve_runtime_from_pool_entry(
            provider="anthropic",
            entry=_Entry(),
            requested_provider="anthropic",
            model_cfg={
                "provider": "anthropic",
                "api_mode": "chat_completions",
            },
        )

        assert resolved["api_mode"] == "anthropic_messages"


class TestCustomProviderUrlFallback:
    """The detector fix's actual reachable path: a user-defined
    ``providers:`` / ``custom_providers:`` entry whose ``api`` URL
    points at ``api.anthropic.com``, with no explicit ``api_mode`` /
    ``transport`` field.

    Pre-fix: this falls through ``_try_resolve_from_custom_pool`` →
    ``_detect_api_mode_for_url("https://api.anthropic.com")`` → None →
    default ``chat_completions`` → request lands on the OpenAI-compat
    shim → "out of extra usage" 400.

    Post-fix: the detector returns ``anthropic_messages`` so the same
    config routes to ``/v1/messages`` where Pro/Max OAuth is billed.
    """

    def test_url_fallback_picks_messages_api(self, monkeypatch):
        class _Entry:
            access_token = "sk-ant-oat01-custom-pool"
            runtime_api_key = "sk-ant-oat01-custom-pool"
            source = "custom-pool"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

        monkeypatch.setattr(rp, "get_custom_provider_pool_key", lambda *a, **k: "custom:my-claude")
        monkeypatch.setattr(rp, "load_pool", lambda key: _Pool())

        resolved = rp._try_resolve_from_custom_pool(
            "https://api.anthropic.com",
            "custom",
        )

        assert resolved is not None
        assert resolved["api_mode"] == "anthropic_messages"

    def test_explicit_api_mode_override_still_wins(self, monkeypatch):
        # The detector is only consulted as a fallback — when the
        # custom-pool caller passes an explicit api_mode (e.g. from a
        # ``transport: chat_completions`` config entry), that takes
        # priority.  Pinned so the fix doesn't accidentally hijack a
        # user who DELIBERATELY pointed a chat_completions transport
        # at api.anthropic.com (uncommon but valid for OpenAI-compat
        # experiments).
        class _Entry:
            access_token = "k"
            runtime_api_key = "k"
            source = "x"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

        monkeypatch.setattr(rp, "get_custom_provider_pool_key", lambda *a, **k: "custom:my-claude")
        monkeypatch.setattr(rp, "load_pool", lambda key: _Pool())

        resolved = rp._try_resolve_from_custom_pool(
            "https://api.anthropic.com",
            "custom",
            api_mode_override="chat_completions",
        )

        assert resolved is not None
        assert resolved["api_mode"] == "chat_completions"
