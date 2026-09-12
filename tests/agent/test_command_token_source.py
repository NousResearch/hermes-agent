"""``key_cmd``: derive a provider API key by running a command.

Gateways that issue short-lived bearers (SSO/OIDC brokers, cloud IAM, internal
auth proxies) make a stored key go stale mid-session. These tests pin the three
behaviours that make the feature work:

* resolution yields a CALLABLE (invoked per request) rather than a resolved
  string, so a long session never sends a stale token;
* the token is cached until shortly before expiry, so the command is not run
  once per request;
* a failure never leaks the helper's output or the command string, either of
  which can contain a credential.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from agent.command_token_source import (
    CommandTokenError,
    CommandTokenSource,
    _mint,
    build_command_token_provider,
)


class TestMinting:
    def test_bare_token_stdout(self):
        source = CommandTokenSource("printf 'tok-abc'", "dbx")
        assert source() == "tok-abc"

    def test_json_access_token(self):
        """The OAuth 2.0 token-endpoint response shape."""
        source = CommandTokenSource(
            """printf '{"access_token":"tok-json","expires_in":3600}'""", "dbx"
        )
        assert source() == "tok-json"

    def test_trailing_newline_is_stripped(self):
        """A raw newline in the credential would corrupt the auth header."""
        assert CommandTokenSource("echo tok-nl", "dbx")() == "tok-nl"

    def test_multiline_output_is_rejected_not_guessed(self):
        """Only the token may land on stdout.

        Silently taking the first line turns a misconfigured helper (banner,
        warning, two tokens) into a corrupt-credential 401 that is much harder
        to diagnose than an explicit refusal.
        """
        source = CommandTokenSource("printf 'banner\\ntok-real'", "dbx")
        with pytest.raises(CommandTokenError, match="multiple lines"):
            source()

    def test_json_without_access_token_is_an_error(self):
        source = CommandTokenSource("""printf '{"nope":1}'""", "dbx")
        with pytest.raises(CommandTokenError, match="access_token"):
            source()

    def test_empty_output_is_an_error(self):
        with pytest.raises(CommandTokenError, match="no output"):
            CommandTokenSource("true", "dbx")()

    def test_nonzero_exit_is_an_error(self):
        with pytest.raises(CommandTokenError, match="exited 3"):
            CommandTokenSource("exit 3", "dbx")()

    def test_failure_message_is_actionable_without_echoing_the_command(self):
        """Actionable, but never echoes the command (it may embed a secret)."""
        secret_cmd = "print-token --client-secret=SENTINEL-SECRET; exit 1"
        with pytest.raises(CommandTokenError) as excinfo:
            CommandTokenSource(secret_cmd, "dbx")()
        message = str(excinfo.value)
        assert "SENTINEL-SECRET" not in message
        assert "dbx" in message          # names the provider to fix
        assert "exited" in message       # states what happened


class TestNoCredentialLeak:
    def test_failure_message_excludes_command_output(self):
        """A failing auth helper may print a token — it must not be surfaced."""
        source = CommandTokenSource(
            "printf 'SENTINEL-SECRET'; printf 'stderr-SENTINEL' >&2; exit 1",
            "dbx",
        )
        with pytest.raises(CommandTokenError) as excinfo:
            source()
        assert "SENTINEL" not in str(excinfo.value)


class TestCaching:
    def test_token_is_cached_between_calls(self):
        """Without caching the command would run on every request."""
        # A command whose output changes each run: equal results prove caching.
        source = CommandTokenSource("date +%s%N", "dbx")
        assert source() == source()

    def test_expired_token_is_reminted(self):
        # date +%s%N changes every run; $RANDOM would be bash-only (empty
        # under dash, which is what /bin/sh is on Debian-family CI).
        source = CommandTokenSource(
            """printf '{"access_token":"tok-%s","expires_in":3600}' "$(date +%s%N)" """,
            "dbx",
        )
        first = source()
        # Force the cache past its expiry.
        source._expires_at = 0.0
        assert source() != first

    def test_no_advertised_ttl_caches_on_a_bounded_window(self):
        """No TTL means a bounded cache, not a process-lifetime one.

        Nothing in the request path re-mints on 401 (SDK retries cover
        429/5xx only), so caching forever would wedge an expired token until
        restart. The window keeps the helper from running per-request while
        guaranteeing an eventual re-mint.
        """
        from agent.command_token_source import _NO_TTL_REFRESH_SECONDS

        source = CommandTokenSource("date +%s%N", "dbx")
        first = source()
        assert 0 < source._expires_at - time.monotonic() <= _NO_TTL_REFRESH_SECONDS
        assert source() == first  # cached inside the window
        source._expires_at = time.monotonic() - 1  # cross the window
        assert source() != first  # re-minted after it

    def test_advertised_ttl_sets_an_expiry(self):
        source = CommandTokenSource(
            """printf '{"access_token":"tok","expires_in":3600}'""", "dbx"
        )
        source()
        assert source._expires_at is not None

    def test_ttl_shorter_than_the_leeway_still_caches_briefly(self):
        """A leeway larger than the TTL must not disable caching entirely."""
        source = CommandTokenSource(
            """printf '{"access_token":"tok","expires_in":1}'""", "dbx"
        )
        source()
        assert source._expires_at is not None
        assert source._expires_at > 0.0


class TestBuilder:
    def test_returns_none_when_unset(self):
        assert build_command_token_provider("") is None
        assert build_command_token_provider("   ") is None

    def test_returns_callable_when_set(self):
        provider = build_command_token_provider("printf tok", "dbx")
        assert callable(provider)
        assert provider() == "tok"


class TestResolutionYieldsACallable:
    """The integration contract: a callable reaches the wire client."""

    def test_key_cmd_entry_resolves_to_a_callable(self, monkeypatch):
        from hermes_cli import runtime_provider as rp

        config = {
            "providers": {
                "dbx": {
                    "base_url": "https://example.invalid/v1",
                    "api_mode": "chat_completions",
                    "model": "m1",
                    "key_cmd": "printf minted-token",
                }
            }
        }
        monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)

        runtime = rp.resolve_runtime_provider(requested="custom:dbx")
        api_key = runtime["api_key"]
        assert callable(api_key), "key_cmd must resolve to a per-request callable"
        assert api_key() == "minted-token"

    def test_explicit_api_key_still_wins(self, monkeypatch):
        """``--api-key`` stays the one-off recovery escape hatch."""
        from hermes_cli import runtime_provider as rp

        config = {
            "providers": {
                "dbx": {
                    "base_url": "https://example.invalid/v1",
                    "api_mode": "chat_completions",
                    "model": "m1",
                    "key_cmd": "printf minted-token",
                }
            }
        }
        monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)

        runtime = rp.resolve_runtime_provider(
            requested="custom:dbx", explicit_api_key="sk-explicit-override"
        )
        assert runtime["api_key"] == "sk-explicit-override"


class TestCallableKeyGetsBearerAuth:
    """A callable api_key must reach the Anthropic bearer-hook client path.

    This is why key_cmd needs no per-vendor auth wiring: a static string is
    sent as ``x-api-key`` (which OAuth-gated gateways reject with 401), while a
    callable routes through the per-request ``Authorization: Bearer`` hook the
    Entra ID path already established. Verified against a live gateway with the
    SAME token value: static -> 401, callable -> 200.
    """

    def test_callable_takes_the_bearer_hook_path(self, monkeypatch):
        import agent.anthropic_adapter as aa

        seen = {}

        def _fake_hook(api_key, base_url, timeout, **kw):
            seen["callable"] = callable(api_key)
            return object()

        monkeypatch.setattr(
            aa, "_build_anthropic_client_with_bearer_hook", _fake_hook
        )
        aa.build_anthropic_client(
            lambda: "minted-token", "https://gateway.invalid/anthropic"
        )
        assert seen.get("callable") is True


class TestAbsoluteExpiry:
    """Helpers that advertise a deadline instead of a lifetime.

    OAuth 2.0 token endpoints send a relative ``expires_in``, but CLI token
    helpers commonly print an absolute ISO 8601 timestamp instead (Databricks
    ``expiry``, older Azure ``expiresOn``). Reading only ``expires_in`` treats
    those as "no TTL advertised", caches the token for the life of the process,
    and every request 401s once the real deadline passes.
    """

    @staticmethod
    def _iso(seconds_from_now: float) -> str:
        from datetime import datetime, timedelta, timezone

        return (
            datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now)
        ).isoformat()

    def test_iso_expiry_yields_a_ttl(self):
        deadline = self._iso(3600)
        _, ttl = _mint(f"printf '%s' '{{\"access_token\":\"t\",\"expiry\":\"{deadline}\"}}'", "p")
        assert ttl is not None, "an advertised deadline must produce a TTL"
        assert 3500 < ttl <= 3600

    def test_azure_expires_on_spelling(self):
        deadline = self._iso(1800)
        _, ttl = _mint(f"printf '%s' '{{\"access_token\":\"t\",\"expiresOn\":\"{deadline}\"}}'", "p")
        assert ttl is not None and 1700 < ttl <= 1800

    def test_expires_in_still_wins_when_both_present(self):
        """The RFC 6749 field is authoritative where a helper sends both."""
        deadline = self._iso(3600)
        _, ttl = _mint(
            f"printf '%s' '{{\"access_token\":\"t\",\"expires_in\":120,\"expiry\":\"{deadline}\"}}'",
            "p",
        )
        assert ttl == 120.0

    def test_unparseable_expiry_is_not_a_ttl(self):
        """Junk must fall back to refresh-on-401, never to a guessed deadline."""
        _, ttl = _mint('printf \'%s\' \'{"access_token":"t","expiry":"whenever"}\'', "p")
        assert ttl is None

    def test_already_past_expiry_is_not_a_ttl(self):
        """A stale deadline must not become a negative or zero TTL."""
        _, ttl = _mint(
            f"printf '%s' '{{\"access_token\":\"t\",\"expiry\":\"{self._iso(-60)}\"}}'", "p"
        )
        assert ttl is None

    def test_the_token_actually_gets_re_minted(self, tmp_path):
        """The regression that mattered: a deadline must expire the cache."""
        counter = tmp_path / "calls"
        cmd = (
            f"printf x >> {counter}; "
            f"printf '%s' '{{\"access_token\":\"t\",\"expiry\":\"{self._iso(1)}\"}}'"
        )
        src = CommandTokenSource(cmd, "p")
        src()
        assert src._expires_at is not None, "cache must carry a deadline"
        src._expires_at = time.monotonic() - 1  # simulate crossing it
        src()
        assert len(counter.read_text()) == 2, "expired cache must re-run the helper"


class TestAuxiliaryResolverHonoursKeyCmd:
    """Auxiliary tasks resolve credentials on their own path.

    ``agent.auxiliary_client.resolve_provider_client`` does not go through
    ``_resolve_named_custom_runtime``, so a key_cmd honoured only there leaves
    title generation, compression, vision and embedding falling back to the
    ``no-key-required`` placeholder — the main agent turn succeeds while every
    auxiliary call 401s.
    """

    @staticmethod
    def _resolve(monkeypatch, entry, explicit_api_key: object = None):
        """Resolve *entry* as a named custom provider; return the api_key seen.

        ``explicit_api_key`` is typed loosely on purpose: the ``key_cmd``
        contract hands ``resolve_provider_client`` a callable, not a str.
        """
        import agent.auxiliary_client as ac
        from hermes_cli import runtime_provider as rp

        monkeypatch.setattr(
            rp, "_get_named_custom_provider",
            lambda name: dict(entry, name="dbx") if name == "dbx" else None,
        )
        seen = {}

        def _spy(*, api_key, base_url, **kw):
            seen["api_key"] = api_key
            return SimpleNamespace(api_key=api_key, base_url=base_url)

        monkeypatch.setattr(ac, "_create_openai_client", _spy)
        ac.resolve_provider_client("dbx", explicit_api_key=explicit_api_key)
        return seen.get("api_key")

    BASE = {"base_url": "https://example.invalid/v1", "model": "m1"}

    def test_key_cmd_resolves_to_a_callable(self, monkeypatch):
        api_key = self._resolve(monkeypatch, {**self.BASE, "key_cmd": "printf minted-token"})
        assert callable(api_key), "auxiliary tasks must mint per request too"
        assert api_key() == "minted-token"

    def test_key_cmd_beats_static_credentials(self, monkeypatch):
        """Precedence matches the runtime resolver, so both agree on one entry."""
        api_key = self._resolve(
            monkeypatch,
            {**self.BASE, "api_key": "stale-static", "key_cmd": "printf minted-token"},
        )
        assert callable(api_key) and api_key() == "minted-token"

    def test_static_credentials_still_resolve(self, monkeypatch):
        assert self._resolve(monkeypatch, {**self.BASE, "api_key": "static"}) == "static"

    def test_blank_key_cmd_keeps_the_placeholder(self, monkeypatch):
        """A blank command must not become a callable that mints nothing."""
        assert self._resolve(
            monkeypatch, {**self.BASE, "key_cmd": "   "}
        ) == "no-key-required"


class TestExplicitCallableSurvivesCustomResolution:
    """Callers hand ``resolve_provider_client`` a callable (key_cmd token,
    vision auto-routing through the main runtime's callable credential) and
    every custom branch must pass it through *uncalled*: ``.strip()`` on it
    raised AttributeError (issue #88667), ``str()`` would send the object repr
    as the bearer (#104460's bug class)."""

    TOK = CommandTokenSource("printf minted-token", "dbx")

    @staticmethod
    def _spy_client(monkeypatch):
        import agent.auxiliary_client as ac

        seen = {}

        def _spy(*, api_key, base_url, **kw):
            seen["api_key"] = api_key
            return SimpleNamespace(api_key=api_key, base_url=base_url)

        monkeypatch.setattr(ac, "_create_openai_client", _spy)
        return seen

    def test_named_custom_branch_honours_explicit_callable(self, monkeypatch):
        from hermes_cli import runtime_provider as rp

        monkeypatch.setattr(
            rp, "_get_named_custom_provider",
            lambda name: {"base_url": "https://example.invalid/v1", "model": "m1",
                          "name": "dbx"} if name == "dbx" else None,
        )
        seen = self._spy_client(monkeypatch)
        import agent.auxiliary_client as ac

        ac.resolve_provider_client("dbx", explicit_api_key=self.TOK)
        assert seen.get("api_key") is self.TOK, (
            "an explicit callable must reach the client unchanged, not be "
            "stripped (crash) or stringified (stale-garbage bearer)"
        )

    def test_bare_custom_branch_honours_explicit_callable(self, monkeypatch):
        seen = self._spy_client(monkeypatch)
        import agent.auxiliary_client as ac

        ac.resolve_provider_client(
            "custom",
            explicit_base_url="https://example.invalid/v1",
            explicit_api_key=self.TOK,
        )
        assert seen.get("api_key") is self.TOK

    def test_main_runtime_callable_is_not_stringified(self, monkeypatch):
        """The main-runtime reuse branch must keep the callable; ``str()`` on
        it would send the object repr as the bearer (the #104460 class)."""
        seen = self._spy_client(monkeypatch)
        import agent.auxiliary_client as ac

        ac.resolve_provider_client(
            "custom",
            main_runtime={"base_url": "https://example.invalid/v1",
                          "api_key": self.TOK, "model": "m1"},
        )
        assert seen.get("api_key") is self.TOK


class TestAsyncWrapPreservesCallableKey:
    """``_to_async_client`` must carry the callable key across the sync→async hop.

    The OpenAI SDK stores a callable api_key as ``_api_key_provider`` and blanks
    ``client.api_key`` until the first request refreshes it. Rebuilding the async
    client from the (still empty) attribute silently sends no Authorization
    header — every async auxiliary call (vision) 401s against a key_cmd provider.
    The async SDK awaits the provider, so the bridge must be awaitable.
    """

    def test_async_client_keeps_a_working_key_provider(self):
        import asyncio

        import agent.auxiliary_client as ac

        calls = []

        def _provider():
            calls.append(1)
            return "tok-from-provider"

        class _StubSync:
            api_key = ""  # what the SDK leaves behind after callable init
            base_url = "https://example.invalid/v1"
            _api_key_provider = staticmethod(_provider)

        async_client, _ = ac._to_async_client(_StubSync(), "m1")
        bridge = getattr(async_client, "_api_key_provider", None)
        assert bridge is not None, (
            "a sync client with a callable key must rebuild the async client "
            "with an api-key provider, not the empty api_key attribute"
        )

        async def _mint_via_bridge():
            # Mirror one request: the async SDK calls _refresh_api_key() in
            # _prepare_options before building auth headers.
            await getattr(async_client, "_refresh_api_key")()
            return getattr(async_client, "auth_headers")

        assert asyncio.run(_mint_via_bridge()) == {
            "Authorization": "Bearer " + "tok-from-provider"
        }
        assert calls == [1]


