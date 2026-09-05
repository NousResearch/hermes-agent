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

Since #98831 the command runs as NATIVE argv (shell=False): the helper is an
executable plus flags. Tests that need multi-step helpers (exit codes, side
effects) write a small Python helper script instead of relying on shell
chaining — which is now correctly rejected.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.command_token_source import (
    CommandTokenError,
    CommandTokenSource,
    _mint,
    _parse_key_cmd_argv,
    build_command_token_provider,
)


def _helper(tmp_path: Path, body: str, name: str = "helper.py") -> str:
    """Write a python helper script; return an argv-safe command string for it."""
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return f"{sys.executable} {p}"


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
        assert CommandTokenSource("printf tok-nl\\n", "dbx")() == "tok-nl"

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

    def test_empty_output_is_an_error(self, tmp_path):
        cmd = _helper(tmp_path, "pass\n")
        with pytest.raises(CommandTokenError, match="no output"):
            CommandTokenSource(cmd, "dbx")()

    def test_nonzero_exit_is_an_error(self, tmp_path):
        cmd = _helper(tmp_path, "raise SystemExit(3)\n")
        with pytest.raises(CommandTokenError, match="exited 3"):
            CommandTokenSource(cmd, "dbx")()

    def test_failure_message_is_actionable_without_echoing_the_command(self, tmp_path):
        """Actionable, but never echoes the command (it may embed a secret)."""
        cmd = _helper(tmp_path, "raise SystemExit(1)\n", name="print-token.py")
        cmd += " --client-secret=SENTINEL-SECRET"
        with pytest.raises(CommandTokenError) as excinfo:
            CommandTokenSource(cmd, "dbx")()
        message = str(excinfo.value)
        assert "SENTINEL-SECRET" not in message
        assert "dbx" in message          # names the provider to fix
        assert "exited" in message       # states what happened


class TestNoCredentialLeak:
    def test_failure_message_excludes_command_output(self, tmp_path):
        """A failing auth helper may print a token — it must not be surfaced."""
        cmd = _helper(
            tmp_path,
            'import sys\nprint("SENTINEL-SECRET")\n'
            'print("stderr-SENTINEL", file=sys.stderr)\nraise SystemExit(1)\n',
        )
        source = CommandTokenSource(cmd, "dbx")
        with pytest.raises(CommandTokenError) as excinfo:
            source()
        assert "SENTINEL" not in str(excinfo.value)


class TestArgvOnlyContract:
    """#98831: the command runs as native argv — shell semantics are rejected."""

    def test_shell_operators_are_rejected(self):
        for cmd in (
            "print-token --client-secret=S; exit 1",
            "cat creds | grep tok",
            "echo x > /tmp/t",
            "printf $(date)",
            "mint && echo done",
        ):
            with pytest.raises(CommandTokenError):
                CommandTokenSource(cmd, "dbx")

    def test_shells_and_dispatch_wrappers_are_rejected(self):
        for cmd in (
            "bash -c 'mint-token'",
            "sh -c mint-token",
            "env KEY=1 mint-token",
            "sudo mint-token",
            "docker run mint-token",
            "nice -n1 mint-token",
        ):
            with pytest.raises(CommandTokenError):
                _parse_key_cmd_argv(cmd, "dbx")

    def test_quoted_arguments_are_not_shell_syntax(self):
        """Quoting is argument parsing, not shell execution — canonical shapes stay valid."""
        for cmd in (
            "printf 'tok-abc'",
            'printf \'{"access_token":"t","expires_in":3600}\'',
            "my-auth-cli print-token --profile 'prod lane'",
        ):
            argv = _parse_key_cmd_argv(cmd, "dbx")
            assert argv and argv[0]

    def test_argv_is_bounded(self):
        with pytest.raises(CommandTokenError):
            _parse_key_cmd_argv(" ".join(f"a{i}" for i in range(20)), "dbx")


class TestCaching:
    def test_token_is_cached_between_calls(self, tmp_path):
        """Without caching the command would run on every request."""
        # A helper whose output changes each run: equal results prove caching.
        cmd = _helper(tmp_path, "import time\nprint(int(time.time()*1000))\n")
        source = CommandTokenSource(cmd, "dbx")
        assert source() == source()

    def test_expired_token_is_reminted(self, tmp_path):
        cmd = _helper(
            tmp_path,
            'import time\nprint(\'{"access_token":"tok-%d","expires_in":3600}\' % (time.time()*1000,))\n',
        )
        source = CommandTokenSource(cmd, "dbx")
        first = source()
        # Force the cache past its expiry.
        source._expires_at = 0.0
        assert source() != first

    def test_no_advertised_ttl_caches_on_a_bounded_window(self, tmp_path):
        """No TTL means a bounded cache, not a process-lifetime one.

        Nothing in the request path re-mints on 401 (SDK retries cover
        429/5xx only), so caching forever would wedge an expired token until
        restart. The window keeps the helper from running per-request while
        guaranteeing an eventual re-mint.
        """
        from agent.command_token_source import _NO_TTL_REFRESH_SECONDS

        cmd = _helper(tmp_path, "import time\nprint(int(time.time()*1000))\n")
        source = CommandTokenSource(cmd, "dbx")
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
        deadline = self._iso(1)
        cmd = _helper(
            tmp_path,
            f'print(\'{{"access_token":"t","expiry":"{deadline}"}}\')\n',
        )
        src = CommandTokenSource(cmd, "p")
        src()
        assert src._expires_at is not None, "cache must carry a deadline"
        src._expires_at = time.monotonic() - 1  # simulate crossing it
        src()
        # Re-minted: the helper ran again (its token is identical, but the
        # cache was crossed and _mint executed a second time without error).
        assert src() == "t"


class TestAuxiliaryResolverHonoursKeyCmd:
    """Auxiliary tasks resolve credentials on their own path.

    ``agent.auxiliary_client.resolve_provider_client`` does not go through
    ``_resolve_named_custom_runtime``, so a key_cmd honoured only there leaves
    title generation, compression, vision and embedding falling back to the
    ``no-key-required`` placeholder — the main agent turn succeeds while every
    auxiliary call 401s.
    """

    @staticmethod
    def _resolve(monkeypatch, entry):
        """Resolve *entry* as a named custom provider; return the api_key seen."""
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
        ac.resolve_provider_client("dbx")
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
