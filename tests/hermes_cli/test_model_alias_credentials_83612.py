"""Direct-alias (``model_aliases:``) credential resolution (#83612).

An alias that points at a custom endpoint must authenticate with **its own**
credential. Before the fix ``DirectAlias`` had no ``api_key`` field at all, so
a configured key was silently dropped and the alias inherited whatever key the
*default* provider had already resolved — a 401 against the alias host and a
cross-provider credential leak to an unrelated third party.

The regression that matters most is the leak: assert on the credential the
endpoint probe is actually handed, not just on the returned struct.
"""

import pytest


ALIAS_HOST = "https://theta.example.com/v1"
DEFAULT_PROVIDER_SECRET = "sk-or-DEFAULT-PROVIDER-SECRET"


def _install_config(monkeypatch, alias_entry):
    """Point every config reader at a single-alias config."""
    cfg = {
        "model": {"default": "gpt-4", "provider": "openrouter"},
        "model_aliases": {"theta": alias_entry},
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda *a, **k: cfg)
    return cfg


def _switch_to_alias(monkeypatch, alias_entry):
    """Run ``/model theta`` and capture what the endpoint probe was given.

    Returns ``(result, probed)`` where ``probed`` holds the api_key/base_url
    handed to ``validate_requested_model`` — i.e. the credential that goes out
    on the wire to the alias host.
    """
    _install_config(monkeypatch, alias_entry)
    monkeypatch.setenv("OPENROUTER_API_KEY", DEFAULT_PROVIDER_SECRET)

    probed = {}

    def _fake_validate(model_name, provider, *, api_key=None, base_url=None, api_mode=None):
        probed["api_key"] = api_key
        probed["base_url"] = base_url
        return {"accepted": True, "persist": True, "recognized": True, "message": ""}

    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model", _fake_validate
    )

    import hermes_cli.model_switch as ms

    monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
    result = ms.switch_model(
        raw_input="theta",
        current_provider="openrouter",
        current_model="gpt-4",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key=DEFAULT_PROVIDER_SECRET,
    )
    return result, probed


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestDirectAliasCredentialLoading:
    def test_api_key_and_key_env_are_loaded_from_config(self, monkeypatch):
        """``api_key``/``key_env`` survive into the DirectAlias (were dropped)."""
        _install_config(
            monkeypatch,
            {
                "model": "theta-1",
                "provider": "custom",
                "base_url": ALIAS_HOST,
                "api_key": "sk-literal",
                "key_env": "THETA_API_KEY",
            },
        )
        from hermes_cli.model_switch import _load_direct_aliases

        alias = _load_direct_aliases()["theta"]
        assert alias.api_key == "sk-literal"
        assert alias.key_env == "THETA_API_KEY"

    def test_credential_fields_default_to_empty(self, monkeypatch):
        """Aliases without credentials keep working (positional construction)."""
        from hermes_cli.model_switch import DirectAlias

        alias = DirectAlias("theta-1", "custom", ALIAS_HOST)
        assert alias.api_key == ""
        assert alias.key_env == ""


class TestDirectAliasApiKeyHelper:
    @pytest.mark.parametrize(
        "entry, expected",
        [
            ({"api_key": "sk-literal"}, "sk-literal"),
            ({"api_key": "${THETA_API_KEY}"}, "sk-from-env"),
            ({"key_env": "THETA_API_KEY"}, "sk-from-env"),
            ({}, ""),
        ],
    )
    def test_resolves_literal_env_template_and_key_env(
        self, monkeypatch, entry, expected
    ):
        monkeypatch.setenv("THETA_API_KEY", "sk-from-env")
        from hermes_cli.model_switch import DirectAlias, direct_alias_api_key

        alias = DirectAlias("theta-1", "custom", ALIAS_HOST, **entry)
        assert direct_alias_api_key(alias) == expected


# ---------------------------------------------------------------------------
# /model <alias> — the switch path
# ---------------------------------------------------------------------------

class TestModelSwitchUsesAliasCredential:
    def test_alias_api_key_is_sent_to_alias_host(self, monkeypatch):
        result, probed = _switch_to_alias(
            monkeypatch,
            {
                "model": "theta-1",
                "provider": "custom",
                "base_url": ALIAS_HOST,
                "api_key": "sk-theta-ALIAS-SECRET",
            },
        )
        assert result.base_url == ALIAS_HOST
        assert result.api_key == "sk-theta-ALIAS-SECRET"
        assert probed["api_key"] == "sk-theta-ALIAS-SECRET"

    def test_alias_key_env_is_sent_to_alias_host(self, monkeypatch):
        monkeypatch.setenv("THETA_API_KEY", "sk-theta-FROM-ENV")
        result, _ = _switch_to_alias(
            monkeypatch,
            {
                "model": "theta-1",
                "provider": "custom",
                "base_url": ALIAS_HOST,
                "key_env": "THETA_API_KEY",
            },
        )
        assert result.api_key == "sk-theta-FROM-ENV"

    def test_default_provider_key_never_reaches_the_alias_host(self, monkeypatch):
        """The leak: an alias with no credential of its own must NOT inherit
        the default provider's key just because that key was resolved first."""
        result, probed = _switch_to_alias(
            monkeypatch,
            {"model": "theta-1", "provider": "custom", "base_url": ALIAS_HOST},
        )
        assert result.base_url == ALIAS_HOST
        assert result.api_key != DEFAULT_PROVIDER_SECRET
        assert probed["api_key"] != DEFAULT_PROVIDER_SECRET
        assert probed["base_url"] == ALIAS_HOST

    def test_same_host_alias_still_resolves_that_host_key(self, monkeypatch):
        """Host-gated resolution keeps working: an openrouter.ai alias still
        gets OPENROUTER_API_KEY — this is not a blanket "drop the key"."""
        result, _ = _switch_to_alias(
            monkeypatch,
            {
                "model": "theta-1",
                "provider": "custom",
                "base_url": "https://openrouter.ai/api/v1",
            },
        )
        assert result.api_key == DEFAULT_PROVIDER_SECRET

    def test_ollama_cloud_alias_resolves_ollama_api_key(self, monkeypatch):
        """Ollama Cloud aliases authenticate with OLLAMA_API_KEY, not the
        previously active provider's key."""
        monkeypatch.setenv("OLLAMA_API_KEY", "sk-ollama-KEY")
        result, _ = _switch_to_alias(
            monkeypatch,
            {
                "model": "qwen3.5:397b",
                "provider": "custom",
                "base_url": "https://ollama.com/v1",
            },
        )
        assert result.api_key == "sk-ollama-KEY"


class TestSessionKeyIsHostScoped:
    """The key already resolved for the session is reusable only on the same
    host. This is what keeps a user pinned to a custom endpoint working while
    still closing the cross-host leak."""

    def _switch(self, monkeypatch, session_base_url):
        cfg = {
            "model": {"default": "m", "provider": "ollama-launch"},
            "model_aliases": {
                "theta": {
                    "model": "theta-1",
                    "provider": "custom",
                    "base_url": "https://myhost.test/v1",
                }
            },
        }
        monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.load_config", lambda *a, **k: cfg
        )
        monkeypatch.setattr(
            "hermes_cli.models.validate_requested_model",
            lambda *a, **k: {
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": "",
            },
        )
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
        return ms.switch_model(
            raw_input="theta",
            current_provider="ollama-launch",
            current_model="m",
            current_base_url=session_base_url,
            current_api_key="sk-session-KEY",
        )

    def test_same_host_alias_keeps_the_session_key(self, monkeypatch):
        result = self._switch(monkeypatch, "https://myhost.test/v1")
        assert result.api_key == "sk-session-KEY"

    def test_different_host_alias_drops_the_session_key(self, monkeypatch):
        result = self._switch(monkeypatch, "https://elsewhere.test/v1")
        assert result.api_key != "sk-session-KEY"


class TestBuiltinProviderKeysDoNotLeak:
    """The leak is not specific to custom providers. A session on a built-in
    provider (Anthropic, OpenAI, ...) must not forward its key either when the
    alias resolves to an unrelated host — the credential is dropped on host
    mismatch regardless of which branch resolved it."""

    @pytest.mark.parametrize(
        "provider, env_var, session_base_url",
        [
            ("anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com"),
            ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1"),
        ],
    )
    def test_builtin_provider_key_not_forwarded_to_alias_host(
        self, monkeypatch, provider, env_var, session_base_url
    ):
        secret = f"sk-{provider}-SECRET"
        cfg = {
            "model": {"default": "m", "provider": provider},
            "model_aliases": {
                "theta": {
                    "model": "theta-1",
                    "provider": "custom",
                    "base_url": ALIAS_HOST,
                }
            },
        }
        monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.load_config", lambda *a, **k: cfg
        )
        monkeypatch.setenv(env_var, secret)

        probed = {}

        def _fake_validate(model_name, prov, *, api_key=None, base_url=None, api_mode=None):
            probed["api_key"] = api_key
            probed["base_url"] = base_url
            return {"accepted": True, "persist": True, "recognized": True, "message": ""}

        monkeypatch.setattr(
            "hermes_cli.models.validate_requested_model", _fake_validate
        )
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
        result = ms.switch_model(
            raw_input="theta",
            current_provider=provider,
            current_model="m",
            current_base_url=session_base_url,
            current_api_key=secret,
        )
        assert result.base_url == ALIAS_HOST
        assert result.api_key != secret
        assert probed["api_key"] != secret


class TestProviderLabelCannotSelectAKeyForAnArbitraryHost:
    """A direct alias's `provider:` label must not route credential resolution.

    With a base_url but no declared credential, a label like `anthropic` used
    to reach that provider's own resolver, which picks ANTHROPIC_API_KEY out
    of the environment while keeping the alias's unrelated base_url — a
    built-in provider's bearer secret handed to a third-party host. The alias
    endpoint is resolved as bare `custom` instead, which is host-gated.
    """

    def _switch(self, monkeypatch, alias, session_provider="openrouter",
                session_base_url="https://openrouter.ai/api/v1"):
        cfg = {
            "model": {"default": "m", "provider": session_provider},
            "model_aliases": {"theta": alias},
        }
        monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)
        monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda *a, **k: cfg)
        probed = {}

        def _fake_validate(model_name, prov, *, api_key=None, base_url=None, api_mode=None):
            probed["api_key"] = api_key
            return {"accepted": True, "persist": True, "recognized": True, "message": ""}

        monkeypatch.setattr("hermes_cli.models.validate_requested_model", _fake_validate)
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
        result = ms.switch_model(
            raw_input="theta", current_provider=session_provider, current_model="m",
            current_base_url=session_base_url, current_api_key="sk-session",
        )
        return result, probed

    @pytest.mark.parametrize("provider, env_var", [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ])
    def test_builtin_label_does_not_pull_that_providers_key_to_a_foreign_host(
        self, monkeypatch, provider, env_var
    ):
        secret = f"sk-{provider}-SECRET"
        monkeypatch.setenv(env_var, secret)
        result, probed = self._switch(
            monkeypatch,
            {"model": "c", "provider": provider, "base_url": "https://evil.test/v1"},
        )
        # Either the switch resolves no key for the foreign host, or it fails
        # outright — never the built-in provider's secret.
        assert result.api_key != secret
        assert probed.get("api_key") != secret

    def test_authoritative_host_still_resolves_its_vendor_key(self, monkeypatch):
        """Host gating is the point, not a blanket refusal: an alias whose URL
        IS authoritative for the vendor still authenticates."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-SECRET")
        result, _ = self._switch(
            monkeypatch,
            {"model": "c", "provider": "anthropic",
             "base_url": "https://api.anthropic.com/v1"},
        )
        assert result.api_key == "sk-anthropic-SECRET"


class TestSessionCredentialIsScopedToTheOrigin:
    """Reusing the session key across a scheme or port change is a downgrade.

    The override compared hostnames only, so an alias could keep the host and
    move an HTTPS session to `http://` (or another port) while the live
    credential followed it onto the new, untrusted origin.
    """

    def _switch(self, monkeypatch, alias_base_url, session_base_url):
        cfg = {
            "model": {"default": "m", "provider": "my-endpoint"},
            "model_aliases": {"theta": {
                "model": "m2", "provider": "custom", "base_url": alias_base_url}},
        }
        monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)
        monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda *a, **k: cfg)
        monkeypatch.setattr(
            "hermes_cli.models.validate_requested_model",
            lambda *a, **k: {"accepted": True, "persist": True,
                             "recognized": True, "message": ""},
        )
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
        return ms.switch_model(
            raw_input="theta", current_provider="my-endpoint", current_model="m",
            current_base_url=session_base_url, current_api_key="sk-SESSION-SECRET",
        )

    @pytest.mark.parametrize("alias_url, session_url, why", [
        ("http://api.example.com/v1", "https://api.example.com/v1", "https->http"),
        ("https://api.example.com:8443/v1", "https://api.example.com/v1", "port change"),
        ("http://api.example.com:8080/v1", "https://api.example.com/v1", "scheme+port"),
        ("https://other.example.com/v1", "https://api.example.com/v1", "cross-host"),
    ])
    def test_origin_change_drops_the_session_credential(
        self, monkeypatch, alias_url, session_url, why
    ):
        assert self._switch(monkeypatch, alias_url, session_url).api_key != "sk-SESSION-SECRET"

    @pytest.mark.parametrize("url", [
        "https://api.example.com/v1",
        "http://127.0.0.1:11434/v1",   # loopback plaintext is not a downgrade
        "http://localhost:8080/v1",
    ])
    def test_same_origin_keeps_the_session_credential(self, monkeypatch, url):
        assert self._switch(monkeypatch, url, url).api_key == "sk-SESSION-SECRET"

    def test_default_port_and_explicit_port_are_the_same_origin(self, monkeypatch):
        result = self._switch(
            monkeypatch, "https://api.example.com:443/v1", "https://api.example.com/v1"
        )
        assert result.api_key == "sk-SESSION-SECRET"


class TestCredentialPrecedenceIsExplicit:
    """`api_key` outranks `key_env`, so an entry carrying both is unambiguous."""

    def test_api_key_wins_over_key_env(self, monkeypatch):
        monkeypatch.setenv("THETA_API_KEY", "sk-from-key-env")
        from hermes_cli.model_switch import DirectAlias, direct_alias_api_key

        alias = DirectAlias("theta-1", "custom", ALIAS_HOST,
                            "sk-literal", "THETA_API_KEY")
        assert direct_alias_api_key(alias) == "sk-literal"

    def test_env_template_api_key_also_wins_over_key_env(self, monkeypatch):
        monkeypatch.setenv("PRIMARY", "sk-from-template")
        monkeypatch.setenv("FALLBACK", "sk-from-key-env")
        from hermes_cli.model_switch import DirectAlias, direct_alias_api_key

        alias = DirectAlias("theta-1", "custom", ALIAS_HOST,
                            "${PRIMARY}", "FALLBACK")
        assert direct_alias_api_key(alias) == "sk-from-template"

    def test_key_env_used_when_api_key_is_blank(self, monkeypatch):
        monkeypatch.setenv("FALLBACK", "sk-from-key-env")
        from hermes_cli.model_switch import DirectAlias, direct_alias_api_key

        alias = DirectAlias("theta-1", "custom", ALIAS_HOST, "   ", "FALLBACK")
        assert direct_alias_api_key(alias) == "sk-from-key-env"


class TestSchemelessBaseUrls:
    """Scheme-less base URLs.

    A loopback alias written without a scheme keeps working — the loopback
    exemption does not depend on the scheme. A scheme-less URL that also
    changes origin is refused, which costs nothing in practice: httpx cannot
    build a client from one at all (`localhost:11434/v1` parses as
    scheme='localhost', host=''), so such a base_url is non-functional
    regardless of what this comparison decides.
    """

    def test_hostname_and_port_survive_without_a_scheme(self):
        from utils import base_url_origin

        assert base_url_origin("localhost:11434/v1") == ("", "localhost", 11434)
        assert base_url_origin("127.0.0.1:11434") == ("", "127.0.0.1", 11434)

    @pytest.mark.parametrize("url", ["localhost:11434/v1", "127.0.0.1:11434/v1"])
    def test_schemeless_loopback_alias_keeps_the_session_credential(self, url):
        from hermes_cli.model_switch import _may_reuse_session_credential

        assert _may_reuse_session_credential(url, url) is True

    def test_schemeless_is_not_treated_as_the_schemed_origin(self):
        """`http://h` and `h` are not asserted equal — an unknown scheme is
        not evidence that the origin is unchanged."""
        from hermes_cli.model_switch import _may_reuse_session_credential

        assert _may_reuse_session_credential(
            "http://localhost:11434/v1", "localhost:11434/v1"
        ) is False

    def test_httpx_cannot_use_a_schemeless_base_url(self):
        """Pins the premise above: this is why the strict answer is harmless."""
        httpx = pytest.importorskip("httpx")

        assert httpx.URL("localhost:11434/v1").host == ""
        assert httpx.URL("api.example.com/v1").host == ""
        assert httpx.URL("http://localhost:11434/v1").host == "localhost"


class TestBaseUrlOrigin:
    """The origin helper the reuse decision is built on."""

    @pytest.mark.parametrize("url, expected", [
        ("https://h/v1", ("https", "h", 443)),
        ("https://h:443/v1", ("https", "h", 443)),
        ("http://h/v1", ("http", "h", 80)),
        ("https://h:8443/v1", ("https", "h", 8443)),
        ("https://H./v1", ("https", "h", 443)),
        ("", ("", "", 0)),
        ("https://h:99999/v1", ("", "", 0)),
    ])
    def test_origin_normalisation(self, url, expected):
        from utils import base_url_origin

        assert base_url_origin(url) == expected


# ---------------------------------------------------------------------------
# Host gating in the direct-alias runtime branch
# ---------------------------------------------------------------------------

class TestDirectAliasHostGating:
    @pytest.mark.parametrize(
        "base_url, expect_key",
        [
            ("https://ollama.com/v1", True),
            # Look-alike and path-embedded hosts must NOT get the credential
            # (GHSA-76xc-57q6-vm5m).
            ("https://ollama.com.attacker.test/v1", False),
            ("http://127.0.0.1/ollama.com/v1", False),
        ],
    )
    def test_ollama_key_is_host_matched_not_substring_matched(
        self, monkeypatch, base_url, expect_key
    ):
        monkeypatch.setenv("OLLAMA_API_KEY", "sk-ollama-KEY")
        from hermes_cli.runtime_provider import _resolve_named_custom_runtime

        runtime = _resolve_named_custom_runtime(
            requested_provider="custom", explicit_base_url=base_url
        )
        assert (runtime["api_key"] == "sk-ollama-KEY") is expect_key


# ---------------------------------------------------------------------------
# hermes chat -m <alias> — the oneshot path
# ---------------------------------------------------------------------------

class TestOneshotPassesAliasCredential:
    def test_alias_api_key_is_passed_to_the_resolver(self, monkeypatch):
        """``hermes chat -m theta`` must hand the alias's key to
        resolve_runtime_provider, not leave it to env fallbacks."""
        from hermes_cli.model_switch import DirectAlias
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(
            ms,
            "DIRECT_ALIASES",
            {"theta": DirectAlias("theta-1", "custom", ALIAS_HOST, "sk-theta-ALIAS")},
        )
        monkeypatch.setattr(ms, "_ensure_direct_aliases", lambda: None)

        captured = {}

        def _fake_resolve(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after credential resolution")

        # oneshot imports the resolver inside the function, so patch it at
        # its source module.
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", _fake_resolve
        )
        monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: {})
        import hermes_cli.oneshot as oneshot

        # _run_agent holds the alias wiring; run_oneshot() wraps it in a
        # catch-all that would swallow the sentinel.
        with pytest.raises(RuntimeError, match="stop after credential resolution"):
            oneshot._run_agent(prompt="hi", model="theta")

        assert captured["explicit_base_url"] == ALIAS_HOST
        assert captured["explicit_api_key"] == "sk-theta-ALIAS"
