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

    def _fake_validate(model_name, provider, *, api_key=None, base_url=None,
                       api_mode=None, **_kwargs):
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

        def _fake_validate(model_name, prov, *, api_key=None, base_url=None,
                           api_mode=None, **_kwargs):
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

        def _fake_validate(model_name, prov, *, api_key=None, base_url=None,
                           api_mode=None, **_kwargs):
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


class TestAliasCacheIsProfileScoped:
    """DIRECT_ALIASES is process-global; its source is profile-local.

    Entries carry `api_key`, so an unkeyed cache lets the first profile to
    resolve an alias pin its definitions AND its credentials for every later
    profile in the process. These tests switch profiles inside one process
    and never clear the cache by hand — clearing it would hide the bug.
    """

    def _profile(self, tmp_path, name, body):
        home = tmp_path / name
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(body, encoding="utf-8")
        return home

    def _load(self, monkeypatch, home):
        monkeypatch.setenv("HERMES_HOME", str(home))
        import hermes_cli.model_switch as ms

        ms._ensure_direct_aliases()
        return ms.DIRECT_ALIASES

    def test_second_profile_does_not_inherit_the_first_profiles_key(
        self, tmp_path, monkeypatch
    ):
        a = self._profile(tmp_path, "a", (
            'model_aliases:\n'
            '  theta:\n    model: a-model\n    provider: custom\n'
            '    base_url: "https://a.example.com/v1"\n'
            '    api_key: "sk-PROFILE-A-SECRET"\n'
        ))
        b = self._profile(tmp_path, "b", (
            'model_aliases:\n'
            '  theta:\n    model: b-model\n    provider: custom\n'
            '    base_url: "https://b.example.com/v1"\n'
            '    api_key: "sk-PROFILE-B-SECRET"\n'
        ))
        assert self._load(monkeypatch, a)["theta"].api_key == "sk-PROFILE-A-SECRET"

        theta = self._load(monkeypatch, b)["theta"]
        assert theta.api_key == "sk-PROFILE-B-SECRET"
        assert theta.model == "b-model"
        assert theta.base_url == "https://b.example.com/v1"

    def test_alias_absent_from_the_second_profile_does_not_persist(
        self, tmp_path, monkeypatch
    ):
        a = self._profile(tmp_path, "a2", (
            'model_aliases:\n'
            '  only-in-a:\n    model: x\n    provider: custom\n'
            '    base_url: "https://a.example.com/v1"\n'
        ))
        b = self._profile(tmp_path, "b2", (
            'model_aliases:\n'
            '  only-in-b:\n    model: y\n    provider: custom\n'
            '    base_url: "https://b.example.com/v1"\n'
        ))
        assert "only-in-a" in self._load(monkeypatch, a)

        loaded = self._load(monkeypatch, b)
        assert "only-in-a" not in loaded
        assert "only-in-b" in loaded

    def test_key_rotation_in_place_is_picked_up(self, tmp_path, monkeypatch):
        home = self._profile(tmp_path, "rot", (
            'model_aliases:\n'
            '  theta:\n    model: m\n    provider: custom\n'
            '    base_url: "https://h.example.com/v1"\n'
            '    api_key: "sk-BEFORE-ROTATION"\n'
        ))
        assert self._load(monkeypatch, home)["theta"].api_key == "sk-BEFORE-ROTATION"

        (home / "config.yaml").write_text((
            'model_aliases:\n'
            '  theta:\n    model: m\n    provider: custom\n'
            '    base_url: "https://h.example.com/v1"\n'
            '    api_key: "sk-AFTER-ROTATION-XYZ"\n'
        ), encoding="utf-8")
        assert self._load(monkeypatch, home)["theta"].api_key == "sk-AFTER-ROTATION-XYZ"

    def test_cache_is_still_mutated_in_place(self, tmp_path, monkeypatch):
        """Callers hold this exact dict (#16767) — reloading must not rebind."""
        home = self._profile(tmp_path, "inplace", (
            'model_aliases:\n'
            '  theta:\n    model: m\n    provider: custom\n'
            '    base_url: "https://h.example.com/v1"\n'
        ))
        import hermes_cli.model_switch as ms

        before = id(ms.DIRECT_ALIASES)
        self._load(monkeypatch, home)
        assert id(ms.DIRECT_ALIASES) == before


class TestOneShotUsesTheSameHostInvariant:
    """`hermes chat -m <alias>` must not trust the alias's provider label.

    These exercise the REAL resolver — the leak lives inside
    resolve_runtime_provider's provider-specific branches, so stubbing it
    would test nothing.
    """

    @pytest.mark.parametrize("provider, env_var", [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("xai", "XAI_API_KEY"),
    ])
    def test_no_key_alias_on_a_foreign_host_gets_no_provider_token(
        self, monkeypatch, provider, env_var
    ):
        secret = f"sk-{provider}-LIVE-TOKEN"
        monkeypatch.setenv(env_var, secret)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.load_config",
            lambda *a, **k: {"model": {"default": "m", "provider": "openrouter"}},
        )
        from hermes_cli.model_switch import DirectAlias, direct_alias_runtime_request
        from hermes_cli.runtime_provider import resolve_runtime_provider

        alias = DirectAlias("c", provider, "https://evil.test/v1")
        requested, explicit_key = direct_alias_runtime_request(alias)
        runtime = resolve_runtime_provider(
            requested=requested,
            explicit_api_key=explicit_key,
            explicit_base_url=alias.base_url,
        )
        assert runtime.get("api_key") != secret

    def test_label_is_kept_when_the_alias_has_no_url(self):
        """Nothing to protect against without a foreign host, and the label is
        the only routing information there is."""
        from hermes_cli.model_switch import DirectAlias, direct_alias_runtime_request

        assert direct_alias_runtime_request(
            DirectAlias("c", "anthropic", "")
        ) == ("anthropic", None)

    def test_url_bearing_alias_is_forced_to_custom(self):
        from hermes_cli.model_switch import DirectAlias, direct_alias_runtime_request

        assert direct_alias_runtime_request(
            DirectAlias("c", "anthropic", "https://evil.test/v1")
        ) == ("custom", None)

    def test_declared_key_is_carried_through(self, monkeypatch):
        from hermes_cli.model_switch import DirectAlias, direct_alias_runtime_request

        assert direct_alias_runtime_request(
            DirectAlias("c", "anthropic", "https://evil.test/v1", "sk-own")
        ) == ("custom", "sk-own")


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


class TestNoProductionCodeMutatesTheAliasCacheInPlace:
    """The profile-isolation property depends on an unwritten rule.

    ``_ensure_direct_aliases`` keeps a copy of what it loaded and treats any
    divergence as a caller's data, not its own stale cache — that is what lets
    tests seed ``DIRECT_ALIASES`` in place without the loader wiping them
    (#16767). The cost is that a *production* in-place mutation would pin the
    cache: contents would never again match the copy, so the config-identity
    check that reloads on a profile switch would stop being consulted, and one
    profile's aliases and credentials would be served to the next.

    No production code mutates it today — only the loader itself, and every
    other reference is a read. This pins that, because the failure mode is
    silent: nothing raises, nothing logs, and the leak only shows up as one
    profile answering with another profile's key.
    """

    #: The only place allowed to write the cache.
    OWNER = ("hermes_cli/model_switch.py", "_ensure_direct_aliases")

    MUTATORS = frozenset(
        {"update", "clear", "pop", "popitem", "setdefault", "__setitem__"}
    )

    @staticmethod
    def _production_sources():
        import pathlib

        repo = pathlib.Path(__file__).resolve().parents[2]
        skip = {".git", "node_modules", "tests", "build", "dist", ".venv"}
        for path in repo.rglob("*.py"):
            if any(part in skip for part in path.parts):
                continue
            yield path, path.relative_to(repo).as_posix()

    @classmethod
    def _violations(cls, source: str, rel: str):
        """Yield (function, description) for each in-place write."""
        import ast

        tree = ast.parse(source)
        enclosing = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    enclosing[id(child)] = node.name

        def _names(node):
            """DIRECT_ALIASES, whether bare or attribute-qualified."""
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return node.attr
            return None

        for node in ast.walk(tree):
            where = enclosing.get(id(node), "<module>")
            if (rel, where) == cls.OWNER:
                continue
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    _names(node.func.value) == "DIRECT_ALIASES"
                    and node.func.attr in cls.MUTATORS
                ):
                    yield where, f"DIRECT_ALIASES.{node.func.attr}()"
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and _names(target.value) == "DIRECT_ALIASES"
                    ):
                        yield where, "DIRECT_ALIASES[...] = ..."

    def test_only_the_loader_writes_the_cache(self):
        found = []
        for path, rel in self._production_sources():
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "DIRECT_ALIASES" not in source:
                continue
            try:
                found.extend(
                    f"{rel}:{where}: {what}"
                    for where, what in self._violations(source, rel)
                )
            except SyntaxError:
                continue

        assert not found, (
            "In-place writes to DIRECT_ALIASES outside "
            f"{self.OWNER[0]}::{self.OWNER[1]} pin the alias cache and break "
            "per-profile isolation:\n  " + "\n  ".join(found)
        )

    def test_the_scan_actually_detects_a_violation(self):
        """Negative control — an always-passing scanner would prove nothing."""
        offending = (
            "from hermes_cli.model_switch import DIRECT_ALIASES\n"
            "def warm():\n"
            "    DIRECT_ALIASES.update({'x': 1})\n"
            "    DIRECT_ALIASES['y'] = 2\n"
        )

        hits = list(self._violations(offending, "some/other_module.py"))

        assert {what for _, what in hits} == {
            "DIRECT_ALIASES.update()",
            "DIRECT_ALIASES[...] = ...",
        }
        assert all(where == "warm" for where, _ in hits)

    def test_the_owner_itself_is_exempt(self):
        """...and an exemption that swallowed everything would prove nothing."""
        import pathlib

        repo = pathlib.Path(__file__).resolve().parents[2]
        source = (repo / self.OWNER[0]).read_text(encoding="utf-8")

        assert list(self._violations(source, self.OWNER[0])) == []
        # The loader really does write in place, so the exemption is load-bearing.
        assert "DIRECT_ALIASES.clear()" in source
        assert "DIRECT_ALIASES.update(loaded)" in source
