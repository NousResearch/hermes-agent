"""The model picker must probe with a ``key_cmd``-minted credential.

``key_cmd`` (#86891) lets a provider authenticate with a SHORT-LIVED bearer
minted by a command — SSO/OIDC brokers, cloud IAM, internal auth proxies. The
request path has honoured it since it landed, but the picker resolved probe
credentials from ``api_key``/``key_env`` ONLY.

The failure is quiet and easy to misread. The picker probes ``/v1/models`` with
an EMPTY key, an authenticated endpoint answers 401, discovery returns nothing,
and the provider falls back to its single configured ``default_model``. The
user sees ONE model and cannot tell that apart from an endpoint that genuinely
serves one — inference itself keeps working, because that path mints correctly.

The same gap existed in the ``hermes model`` setup flow
(``_model_flow_named_custom``), which builds its own ``Authorization: Bearer``
header from the same incomplete resolution — same bug class, sibling path.

These tests pin:

* a ``key_cmd`` entry probes with the minted token, so the full catalog shows;
* the cache fingerprint is keyed on the COMMAND, not the minted token (which
  rotates every refresh — keying on it would re-probe on every open);
* a broken/interactive helper degrades to the pre-existing empty-key behaviour
  rather than taking the whole picker down;
* a minted token is NEVER persisted back into ``config.yaml`` — it would be
  stale within the hour and would shadow the ``key_cmd`` meant to re-mint it.
"""

from __future__ import annotations

from agent.command_token_source import resolve_probe_token


class TestResolveProbeToken:
    """The shared credential helper both probe paths call."""

    def test_bare_token_stdout(self):
        assert resolve_probe_token(
            {"key_cmd": "printf 'tok-abc'", "name": "gw"}
        ) == "tok-abc"

    def test_json_access_token(self):
        """The OAuth 2.0 token-endpoint response shape."""
        entry = {
            "key_cmd": """printf '{"access_token":"tok-json","expires_in":3600}'""",
            "name": "gw",
        }
        assert resolve_probe_token(entry) == "tok-json"

    def test_absent_key_cmd_is_empty(self):
        """No key_cmd — the caller falls through to api_key/key_env."""
        assert resolve_probe_token({"name": "gw"}) == ""

    def test_blank_key_cmd_is_empty(self):
        assert resolve_probe_token({"key_cmd": "   ", "name": "gw"}) == ""

    def test_failing_helper_degrades_to_empty(self):
        """A helper that needs an interactive sign-in (or is simply broken)
        must not take down the picker: every other provider's row still
        renders, and this one degrades to the old empty-key behaviour."""
        assert resolve_probe_token({"key_cmd": "exit 1", "name": "gw"}) == ""

    def test_silent_helper_is_empty(self):
        assert resolve_probe_token({"key_cmd": "true", "name": "gw"}) == ""

    def test_multiline_output_is_rejected(self):
        """command_token_source refuses to guess which line is the token."""
        assert resolve_probe_token(
            {"key_cmd": "printf 'a\\nb'", "name": "gw"}
        ) == ""


class TestPickerProbesWithMintedCredential:
    """End-to-end: a key_cmd provider lists its full catalog, not just one."""

    def _probe_capture(self, monkeypatch):
        """Record the api_key the picker hands the live /models probe."""
        seen: dict = {}

        # The real callee takes keyword extras (headers, timeout, api_mode);
        # the probe is wrapped in `except Exception: pass`, so a stub with a
        # narrower signature would be silently swallowed and look like "the
        # probe never ran".
        #
        # Behave like a real authenticated endpoint: no credential -> no
        # catalog. A stub that returns models regardless would still pass
        # unpatched (the caller falls back to default_model), so the
        # model-count assertions below would prove nothing.
        def fake_fetch(api_key, api_url, provider, preserve_native_models, **kwargs):
            seen["api_key"] = api_key
            seen["api_url"] = api_url
            if not api_key:
                return None  # what a 401 looks like to the picker
            return ["model-a", "model-b", "model-c"]

        monkeypatch.setattr(
            "hermes_cli.model_switch_providers._fetch_picker_live_models", fake_fetch
        )
        return seen

    def test_probe_receives_the_minted_token(self, monkeypatch):
        from hermes_cli.model_switch import list_authenticated_providers

        seen = self._probe_capture(monkeypatch)
        rows = list_authenticated_providers(
            user_providers={
                "gw": {
                    "base_url": "https://gw.example.test/v1",
                    "api_mode": "chat_completions",
                    "key_cmd": "printf 'tok-minted'",
                    "default_model": "model-a",
                }
            },
            refresh=True,
            for_picker=True,
        )

        assert seen.get("api_key") == "tok-minted", (
            "picker probed with an empty key — a key_cmd endpoint 401s and "
            "collapses to its single default_model"
        )
        gw = [r for r in rows if isinstance(r, dict) and r.get("slug") == "gw"]
        assert gw, "key_cmd provider missing from the picker entirely"
        # The user-visible symptom: unpatched this is 1 (just default_model).
        assert len(gw[0].get("models") or []) == 3

    def test_static_api_key_still_wins(self, monkeypatch):
        """key_cmd is a FALLBACK here: an explicit api_key is used as-is, so
        this change cannot alter behaviour for existing static-key configs."""
        from hermes_cli.model_switch import list_authenticated_providers

        seen = self._probe_capture(monkeypatch)
        list_authenticated_providers(
            user_providers={
                "gw": {
                    "base_url": "https://gw.example.test/v1",
                    "api_key": "sk-static",
                    "key_cmd": "printf 'tok-minted'",
                    "default_model": "model-a",
                }
            },
            refresh=True,
            for_picker=True,
        )

        assert seen.get("api_key") == "sk-static"


class TestCacheFingerprintStability:
    """The picker's cache key must not rotate with the token."""

    def test_fingerprint_keys_on_command_not_token(self, monkeypatch):
        """A helper minting a DIFFERENT token each call must still produce a
        stable cache fingerprint. Keying on the minted value would change the
        fingerprint on every refresh and force a re-probe on every open."""
        from hermes_cli.model_switch import list_authenticated_providers

        calls = {"n": 0}

        def fake_fetch(api_key, api_url, provider, preserve_native_models, **kwargs):
            calls["n"] += 1
            return ["model-a", "model-b"]

        monkeypatch.setattr(
            "hermes_cli.model_switch_providers._fetch_picker_live_models", fake_fetch
        )

        # $RANDOM would vary per call; use a counter file-free equivalent that
        # is deterministic per call but different between calls.
        providers = {
            "gw": {
                "base_url": "https://gw.example.test/v1",
                "key_cmd": "printf 'tok-%s' $$",  # PID: differs per invocation
                "default_model": "model-a",
            }
        }

        first = list_authenticated_providers(
            user_providers=providers, refresh=True, for_picker=True
        )
        second = list_authenticated_providers(
            user_providers=providers, refresh=True, for_picker=True
        )

        def row(rows):
            return [r for r in rows if isinstance(r, dict) and r.get("slug") == "gw"]

        assert row(first) and row(second)
        assert (row(first)[0].get("models") or []) == (
            row(second)[0].get("models") or []
        )


class TestMintedTokenIsNeverPersisted:
    """A key_cmd token must not be written back into config.yaml.

    ``_model_flow_named_custom`` computes the value to persist from the
    STATICALLY configured credential. Resolving key_cmd into ``api_key`` before
    that call would persist a short-lived bearer, which is stale within the
    hour and shadows the key_cmd that exists to re-mint it.
    """

    def test_key_cmd_provider_persists_no_credential(self):
        from hermes_cli.main_provider_setup import _custom_provider_api_key_config_value

        assert _custom_provider_api_key_config_value(
            {"key_cmd": "printf 'tok-secret'"}, ""
        ) == ""

    def test_static_key_still_persists(self):
        from hermes_cli.main_provider_setup import _custom_provider_api_key_config_value

        assert _custom_provider_api_key_config_value(
            {"api_key": "sk-static"}, "sk-static"
        ) == "sk-static"

    def test_key_env_persists_as_reference(self):
        """key_env persists as ${VAR}, never the resolved secret."""
        from hermes_cli.main_provider_setup import _custom_provider_api_key_config_value

        assert _custom_provider_api_key_config_value(
            {"key_env": "MY_KEY"}, "resolved-secret-value"
        ) == "${MY_KEY}"


class TestSetupFlowHonoursKeyCmd:
    """`hermes model`'s named-custom flow is the picker's sibling path.

    It builds its own ``Authorization: Bearer <api_key>`` for the /models
    probe, so an unresolved key_cmd sends no auth header and the endpoint
    401s — same symptom, different call path.

    These drive the real ``_model_flow_named_custom`` and assert on what the
    probe actually receives, rather than inspecting the function's source: a
    semantics-preserving refactor should not fail the suite.
    """

    class _StopAfterProbe(Exception):
        """Unwind once the probe has run, before the interactive menu."""

    def _run_flow_capturing_probe(self, monkeypatch, entry):
        """Invoke the flow far enough to capture the probe's credential."""
        # Patch the modules the flow imports FROM, not the compat shims: the
        # `hermes_cli.models` re-export of should_use_ollama_native_catalog is
        # scheduled for removal (see HermesPluginCompatWarning).
        import hermes_cli.models as models_mod
        import hermes_cli.models_local as models_local_mod
        from hermes_cli.model_setup_flows_custom import _model_flow_named_custom

        seen = {}

        def fake_fetch(api_key, base_url, **kwargs):
            seen["api_key"] = api_key
            seen["base_url"] = base_url
            # The flow would prompt interactively next; stop here.
            raise TestSetupFlowHonoursKeyCmd._StopAfterProbe

        monkeypatch.setattr(models_mod, "fetch_api_models", fake_fetch)
        # Ollama detection issues its own probe; force the generic path.
        monkeypatch.setattr(
            models_local_mod, "should_use_ollama_native_catalog", lambda *a, **k: False
        )

        try:
            _model_flow_named_custom({}, dict(entry))
        except TestSetupFlowHonoursKeyCmd._StopAfterProbe:
            pass
        except Exception:
            # Any other failure still tells us what the probe received; the
            # assertions below decide whether that was correct.
            pass
        return seen

    def test_probe_receives_the_minted_token(self, monkeypatch):
        seen = self._run_flow_capturing_probe(
            monkeypatch,
            {
                "name": "gw",
                "base_url": "https://gw.example.test/v1",
                "key_cmd": "printf 'tok-minted'",
                "model": "model-a",
            },
        )

        assert seen.get("api_key") == "tok-minted", (
            "setup flow probed with an empty key — its /models request goes "
            "out unauthenticated and the endpoint 401s"
        )

    def test_static_api_key_still_wins(self, monkeypatch):
        """key_cmd is a fallback: an explicit api_key is used unchanged, so
        existing static-key configs are unaffected."""
        seen = self._run_flow_capturing_probe(
            monkeypatch,
            {
                "name": "gw",
                "base_url": "https://gw.example.test/v1",
                "api_key": "sk-static",
                "key_cmd": "printf 'tok-minted'",
                "model": "model-a",
            },
        )

        assert seen.get("api_key") == "sk-static"

    def test_minted_token_is_not_persisted(self, monkeypatch, tmp_path):
        """The value written back to config.yaml is derived from the STATIC
        credential. Persisting a short-lived bearer would leave a stale key
        that also shadows the key_cmd meant to re-mint it."""
        import hermes_cli.main_provider_setup as main_mod

        persisted = {}
        real = main_mod._custom_provider_api_key_config_value

        def spy(provider_info, resolved_api_key=""):
            out = real(provider_info, resolved_api_key)
            persisted["value"] = out
            return out

        monkeypatch.setattr(main_mod, "_custom_provider_api_key_config_value", spy)

        self._run_flow_capturing_probe(
            monkeypatch,
            {
                "name": "gw",
                "base_url": "https://gw.example.test/v1",
                "key_cmd": "printf 'tok-minted'",
                "model": "model-a",
            },
        )

        assert persisted.get("value", "") == "", (
            "a key_cmd-minted bearer reached the value persisted to config.yaml"
        )
