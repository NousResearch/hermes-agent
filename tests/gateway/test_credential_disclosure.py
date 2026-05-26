"""Tests for ``gateway.credential_disclosure``.

Regression coverage for issue #32524 — the gateway must announce which
provider it has adopted at startup and, when that adoption came from a
process env var for a paid cloud API with no deliberate
``hermes auth add`` / config-pin signal, the announcement must escalate
to a WARNING-level multi-line block so operators can't miss it
scrolling past in ``gateway.log``.

The tests below are deliberately classifier-level (pure-Python, no
gateway/asyncio bootstrap). The intent is that any future refactor of
the gateway startup wiring still has to satisfy these invariants —
the classifier is the contract.
"""

from __future__ import annotations

import logging

import pytest

from gateway.credential_disclosure import (
    PAID_CLOUD_PROVIDERS,
    CredentialDisclosure,
    build_disclosure_lines,
    classify_runtime_credential,
    should_emit_disclosure,
)


# ─── classify_runtime_credential ────────────────────────────────────────


class TestImplicitEnvDetection:
    """Implicit-env detection is the heart of the #32524 fix.

    "Implicit" means: the credential resolver landed on a paid cloud
    provider, the credential came from a process env var, and the user
    never deliberately wired that provider through ``hermes auth add``
    or pinned it in ``config.yaml``.
    """

    def test_anthropic_api_key_with_no_auth_store_or_config_is_implicit(self, monkeypatch):
        """The exact incident shape from #32524 — bare ANTHROPIC_API_KEY
        in the environment, no auth store entry, no config pin. Must
        classify as implicit and warning-worthy.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-1234567890")
        runtime = {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.anthropic.com",
            "api_key": "sk-ant-test-1234567890",
            "source": "env",
        }

        info = classify_runtime_credential(
            runtime,
            model="claude-sonnet-4-20250514",
            auth_store={"providers": {}, "credential_pool": {}},
            user_config={},  # no model.provider pin, no providers: entry
        )

        assert info.is_paid_cloud is True
        assert info.is_implicit_env is True
        assert info.is_warning_worthy is True
        assert info.env_var == "ANTHROPIC_API_KEY"
        assert info.provider == "anthropic"
        assert info.model == "claude-sonnet-4-20250514"

    def test_explicit_hermes_auth_add_suppresses_warning(self, monkeypatch):
        """User ran ``hermes auth add anthropic`` → auth store carries
        a providers entry → the env-var pickup is no longer "implicit"."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-1234567890")
        runtime = {
            "provider": "anthropic",
            "api_key": "sk-ant-test-1234567890",
            "source": "env",
        }

        info = classify_runtime_credential(
            runtime,
            auth_store={"providers": {"anthropic": {"api_key": "..."}}},
            user_config={},
        )

        assert info.is_paid_cloud is True
        assert info.is_implicit_env is False
        assert info.is_warning_worthy is False

    def test_credential_pool_entry_suppresses_warning(self, monkeypatch):
        """``hermes auth pool add anthropic ...`` is also explicit."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        runtime = {
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "source": "env",
        }

        info = classify_runtime_credential(
            runtime,
            auth_store={
                "providers": {},
                "credential_pool": {"anthropic": [{"api_key": "..."}]},
            },
            user_config={},
        )

        assert info.is_implicit_env is False
        assert info.is_warning_worthy is False

    def test_model_provider_pinned_to_anthropic_suppresses_warning(self, monkeypatch):
        """A user who set ``model.provider: anthropic`` in config.yaml
        has explicitly chosen the provider — env-var pickup is no
        longer surprising."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        runtime = {
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "source": "env",
        }

        info = classify_runtime_credential(
            runtime,
            auth_store={"providers": {}},
            user_config={"model": {"provider": "anthropic", "default": "claude-3-5-sonnet"}},
        )

        assert info.is_implicit_env is False
        assert info.is_warning_worthy is False

    def test_providers_dict_entry_suppresses_warning(self, monkeypatch):
        """A keyed ``providers: anthropic:`` entry counts as explicit."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        runtime = {
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "source": "env",
        }

        info = classify_runtime_credential(
            runtime,
            auth_store={"providers": {}},
            user_config={
                "providers": {
                    "anthropic": {
                        "base_url": "https://api.anthropic.com",
                        "key_env": "ANTHROPIC_API_KEY",
                    }
                }
            },
        )

        assert info.is_implicit_env is False
        assert info.is_warning_worthy is False

    def test_custom_providers_entry_suppresses_warning(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        runtime = {
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "source": "env",
        }

        info = classify_runtime_credential(
            runtime,
            auth_store={"providers": {}},
            user_config={
                "custom_providers": [
                    {"name": "Anthropic", "base_url": "https://api.anthropic.com"},
                ]
            },
        )

        assert info.is_implicit_env is False
        assert info.is_warning_worthy is False

    def test_explicit_provider_match_is_case_insensitive(self, monkeypatch):
        """User-typed config entries may differ in case from the
        canonical slug. The check must normalise both sides."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        runtime = {
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "source": "env",
        }
        info = classify_runtime_credential(
            runtime,
            auth_store={"providers": {"Anthropic": {}}},
            user_config={},
        )
        assert info.is_implicit_env is False


class TestPaidCloudAllowlist:
    """Free / local / OAuth-subscription providers must NOT trigger the
    paid-cloud warning even when picked up from an env var."""

    def test_lmstudio_local_endpoint_is_not_paid_cloud(self):
        runtime = {
            "provider": "lmstudio",
            "base_url": "http://127.0.0.1:1234/v1",
            "api_key": "lm-studio",
            "source": "env",
        }
        info = classify_runtime_credential(runtime)
        assert info.is_paid_cloud is False
        assert info.is_warning_worthy is False

    def test_nous_subscription_is_not_paid_cloud(self):
        # Nous billing is portal-subscription, not metered per call.
        runtime = {
            "provider": "nous",
            "api_key": "nous-test",
            "source": "portal",
        }
        info = classify_runtime_credential(runtime)
        assert info.is_paid_cloud is False
        assert info.is_warning_worthy is False

    def test_bedrock_is_excluded_from_paid_cloud_check(self):
        """AWS credentials come through the operator's IAM console, not
        a stray ``ANTHROPIC_API_KEY`` discovered on PATH — qualitatively
        different from raw cloud API keys and out of scope for #32524."""
        runtime = {
            "provider": "bedrock",
            "api_key": "aws-sdk",
            "source": "AWS_ACCESS_KEY_ID",
        }
        info = classify_runtime_credential(runtime)
        assert info.is_paid_cloud is False
        assert info.is_warning_worthy is False

    def test_oauth_token_source_is_not_implicit_env(self, monkeypatch):
        """Claude Code OAuth (``ANTHROPIC_TOKEN`` / refresh-token file)
        is a deliberate subscription — even though the source label is
        ``env``, the user went through the OAuth dance. We treat the
        ``oauth`` source label as non-env regardless of provider.
        """
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-test")
        runtime = {
            "provider": "anthropic",
            "api_key": "sk-ant-oat01-test",
            "source": "claude-code-oauth",
        }
        info = classify_runtime_credential(runtime)
        # source is not in the ENV token set → not implicit-env.
        assert info.is_implicit_env is False
        assert info.is_warning_worthy is False


class TestEnvVarIdentification:
    """The warning prose should name the actual env var that supplied
    the credential so the operator can grep their shell rc files for
    the offending export."""

    def test_anthropic_api_key_round_trips(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc123")
        runtime = {"provider": "anthropic", "api_key": "sk-ant-abc123", "source": "env"}
        info = classify_runtime_credential(runtime, auth_store={"providers": {}}, user_config={})
        assert info.env_var == "ANTHROPIC_API_KEY"

    def test_anthropic_token_takes_precedence_in_lookup(self, monkeypatch):
        """The registry lists ANTHROPIC_API_KEY first, but ANTHROPIC_TOKEN
        is also valid — whichever env var actually matches the key wins."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat-xyz")
        runtime = {"provider": "anthropic", "api_key": "sk-ant-oat-xyz", "source": "env"}
        info = classify_runtime_credential(runtime, auth_store={"providers": {}}, user_config={})
        assert info.env_var == "ANTHROPIC_TOKEN"

    def test_openai_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        runtime = {"provider": "openai", "api_key": "sk-openai-test", "source": "env"}
        info = classify_runtime_credential(runtime, auth_store={"providers": {}}, user_config={})
        # ``openai`` isn't in PROVIDER_REGISTRY by that slug (the
        # registry uses ``openai-api``), so we fall through to the
        # convention-based lookup.
        assert info.env_var == "OPENAI_API_KEY"

    def test_missing_env_match_returns_none(self):
        """When we can't confirm which env var sourced the key (e.g. the
        gateway is reading from a credential pool that mangled the
        value), env_var is None — the warning still fires, it just
        can't name the variable."""
        runtime = {
            "provider": "anthropic",
            "api_key": "no-matching-env-var",
            "source": "env",
        }
        info = classify_runtime_credential(
            runtime, auth_store={"providers": {}}, user_config={}, env={}
        )
        assert info.env_var is None
        assert info.is_warning_worthy is True


# ─── build_disclosure_lines ─────────────────────────────────────────────


class TestBuildDisclosureLines:
    def test_routine_disclosure_is_single_info_line(self):
        info = CredentialDisclosure(
            provider="lmstudio",
            model="qwen3-coder",
            base_url="http://127.0.0.1:1234/v1",
            source="env",
            env_var=None,
            is_paid_cloud=False,
            is_implicit_env=False,
        )
        lines = build_disclosure_lines(info)
        assert len(lines) == 1
        level, msg = lines[0]
        assert level == logging.INFO
        assert "lmstudio" in msg
        assert "qwen3-coder" in msg
        assert "127.0.0.1" in msg

    def test_warning_worthy_disclosure_includes_remediation(self):
        info = CredentialDisclosure(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            base_url="https://api.anthropic.com",
            source="env",
            env_var="ANTHROPIC_API_KEY",
            is_paid_cloud=True,
            is_implicit_env=True,
        )
        lines = build_disclosure_lines(info)
        levels = [lvl for lvl, _ in lines]
        text = "\n".join(msg for _, msg in lines)

        # Exactly one INFO summary + three WARNING lines (banner +
        # rationale + remediation).  Pinning the count keeps the test
        # honest about what we promise to log — drift here without a
        # test update breaks operator playbooks.
        assert levels[0] == logging.INFO
        assert levels[1:] == [logging.WARNING, logging.WARNING, logging.WARNING]
        assert "ANTHROPIC_API_KEY" in text
        assert "hermes auth add anthropic" in text
        assert "gateway.warn_implicit_paid_credentials" in text
        # The reporter's concrete pain point — must mention billing
        # explicitly so a skimming operator can't miss the implication.
        assert "billed" in text.lower()

    def test_empty_provider_emits_no_lines(self):
        """If credential resolution failed entirely (no provider known),
        the banner should be silent rather than emit a misleading line."""
        info = CredentialDisclosure(
            provider="",
            model="",
            base_url="",
            source="(unknown)",
            env_var=None,
            is_paid_cloud=False,
            is_implicit_env=False,
        )
        assert build_disclosure_lines(info) == []

    def test_warning_handles_unknown_env_var_gracefully(self):
        info = CredentialDisclosure(
            provider="anthropic",
            model="claude-3-5-sonnet",
            base_url="https://api.anthropic.com",
            source="env",
            env_var=None,
            is_paid_cloud=True,
            is_implicit_env=True,
        )
        lines = build_disclosure_lines(info)
        text = "\n".join(msg for _, msg in lines)
        assert "an environment variable" in text


# ─── should_emit_disclosure ─────────────────────────────────────────────


class TestShouldEmitDisclosure:
    """Operators must be able to silence the warning via config — but
    the default has to be ON so a fresh install can never be #32524'd."""

    def test_missing_config_defaults_to_on(self):
        assert should_emit_disclosure(None) is True
        assert should_emit_disclosure({}) is True

    def test_missing_gateway_section_defaults_to_on(self):
        assert should_emit_disclosure({"model": {"provider": "anthropic"}}) is True

    def test_explicit_true_passes_through(self):
        cfg = {"gateway": {"warn_implicit_paid_credentials": True}}
        assert should_emit_disclosure(cfg) is True

    def test_explicit_false_silences(self):
        cfg = {"gateway": {"warn_implicit_paid_credentials": False}}
        assert should_emit_disclosure(cfg) is False

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", "OFF"])
    def test_string_falsy_values_silence(self, value):
        cfg = {"gateway": {"warn_implicit_paid_credentials": value}}
        assert should_emit_disclosure(cfg) is False

    @pytest.mark.parametrize("value", ["true", "True", "1", "yes", "on", ""])
    def test_string_truthy_values_enable(self, value):
        # Empty string is treated as ON — for a security disclosure the
        # safer failure mode is to err on the side of more logging when
        # the operator's config value is malformed/blank.
        cfg = {"gateway": {"warn_implicit_paid_credentials": value}}
        assert should_emit_disclosure(cfg) is True


# ─── Allowlist contract ─────────────────────────────────────────────────


class TestPaidCloudAllowlistContract:
    """Don't change-detector the exact list — but keep the invariants
    that motivated the allowlist in the first place. Drift here without
    a test update would silently re-open #32524 for whatever provider
    quietly slipped out of the set."""

    def test_anthropic_must_be_paid_cloud(self):
        # The literal incident provider.
        assert "anthropic" in PAID_CLOUD_PROVIDERS

    def test_openai_must_be_paid_cloud(self):
        # Same threat shape: env-discovered OPENAI_API_KEY → silent metered billing.
        assert "openai" in PAID_CLOUD_PROVIDERS
        assert "openai-api" in PAID_CLOUD_PROVIDERS

    def test_local_providers_must_not_be_paid_cloud(self):
        for slug in ("lmstudio", "ollama", "ollama-cloud", "llama-cpp"):
            assert slug not in PAID_CLOUD_PROVIDERS, (
                f"{slug} is local / free-tier and must not trigger "
                "the paid-cloud disclosure warning."
            )

    def test_oauth_subscription_providers_must_not_be_paid_cloud(self):
        # OAuth-backed subscription endpoints — user already went
        # through a deliberate auth flow, billing is subscription not
        # per-request.
        for slug in ("nous", "openai-codex", "xai-oauth", "google-gemini-cli", "qwen-oauth"):
            assert slug not in PAID_CLOUD_PROVIDERS, (
                f"{slug} is subscription-billed; warning would be noise."
            )

    def test_cloud_iam_providers_must_not_be_paid_cloud(self):
        # AWS / GCP cred chains are governed by the cloud console — out of
        # scope for the "stray env var" threat model #32524 describes.
        for slug in ("bedrock",):
            assert slug not in PAID_CLOUD_PROVIDERS
