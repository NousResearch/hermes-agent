"""Deployment configuration: the operator's half, and the line NOVA must not cross.

The property under test throughout is the same one: **NOVA writes the name of a credential
and never its value.** Everything else here — the vocabulary split, the reserved keys, the
readiness report — exists to make that property survivable in a real deployment rather than
merely true in principle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nova._fields import Doc
from nova.errors import SpecError
from nova.runtime.hermes.provider import (
    MINIMUM_CONTEXT_WINDOW,
    build_provider_config,
    required_env,
)
from nova.runtime.hermes.readiness import check, read_env_file
from nova.spec import load_bundle
from nova.spec.deployment import (
    DeploymentSpec,
    ProviderSpec,
    env_references,
    load_deployment,
)

from .conftest import EXAMPLE_BUNDLE


# -- secrets never enter a declaration -------------------------------------


@pytest.mark.parametrize(
    "value",
    ["sk-live-abcdef123456", "AKIAIOSFODNN7EXAMPLE", "hunter2!", "ghp_aaaabbbbcccc-x"],
)
def test_a_literal_credential_is_refused_where_a_name_belongs(value):
    """Parse time is the only moment refusing helps: once the key is in version control it
    is in every clone and in the history after it is removed."""
    with pytest.raises(SpecError, match="must never contain the credential"):
        ProviderSpec.parse(Doc({"api_key_env": value}))


def test_a_variable_name_is_accepted():
    assert ProviderSpec.parse(Doc({"api_key_env": "ACME_LLM_KEY"})).api_key_env == "ACME_LLM_KEY"


def test_a_literal_credential_in_the_passthrough_is_refused():
    with pytest.raises(SpecError, match="looks like a credential"):
        DeploymentSpec.parse({"runtime_config": {"gateway": {"api_token": "abc123secret"}}})


def test_a_referenced_credential_in_the_passthrough_is_allowed():
    spec = DeploymentSpec.parse(
        {"runtime_config": {"gateway": {"api_token": "${ACME_GATEWAY_TOKEN}"}}}
    )
    assert spec.runtime_config["gateway"]["api_token"] == "${ACME_GATEWAY_TOKEN}"


def test_nova_never_writes_a_credential_into_a_profile():
    """The end-to-end form of the rule, asserted on what materialization actually emits."""
    provider = ProviderSpec(
        provider="acme-gateway", endpoint="${ACME_LLM_URL}", api_key_env="ACME_LLM_KEY",
        model="m1", context_window=200_000,
    )
    config, _ = build_provider_config(provider)
    entry = config["custom_providers"][0]
    assert entry["key_env"] == "ACME_LLM_KEY"
    # api_key would put the resolved secret into the loaded config object. key_env does not.
    assert "api_key" not in entry


# -- the passthrough may not undo a governance control ----------------------


@pytest.mark.parametrize(
    "key", ["plugins", "approvals", "agent", "delegation", "kanban", "nova", "model"]
)
def test_the_passthrough_refuses_nova_compiled_keys(key):
    """An ``approvals`` block that re-permits a denied tool, or a ``plugins`` block that
    disables the enforcement plugin, would be invisible to every NOVA-side check."""
    with pytest.raises(SpecError, match="may not set"):
        DeploymentSpec.parse({"runtime_config": {key: {"anything": 1}}})


def test_the_passthrough_allows_what_nova_does_not_model():
    spec = DeploymentSpec.parse({"runtime_config": {"tools": {"tool_search": {"enabled": "off"}}}})
    assert spec.runtime_config["tools"]["tool_search"]["enabled"] == "off"


def test_nova_compiled_keys_survive_a_passthrough_that_shares_their_shape():
    """Second half of the guarantee: the parse refuses reserved keys, and the write order
    puts NOVA's own compilation last regardless."""
    from nova.runtime.hermes.materialize import build_config

    bundle = load_bundle(EXAMPLE_BUNDLE)
    config = build_config(
        bundle.agent("customer-support"),
        policy=True,
        knowledge=True,
        runtime_config={"tools": {"tool_search": {"enabled": "off"}}},
    )
    assert config["plugins"]["enabled"]
    assert config["tools"]["tool_search"]["enabled"] == "off"


# -- tenant defaults, per-agent overrides ----------------------------------


def test_an_agent_override_wins_field_by_field():
    """Wholesale replacement would force an agent changing one value to restate the whole
    block, and restated config is where deployments drift."""
    tenant = ProviderSpec(
        provider="acme-gateway", endpoint="${URL}", api_key_env="K", model="big", region="eu"
    )
    merged = tenant.merged_with(ProviderSpec(model="small"))
    assert merged.model == "small"
    assert (merged.endpoint, merged.api_key_env, merged.region) == ("${URL}", "K", "eu")


def test_an_absent_override_means_inherit_not_clear():
    tenant = ProviderSpec(provider="p", endpoint="${U}", context_window=200_000)
    assert tenant.merged_with(ProviderSpec()).context_window == 200_000


def test_the_bundle_resolves_one_providers_answer_per_agent():
    bundle = load_bundle(EXAMPLE_BUNDLE)
    resolved = bundle.provider_for("customer-support")
    assert resolved.provider
    assert resolved.api_key_env == "ACME_LLM_KEY"


# -- translation into the runtime's vocabulary -----------------------------


def test_a_custom_provider_is_addressed_by_its_own_name():
    """``provider: custom`` resolves to nothing and fails as "No LLM provider configured" —
    the runtime routes by the custom entry's name."""
    config, _ = build_provider_config(
        ProviderSpec(provider="acme-gateway", endpoint="http://x/v1", model="m")
    )
    assert config["custom_providers"][0]["name"] == "acme-gateway"
    assert config["model"]["provider"] == "acme-gateway"


def test_a_native_provider_gets_no_custom_entry():
    config, _ = build_provider_config(ProviderSpec(provider="bedrock", region="eu-west-1"))
    assert "custom_providers" not in config
    assert config["model"]["region"] == "eu-west-1"


def test_a_native_provider_warns_about_fields_it_will_ignore():
    """Recorded-but-unread configuration looks configured and changes nothing."""
    _, warnings = build_provider_config(
        ProviderSpec(provider="bedrock", endpoint="http://x", api_key_env="K")
    )
    assert any("ignores a declared endpoint" in w for w in warnings)
    assert any("will not read it" in w for w in warnings)


def test_a_native_providers_key_is_not_demanded_by_readiness():
    """It would send an operator to provision a variable nothing reads, and a readiness
    report that cries wolf is one nobody reads the third time."""
    assert "K" not in required_env(ProviderSpec(provider="bedrock", api_key_env="K"))


def test_a_custom_provider_with_no_endpoint_warns():
    _, warnings = build_provider_config(ProviderSpec(provider="mine", model="m"))
    assert any("nowhere to send a request" in w for w in warnings)


def test_a_context_window_below_the_runtime_floor_warns_at_apply():
    """The runtime refuses it at startup with a message that reads as unrelated to the
    config someone just changed."""
    _, warnings = build_provider_config(
        ProviderSpec(provider="p", endpoint="http://x", context_window=32_000)
    )
    assert any(str(MINIMUM_CONTEXT_WINDOW // 1000) in w or "below" in w for w in warnings)


def test_an_undeclared_provider_writes_nothing():
    """The state every deployment was in before this existed."""
    config, warnings = build_provider_config(ProviderSpec())
    assert config == {} and warnings == []


# -- env references ---------------------------------------------------------


def test_a_defaulted_reference_is_not_required():
    assert env_references("${REGION:-eu-west-1}", required_only=True) == ()
    assert env_references("${REGION:-eu-west-1}") == ("REGION",)


def test_an_endpoint_reference_is_required():
    spec = ProviderSpec(endpoint="${ACME_LLM_URL}", api_key_env="K")
    assert set(spec.required_env) == {"ACME_LLM_URL", "K"}


def test_an_endpoint_is_not_expanded_at_load():
    """It must reach the worker as a reference and resolve in the worker's environment —
    a different machine from whoever ran ``nova apply``."""
    spec = ProviderSpec.parse(Doc({"endpoint": "${ACME_LLM_URL}"}))
    assert spec.endpoint == "${ACME_LLM_URL}"


# -- readiness --------------------------------------------------------------


def test_readiness_reports_missing_variables_and_where_they_go(tmp_path):
    result = check("a", ["ACME_LLM_KEY"], profile_dir=tmp_path, environ={})
    assert not result.ready
    assert result.missing == ("ACME_LLM_KEY",)
    assert ".env" in result.explain()


def test_readiness_finds_a_variable_in_the_profile_env_file(tmp_path):
    (tmp_path / ".env").write_text("ACME_LLM_KEY=whatever\n", encoding="utf-8")
    result = check("a", ["ACME_LLM_KEY"], profile_dir=tmp_path, environ={})
    assert result.ready
    assert result.resolved_from["ACME_LLM_KEY"].endswith(".env")


def test_readiness_distinguishes_host_wide_from_per_agent(tmp_path):
    """A process-env value is the same for every agent on the box — worth saying, because
    it stops being per-agent the moment a second tenant lands on the host."""
    result = check("a", ["K"], profile_dir=tmp_path, environ={"K": "v"})
    assert result.ready
    assert result.resolved_from["K"] == "process environment"


def test_the_env_file_parser_keeps_names_and_discards_values(tmp_path):
    """So nothing in NOVA can log, audit or display a credential it parsed on the way to
    answering "is this set?"."""
    path = tmp_path / ".env"
    path.write_text(
        "# comment\nexport ACME_LLM_KEY=super-secret\nEMPTY=\nnot_an_assignment\n",
        encoding="utf-8",
    )
    parsed = read_env_file(path)
    assert set(parsed) == {"ACME_LLM_KEY", "EMPTY"}
    assert all(value == "" for value in parsed.values())
    assert "super-secret" not in str(parsed)


def test_a_missing_env_file_is_not_an_error(tmp_path):
    assert read_env_file(tmp_path / "absent") == {}


# -- the example bundle -----------------------------------------------------


def test_the_example_declares_deployment_settings():
    bundle = load_bundle(EXAMPLE_BUNDLE)
    assert bundle.deployment.provider.declared
    # The endpoint reference has no default, so it is genuinely required too.
    assert set(bundle.deployment.required_env) == {"ACME_LLM_KEY", "ACME_LLM_URL"}


def test_the_example_contains_no_literal_credential():
    text = (Path(EXAMPLE_BUNDLE) / "deployment.yaml").read_text(encoding="utf-8")
    assert "api_key:" not in text
    assert "api_key_env: ACME_LLM_KEY" in text


def test_an_absent_deployment_file_is_a_valid_state(tmp_path):
    assert load_deployment(tmp_path).to_dict() == {}


@pytest.mark.parametrize(
    "value,why",
    [
        ("AKIAIOSFODNN7EXAMPLE", "an AWS key id is a valid env-var shape"),
        ("ghp_1234567890abcdefghij", "a GitHub token"),
        ("AIzaSyD1234567890abcdefgh", "a Google API key"),
        ("xoxb-123456789012-abcdef", "a Slack token"),
    ],
)
def test_credentials_that_are_shaped_like_variable_names_are_still_refused(value, why):
    """The shape check alone is not enough, and this is the case that proves it: an AWS
    access key id passes ``[A-Z_][A-Z0-9_]*`` cleanly."""
    with pytest.raises(SpecError, match="looks like a credential"):
        ProviderSpec.parse(Doc({"api_key_env": value}))


@pytest.mark.parametrize("name", ["ACME_LLM_KEY", "OPENAI_API_KEY", "K", "AWS_REGION", "MY_VAR_2"])
def test_real_variable_names_are_not_mistaken_for_credentials(name):
    """A check that rejects legitimate names is one people work around."""
    assert ProviderSpec.parse(Doc({"api_key_env": name})).api_key_env == name
