"""The channel layer: the grant, the compile, and what must never be writable.

The audit (``docs/NOVA_CHANNEL_AUDIT.md``) found the runtime already has adapters, routing,
webhook verification and a durable outbound ledger. So almost nothing here tests messaging.
What it tests is the part the runtime deliberately does not have — *which agents a connection
is allowed to reach* — and the part a commercial product must never get wrong: a credential
in a file somebody commits.
"""

from __future__ import annotations

import pytest

from nova.channels import (
    PROVIDERS,
    ChannelSpec,
    Verification,
    check_agents_exist,
    get_provider,
    parse_channels,
)
from nova.errors import SpecError


def declaration(**overrides):
    entry = {
        "id": "support-telegram",
        "provider": "telegram",
        "allowed_agents": ["customer-support"],
        "routes": [{"agent": "customer-support"}],
    }
    entry.update(overrides)
    return {"channels": [entry]}


# -- the grant ---------------------------------------------------------------


def test_a_route_outside_the_grant_is_refused():
    """The centre of the whole layer. A route says where a message goes; the grant says
    where it is *allowed* to go, and those diverge the moment somebody edits a route."""
    with pytest.raises(SpecError, match="does not grant"):
        parse_channels(
            declaration(allowed_agents=["customer-support"], routes=[{"agent": "operations"}])
        )


def test_a_connection_with_no_grant_is_refused_rather_than_defaulted():
    """Empty could mean 'no agent' or 'every agent'. One of those is a channel into the
    entire workforce, so neither is inferred."""
    with pytest.raises(SpecError, match="needs allowed_agents"):
        parse_channels(declaration(allowed_agents=[]))


def test_a_grant_naming_an_unknown_agent_is_refused():
    channels = parse_channels(declaration(allowed_agents=["ghost"], routes=[]))
    with pytest.raises(SpecError, match="does not contain"):
        check_agents_exist(channels, ["customer-support"])


def test_the_grant_reaches_the_runtime_allowlist():
    """NOVA's parser refusing is the first gate. Compiling the grant into the served set
    makes the runtime's own fail-closed check the second one."""
    from nova.runtime.hermes.channels import plan

    channels = parse_channels(
        declaration(allowed_agents=["customer-support", "operations"])
    )
    assert plan(channels).served_agents == ("customer-support", "operations")


# -- credentials -------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["token", "api_key", "app_secret", "password", "bot_secret", "credential"]
)
def test_no_credential_field_can_be_declared(field):
    """A key in this file is in version control, in every clone, and in the history after
    it is removed. Refusing at parse time is the only moment where saying no still helps."""
    with pytest.raises(SpecError, match="carry a credential"):
        parse_channels(declaration(**{field: "xoxb-not-a-real-token"}))


def test_a_credential_nested_in_settings_is_also_refused():
    with pytest.raises(SpecError, match="carry a credential"):
        parse_channels(declaration(settings={"auth": {"api_key": "sk-nope"}}))


def test_the_compiler_refuses_to_pass_a_credential_through():
    """Belt and braces: even if a settings key escaped the parser, the compiler will not
    hand it to the runtime."""
    from nova.errors import RuntimeAdapterError
    from nova.runtime.hermes.channels import plan

    smuggled = ChannelSpec(
        id="x-chan", provider="telegram", allowed_agents=("a",), settings={"token": "xoxb-x"}
    )
    with pytest.raises(RuntimeAdapterError, match="credential never belongs"):
        plan([smuggled])


def test_nova_reports_credential_names_and_never_values():
    provider = get_provider("slack")
    assert provider.required_env == ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN")
    assert all(isinstance(name, str) and name.isupper() for name in provider.required_env)


# -- honesty about what is supported -----------------------------------------


def test_no_provider_claims_field_validation_it_does_not_have():
    """A tick that means 'a plugin directory exists' is the tick a customer signs a contract
    on. Promoting one requires connecting a real provider, not editing this catalogue."""
    for provider in PROVIDERS:
        assert provider.verification is Verification.SOURCE_READ, (
            f"{provider.id} claims {provider.verification.value} — if that is now true, the "
            f"evidence belongs in docs/PHASE_9.md before this test changes"
        )


def test_unknown_capabilities_are_unknown_not_false():
    """The audit's rule: unknown is never rendered as yes, and never quietly as no either."""
    whatsapp = get_provider("whatsapp")
    assert whatsapp.capabilities["groups"].supported is None


def test_a_provider_nobody_opened_is_not_offered():
    """The runtime bundles 22 adapters. Offering all of them would be the checklist support
    the phase brief forbids."""
    with pytest.raises(SpecError, match="is not a channel NOVA offers"):
        parse_channels(declaration(provider="matrix"))


def test_a_webhook_provider_declares_that_it_needs_a_public_endpoint():
    """A deployment decision with a security boundary, not a checkbox."""
    assert get_provider("whatsapp").needs_public_endpoint is True
    assert get_provider("telegram").needs_public_endpoint is False


# -- routing -----------------------------------------------------------------


def test_routes_compile_most_specific_first():
    from nova.runtime.hermes.channels import plan

    channels = parse_channels(
        declaration(
            allowed_agents=["customer-support", "operations"],
            routes=[
                {"agent": "customer-support"},
                {"agent": "operations", "conversation": "-100123"},
            ],
        )
    )
    routes = plan(channels).routes
    assert "chat_id" in routes[0], "a named conversation must outrank the catch-all"
    assert routes[0]["profile"] == "operations"


def test_a_numeric_conversation_id_is_compiled_as_a_string():
    """The runtime warns that an unquoted numeric id loaded as an int 'can never match an
    inbound id'. Compiling it as a string is how NOVA makes that impossible."""
    from nova.runtime.hermes.channels import plan

    channels = parse_channels(
        declaration(routes=[{"agent": "customer-support", "conversation": 12345}])
    )
    assert plan(channels).routes[0]["chat_id"] == "12345"


def test_two_catch_all_routes_are_refused():
    with pytest.raises(SpecError, match="more than one catch-all"):
        parse_channels(
            declaration(routes=[{"agent": "customer-support"}, {"agent": "customer-support"}])
        )


def test_a_thread_without_its_conversation_is_refused():
    with pytest.raises(SpecError, match="can never match"):
        parse_channels(
            declaration(routes=[{"agent": "customer-support", "thread": "77"}])
        )


def test_a_connected_channel_with_no_routes_warns_rather_than_silently_defaulting():
    from nova.runtime.hermes.channels import plan

    channels = parse_channels(declaration(routes=[]))
    warnings = plan(channels).warnings
    assert any("no routes" in w for w in warnings)


def test_a_disabled_channel_compiles_no_routes():
    from nova.runtime.hermes.channels import plan

    channels = parse_channels(declaration(enabled=False))
    assert plan(channels).routes == ()
    assert plan(channels).served_agents == ()


# -- the merge ---------------------------------------------------------------


def test_operator_owned_keys_survive_a_channel_apply():
    """An operator who loses a setting to an apply will not trust the next one."""
    from nova.runtime.hermes.channels import merge_into, plan

    existing = {
        "gateway": {"max_concurrent_sessions": 7, "bind": "127.0.0.1"},
        "quick_commands": {"hello": "say hi"},
    }
    merged = merge_into(existing, plan(parse_channels(declaration())))
    assert merged["gateway"]["max_concurrent_sessions"] == 7
    assert merged["gateway"]["bind"] == "127.0.0.1"
    assert merged["quick_commands"] == {"hello": "say hi"}


def test_a_revoked_channel_stops_delivering():
    """The one outcome this layer must never produce is a removed channel that still
    delivers, so the owned keys are replaced rather than merged."""
    from nova.runtime.hermes.channels import merge_into, plan

    connected = merge_into({}, plan(parse_channels(declaration())))
    assert connected["gateway"]["profile_routes"]

    revoked = merge_into(connected, plan(()))
    assert revoked["gateway"]["profile_routes"] == []
    assert "multiplex_profile_allowlist" not in revoked["gateway"]


def test_a_token_in_an_existing_config_is_stripped_on_merge():
    """The runtime's PlatformConfig can carry a raw token from YAML. NOVA will not preserve
    one into a file it writes."""
    from nova.runtime.hermes.channels import merge_into, plan

    existing = {"gateway": {"platforms": {"telegram": {"token": "leaked-value"}}}}
    merged = merge_into(existing, plan(parse_channels(declaration())))
    assert "token" not in merged["gateway"]["platforms"]["telegram"]


def test_the_owned_key_list_is_what_the_merge_actually_touches():
    """Pins the promise: everything outside OWNED_GATEWAY_KEYS is preserved byte-for-byte."""
    from nova.runtime.hermes.channels import OWNED_GATEWAY_KEYS, merge_into, plan

    existing = {"gateway": {f"operator_key_{i}": i for i in range(5)}}
    merged = merge_into(existing, plan(parse_channels(declaration())))
    changed = {
        key for key in merged["gateway"]
        if existing["gateway"].get(key) != merged["gateway"][key]
    }
    assert changed <= set(OWNED_GATEWAY_KEYS), f"unexpectedly wrote {changed - set(OWNED_GATEWAY_KEYS)}"


# -- identity ----------------------------------------------------------------


def test_duplicate_connection_ids_are_refused():
    doc = declaration()
    doc["channels"].append(dict(doc["channels"][0]))
    with pytest.raises(SpecError, match="declared twice"):
        parse_channels(doc)


def test_a_connection_id_must_be_usable_in_a_url_and_a_log():
    with pytest.raises(SpecError, match="not a connection id"):
        parse_channels(declaration(id="Not An Id"))


def test_absent_channels_means_no_channels_not_an_error(tmp_path):
    from nova.channels import load_channels

    assert load_channels(tmp_path) == ()


def test_a_float_conversation_id_is_refused_rather_than_stringified():
    """Found the same way the runtime found it. ``12345.0`` stringifies to something that can
    never equal an inbound id, so coercing it would recreate the silent no-match that the
    coercion exists to prevent."""
    with pytest.raises(SpecError, match="can never match"):
        parse_channels(
            declaration(routes=[{"agent": "customer-support", "conversation": 1234.5}])
        )


def test_a_boolean_is_not_a_conversation_id():
    """``bool`` is an ``int`` subclass and never a valid id."""
    with pytest.raises(SpecError, match="not a conversation id"):
        parse_channels(
            declaration(routes=[{"agent": "customer-support", "conversation": True}])
        )


# -- failure and recovery (Phase J) ------------------------------------------


def test_a_corrupt_config_is_refused_rather_than_overwritten(tmp_path):
    """An operator's config.yaml carries their bind address and their session limits.
    Overwriting one we could not parse would destroy settings to fix a channel."""
    from nova.errors import RuntimeAdapterError
    from nova.runtime.hermes.channels import apply

    (tmp_path / "config.yaml").write_text("gateway: [unclosed\n", encoding="utf-8")
    with pytest.raises(RuntimeAdapterError, match="could not be read as YAML"):
        apply(tmp_path, parse_channels(declaration()))
    assert "unclosed" in (tmp_path / "config.yaml").read_text(encoding="utf-8")


def test_a_config_that_is_not_a_mapping_is_refused(tmp_path):
    from nova.errors import RuntimeAdapterError
    from nova.runtime.hermes.channels import apply

    (tmp_path / "config.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(RuntimeAdapterError, match="not a YAML mapping"):
        apply(tmp_path, parse_channels(declaration()))


def test_applying_twice_changes_nothing(tmp_path):
    """An operator re-running apply after an interrupted deploy must not see drift."""
    from nova.runtime.hermes.channels import apply

    channels = parse_channels(declaration())
    apply(tmp_path, channels)
    first = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    apply(tmp_path, channels)
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == first


def test_a_missing_agent_env_reports_rather_than_raises(tmp_path):
    """A channel waiting on a credential is not broken configuration; it is configuration
    waiting for the operator step NOVA deliberately cannot take."""
    from nova.runtime.hermes.channels import readiness

    rows = readiness(parse_channels(declaration()), home=tmp_path)
    assert rows[0]["ready"] is False
    assert rows[0]["missing_by_agent"]["customer-support"] == ["TELEGRAM_BOT_TOKEN"]


def test_readiness_reports_names_never_values(tmp_path):
    from nova.runtime.hermes.channels import readiness

    profile = tmp_path / "profiles" / "customer-support"
    profile.mkdir(parents=True)
    (profile / ".env").write_text("TELEGRAM_BOT_TOKEN=super-secret-value\n", encoding="utf-8")

    rows = readiness(parse_channels(declaration()), home=tmp_path)
    assert rows[0]["missing_by_agent"]["customer-support"] == []
    assert "super-secret-value" not in repr(rows), "a credential value reached a report"
