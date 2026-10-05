"""Single-owner /model argument parsing (hermes_cli.model_switch.parse_model_switch_args)."""


from hermes_cli.model_switch import (
    MODEL_SWITCH_ERR_LIST_BY_PROVIDER_GLOBAL,
    MODEL_SWITCH_ERR_LIST_BY_PROVIDER_VALUE,
    MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL,
    MODEL_SWITCH_ERROR_TEXT,
    parse_model_switch_args,
)


# ---------------------------------------------------------------------------
# parse_model_switch_args — the ONE parser
# ---------------------------------------------------------------------------


def test_provider_flag_and_scopes():
    req = parse_model_switch_args("sonnet --provider anthropic --global")
    assert req.target == "sonnet"
    assert req.explicit_provider == "anthropic"
    assert req.is_global is True
    assert req.scope == "global"
    assert req.errors == ()

    assert parse_model_switch_args("sonnet").scope == "default"
    assert parse_model_switch_args("sonnet --session").scope == "session"
    assert parse_model_switch_args("sonnet \u2013session").scope == "session"  # iOS/Telegram en-dash
    assert parse_model_switch_args("sonnet --once").scope == "once"
    assert parse_model_switch_args("--refresh").force_refresh is True


def test_once_with_global_conflict():
    req = parse_model_switch_args("sonnet --once --global")
    assert MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL in req.errors
    assert MODEL_SWITCH_ERROR_TEXT[MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL] in req.error_messages()


# ---------------------------------------------------------------------------
# --list-by-provider — the per-user /model list cap
# ---------------------------------------------------------------------------


def test_list_by_provider_takes_a_positive_int():
    assert parse_model_switch_args("--list-by-provider 5").list_by_provider == 5
    assert parse_model_switch_args("--list-by-provider 1").list_by_provider == 1
    # A cap-only call is a listing, not a switch: no target, no persistence scope.
    req = parse_model_switch_args("--list-by-provider 5")
    assert req.target == "" and req.scope == "default" and req.errors == ()


def test_list_by_provider_rejects_values_that_are_not_a_count():
    for raw in ("--list-by-provider 0", "--list-by-provider -3", "--list-by-provider many"):
        req = parse_model_switch_args(raw)
        assert MODEL_SWITCH_ERR_LIST_BY_PROVIDER_VALUE in req.errors, raw
        assert req.list_by_provider is None, raw
        assert MODEL_SWITCH_ERROR_TEXT[MODEL_SWITCH_ERR_LIST_BY_PROVIDER_VALUE] in req.error_messages()


def test_list_by_provider_is_per_user_so_global_is_refused():
    # --global means "write config.yaml"; a cap must never reach another user's list.
    req = parse_model_switch_args("sonnet --list-by-provider 5 --global")
    assert MODEL_SWITCH_ERR_LIST_BY_PROVIDER_GLOBAL in req.errors
    assert "config.yaml" in MODEL_SWITCH_ERROR_TEXT[MODEL_SWITCH_ERR_LIST_BY_PROVIDER_GLOBAL]


def test_list_by_provider_coexists_with_the_other_flags():
    req = parse_model_switch_args("sonnet --provider anthropic --list-by-provider 12 --session")
    assert req.errors == ()
    assert req.list_by_provider == 12
    assert req.explicit_provider == "anthropic"
    assert req.scope == "session"

    req = parse_model_switch_args("sonnet --once --list-by-provider 3")
    assert req.errors == () and req.list_by_provider == 3 and req.scope == "once"


def test_list_by_provider_survives_the_ios_dash_normalisation():
    # Telegram/iOS turn ``--`` into an en dash; the flag keyword must still parse.
    req = parse_model_switch_args("\u2013list-by-provider 7")
    assert req.errors == () and req.list_by_provider == 7


def test_legacy_args_are_unchanged_by_the_new_flag():
    req = parse_model_switch_args("sonnet --reasoning high")
    assert req.list_by_provider is None and req.errors == ()
    assert req.reasoning_effort == "high"
    # A model id containing the flag's name is still a model id, not a flag.
    assert parse_model_switch_args("openai/list-by-provider").target == "openai/list-by-provider"
