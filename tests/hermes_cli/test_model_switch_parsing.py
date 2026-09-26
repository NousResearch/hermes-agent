"""Single-owner /model argument parsing (hermes_cli.model_switch.parse_model_switch_args)."""


import pytest

from hermes_cli.model_switch import (
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

    assert parse_model_switch_args("sonnet --session").scope == "session"
    assert parse_model_switch_args("sonnet --once").scope == "once"
    assert parse_model_switch_args("--refresh").force_refresh is True


def test_once_with_global_conflict():
    req = parse_model_switch_args("sonnet --once --global")
    assert MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL in req.errors
    assert MODEL_SWITCH_ERROR_TEXT[MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL] in req.error_messages()


# Discord's native /model shows its option as ``name:<value>``; that text pasted into another
# surface must parse to the same target as the bare value.
@pytest.mark.parametrize("raw,target", [
    ("name:my-provider/some-model", "my-provider/some-model"),
    ("NAME=my-provider/some-model", "my-provider/some-model"),
    ("model:some-model", "some-model"),
    ("model=some-model", "some-model"),
    ("name: some-model", "some-model"),
    ("name:some-model --provider my-provider", "some-model"),
    ("some-model", "some-model"),
    # a colon that is not an option label is left for switch_model
    ("my-provider:some-model", "my-provider:some-model"),
    ("anthropic/claude-sonnet-4.5:extended", "anthropic/claude-sonnet-4.5:extended"),
    ("qwen3.5:4b", "qwen3.5:4b"),
])
def test_parser_strips_discord_option_label(raw, target):
    assert parse_model_switch_args(raw).target == target


def test_parser_keeps_flags_with_label():
    req = parse_model_switch_args("name:some-model --provider my-provider --global")
    assert req.target == "some-model"
    assert req.explicit_provider == "my-provider"
    assert req.is_global is True
    assert req.errors == ()
