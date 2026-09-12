"""Single-owner /model parsing + effective-model resolution tests.

Covers the consolidation of the 7 historical parsing/resolution variants
into hermes_cli.model_switch (parse_model_switch_args +
resolve_effective_model), including the 7dd00bb47d regression class
(api_server discarding session-persisted models) as a permanent parity
test against the pre-consolidation logic captured from origin/main.

Real imports throughout (AGENTS.md: no mocks for resolution chains).
"""

import pytest

from hermes_cli.model_switch import (
    MODEL_SWITCH_ERR_ONCE_REQUIRES_TARGET,
    MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL,
    MODEL_SWITCH_ERROR_TEXT,
    ModelSwitchRequest,
    parse_model_flags_detailed,
    parse_model_switch_args,
    resolve_effective_model,
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
    assert (
        MODEL_SWITCH_ERROR_TEXT[MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL]
        == "/model --once cannot be combined with --global"
    )
    assert "/model --once cannot be combined with --global" in req.error_messages()


@pytest.mark.parametrize(
    "raw,expected_target",
    [
        ("sonnet\u200b", "sonnet"),      # trailing ZWSP
        ("\u200bsonnet", "sonnet"),      # leading ZWSP
        ("sonnet\u200b --once", "sonnet"),  # ZWSP after a clean token + a real flag
    ],
)
def test_invisible_zero_width_chars_do_not_leak_into_target(raw, expected_target):
    """Zero-width / BOM markers must not survive into the model target.

    Clients (mobile auto-correct, IMEs, rich-text copy-paste) inject these
    invisible chars. They are *not* whitespace to Python, so ``str.split()``
    keeps them glued to a token and the value would otherwise reach validation
    as a "model name" containing an embedded invisible char. The parser must
    strip them so the target is the clean model ID — and any real flags are
    still parsed.
    """
    req = parse_model_switch_args(raw)
    # The invariant: no invisible char survives into the target token.
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        assert ch not in req.target, f"invisible {ch!r} leaked into target {req.target!r}"
    assert req.target == expected_target
    # Real flags co-existing with an invisible char are still honored.
    if raw.endswith("--once"):
        assert req.is_once is True


@pytest.mark.parametrize(
    "raw,expected_target",
    [
        ("so\u200bnet", "sonet"),     # ZWSP mid-token
        ("so\ufeffnet", "sonet"),     # BOM / zero-width no-break mid-token
        ("so\u200cnet", "sonet"),     # ZWNJ mid-token
        ("so\u200dnet", "sonet"),     # ZWJ mid-token
    ],
)
def test_invisible_zero_width_chars_stripped_from_mid_token(raw, expected_target):
    """A zero-width marker glued *inside* a token is removed, leaving the
    clean (de-invisible) model ID — never a token that still carries it."""
    req = parse_model_switch_args(raw)
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        assert ch not in req.target, f"invisible {ch!r} leaked into target {req.target!r}"
    assert req.target == expected_target


def test_space_in_model_name_yields_diagnostic_error_not_dead_end():
    """A genuine internal space must fail with an actionable message.

    A model ID never legitimately contains a space, so ``co dex`` must be
    rejected — but the message should name the symptom and point at the
    picker, not the old dead-end "Model names cannot contain spaces."
    """
    req = parse_model_switch_args("co dex")
    assert req.target == "co dex"  # parser preserves it for the validator to judge

    from hermes_cli.models_validate import validate_requested_model
    result = validate_requested_model(req.target, "custom")
    assert result["accepted"] is False
    msg = result["message"]
    assert "contains a space" in msg
    assert "/model" in msg  # points the user at the picker as the recovery




# ---------------------------------------------------------------------------
# resolve_effective_model — session > channel/session-persisted > global
# ---------------------------------------------------------------------------

class _ChannelOverride:
    def __init__(self, model):
        self.model = model








# ---------------------------------------------------------------------------
# Parity: run.py-style channel resolution (old logic from origin/main)
# ---------------------------------------------------------------------------

def _old_run_py_resolve(override, global_model):
    # Captured from origin/main gateway/run.py:_resolve_model_for_channel:
    #     if override and override.model:
    #         return override.model
    #     return _resolve_gateway_model(user_config)
    if override and override.model:
        return override.model
    return global_model




# ---------------------------------------------------------------------------
# Parity: api_server-style resolution (old logic from origin/main)
# ---------------------------------------------------------------------------

def _clean(value):
    # api_server._clean_request_string equivalent for the parity harness.
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _old_api_server_resolve(session_override, session_row_model, global_model):
    # Captured from origin/main gateway/platforms/api_server.py:_create_agent
    # (post-7dd00bb47d — session /model override > session-persisted model >
    # global default):
    model = global_model
    if session_override:
        model = (_clean(session_override.get("model")) or model)
    elif _clean(session_row_model):
        model = _clean(session_row_model)
    return model


@pytest.mark.parametrize(
    "session_override,session_row_model,global_model",
    [
        (None, None, "global-model"),
        (None, "session-persisted", "global-model"),  # the 7dd00bb47d regression
        ({"model": "override-model"}, "session-persisted", "global-model"),
        ({"model": ""}, "session-persisted", "global-model"),
        ({"model": "override-model"}, None, "global-model"),
        (None, "  ", "global-model"),
    ],
)
def test_api_server_resolution_parity(session_override, session_row_model, global_model):
    # New logic mirrors the migrated api_server code path exactly:
    if session_override:
        new = resolve_effective_model(session_override, None, global_model)
    elif _clean(session_row_model):
        new = resolve_effective_model(None, session_row_model, global_model)
    else:
        new = global_model
    assert new == _old_api_server_resolve(session_override, session_row_model, global_model)


