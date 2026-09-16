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


