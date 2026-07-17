"""Tests for the terminal-CLI Claude Agent SDK inference-method helpers.

The desktop app configures ``model.claude_agent_sdk`` via
``hermes_cli.web_server`` (see test_web_server_claude_agent_sdk.py). These cover
the parallel terminal-CLI path in ``hermes_cli.model_setup_flows`` used by
``hermes model`` → Anthropic, so both surfaces write the same config shape.
"""

from hermes_cli.model_setup_flows import (
    _apply_claude_agent_sdk_mode,
    _read_claude_agent_sdk_mode,
)


# ---------------------------------------------------------------------------
# _read_claude_agent_sdk_mode
# ---------------------------------------------------------------------------
def test_read_unset_is_off():
    assert _read_claude_agent_sdk_mode({}) == "off"
    assert _read_claude_agent_sdk_mode({"model": "claude-sonnet-4-6"}) == "off"
    assert _read_claude_agent_sdk_mode({"model": {"default": "x"}}) == "off"


def test_read_string_mode():
    assert _read_claude_agent_sdk_mode({"model": {"claude_agent_sdk": "hybrid"}}) == "hybrid"
    # Unknown modes fall back to off, never crash.
    assert _read_claude_agent_sdk_mode({"model": {"claude_agent_sdk": "bogus"}}) == "off"


def test_read_dict_mode():
    cfg = {"model": {"claude_agent_sdk": {"mode": "delegate", "max_turns": 12}}}
    assert _read_claude_agent_sdk_mode(cfg) == "delegate"


def test_read_enabled_false_is_off():
    cfg = {"model": {"claude_agent_sdk": {"mode": "hybrid", "enabled": False}}}
    assert _read_claude_agent_sdk_mode(cfg) == "off"


# ---------------------------------------------------------------------------
# _apply_claude_agent_sdk_mode
# ---------------------------------------------------------------------------
def test_apply_off_removes_key():
    model = {"default": "claude-sonnet-4-6", "claude_agent_sdk": {"mode": "hybrid"}}
    _apply_claude_agent_sdk_mode(model, "off")
    assert "claude_agent_sdk" not in model
    assert model["default"] == "claude-sonnet-4-6"


def test_apply_mode_writes_dict():
    model = {"default": "claude-sonnet-4-6"}
    _apply_claude_agent_sdk_mode(model, "inference")
    assert model["claude_agent_sdk"] == {"mode": "inference"}


def test_apply_mode_preserves_advanced_keys():
    model = {
        "claude_agent_sdk": {
            "mode": "hybrid",
            "permission_mode": "bypassPermissions",
            "max_turns": 30,
            "max_budget_usd": 2.5,
        }
    }
    _apply_claude_agent_sdk_mode(model, "delegate")
    assert model["claude_agent_sdk"] == {
        "mode": "delegate",
        "permission_mode": "bypassPermissions",
        "max_turns": 30,
        "max_budget_usd": 2.5,
    }


def test_read_roundtrips_apply():
    model = {}
    for mode in ("inference", "delegate", "hybrid", "off"):
        _apply_claude_agent_sdk_mode(model, mode)
        assert _read_claude_agent_sdk_mode({"model": model}) == mode
