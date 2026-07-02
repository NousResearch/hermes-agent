"""Tests for the desktop /api/model/claude-agent-sdk read/write helpers."""

from hermes_cli.web_server import _apply_claude_agent_sdk, _claude_agent_sdk_to_ui


# ---------------------------------------------------------------------------
# _claude_agent_sdk_to_ui
# ---------------------------------------------------------------------------
def test_ui_unset_is_off():
    assert _claude_agent_sdk_to_ui(None) == {
        "mode": "off", "permission_mode": None, "max_turns": None, "max_budget_usd": None,
    }


def test_ui_string_mode():
    assert _claude_agent_sdk_to_ui("hybrid")["mode"] == "hybrid"
    assert _claude_agent_sdk_to_ui("bogus")["mode"] == "off"


def test_ui_dict_mode_and_options():
    out = _claude_agent_sdk_to_ui({"mode": "delegate", "max_turns": 12, "max_budget_usd": 2.5,
                                   "permission_mode": "bypassPermissions"})
    assert out == {"mode": "delegate", "permission_mode": "bypassPermissions",
                   "max_turns": 12, "max_budget_usd": 2.5}


def test_ui_enabled_false_is_off():
    assert _claude_agent_sdk_to_ui({"mode": "hybrid", "enabled": False})["mode"] == "off"


# ---------------------------------------------------------------------------
# _apply_claude_agent_sdk
# ---------------------------------------------------------------------------
def test_apply_promotes_bare_string_model():
    out = _apply_claude_agent_sdk("anthropic/claude-opus-4.6", mode="inference")
    assert out == {"default": "anthropic/claude-opus-4.6", "claude_agent_sdk": {"mode": "inference"}}


def test_apply_preserves_existing_model_dict():
    model = {"default": "anthropic/claude-opus-4.6", "provider": "anthropic", "base_url": "x"}
    out = _apply_claude_agent_sdk(model, mode="hybrid", max_turns=20, max_budget_usd=3.0)
    assert out["provider"] == "anthropic" and out["base_url"] == "x"
    assert out["claude_agent_sdk"] == {"mode": "hybrid", "max_turns": 20, "max_budget_usd": 3.0}
    # input not mutated
    assert "claude_agent_sdk" not in model


def test_apply_off_removes_key():
    model = {"default": "m", "claude_agent_sdk": {"mode": "hybrid"}}
    out = _apply_claude_agent_sdk(model, mode="off")
    assert "claude_agent_sdk" not in out
    assert out["default"] == "m"


def test_apply_empty_model():
    out = _apply_claude_agent_sdk("", mode="inference")
    assert out == {"claude_agent_sdk": {"mode": "inference"}}


def test_apply_permission_mode_only_when_set():
    out = _apply_claude_agent_sdk({}, mode="delegate")
    assert out["claude_agent_sdk"] == {"mode": "delegate"}  # no None keys leaked
