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


# ---------------------------------------------------------------------------
# Auto-seed a default Anthropic model when enabling Anthropic inference with
# none set (otherwise the SDK/OAuth method is inert → "inference not ready").
# ---------------------------------------------------------------------------
import hermes_cli.web_server as ws


def test_current_main_model_name_shapes():
    assert ws._current_main_model_name({}) == ""
    assert ws._current_main_model_name({"model": "claude-x"}) == "claude-x"
    assert ws._current_main_model_name({"model": {"default": "claude-y"}}) == "claude-y"
    assert ws._current_main_model_name({"model": {"name": "claude-z"}}) == "claude-z"
    assert ws._current_main_model_name({"model": {"claude_agent_sdk": {"mode": "hybrid"}}}) == ""


def test_ensure_anthropic_model_fills_only_empty(monkeypatch):
    saved = {}
    monkeypatch.setattr(ws, "_recommended_anthropic_model", lambda: "claude-fable-5")
    monkeypatch.setattr(ws, "save_config", lambda cfg: saved.update(cfg))

    # No main model → seed the recommended Anthropic model + provider.
    monkeypatch.setattr(ws, "read_raw_config", lambda: {"model": {"claude_agent_sdk": {"mode": "hybrid"}}})
    assert ws._ensure_anthropic_main_model_if_unset() == "claude-fable-5"
    assert saved["model"]["default"] == "claude-fable-5"
    assert saved["model"]["provider"] == "anthropic"
    # Preserves the existing claude_agent_sdk block.
    assert saved["model"]["claude_agent_sdk"] == {"mode": "hybrid"}


def test_ensure_anthropic_model_never_clobbers_existing(monkeypatch):
    monkeypatch.setattr(ws, "_recommended_anthropic_model", lambda: "claude-fable-5")
    monkeypatch.setattr(ws, "save_config", lambda cfg: (_ for _ in ()).throw(AssertionError("must not save")))
    monkeypatch.setattr(ws, "read_raw_config",
                        lambda: {"model": {"default": "gpt-5", "provider": "openrouter"}})
    # A model is already set → no-op, returns None, never writes.
    assert ws._ensure_anthropic_main_model_if_unset() is None


# ---------------------------------------------------------------------------
# Latest-Sonnet default selection (auto-seed prefers newest Sonnet by version).
# ---------------------------------------------------------------------------
def test_sonnet_version_key_parses_and_ignores_dates():
    assert ws._sonnet_version_key("claude-sonnet-5") == (5, 0, 1)
    assert ws._sonnet_version_key("claude-sonnet-4-6") == (4, 6, 1)
    # 8-digit date snapshot is not a minor version.
    assert ws._sonnet_version_key("claude-sonnet-4-20250514") == (4, 0, 0)
    assert ws._sonnet_version_key("claude-sonnet-4-5-20250929") == (4, 5, 0)
    # legacy "<major>-<minor>-sonnet-<date>" — version precedes "sonnet".
    assert ws._sonnet_version_key("claude-3-5-sonnet-20241022") == (3, 5, 0)
    # non-Sonnet → None
    assert ws._sonnet_version_key("claude-opus-4-8") is None


def test_latest_sonnet_picks_highest_version():
    models = [
        "claude-fable-5", "claude-opus-4-8", "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514", "claude-sonnet-5",
    ]
    assert ws._latest_sonnet(models) == "claude-sonnet-5"
    # Undated alias beats the pinned snapshot at the same version.
    assert ws._latest_sonnet(["claude-sonnet-5-20260101", "claude-sonnet-5"]) == "claude-sonnet-5"
    # No Sonnet present → "".
    assert ws._latest_sonnet(["claude-opus-4-8", "claude-haiku-4-5"]) == ""
