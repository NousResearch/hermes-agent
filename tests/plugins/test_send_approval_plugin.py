"""Tests for the send-approval plugin.

Covers ``plugins/send-approval/``:

  * Classification table — effect classes {send-to-person, spend,
    hard-to-undo} over the parity families: the AP piece runner
    (``mcp__activepieces__ap_run_action``, classified from the piece action
    name in its args), ``gmail_send*``, ``slack_send*``, ``*_send_message``,
    ``stripe_*``, calendar create/delete, and file/repo destructive piece
    actions. Read-only piece actions and native tools are NOT flagged.
  * Gate integration through the REAL ``pre_tool_call`` dispatch —
    ``hermes_cli.plugins._dispatch_pre_tool_call_hooks`` reuses
    ``tools/approval.py:request_tool_approval``: approve → the tool
    proceeds, deny → refused, unattended (no human) → refused (never
    fails open).
  * Bundled-plugin discovery via ``PluginManager.discover_and_load``.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_plugin_init():
    """Import the plugin __init__.py in isolation under a plugin namespace."""
    plugin_dir = _repo_root() / "plugins" / "send-approval"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.send_approval_under_test",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.send_approval_under_test"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.send_approval_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


AP_SEND = {
    "pieceName": "gmail",
    "actionName": "send_email",
    "input": {"to": "dana@example.com", "subject": "Lunch?"},
}


# ---------------------------------------------------------------------------
# Classification table
# ---------------------------------------------------------------------------

class TestClassificationTable:
    def _flag(self, mod, tool_name, args=None):
        return mod._classify(tool_name, args if args is not None else {})

    def test_runner_args_send_to_person(self):
        """ap_run_action with a send-flavored piece action name is flagged as send-to-person."""
        mod = _load_plugin_init()
        out = self._flag(mod, "mcp__activepieces__ap_run_action", AP_SEND)
        assert out is not None
        assert out["effect"] == "send-to-person"

    def test_runner_args_spend(self):
        mod = _load_plugin_init()
        out = self._flag(mod, "mcp__activepieces__ap_run_action",
                         {"pieceName": "stripe", "actionName": "create_charge"})
        assert out is not None
        assert out["effect"] == "spend"

    def test_runner_args_hard_to_undo(self):
        mod = _load_plugin_init()
        out = self._flag(mod, "mcp__activepieces__ap_run_action",
                         {"pieceName": "google-drive", "actionName": "delete_file"})
        assert out is not None
        assert out["effect"] == "hard-to-undo"

    def test_runner_read_only_piece_action_not_flagged(self):
        mod = _load_plugin_init()
        assert self._flag(mod, "mcp__activepieces__ap_run_action",
                          {"pieceName": "gmail", "actionName": "list_threads"}) is None

    def test_runner_without_action_args_not_flagged(self):
        mod = _load_plugin_init()
        assert self._flag(mod, "mcp__activepieces__ap_run_action", {}) is None
        assert self._flag(mod, "mcp__activepieces__ap_run_action", None) is None

    def test_piece_action_tool_name_send(self):
        """Piece actions surfacing as <piece>_<action> MCP tool names are classified by name."""
        mod = _load_plugin_init()
        out = self._flag(mod, "mcp__activepieces__gmail_send_email")
        assert out is not None
        assert out["effect"] == "send-to-person"

    def test_legacy_single_underscore_mcp_form(self):
        mod = _load_plugin_init()
        out = self._flag(mod, "mcp_activepieces_gmail_send_email")
        assert out is not None
        assert out["effect"] == "send-to-person"

    def test_gmail_send_family(self):
        mod = _load_plugin_init()
        out = self._flag(mod, "gmail_send_email")
        assert out is not None
        assert out["effect"] == "send-to-person"

    def test_slack_send_family(self):
        mod = _load_plugin_init()
        out = self._flag(mod, "slack_send_channel_message")
        assert out is not None
        assert out["effect"] == "send-to-person"

    def test_send_message_suffix_family(self):
        """Any *_send_message (twilio, whatsapp, chat platforms, ...) is flagged."""
        mod = _load_plugin_init()
        for name in ("twilio_send_message", "whatsapp_send_message", "send_message"):
            out = self._flag(mod, name)
            assert out is not None, name
            assert out["effect"] == "send-to-person", name

    def test_stripe_family(self):
        """Every stripe_* action moves money or card data: flagged as spend."""
        mod = _load_plugin_init()
        for name in ("stripe_create_charge", "stripe_create_payment_intent", "stripe_refund"):
            out = self._flag(mod, name)
            assert out is not None, name
            assert out["effect"] == "spend", name

    def test_calendar_create_flagged(self):
        mod = _load_plugin_init()
        out = self._flag(mod, "google_calendar_create_event")
        assert out is not None
        assert out["effect"] == "send-to-person"

    def test_outlook_calendar_create_flagged(self):
        mod = _load_plugin_init()
        out = self._flag(mod, "microsoft_outlook_calendar_create_event")
        assert out is not None
        assert out["effect"] == "send-to-person"

    def test_calendar_delete_hard_to_undo(self):
        mod = _load_plugin_init()
        out = self._flag(mod, "microsoft_outlook_calendar_delete_event")
        assert out is not None
        assert out["effect"] == "hard-to-undo"

    def test_repo_file_destructive_hard_to_undo(self):
        mod = _load_plugin_init()
        for name in ("github_delete_branch", "google_drive_delete_file",
                     "mcp__activepieces__github_delete_file"):
            out = self._flag(mod, name)
            assert out is not None, name
            assert out["effect"] == "hard-to-undo", name

    def test_read_only_piece_actions_not_flagged(self):
        mod = _load_plugin_init()
        for name in ("gmail_list_threads", "slack_read_channel",
                     "google_calendar_list_events", "github_get_issue"):
            assert self._flag(mod, f"mcp__activepieces__{name}") is None, name

    def test_native_non_mcp_tools_not_flagged(self):
        """Native toolsets (file/terminal/memory) are NOT gated here — destructive
        shell commands already run through the core approval machinery."""
        mod = _load_plugin_init()
        for name in ("read_file", "write_file", "terminal", "memory", "browser", "todo"):
            assert self._flag(mod, name, {"command": "rm -rf /"}) is None, name

    def test_distinct_rule_keys_per_family(self):
        mod = _load_plugin_init()
        a = self._flag(mod, "gmail_send_email")
        b = self._flag(mod, "stripe_create_charge")
        assert a["rule_key"] != b["rule_key"]

    def test_reason_names_effect_class(self):
        mod = _load_plugin_init()
        out = self._flag(mod, "mcp__activepieces__ap_run_action", AP_SEND)
        assert "send-to-person" in out["message"]
        assert "gmail" in out["message"].lower() or "send" in out["message"].lower()


class TestHookDirective:
    def test_returns_approve_directive_for_flagged_call(self):
        mod = _load_plugin_init()
        out = mod._on_pre_tool_call(tool_name="mcp__activepieces__ap_run_action", args=AP_SEND)
        assert isinstance(out, dict)
        assert out["action"] == "approve"
        assert out["message"]
        assert out["rule_key"]

    def test_returns_none_for_unflagged_call(self):
        mod = _load_plugin_init()
        assert mod._on_pre_tool_call(tool_name="gmail_list_threads", args={}) is None


# ---------------------------------------------------------------------------
# Gate integration — through the real pre_tool_call dispatch
# ---------------------------------------------------------------------------

@pytest.fixture
def _hermes_home(tmp_path, monkeypatch):
    """Temp HERMES_HOME with send-approval enabled through the real config path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    import yaml

    config = {"plugins": {"enabled": ["send-approval"]}}
    (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.safe_dump(config))
    for k in list(sys.modules):
        if k.startswith(("hermes_plugins", "hermes_cli.plugins")):
            del sys.modules[k]
    from hermes_cli.plugins import _ensure_plugins_discovered

    mgr = _ensure_plugins_discovered(force=True)
    loaded = set(getattr(mgr, "_plugins", {}).keys())
    assert "send-approval" in loaded
    yield tmp_path


@pytest.fixture(autouse=True)
def _isolate_approval_state(monkeypatch):
    """Clean approval state: fixed session key, empty allowlists, not yolo."""
    import tools.approval as approval
    import tools.approval_context as approval_context

    monkeypatch.setattr(approval, "get_current_session_key", lambda default="default": "test-session")
    monkeypatch.setattr(approval_context, "get_current_session_key", lambda default="default": "test-session")
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False, raising=False)
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: None, raising=False)
    yield


class TestGateIntegration:
    """The plugin's approve directive must reuse tools/approval.py's human gate."""

    def _dispatch(self, tool_name="mcp__activepieces__ap_run_action", args=None):
        from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

        return _dispatch_pre_tool_call_hooks(tool_name, args if args is not None else AP_SEND)

    def test_approved_call_proceeds(self, _hermes_home, monkeypatch):
        """Gate verdict 'once' → the tool call proceeds (block message None)."""
        import tools.approval as approval
        import tools.approval_context as approval_context
        import tools.approval_prompt as approval_prompt
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "once")
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *a, **k: "once")
        assert self._dispatch() == (None, None)

        import tools.approval as approval
        import tools.approval_context as approval_context
        import tools.approval_prompt as approval_prompt

        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *a, **k: "deny")
        block_message, _modified = self._dispatch()
        assert block_message is not None
        assert "denied" in block_message.lower()

    def test_unattended_refused_never_fails_open(self, _hermes_home, monkeypatch):
        """No interactive user / gateway / cron present → fail CLOSED."""
        import tools.approval as approval
        import tools.approval_context as approval_context

        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval_context, "_is_cron_approval_context", lambda: False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        block_message, _modified = self._dispatch()
        assert block_message is not None
        assert "no interactive user or gateway" in block_message.lower()

    def test_unflagged_call_never_enters_gate(self, _hermes_home, monkeypatch):
        """Read-only piece actions run without any approval round-trip."""
        prompts = []
        import tools.approval as approval
        import tools.approval_prompt as approval_prompt

        monkeypatch.setattr(approval, "prompt_dangerous_approval",
                            lambda *a, **k: prompts.append(1) or "deny")
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval",
                            lambda *a, **k: prompts.append(1) or "deny")
        block_message, _modified = self._dispatch(
            tool_name="mcp__activepieces__ap_run_action",
            args={"pieceName": "gmail", "actionName": "list_threads"},
        )
        assert block_message is None
        assert prompts == []


# ---------------------------------------------------------------------------
# Bundled-plugin discovery
# ---------------------------------------------------------------------------

class TestPluginDiscovery:
    def test_manifest_declares_pre_tool_call_hook(self):
        import yaml

        manifest = yaml.safe_load(
            (_repo_root() / "plugins" / "send-approval" / "plugin.yaml").read_text())
        assert manifest["name"] == "send-approval"
        assert manifest["hooks"] == ["pre_tool_call"]

    def test_loads_via_plugin_manager(self, _hermes_home):
        """End-to-end: enabled in config.yaml → PluginManager discovers it."""
        from hermes_cli.plugins import get_plugin_manager

        mgr = get_plugin_manager()
        assert "send-approval" in set(getattr(mgr, "_plugins", {}).keys())