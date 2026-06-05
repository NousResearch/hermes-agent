"""Tests for plugins/web_browser/.

These tests cover the subprocess wiring + approval gating contract. The
actual ``agent-browser`` CLI is never invoked — we mock ``subprocess.run``.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is importable when pytest is invoked from arbitrary cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    cp = MagicMock()
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = returncode
    return cp


def _approve_all():
    """Patch the approval gate to always approve."""
    return patch(
        "plugins.web_browser.web_browser_plugin._request_approval",
        return_value={"approved": True, "message": None},
    )


def _deny_all():
    return patch(
        "plugins.web_browser.web_browser_plugin._request_approval",
        return_value={"approved": False, "message": "denied"},
    )


def _stub_binary():
    return patch(
        "plugins.web_browser.web_browser_plugin._resolve_binary",
        return_value="/usr/local/bin/agent-browser",
    )


# ---------------------------------------------------------------------------
# Schemas + label_he contract
# ---------------------------------------------------------------------------


def test_every_tool_has_label_he_and_category_web():
    from plugins.web_browser import schemas

    for attr in dir(schemas):
        if not attr.endswith("_SCHEMA"):
            continue
        schema = getattr(schemas, attr)
        assert "label_he" in schema, f"{attr} missing label_he"
        assert schema["label_he"], f"{attr} has empty label_he"
        assert schema.get("category") == "web", f"{attr} category != web"
        assert "name" in schema and schema["name"].startswith("browser_")


def test_expected_ten_tool_names_are_registered_in_tool_defs():
    from plugins.web_browser.web_browser_plugin import TOOL_DEFS

    names = {t["name"] for t in TOOL_DEFS}
    assert names == {
        "browser_navigate", "browser_screenshot", "browser_snapshot",
        "browser_click", "browser_fill", "browser_type", "browser_press",
        "browser_get", "browser_find", "browser_close",
    }


# ---------------------------------------------------------------------------
# Binary detection
# ---------------------------------------------------------------------------


def test_check_requirements_returns_false_when_binary_missing():
    from plugins.web_browser import web_browser_plugin as wbp

    with patch.object(wbp, "_resolve_binary", return_value=None):
        assert wbp.check_web_browser_requirements() is False


def test_check_requirements_returns_true_when_binary_present():
    from plugins.web_browser import web_browser_plugin as wbp

    with patch.object(wbp, "_resolve_binary", return_value="/usr/local/bin/agent-browser"):
        assert wbp.check_web_browser_requirements() is True


# ---------------------------------------------------------------------------
# Approval gating — first-use emits approval.request, denial blocks subprocess
# ---------------------------------------------------------------------------


def test_navigate_consults_approval_with_browse_url_command():
    from plugins.web_browser import web_browser_plugin as wbp

    captured = {}

    def _fake_approve(command, description):
        captured["command"] = command
        captured["description"] = description
        return {"approved": True, "message": None}

    with patch.object(wbp, "_request_approval", side_effect=_fake_approve), \
         _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout='{"url":"https://x"}')):
        wbp.handle_browser_navigate({"url": "https://example.com"})

    assert captured["command"] == "browse https://example.com"
    assert "example.com" in captured["description"]


def test_denied_approval_skips_subprocess():
    from plugins.web_browser import web_browser_plugin as wbp

    with _deny_all(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run") as run_mock:
        out = wbp.handle_browser_navigate({"url": "https://x"})

    run_mock.assert_not_called()
    parsed = json.loads(out)
    assert parsed["success"] is False
    assert "approval denied" in parsed["error"]


# ---------------------------------------------------------------------------
# Subprocess argv contracts (one per tool)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, expected_tail",
    [
        ({"url": "https://x"}, ["open", "https://x", "--json"]),
        (
            {"url": "https://x", "task": "log in"},
            ["open", "https://x", "--task", "log in", "--json"],
        ),
        (
            {"url": "https://x", "basic_auth": "u:p"},
            ["open", "https://x", "--basic-auth", "u:p", "--json"],
        ),
    ],
)
def test_navigate_argv(args, expected_tail):
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout='{"ok": true}')) as run_mock:
        wbp.handle_browser_navigate(args)

    argv = run_mock.call_args[0][0]
    assert argv[0] == "/usr/local/bin/agent-browser"
    assert argv[1:] == expected_tail


def test_screenshot_argv_with_path_and_selector():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout='{"path":"/tmp/x.png"}')) as run_mock:
        wbp.handle_browser_screenshot({"path": "/tmp/x.png", "selector": "#main"})

    argv = run_mock.call_args[0][0]
    assert argv[1:] == ["screenshot", "/tmp/x.png", "--selector", "#main", "--json"]


def test_snapshot_full_flag():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout='{"tree":[]}')) as run_mock:
        wbp.handle_browser_snapshot({"full": True})

    argv = run_mock.call_args[0][0]
    assert "--full" in argv and "snapshot" in argv


def test_click_requires_selector():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run") as run_mock:
        out = wbp.handle_browser_click({"selector": ""})

    run_mock.assert_not_called()
    assert json.loads(out)["success"] is False


def test_click_argv():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout='{"ok": true}')) as run_mock:
        wbp.handle_browser_click({"selector": "@e5"})

    argv = run_mock.call_args[0][0]
    assert argv[1:] == ["click", "@e5", "--json"]


def test_fill_argv_and_required_value():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run") as run_mock:
        bad = wbp.handle_browser_fill({"selector": "#q"})
        run_mock.assert_not_called()
        assert json.loads(bad)["success"] is False

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout='{"ok":true}')) as run_mock:
        wbp.handle_browser_fill({"selector": "#q", "value": "hello"})

    argv = run_mock.call_args[0][0]
    assert argv[1:] == ["fill", "#q", "hello", "--json"]


def test_type_argv():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout='{"ok":true}')) as run_mock:
        wbp.handle_browser_type({"selector": "#q", "text": "hi"})

    assert run_mock.call_args[0][0][1:] == ["type", "#q", "hi", "--json"]


def test_press_argv():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout='{"ok":true}')) as run_mock:
        wbp.handle_browser_press({"key": "Enter"})

    assert run_mock.call_args[0][0][1:] == ["press", "Enter", "--json"]


def test_get_argv_attr_with_selector():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout='{"value":"foo"}')) as run_mock:
        wbp.handle_browser_get({"what": "attr", "attr": "href", "selector": "a.cta"})

    assert run_mock.call_args[0][0][1:] == ["get", "attr", "href", "a.cta", "--json"]


def test_find_argv_with_action_and_text():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout='{"ok":true}')) as run_mock:
        wbp.handle_browser_find({"locator": "role", "value": "button", "action": "click"})
        argv1 = run_mock.call_args[0][0]
        assert argv1[1:] == ["find", "role", "button", "click", "--json"]

        wbp.handle_browser_find({"locator": "label", "value": "Email", "action": "fill", "text": "x@y"})
        argv2 = run_mock.call_args[0][0]
        assert argv2[1:] == ["find", "label", "Email", "fill", "x@y", "--json"]


def test_close_all_flag():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout='{"closed":2}')) as run_mock:
        wbp.handle_browser_close({"all": True})

    assert run_mock.call_args[0][0][1:] == ["close", "--all", "--json"]


# ---------------------------------------------------------------------------
# Subprocess failure modes
# ---------------------------------------------------------------------------


def test_nonzero_exit_returned_as_error():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               return_value=_completed(stdout="", stderr="boom", returncode=2)):
        out = wbp.handle_browser_navigate({"url": "https://x"})

    parsed = json.loads(out)
    assert parsed["success"] is False
    assert parsed["exit_code"] == 2
    assert "boom" in parsed["stderr"]


def test_missing_binary_returns_install_hint():
    from plugins.web_browser import web_browser_plugin as wbp

    with _approve_all(), \
         patch.object(wbp, "_resolve_binary", return_value=None):
        out = wbp.handle_browser_navigate({"url": "https://x"})

    parsed = json.loads(out)
    assert parsed["success"] is False
    assert "agent-browser" in parsed["error"]
    assert "npm" in parsed["error"]


def test_timeout_surfaced_cleanly():
    from plugins.web_browser import web_browser_plugin as wbp
    import subprocess as _sp

    with _approve_all(), _stub_binary(), \
         patch("plugins.web_browser.web_browser_plugin.subprocess.run",
               side_effect=_sp.TimeoutExpired(cmd=["x"], timeout=1)):
        out = wbp.handle_browser_navigate({"url": "https://x"})

    parsed = json.loads(out)
    assert parsed["success"] is False
    assert "timed out" in parsed["error"]


# ---------------------------------------------------------------------------
# register() wires every tool with override=True
# ---------------------------------------------------------------------------


def test_register_emits_ten_override_tools():
    import plugins.web_browser as plugin

    seen = []

    class _Ctx:
        def register_tool(self, **kw):
            seen.append(kw)

    plugin.register(_Ctx())

    assert len(seen) == 10
    names = {kw["name"] for kw in seen}
    assert "browser_navigate" in names and "browser_close" in names
    for kw in seen:
        assert kw["toolset"] == "web_browser"
        assert kw["override"] is True
        assert callable(kw["handler"])
        assert kw["schema"]["name"] == kw["name"]
