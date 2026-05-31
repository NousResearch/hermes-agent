"""Tests for plugins/email — the gws subprocess wrapper.

Covers:

  * Binary resolution via AGENTE_GWS_BIN env / PATH fallback.
  * Clear ``gws not bundled`` error when neither is set.
  * Each tool shells the expected ``gws gmail users <subcmd> --params`` argv.
  * Stdout JSON is parsed and returned as a dict/list.
  * TimeoutExpired returns {"error": "gws_timeout"} dict.
  * Plugin __init__ exposes 5 tools through the register() context shim.
"""

from __future__ import annotations

import importlib
import base64
import json
import subprocess
import sys
import types
from email import message_from_bytes
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_email_plugin():
    """Import plugins/email/email_plugin.py without requiring the full hermes_cli stack."""
    # Ensure repo root is on sys.path so `import plugins.email.email_plugin` works.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if "plugins.email.email_plugin" in sys.modules:
        return importlib.reload(sys.modules["plugins.email.email_plugin"])
    return importlib.import_module("plugins.email.email_plugin")


@pytest.fixture
def email_plugin(monkeypatch):
    monkeypatch.setenv("AGENTE_GWS_BIN", "/fake/bin/gws")
    return _load_email_plugin()


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------


class TestResolveGwsBin:
    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("AGENTE_GWS_BIN", "/opt/bundled/gws")
        mod = _load_email_plugin()
        assert mod._resolve_gws_bin() == "/opt/bundled/gws"

    def test_falls_back_to_which(self, monkeypatch):
        monkeypatch.delenv("AGENTE_GWS_BIN", raising=False)
        mod = _load_email_plugin()
        with patch("plugins.email.email_plugin.shutil.which", return_value="/usr/local/bin/gws"):
            assert mod._resolve_gws_bin() == "/usr/local/bin/gws"

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("AGENTE_GWS_BIN", raising=False)
        mod = _load_email_plugin()
        with patch("plugins.email.email_plugin.shutil.which", return_value=None):
            with pytest.raises(mod.GwsUnavailableError) as exc_info:
                mod._resolve_gws_bin()
            assert "gws not bundled" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tool argv + JSON parsing
# ---------------------------------------------------------------------------


def _completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestListEmails:
    def test_shells_expected_argv(self, email_plugin):
        payload = {"messages": [{"id": "abc", "subject": "Hi"}]}
        with patch("plugins.email.email_plugin.subprocess.run", return_value=_completed(json.dumps(payload))) as run:
            out = email_plugin.list_emails(folder="INBOX", limit=10)
        argv = run.call_args[0][0]
        expected_params = json.dumps({"userId": "me", "maxResults": 10, "labelIds": ["INBOX"]})
        assert argv == ["/fake/bin/gws", "gmail", "users", "messages", "list", "--params", expected_params]
        assert out == payload

    def test_default_folder_is_inbox(self, email_plugin):
        with patch("plugins.email.email_plugin.subprocess.run", return_value=_completed("{}")) as run:
            email_plugin.list_emails()
        argv_str = str(run.call_args[0][0])
        assert "INBOX" in argv_str

    def test_since_adds_query_param(self, email_plugin):
        with patch("plugins.email.email_plugin.subprocess.run", return_value=_completed("{}")) as run:
            email_plugin.list_emails(since="2024-01-01")
        argv = run.call_args[0][0]
        params_str = argv[argv.index("--params") + 1]
        params = json.loads(params_str)
        assert params["q"] == "after:2024-01-01"


class TestReadEmail:
    def test_passes_id_and_raw(self, email_plugin):
        payload = {"id": "abc", "body": "שלום"}  # Hebrew round-trip
        with patch("plugins.email.email_plugin.subprocess.run", return_value=_completed(json.dumps(payload, ensure_ascii=False))) as run:
            out = email_plugin.read_email(message_id="abc")
        argv = run.call_args[0][0]
        expected_params = json.dumps({"userId": "me", "id": "abc"})
        assert argv == ["/fake/bin/gws", "gmail", "users", "messages", "get", "--raw", "--params", expected_params]
        assert out["body"] == "שלום"


class TestDraftReply:
    def test_argv(self, email_plugin):
        with patch("plugins.email.email_plugin.subprocess.run", return_value=_completed('{"draft_id": "d1"}')) as run:
            out = email_plugin.draft_reply(message_id="m1", body="thanks")
        argv = run.call_args[0][0]
        assert argv[0] == "/fake/bin/gws"
        assert argv[1:4] == ["gmail", "users", "drafts"]
        assert argv[4] == "create"
        assert "--params" in argv
        params = json.loads(argv[argv.index("--params") + 1])
        assert params["userId"] == "me"
        assert "--body" in argv
        body = json.loads(argv[argv.index("--body") + 1])
        assert body["message"]["threadId"] == "m1"
        raw = base64.urlsafe_b64decode(body["message"]["raw"].encode())
        msg = message_from_bytes(raw)
        assert msg.get_payload(decode=True) == b"thanks"
        assert b"In-Reply-To: m1" in raw
        assert out == {"draft_id": "d1"}


class TestSendEmail:
    def test_argv(self, email_plugin):
        with patch("plugins.email.email_plugin.subprocess.run", return_value=_completed('{"ok": true}')) as run:
            email_plugin.send_email(to="a@b.co", subject="s", body="b")
        argv = run.call_args[0][0]
        assert argv[0] == "/fake/bin/gws"
        assert argv[1:5] == ["gmail", "users", "messages", "send"]
        assert "--params" in argv
        params = json.loads(argv[argv.index("--params") + 1])
        assert params["userId"] == "me"
        assert "--body" in argv
        message_body = json.loads(argv[argv.index("--body") + 1])
        assert "raw" in message_body


class TestMarkEmail:
    def test_add_label(self, email_plugin):
        with patch("plugins.email.email_plugin.subprocess.run", return_value=_completed("{}")) as run:
            email_plugin.mark_email(message_id="m1", add_label="UNREAD")
        argv = run.call_args[0][0]
        assert argv[1:5] == ["gmail", "users", "messages", "modify"]
        body = json.loads(argv[argv.index("--body") + 1])
        assert body["addLabelIds"] == ["UNREAD"]

    def test_remove_label(self, email_plugin):
        with patch("plugins.email.email_plugin.subprocess.run", return_value=_completed("{}")) as run:
            email_plugin.mark_email(message_id="m1", remove_label="UNREAD")
        argv = run.call_args[0][0]
        body = json.loads(argv[argv.index("--body") + 1])
        assert body["removeLabelIds"] == ["UNREAD"]

    def test_label_delta_matches_registered_tool_schema(self, email_plugin):
        with patch("plugins.email.email_plugin.subprocess.run", return_value=_completed("{}")) as run:
            email_plugin.mark_email(message_id="m1", add_label="STARRED", remove_label="INBOX")
        argv = run.call_args[0][0]
        body = json.loads(argv[argv.index("--body") + 1])
        assert body["addLabelIds"] == ["STARRED"]
        assert body["removeLabelIds"] == ["INBOX"]


class TestErrorPropagation:
    def test_nonzero_exit_returns_error_dict(self, email_plugin):
        with patch("plugins.email.email_plugin.subprocess.run", return_value=_completed("", returncode=2, stderr="boom")):
            result = email_plugin.list_emails()
        assert result["error"] == "gws_error"
        assert result["returncode"] == 2
        assert "boom" in result["stderr"]

    def test_timeout_returns_timeout_dict(self, email_plugin):
        with patch("plugins.email.email_plugin.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="gws", timeout=30)):
            result = email_plugin.list_emails()
        assert result["error"] == "gws_timeout"
        assert result["timeout"] == 30

    def test_bad_json_returns_error_dict(self, email_plugin):
        with patch("plugins.email.email_plugin.subprocess.run", return_value=_completed("not json")):
            result = email_plugin.list_emails()
        assert result["error"] == "gws_bad_json"
        assert result["raw"] == "not json"

    def test_missing_binary_surfaces_unavailable(self, monkeypatch):
        monkeypatch.delenv("AGENTE_GWS_BIN", raising=False)
        mod = _load_email_plugin()
        with patch("plugins.email.email_plugin.shutil.which", return_value=None):
            with pytest.raises(mod.GwsUnavailableError):
                mod.list_emails()


# ---------------------------------------------------------------------------
# Plugin registration — discoverable + 5 tools
# ---------------------------------------------------------------------------


class _FakeCtx:
    def __init__(self) -> None:
        self.tools: list[dict] = []

    def register_tool(self, **kw) -> None:
        self.tools.append(kw)


def test_register_exposes_five_tools(monkeypatch):
    """plugins.email.register() should publish exactly five tools in the `email` toolset."""
    monkeypatch.setenv("AGENTE_GWS_BIN", "/fake/bin/gws")
    # Lazy import so the test does not require the full hermes_cli during collection.
    sys.path.insert(0, str(REPO_ROOT))

    # Stub tools.registry symbols the package imports at module top-level.
    if "tools.registry" not in sys.modules:
        registry_stub = types.ModuleType("tools.registry")
        registry_stub.tool_result = lambda data=None, **kw: json.dumps({"data": data, **kw})
        registry_stub.tool_error = lambda message, **kw: json.dumps({"error": message, **kw})
        tools_pkg = types.ModuleType("tools")
        tools_pkg.registry = registry_stub
        sys.modules["tools"] = tools_pkg
        sys.modules["tools.registry"] = registry_stub

    if "plugins.email" in sys.modules:
        del sys.modules["plugins.email"]
    pkg = importlib.import_module("plugins.email")
    schemas = importlib.import_module("plugins.email.schemas")

    ctx = _FakeCtx()
    pkg.register(ctx)

    names = [t["name"] for t in ctx.tools]
    assert names == ["list_emails", "read_email", "draft_reply", "send_email", "mark_email"]
    for t in ctx.tools:
        assert t["toolset"] == "email"
        assert "parameters" in t["schema"]
    mark_schema = cast(dict[str, Any], schemas.MARK_EMAIL_SCHEMA)
    params = cast(dict[str, Any], mark_schema["parameters"])
    props = cast(dict[str, Any], params["properties"])
    add_label = cast(dict[str, Any], props["add_label"])
    assert add_label["type"] == "string"
