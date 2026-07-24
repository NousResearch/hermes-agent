"""Tests for the gotify-notify plugin.

Covers the bundled plugin at ``plugins/gotify-notify/``:

  * Payload construction: markdown extras, title truncation, priority
    clamping, message truncation.
  * Missing environment variables: graceful error JSON.
  * ``GOTIFY_CONTENT_TYPE`` override: text/plain and custom values.
  * HTTP success and failure paths (mocked ``urllib.request.urlopen``).
  * Tool registration contract: name, schema, required fields.
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --------------------------------------------------------------------------- #
# Plugin loader
# --------------------------------------------------------------------------- #

def _load_plugin():
    """Import the gotify-notify plugin's __init__.py directly."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "gotify-notify"
    spec = importlib.util.spec_from_file_location(
        "gotify_notify_under_test",
        plugin_dir / "__init__.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_ctx():
    """Fake plugin context that captures the registered tool."""
    class FakeCtx:
        def __init__(self):
            self.tools = {}

        def register_tool(self, name, toolset, schema, handler, description):
            self.tools[name] = {
                "toolset": toolset,
                "schema": schema,
                "handler": handler,
                "description": description,
            }

    return FakeCtx()


@pytest.fixture
def gotify_plugin(fake_ctx):
    """Load the plugin and register its tool against the fake context."""
    mod = _load_plugin()
    mod.register(fake_ctx)
    return fake_ctx


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove GOTIFY_* env vars before each test for predictability."""
    for var in ("GOTIFY_URL", "GOTIFY_APP_TOKEN", "GOTIFY_CONTENT_TYPE"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def env_set(monkeypatch):
    """Set standard Gotify env vars."""
    monkeypatch.setenv("GOTIFY_URL", "http://gotify.test")
    monkeypatch.setenv("GOTIFY_APP_TOKEN", "testtoken123")


# --------------------------------------------------------------------------- #
# Registration contract
# --------------------------------------------------------------------------- #

class TestRegistration:
    def test_registers_gotify_send(self, gotify_plugin):
        assert "gotify_send" in gotify_plugin.tools

    def test_toolset_name(self, gotify_plugin):
        assert gotify_plugin.tools["gotify_send"]["toolset"] == "gotify"

    def test_schema_name_matches(self, gotify_plugin):
        schema = gotify_plugin.tools["gotify_send"]["schema"]
        assert schema["name"] == "gotify_send"

    def test_schema_requires_message(self, gotify_plugin):
        schema = gotify_plugin.tools["gotify_send"]["schema"]
        assert "message" in schema["parameters"]["required"]

    def test_schema_has_title_priority(self, gotify_plugin):
        props = gotify_plugin.tools["gotify_send"]["schema"]["parameters"]["properties"]
        assert "title" in props
        assert "priority" in props
        assert props["priority"]["default"] == 5


# --------------------------------------------------------------------------- #
# Missing env vars
# --------------------------------------------------------------------------- #

class TestMissingEnv:
    def test_missing_url_returns_error(self, gotify_plugin, monkeypatch):
        monkeypatch.setenv("GOTIFY_APP_TOKEN", "tok")
        monkeypatch.delenv("GOTIFY_URL", raising=False)
        result = gotify_plugin.tools["gotify_send"]["handler"]({"message": "hi"})
        data = json.loads(result)
        assert data["success"] is False
        assert "GOTIFY_URL" in data["error"]

    def test_missing_token_returns_error(self, gotify_plugin, monkeypatch):
        monkeypatch.setenv("GOTIFY_URL", "http://gotify.test")
        monkeypatch.delenv("GOTIFY_APP_TOKEN", raising=False)
        result = gotify_plugin.tools["gotify_send"]["handler"]({"message": "hi"})
        data = json.loads(result)
        assert data["success"] is False
        assert "GOTIFY_APP_TOKEN" in data["error"]


# --------------------------------------------------------------------------- #
# Payload construction (via mocked HTTP)
# --------------------------------------------------------------------------- #

class TestPayload:
    @patch("urllib.request.urlopen")
    def test_markdown_content_type_in_extras(self, mock_urlopen, gotify_plugin, env_set):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"id":1}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        gotify_plugin.tools["gotify_send"]["handler"]({
            "message": "Hello **world**",
            "title": "Test",
        })

        sent_request = mock_urlopen.call_args[0][0]
        body = json.loads(sent_request.data.decode())
        assert body["extras"]["client::display"]["contentType"] == "text/markdown"

    @patch("urllib.request.urlopen")
    def test_custom_content_type(self, mock_urlopen, gotify_plugin, env_set, monkeypatch):
        monkeypatch.setenv("GOTIFY_CONTENT_TYPE", "text/plain")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"id":1}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        gotify_plugin.tools["gotify_send"]["handler"]({"message": "hi"})

        sent_request = mock_urlopen.call_args[0][0]
        body = json.loads(sent_request.data.decode())
        assert body["extras"]["client::display"]["contentType"] == "text/plain"

    @patch("urllib.request.urlopen")
    def test_title_truncation(self, mock_urlopen, gotify_plugin, env_set):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"id":1}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        long_title = "A" * 200
        gotify_plugin.tools["gotify_send"]["handler"]({
            "message": "hi",
            "title": long_title,
        })

        sent_request = mock_urlopen.call_args[0][0]
        body = json.loads(sent_request.data.decode())
        assert len(body["title"]) == 120

    @patch("urllib.request.urlopen")
    def test_message_truncation(self, mock_urlopen, gotify_plugin, env_set):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"id":1}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        long_message = "x" * 5000
        gotify_plugin.tools["gotify_send"]["handler"]({"message": long_message})

        sent_request = mock_urlopen.call_args[0][0]
        body = json.loads(sent_request.data.decode())
        assert len(body["message"]) == 4000

    @patch("urllib.request.urlopen")
    def test_priority_clamping(self, mock_urlopen, gotify_plugin, env_set):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"id":1}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        # Priority above 10 → clamped to 10
        gotify_plugin.tools["gotify_send"]["handler"]({
            "message": "hi",
            "priority": 99,
        })
        body = json.loads(mock_urlopen.call_args[0][0].data.decode())
        assert body["priority"] == 10

        # Priority below 1 → clamped to 1
        gotify_plugin.tools["gotify_send"]["handler"]({
            "message": "hi",
            "priority": -5,
        })
        body = json.loads(mock_urlopen.call_args[0][0].data.decode())
        assert body["priority"] == 1

    @patch("urllib.request.urlopen")
    def test_default_title(self, mock_urlopen, gotify_plugin, env_set):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"id":1}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        gotify_plugin.tools["gotify_send"]["handler"]({"message": "hi"})
        body = json.loads(mock_urlopen.call_args[0][0].data.decode())
        assert body["title"] == "Hermes"


# --------------------------------------------------------------------------- #
# HTTP success / failure
# --------------------------------------------------------------------------- #

class TestHTTPEndpoints:
    @patch("urllib.request.urlopen")
    def test_success_returns_true(self, mock_urlopen, gotify_plugin, env_set):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"id":42}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = gotify_plugin.tools["gotify_send"]["handler"]({"message": "ok"})
        data = json.loads(result)
        assert data["success"] is True
        assert data["status"] == 200

    @patch("urllib.request.urlopen")
    def test_http_error_returns_false(self, mock_urlopen, gotify_plugin, env_set):
        mock_urlopen.side_effect = Exception("Connection refused")

        result = gotify_plugin.tools["gotify_send"]["handler"]({"message": "fail"})
        data = json.loads(result)
        assert data["success"] is False
        assert "Connection refused" in data["error"]

    @patch("urllib.request.urlopen")
    def test_url_construction(self, mock_urlopen, gotify_plugin, env_set):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"id":1}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        gotify_plugin.tools["gotify_send"]["handler"]({"message": "hi"})
        sent_request = mock_urlopen.call_args[0][0]
        assert sent_request.get_method() == "POST"
        assert "http://gotify.test/message" in sent_request.full_url
        assert "token=testtoken123" in sent_request.full_url

    @patch("urllib.request.urlopen")
    def test_content_type_header(self, mock_urlopen, gotify_plugin, env_set):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"id":1}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        gotify_plugin.tools["gotify_send"]["handler"]({"message": "hi"})
        sent_request = mock_urlopen.call_args[0][0]
        assert sent_request.headers["Content-type"] == "application/json; charset=utf-8"
