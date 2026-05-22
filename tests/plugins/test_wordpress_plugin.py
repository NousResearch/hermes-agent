"""Tests for the bundled WordPress plugin."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from plugins.wordpress.auth import get_credentials, wordpress_requirements_met
from plugins.wordpress.client import WordPressAPIError, WordPressClient, normalize_base_url
from plugins.wordpress.tools import handle_wp_site_info


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "wordpress"


def _load_plugin_init():
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.wordpress",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.wordpress"
    mod.__path__ = [str(PLUGIN_DIR)]
    sys.modules["hermes_plugins.wordpress"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.exists()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
        assert data["name"] == "wordpress"
        assert data["kind"] == "standalone"
        assert data["provides_tools"] == [
            "wp_site_info",
            "wp_post_list",
            "wp_post_get",
            "wp_post_create",
            "wp_post_update",
        ]
        assert data["requires_env"] == [
            "WORDPRESS_USERNAME",
            "WORDPRESS_APP_PASSWORD",
        ]


class TestAuth:
    def test_builds_basic_auth_header(self, monkeypatch):
        monkeypatch.setenv("WORDPRESS_BASE_URL", "example.com")
        monkeypatch.setenv("WORDPRESS_USERNAME", "alice")
        monkeypatch.setenv("WORDPRESS_APP_PASSWORD", "secret")

        creds = get_credentials()
        encoded = base64.b64encode(b"alice:secret").decode("ascii")
        assert creds.authorization_header() == f"Basic {encoded}"

    def test_requirements_gate(self, monkeypatch):
        monkeypatch.delenv("WORDPRESS_USERNAME", raising=False)
        monkeypatch.delenv("WORDPRESS_APP_PASSWORD", raising=False)
        assert wordpress_requirements_met() is False

    def test_requirements_gate_allows_runtime_base_url_override(self, monkeypatch):
        monkeypatch.delenv("WORDPRESS_BASE_URL", raising=False)
        monkeypatch.setenv("WORDPRESS_USERNAME", "alice")
        monkeypatch.setenv("WORDPRESS_APP_PASSWORD", "secret")
        assert wordpress_requirements_met() is True


class TestClient:
    def test_normalize_base_url(self):
        assert normalize_base_url("example.com/") == "https://example.com"
        assert normalize_base_url("http://example.com///") == "http://example.com"

    def test_site_info_reports_auth_state(self):
        calls = []

        def opener(req, timeout=0):
            calls.append(req.full_url)
            if req.full_url.endswith("/wp-json"):
                return _Response(
                    {
                        "name": "Demo",
                        "description": "Demo site",
                        "url": "https://example.com",
                        "home": "https://example.com",
                        "namespaces": ["wp/v2", "oembed/1.0"],
                    }
                )
            if "users/me" in req.full_url:
                auth = req.headers.get("Authorization")
                assert auth and auth.startswith("Basic ")
                return _Response({"id": 7, "slug": "alice", "name": "Alice"})
            raise AssertionError(f"unexpected url {req.full_url}")

        client = WordPressClient(
            credentials=get_credentials(
                base_url="example.com",
                username="alice",
                app_password="secret",
            ),
            opener=opener,
        )

        payload = client.get_site_info()
        assert calls == [
            "https://example.com/wp-json",
            "https://example.com/wp-json/wp/v2/users/me?context=edit",
        ]
        assert payload["site"]["name"] == "Demo"
        assert payload["auth"]["authenticated"] is True
        assert payload["auth"]["current_user"]["slug"] == "alice"

    def test_site_info_tolerates_unauthorized_user_probe(self):
        from urllib.error import HTTPError
        from io import BytesIO

        def opener(req, timeout=0):
            if req.full_url.endswith("/wp-json"):
                return _Response({"name": "Demo", "namespaces": ["wp/v2"]})
            payload = json.dumps({"message": "Unauthorized"}).encode("utf-8")
            raise HTTPError(req.full_url, 401, "Unauthorized", hdrs=None, fp=BytesIO(payload))

        client = WordPressClient(
            credentials=get_credentials(
                base_url="example.com",
                username="alice",
                app_password="secret",
            ),
            opener=opener,
        )

        payload = client.get_site_info()
        assert payload["auth"]["authenticated"] is False
        assert payload["auth"]["current_user"] is None

    def test_raises_for_non_auth_api_errors(self):
        from urllib.error import HTTPError
        from io import BytesIO

        def opener(req, timeout=0):
            payload = json.dumps({"message": "Server error"}).encode("utf-8")
            raise HTTPError(req.full_url, 500, "Server Error", hdrs=None, fp=BytesIO(payload))

        client = WordPressClient(
            credentials=get_credentials(
                base_url="example.com",
                username="alice",
                app_password="secret",
            ),
            opener=opener,
        )

        with pytest.raises(WordPressAPIError) as excinfo:
            client.get_site_info()
        assert excinfo.value.status_code == 500

    def test_site_info_rejects_non_object_root_payload(self):
        def opener(req, timeout=0):
            return _Response(["not", "an", "object"])

        client = WordPressClient(
            credentials=get_credentials(
                base_url="example.com",
                username="alice",
                app_password="secret",
            ),
            opener=opener,
        )

        with pytest.raises(WordPressAPIError) as excinfo:
            client.get_site_info()
        assert "unexpected payload" in str(excinfo.value)

    def test_site_info_rejects_non_object_users_me_payload(self):
        def opener(req, timeout=0):
            if req.full_url.endswith("/wp-json"):
                return _Response({"name": "Demo", "namespaces": ["wp/v2"]})
            return _Response(["not", "an", "object"])

        client = WordPressClient(
            credentials=get_credentials(
                base_url="example.com",
                username="alice",
                app_password="secret",
            ),
            opener=opener,
        )

        with pytest.raises(WordPressAPIError) as excinfo:
            client.get_site_info()
        assert "users/me returned an unexpected payload" in str(excinfo.value)

    def test_list_posts_sends_edit_context(self):
        seen = {}

        def opener(req, timeout=0):
            seen["url"] = req.full_url
            return _Response([{"id": 1, "slug": "hello"}])

        client = WordPressClient(
            credentials=get_credentials(
                base_url="example.com",
                username="alice",
                app_password="secret",
            ),
            opener=opener,
        )

        payload = client.list_posts(query={"status": "draft", "per_page": 5})
        assert seen["url"] == "https://example.com/wp-json/wp/v2/posts?context=edit&status=draft&per_page=5"
        assert payload[0]["id"] == 1

    def test_create_post_uses_post_json_body(self):
        seen = {}

        def opener(req, timeout=0):
            seen["url"] = req.full_url
            seen["method"] = req.get_method()
            seen["body"] = json.loads(req.data.decode("utf-8"))
            return _Response({"id": 10, "status": "draft"})

        client = WordPressClient(
            credentials=get_credentials(
                base_url="example.com",
                username="alice",
                app_password="secret",
            ),
            opener=opener,
        )

        payload = client.create_post({"title": "Demo", "status": "draft"})
        assert seen["url"] == "https://example.com/wp-json/wp/v2/posts"
        assert seen["method"] == "POST"
        assert seen["body"] == {"title": "Demo", "status": "draft"}
        assert payload["id"] == 10

    def test_update_post_posts_to_item_endpoint(self):
        seen = {}

        def opener(req, timeout=0):
            seen["url"] = req.full_url
            seen["method"] = req.get_method()
            seen["body"] = json.loads(req.data.decode("utf-8"))
            return _Response({"id": 10, "status": "publish"})

        client = WordPressClient(
            credentials=get_credentials(
                base_url="example.com",
                username="alice",
                app_password="secret",
            ),
            opener=opener,
        )

        payload = client.update_post(10, {"status": "publish"})
        assert seen["url"] == "https://example.com/wp-json/wp/v2/posts/10"
        assert seen["method"] == "POST"
        assert seen["body"] == {"status": "publish"}
        assert payload["status"] == "publish"


class TestRegister:
    def test_registers_all_wordpress_tools(self):
        module = _load_plugin_init()

        calls = []

        class DummyCtx:
            def register_tool(self, **kwargs):
                calls.append(kwargs)

        module.register(DummyCtx())
        assert [call["name"] for call in calls] == [
            "wp_site_info",
            "wp_post_list",
            "wp_post_get",
            "wp_post_create",
            "wp_post_update",
        ]
        assert all(call["toolset"] == "wordpress" for call in calls)
        assert all(callable(call["check_fn"]) for call in calls)
        assert all(
            call["check_fn"].__name__ == wordpress_requirements_met.__name__
            for call in calls
        )


class TestToolHandler:
    def test_handler_returns_tool_error_on_config_issue(self, monkeypatch):
        monkeypatch.delenv("WORDPRESS_BASE_URL", raising=False)
        monkeypatch.delenv("WORDPRESS_USERNAME", raising=False)
        monkeypatch.delenv("WORDPRESS_APP_PASSWORD", raising=False)

        result = handle_wp_site_info({})
        assert '"error":' in result
        assert "Missing WordPress configuration" in result

    def test_handler_returns_tool_result(self, monkeypatch):
        monkeypatch.setenv("WORDPRESS_BASE_URL", "example.com")
        monkeypatch.setenv("WORDPRESS_USERNAME", "alice")
        monkeypatch.setenv("WORDPRESS_APP_PASSWORD", "secret")

        fake_payload = {"site": {"name": "Demo"}, "auth": {"authenticated": True}}
        mock_get_site_info = MagicMock(return_value=fake_payload)
        monkeypatch.setattr(
            "plugins.wordpress.client.WordPressClient.get_site_info",
            mock_get_site_info,
        )

        result = handle_wp_site_info({})
        assert '"name": "Demo"' in result

    def test_post_get_requires_post_id(self):
        from plugins.wordpress.tools import handle_wp_post_get

        result = handle_wp_post_get({})
        assert '"error": "post_id is required"' in result

    def test_post_create_allows_missing_title(self, monkeypatch):
        from plugins.wordpress.tools import handle_wp_post_create

        monkeypatch.setenv("WORDPRESS_BASE_URL", "example.com")
        monkeypatch.setenv("WORDPRESS_USERNAME", "alice")
        monkeypatch.setenv("WORDPRESS_APP_PASSWORD", "secret")
        monkeypatch.setattr(
            "plugins.wordpress.client.WordPressClient.create_post",
            MagicMock(return_value={"id": 3}),
        )

        result = handle_wp_post_create({"content": "Body only"})
        assert '"id": 3' in result

    def test_post_create_requires_minimum_content(self):
        from plugins.wordpress.tools import handle_wp_post_create

        result = handle_wp_post_create({})
        assert '"error": "Provide at least one of: title, content, excerpt, status"' in result

    def test_post_update_requires_fields(self):
        from plugins.wordpress.tools import handle_wp_post_update

        result = handle_wp_post_update({"post_id": 3})
        assert '"error": "Provide at least one field to update"' in result

    def test_post_handlers_return_client_payloads(self, monkeypatch):
        from plugins.wordpress.tools import (
            handle_wp_post_create,
            handle_wp_post_get,
            handle_wp_post_list,
            handle_wp_post_update,
        )

        monkeypatch.setenv("WORDPRESS_BASE_URL", "example.com")
        monkeypatch.setenv("WORDPRESS_USERNAME", "alice")
        monkeypatch.setenv("WORDPRESS_APP_PASSWORD", "secret")

        monkeypatch.setattr(
            "plugins.wordpress.client.WordPressClient.list_posts",
            MagicMock(return_value=[{"id": 1}]),
        )
        monkeypatch.setattr(
            "plugins.wordpress.client.WordPressClient.get_post",
            MagicMock(return_value={"id": 2}),
        )
        monkeypatch.setattr(
            "plugins.wordpress.client.WordPressClient.create_post",
            MagicMock(return_value={"id": 3}),
        )
        monkeypatch.setattr(
            "plugins.wordpress.client.WordPressClient.update_post",
            MagicMock(return_value={"id": 4}),
        )

        assert '"id": 1' in handle_wp_post_list({"status": "draft"})
        assert '"id": 2' in handle_wp_post_get({"post_id": 2})
        assert '"id": 3' in handle_wp_post_create({"title": "Draft"})
        assert '"id": 4' in handle_wp_post_update({"post_id": 4, "status": "draft"})
