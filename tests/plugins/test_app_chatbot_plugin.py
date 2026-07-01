"""Tests for the app-chatbot CRWD support plugin."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_DIR = _REPO_ROOT / "plugins" / "app-chatbot"
_PKG_NAME = "hermes_plugins.app_chatbot"


def _plugin_dir() -> Path:
    return _PLUGIN_DIR


def _purge_plugin_modules() -> None:
    for key in list(sys.modules):
        if key == _PKG_NAME or key.startswith(f"{_PKG_NAME}."):
            del sys.modules[key]


def _ensure_plugin_package():
    plugin_dir = _plugin_dir()
    if _PKG_NAME not in sys.modules:
        if "hermes_plugins" not in sys.modules:
            ns = types.ModuleType("hermes_plugins")
            ns.__path__ = []
            sys.modules["hermes_plugins"] = ns
        pkg = types.ModuleType(_PKG_NAME)
        pkg.__path__ = [str(plugin_dir)]
        pkg.__package__ = _PKG_NAME
        sys.modules[_PKG_NAME] = pkg

    load_order = ("_utils", "queries", "handlers", "router", "schemas", "prefetch")
    for sub in load_order:
        fq = f"{_PKG_NAME}.{sub}"
        if fq in sys.modules:
            continue
        sub_path = plugin_dir / f"{sub}.py"
        if not sub_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(fq, sub_path)
        submod = importlib.util.module_from_spec(spec)
        submod.__package__ = _PKG_NAME
        sys.modules[fq] = submod
        spec.loader.exec_module(submod)
    return sys.modules[_PKG_NAME]


def _load_module(name: str):
    _ensure_plugin_package()
    fq = f"{_PKG_NAME}.{name}"
    if fq.endswith(".py"):
        fq = fq[:-3]
    if name.endswith(".py"):
        name = name[:-3]
    if not fq.startswith("hermes_plugins"):
        fq = f"{_PKG_NAME}.{name}"
    return sys.modules[fq]


def _load_plugin_init():
    _ensure_plugin_package()
    plugin_dir = _plugin_dir()
    if _PKG_NAME in sys.modules and hasattr(sys.modules[_PKG_NAME], "register"):
        return sys.modules[_PKG_NAME]
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG_NAME
    mod.__path__ = [str(plugin_dir)]
    sys.modules[_PKG_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    _purge_plugin_modules()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("APP_CHATBOT_DEFAULT_USER_ID", "69a6f191cb29b0b371b3a156")
    yield hermes_home


class TestUtils:
    def test_redacts_sensitive_fields(self):
        utils = _load_module("_utils")
        doc = {"email": "a@b.com", "password": "secret", "nested": {"emailOTP": "1234"}}
        redacted = utils.redact_document(doc)
        assert redacted["password"] == "[REDACTED]"
        assert redacted["nested"]["emailOTP"] == "[REDACTED]"

    def test_parse_object_id_valid(self):
        utils = _load_module("_utils")
        oid = utils.parse_object_id("69a6f191cb29b0b371b3a156")
        assert str(oid) == "69a6f191cb29b0b371b3a156"

    def test_parse_object_id_invalid(self):
        utils = _load_module("_utils")
        with pytest.raises(ValueError):
            utils.parse_object_id("not-an-id")

    def test_default_user_id_from_env(self):
        utils = _load_module("_utils")
        assert utils.default_user_id() == "69a6f191cb29b0b371b3a156"


class TestRouter:
    def test_active_gigs_intent(self):
        router = _load_module("router")
        with patch.object(router.queries, "get_active_gigs", return_value={"success": True, "items": []}) as mock_fn:
            result = router.route_intent("What active gigs can I join?", "69a6f191cb29b0b371b3a156")
        assert result["tool"] == "get_active_gigs"
        mock_fn.assert_called_once()

    def test_joined_gigs_intent(self):
        router = _load_module("router")
        with patch.object(router.queries, "get_user_joined_gigs", return_value={"success": True, "items": []}) as mock_fn:
            result = router.route_intent("Show my joined gigs", "69a6f191cb29b0b371b3a156")
        assert result["tool"] == "get_user_joined_gigs"
        mock_fn.assert_called_once()

    def test_no_match_returns_none(self):
        router = _load_module("router")
        assert router.route_intent("hello world random chat", "69a6f191cb29b0b371b3a156") is None

    def test_identity_question_not_routed_as_gig(self):
        router = _load_module("router")
        assert router.route_intent("tell me about u", "69a6f191cb29b0b371b3a156") is None
        assert router.route_intent("who are you", "69a6f191cb29b0b371b3a156") is None

    def test_format_router_context_includes_user_line(self):
        router = _load_module("router")
        ctx = router.format_router_context("hello", default_user_id="69a6f191cb29b0b371b3a156")
        assert "Current CLI user_id" in ctx
        assert "APP_CHATBOT_DEFAULT_USER_ID" in ctx

    def test_format_router_context_includes_data_access_policy(self):
        router = _load_module("router")
        ctx = router.format_router_context("hello", default_user_id="69a6f191cb29b0b371b3a156")
        assert "[Data access policy]" in ctx
        assert "get_active_gigs" in ctx
        assert "Do not attempt direct MongoDB queries" in ctx


class TestQueries:
    def test_get_active_gigs_excludes_enrolled(self, monkeypatch):
        queries = _load_module("queries")
        mock_db = MagicMock()
        mock_crwds = MagicMock()
        mock_members = MagicMock()
        mock_db.added_crwd_members = mock_members
        mock_db.crwds = mock_crwds
        mock_members.find.return_value = [{"crwd_id": queries.parse_object_id("69e6a4d6cea992cbda22b381")}]
        mock_crwds.count_documents.return_value = 1
        mock_crwds.find.return_value.sort.return_value.skip.return_value.limit.return_value = [
            {"_id": queries.parse_object_id("69b8614f1083b9302fd0a9a7"), "name": "Test Gig", "status": "Active"},
        ]
        monkeypatch.setattr(queries, "get_mongo_db", lambda: mock_db)

        result = queries.get_active_gigs("69a6f191cb29b0b371b3a156")
        assert result["success"] is True
        assert len(result["items"]) == 1
        assert result["items"][0]["name"] == "Test Gig"
        query_used = mock_crwds.find.call_args[0][0]
        assert query_used["status"] == "Active"
        assert "$nin" in query_used["_id"]

    def test_get_user_profile_not_found(self, monkeypatch):
        queries = _load_module("queries")
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = None
        monkeypatch.setattr(queries, "get_mongo_db", lambda: mock_db)
        result = queries.get_user_profile_by_id("69a6f191cb29b0b371b3a156")
        assert result["success"] is False


class TestHandlers:
    def test_get_active_gigs_handler(self, monkeypatch):
        handlers = _load_module("handlers")
        with patch.object(handlers.queries, "get_active_gigs", return_value={"success": True, "items": [{"name": "Gig A"}]}):
            raw = handlers.get_active_gigs({})
        data = json.loads(raw)
        assert data["success"] is True
        assert data["items"][0]["name"] == "Gig A"

    def test_get_gig_details_requires_ref(self):
        handlers = _load_module("handlers")
        raw = handlers.get_gig_details({})
        data = json.loads(raw)
        assert data["success"] is False


class TestPluginRegistration:
    def test_register_wires_hook_and_tools(self):
        plugin = _load_plugin_init()
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_hook.assert_called_once_with("pre_llm_call", plugin._prefetch_context)
        assert ctx.register_tool.call_count == 5
        tool_names = [call.kwargs["name"] for call in ctx.register_tool.call_args_list]
        assert tool_names == [
            "get_active_gigs",
            "get_user_profile_by_id",
            "get_gig_details",
            "get_user_gig_history",
            "get_user_joined_gigs",
        ]

    def test_prefetch_context_returns_dict_when_routed(self, monkeypatch):
        plugin = _load_plugin_init()
        with patch.object(
            plugin,
            "format_router_context",
            return_value="[Database Context]\nfoo",
        ):
            result = plugin._prefetch_context(user_message="active gigs")
        assert result == {"context": "[Database Context]\nfoo"}

    def test_prefetch_context_uses_env_default_user_id(self):
        plugin = _load_plugin_init()
        with patch.object(plugin, "format_router_context", return_value="ctx") as mock_fmt:
            plugin._prefetch_context(user_message="active gigs")
        mock_fmt.assert_called_once_with("active gigs", default_user_id="69a6f191cb29b0b371b3a156")
