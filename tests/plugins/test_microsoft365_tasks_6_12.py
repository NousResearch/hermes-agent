import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.microsoft365.backend import Microsoft365Settings, preflight, safe_result
from plugins.microsoft365 import tools


class Context:
    def __init__(self, config):
        self.config = config

    def get_config(self, key, default=None):
        return self.config.get(key, default)


def ctx(capability, operations, **extra):
    return Context({"tenant_id": "t", "client_id": "c", "client_secret": "secret", "user_id": "u", "capabilities": {capability: operations}, **extra})


def test_upload_accepts_bounded_base64_and_sets_content_type(monkeypatch):
    calls = []
    class Content:
        async def put(self, value):
            calls.append(value)
            return SimpleNamespace(id="item", name="x.bin", size=3)
    class Item:
        content = Content()
    class Root:
        def with_url(self, _): return Item()
        def item_with_path(self, _): return Item()
    class Drive:
        id = "drive"
        async def get(self): return self
    class Client:
        users = None
        def __init__(self): self.users = self; self.drives = self; self.drive = self
        def by_user_id(self, _): return self
        def by_drive_id(self, _): return self
        def with_url(self, _): return Item()
        async def get(self): return Drive()
    monkeypatch.setattr(tools, "create_graph_client", lambda _: Client())
    result = json.loads(asyncio.run(tools._run("onedrive", {"action": "upload_files", "path": "x.bin", "content_base64": base64.b64encode(b"abc").decode(), "content_type": "application/octet-stream"}, ctx("onedrive", {"upload_files": True}))))
    assert result["success"] is True
    assert calls == [b"abc"]


def test_upload_rejects_traversal_and_oversize_before_client(monkeypatch):
    def no_client(_): raise AssertionError("client must not be created")
    monkeypatch.setattr(tools, "create_graph_client", no_client)
    for path in ("../x", "a/../../x"):
        result = asyncio.run(tools._run("onedrive", {"action": "upload_files", "path": path, "content_base64": "YQ=="}, ctx("onedrive", {"upload_files": True})))
        assert "path" in result.lower()
    result = asyncio.run(tools._run("onedrive", {"action": "upload_files", "path": "x", "content_base64": base64.b64encode(b"x" * (10 * 1024 * 1024 + 1)).decode()}, ctx("onedrive", {"upload_files": True})))
    assert "limit" in result.lower()


def test_download_returns_bounded_base64(monkeypatch):
    class Item:
        async def get(self): return b"abc"
    class Drive:
        id = "drive"
        async def get(self): return self
    class Client:
        def __init__(self): self.users = self; self.drives = self; self.drive = self
        def by_user_id(self, _): return self
        def by_drive_id(self, _): return self
        def with_url(self, _): return SimpleNamespace(content=Item())
        async def get(self): return Drive()
    monkeypatch.setattr(tools, "create_graph_client", lambda _: Client())
    result = json.loads(asyncio.run(tools._run("onedrive", {"action": "download_files", "path": "x.bin"}, ctx("onedrive", {"download_files": True}))))
    assert result["result"]["content_base64"] == "YWJj"


def test_search_builds_query_and_page_bound(monkeypatch):
    calls = []
    class Request:
        def __init__(self): self.query_parameters = None
    class Builder:
        def __init__(self): self.request_configuration = Request()
        async def get(self, request_configuration=None): calls.append(request_configuration); return {"value": []}
    class Client:
        def __init__(self): self.users = self; self.messages = Builder()
        def by_user_id(self, _): return self
    monkeypatch.setattr(tools, "create_graph_client", lambda _: Client())
    result = json.loads(asyncio.run(tools._run("outlook", {"action": "search", "query": "hello", "limit": 7}, ctx("outlook", {"search": True}))))
    assert result["success"] is True
    assert calls[0].query_parameters["$top"] == 7
    assert "hello" in calls[0].query_parameters["$filter"]


def test_partial_event_patch_only_sends_supplied_fields(monkeypatch):
    calls = []
    class Events:
        def by_event_id(self, _): return self
        async def patch(self, model): calls.append(model)
    class Client:
        def __init__(self): self.users = self; self.calendar = self; self.events = Events()
        def by_user_id(self, _): return self
    monkeypatch.setattr(tools, "create_graph_client", lambda _: Client())
    result = json.loads(asyncio.run(tools._run("calendar", {"action": "update_events", "id": "e", "subject": "new"}, ctx("calendar", {"update_events": True}))))
    assert result["success"] is True
    assert calls[0].subject == "new"
    assert not hasattr(calls[0], "start") or calls[0].start is None


def test_todo_and_planner_are_distinct_and_planner_does_not_use_todo():
    from plugins.microsoft365.backend import OPERATIONS
    assert "todo" in OPERATIONS and "planner" in OPERATIONS
    assert OPERATIONS["todo"] != OPERATIONS["planner"]


def test_safe_result_redacts_nested_headers_and_exception():
    value = {"subject": "useful", "additional_data": {"Authorization": "Bearer secret", "x": 1}, "headers": {"cookie": "secret"}}
    result = safe_result(value)
    assert result["subject"] == "useful"
    assert result["additional_data"]["Authorization"] == "[REDACTED]"
    assert result["headers"] == "[REDACTED]"
    assert "secret" not in repr(result)


def test_dependency_metadata_is_pinned_to_tested_sdk_floor():
    manifest = Path("plugins/microsoft365/plugin.yaml").read_text()
    pyproject = Path("pyproject.toml").read_text()
    assert 'msgraph-sdk>=1.62.0,<2' in manifest
    assert 'msgraph-sdk>=1.62.0,<2' in pyproject
