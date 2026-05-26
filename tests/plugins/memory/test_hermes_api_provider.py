import json

from plugins.memory import discover_memory_providers, load_memory_provider
from plugins.memory.hermes_api import (
    HermesApiMemoryProvider,
    _load_config,
)


def test_provider_discovery_loads_hermes_api():
    providers = {name: (desc, available) for name, desc, available in discover_memory_providers()}

    assert "hermes_api" in providers
    assert providers["hermes_api"][1] is True

    provider = load_memory_provider("hermes_api")
    assert isinstance(provider, HermesApiMemoryProvider)
    assert provider.name == "hermes_api"


def test_initialize_resolves_gateway_user_and_prefetches_contact_context(monkeypatch):
    calls = []

    def fake_request(method, path, *, query=None, body=None, timeout=5):
        calls.append((method, path, query, body))
        if path == "/api/v1/identities":
            assert query == {"kind": "telegram", "value": "42"}
            return {"success": True, "data": [{"contact": {"id": "contact-1"}}]}
        if path == "/api/v1/contacts/contact-1":
            return {
                "success": True,
                "data": {
                    "id": "contact-1",
                    "name": "Neeraj Dalal",
                    "username": "nrjdalal",
                    "contactMd": "Dalonic primary admin.",
                    "memoryMd": "Prefers short Hinglish updates.",
                    "tags": ["admin", "dalonic"],
                },
            }
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr("plugins.memory.hermes_api._request_json", fake_request)

    provider = HermesApiMemoryProvider()
    provider.initialize(session_id="sess-1", platform="telegram", user_id="42")

    context = provider.prefetch("what do you know about me")

    assert calls[:2] == [
        ("GET", "/api/v1/identities", {"kind": "telegram", "value": "42"}, None),
        ("GET", "/api/v1/contacts/contact-1", None, None),
    ]
    assert "<hermes-api-contact-context>" in context
    assert "Current contact: Neeraj Dalal" in context
    assert "Dalonic primary admin." in context
    assert "Prefers short Hinglish updates." in context


def test_tools_list_filter_and_append_contact_memory(monkeypatch):
    patches = []

    def fake_request(method, path, *, query=None, body=None, timeout=5):
        if method == "GET" and path == "/api/v1/contacts":
            return {
                "success": True,
                "data": [
                    {"id": "c1", "name": "Neeraj", "username": "nrjdalal", "memoryMd": "admin"},
                    {"id": "c2", "name": "Disha", "username": "disha", "memoryMd": "core"},
                ],
            }
        if method == "GET" and path == "/api/v1/contacts/c1":
            return {"success": True, "data": {"id": "c1", "name": "Neeraj", "memoryMd": "admin"}}
        if method == "PATCH" and path == "/api/v1/contacts/c1":
            assert body is not None
            patches.append(body)
            return {"success": True, "data": {"id": "c1", "name": "Neeraj", "memoryMd": body["memoryMd"]}}
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr("plugins.memory.hermes_api._request_json", fake_request)
    provider = HermesApiMemoryProvider()

    listed = json.loads(provider.handle_tool_call("hermes_api_list_contacts", {"query": "nrj", "limit": 5}))
    assert listed["success"] is True
    assert [row["id"] for row in listed["data"]] == ["c1"]

    appended = json.loads(
        provider.handle_tool_call(
            "hermes_api_append_contact_memory",
            {"contact_id": "c1", "content": "- likes concise status"},
        )
    )
    assert appended["success"] is True
    assert patches == [{"memoryMd": "admin\n- likes concise status"}]


def test_save_config_persists_base_url(tmp_path):
    provider = HermesApiMemoryProvider()
    provider.save_config({"base_url": "http://127.0.0.1:4000/"}, str(tmp_path))

    assert _load_config(str(tmp_path)) == {"base_url": "http://127.0.0.1:4000"}


def test_on_memory_write_mirrors_active_contact(monkeypatch):
    updates = []

    provider = HermesApiMemoryProvider()
    provider._contact = {"id": "c1", "name": "Neeraj", "memoryMd": "existing"}

    def fake_append(contact_id, content):
        updates.append((contact_id, content))
        return {"id": contact_id, "memoryMd": f"existing\n{content}"}

    monkeypatch.setattr(provider, "_append_contact_memory", fake_append)

    provider.on_memory_write("add", "user", "User prefers concise replies")

    assert updates == [("c1", "- User prefers concise replies")]
