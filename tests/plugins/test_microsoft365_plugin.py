"""Contract and SDK request-builder tests for the Microsoft 365 Plugin."""
import asyncio, json
from types import SimpleNamespace
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

EXPECTED = {
 "outlook": ("search","read","create_draft","send"),
 "sharepoint": ("search","read","download_files","upload_files"),
 "onedrive": ("search","read","download_files","upload_files"),
 "calendar": ("search","create_events","update_events"),
 "teams": ("list_teams","list_channels","search_messages","send_messages"),
 "planner": ("list_task_lists","search","read","create_tasks","update_tasks"),
}

def test_advertised_operations_cover_unified_read_write_scope():
    from plugins.microsoft365.backend import OPERATIONS
    assert OPERATIONS == EXPECTED

def test_permission_mapping_is_least_privilege_for_every_operation():
    from plugins.microsoft365.backend import Microsoft365Settings, OPERATION_PERMISSIONS, required_permissions
    for service, ops in EXPECTED.items():
        assert set(ops) == set(OPERATION_PERMISSIONS[service])
    settings = Microsoft365Settings.from_mapping({"capabilities":{"outlook":{"send":True},"onedrive":{"upload_files":True}}})
    assert required_permissions(settings) == {"Mail.Send", "Files.ReadWrite"}

def test_schema_advertises_writes_and_confirmation_contract():
    from plugins.microsoft365.tools import schema
    assert schema("outlook")["parameters"]["properties"]["action"]["enum"] == list(EXPECTED["outlook"])
    assert "approval" in schema("outlook")["description"]

def test_service_true_enables_all_operations_for_compatibility():
    from plugins.microsoft365.backend import Microsoft365Settings
    assert Microsoft365Settings.from_mapping({"capabilities":{"outlook":True}}).operations("outlook") == set(EXPECTED["outlook"])

def test_disabled_operation_is_rejected_before_graph_client_creation(monkeypatch):
    from plugins.microsoft365 import tools
    class Context:
        def get_config(self, key, default=None): return {"capabilities":{"outlook":{"search":True}}}.get(key, default)
    monkeypatch.setattr(tools, "create_graph_client", lambda _: (_ for _ in ()).throw(AssertionError("client created")))
    result = asyncio.run(tools._run("outlook", {"action":"read"}, Context()))
    assert "disabled" in result.lower()

def test_write_without_host_approval_has_no_side_effect(monkeypatch):
    from plugins.microsoft365 import tools
    class Context:
        def get_config(self, key, default=None): return {"capabilities":{"outlook":{"send":True}}}.get(key, default)
    monkeypatch.setattr(tools, "_approved", lambda *a: False)
    monkeypatch.setattr(tools, "create_graph_client", lambda _: (_ for _ in ()).throw(AssertionError("client created")))
    result = json.loads(asyncio.run(tools._run("outlook", {"action":"send"}, Context())))
    assert result["required_confirmation"] is True

def test_fake_sdk_builder_methods_and_models_are_used(monkeypatch):
    from plugins.microsoft365 import tools
    calls=[]
    class Builder:
        async def post(self, model): calls.append(("post", model)); return SimpleNamespace(id="draft-1")
        async def put(self, content): calls.append(("put", content)); return SimpleNamespace(id="file-1")
        async def get(self): calls.append(("get",)); return {"value":[]}
        def by_message_id(self, ident): calls.append(("by_message_id",ident)); return self
        def send(self): return self
        async def patch(self, model): calls.append(("patch",model)); return model
        def item_with_path(self, path): calls.append(("item_with_path",path)); return self
        @property
        def content(self): return self
        @property
        def root(self): return self
    class Client:
        def __init__(self): self.users=self; self.calendar=self; self.events=self; self.messages=Builder(); self.drive=Builder(); self.root=self.drive; self.teams=self; self.channels=self; self.todo=self; self.lists=self; self.tasks=self
        def by_user_id(self, x): return self
        def by_site_id(self, x): return self
        def by_team_id(self, x): return self
        def by_channel_id(self, x): return self
        def by_event_id(self, x): return self
        def by_todo_task_list_id(self, x): return self
        def by_todo_task_id(self, x): return self
        def search_with_q(self, x): return self
        def __getattr__(self, x): return self
        async def get(self): calls.append(("get",)); return {"value":[]}
        async def post(self, model): calls.append(("post",model)); return SimpleNamespace(id="x")
        async def put(self, content): calls.append(("put", content)); return SimpleNamespace(id="file-1")
    monkeypatch.setattr(tools, "create_graph_client", lambda _: Client())
    monkeypatch.setattr(tools, "_approved", lambda *a: True)
    monkeypatch.setattr(tools, "_model", lambda name, **kw: SimpleNamespace(model_name=name, **kw))
    class Context:
        def get_config(self, key, default=None): return {"capabilities":{"onedrive":{"upload_files":True}}}.get(key, default)
    result=json.loads(asyncio.run(tools._run("onedrive", {"action":"upload_files","path":"a.txt","content":"x"}, Context())))
    assert result["success"] is True
    assert any(c[0] == "put" for c in calls)
    assert any(c[0] == "item_with_path" for c in calls)

def test_preflight_redacts_and_derives_permissions():
    from plugins.microsoft365.backend import Microsoft365Settings, preflight
    result=preflight(Microsoft365Settings.from_mapping({"tenant_id":"t","client_id":"c","client_secret":"secret","capabilities":{"outlook":{"send":True}}}), sdk_available=False)
    assert result["required_permissions"] == ["Mail.Send"] and "secret" not in result["configuration"]["client_secret"]

def test_registers_only_enabled_capability_tools():
    from plugins.microsoft365 import register
    manager=PluginManager(); context=PluginContext(PluginManifest(name="microsoft365"), manager)
    context.get_config=lambda key, default=None: {"capabilities":{"outlook":{"search":True},"calendar":{"update_events":True}}}.get(key,default)
    register(context)
    assert set(manager._plugin_tool_names)=={"microsoft365_preflight","microsoft365_outlook","microsoft365_calendar"}
