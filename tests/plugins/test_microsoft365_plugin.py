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
 "todo": ("list_task_lists","search","read","create_tasks","update_tasks"),
 "planner": ("list_plans","list_buckets","list_tasks","read","create_tasks","update_tasks"),
}

def test_advertised_operations_cover_unified_read_write_scope():
    from plugins.microsoft365.backend import OPERATIONS
    assert OPERATIONS == EXPECTED

def test_permission_mapping_is_least_privilege_for_every_operation():
    from plugins.microsoft365.backend import Microsoft365Settings, OPERATION_PERMISSIONS, required_permissions
    for service, ops in EXPECTED.items():
        assert set(ops) == set(OPERATION_PERMISSIONS[service])
    settings = Microsoft365Settings.from_mapping({"capabilities":{"outlook":{"send":True},"onedrive":{"upload_files":True}}})
    assert required_permissions(settings) == {"Mail.Send", "Files.ReadWrite.All"}


def test_application_permission_mode_does_not_claim_delegated_only_permissions():
    from plugins.microsoft365.backend import (
        APPLICATION_PERMISSION_MODE, Microsoft365Settings, OPERATION_PERMISSIONS,
        required_permissions, unsupported_operations,
    )
    assert APPLICATION_PERMISSION_MODE == "application"
    settings = Microsoft365Settings.from_mapping({"capabilities": {
        "outlook": {"search": True, "send": True},
        "onedrive": {"upload_files": True},
        "teams": {"send_messages": True},
        "planner": {"create_tasks": True},
    }})
    assert required_permissions(settings) == {
        "Mail.Read", "Mail.Send", "Files.ReadWrite.All", "Tasks.ReadWrite.All",
    }
    assert "ChatMessage.Send" not in required_permissions(settings)
    assert "Tasks.ReadWrite" not in required_permissions(settings)
    assert unsupported_operations(settings) == ["teams.send_messages"]
    assert OPERATION_PERMISSIONS["teams"]["send_messages"] == set()


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

def test_fake_sdk_builder_methods_and_models_are_used(monkeypatch):
    from plugins.microsoft365 import tools
    calls=[]
    class Builder:
        async def post(self, model): calls.append(("post", model)); return SimpleNamespace(id="draft-1")
        async def put(self, content): calls.append(("put", content)); return SimpleNamespace(id="file-1")
        async def get(self): calls.append(("get",)); return SimpleNamespace(id="drive-1")
        def by_message_id(self, ident): calls.append(("by_message_id",ident)); return self
        def send(self): return self
        async def patch(self, model): calls.append(("patch",model)); return model
        def item_with_path(self, path): calls.append(("item_with_path",path)); return self
        def with_url(self, url): calls.append(("with_url",url)); return self
        @property
        def content(self): return self
        @property
        def root(self): return self
    class Client:
        def __init__(self): self.users=self; self.calendar=self; self.events=self; self.messages=Builder(); self.drive=Builder(); self.drives=self; self.root=self.drive; self.teams=self; self.channels=self; self.todo=self; self.lists=self; self.tasks=self
        def by_user_id(self, x): return self
        def by_drive_id(self, x): return self
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
    monkeypatch.setattr(tools, "_model", lambda name, **kw: SimpleNamespace(model_name=name, **kw))
    class Context:
        def get_config(self, key, default=None): return {"capabilities":{"onedrive":{"upload_files":True}}}.get(key, default)
    result=json.loads(asyncio.run(tools._run("onedrive", {"action":"upload_files","path":"a.txt","content_base64":"eA=="}, Context())))
    assert result.get("success") is True, result
    assert any(c[0] == "put" for c in calls)
    assert any(c[0] == "item_with_path" for c in calls)

def test_send_uses_generated_send_mail_body_and_action_builder(monkeypatch):
    from plugins.microsoft365 import tools
    calls = []
    class Builder:
        async def post(self, body):
            calls.append(body)
            return SimpleNamespace(id="sent-1")
    class Client:
        def __init__(self):
            self.users = self
            self.messages = self
            self.send_mail = Builder()
        def by_user_id(self, _): return self
    monkeypatch.setattr(tools, "create_graph_client", lambda _: Client())
    class Context:
        def get_config(self, key, default=None):
            return {"capabilities":{"outlook":{"send":True}}}.get(key, default)
    result = json.loads(asyncio.run(tools._run("outlook", {"action":"send", "to":"a@example.com", "subject":"S", "body":"B"}, Context())))
    assert result.get("success") is True, result
    assert calls and type(calls[0]).__name__ == "SendMailPostRequestBody"
    assert calls[0].message.to_recipients[0].email_address.address == "a@example.com"


def test_drive_read_downloads_and_upload_puts_via_drive_item_content(monkeypatch):
    from plugins.microsoft365 import tools
    calls = []
    class Content:
        async def get(self): calls.append(("get",)); return b"data"
        async def put(self, content): calls.append(("put", content)); return SimpleNamespace(id="file")
    class Item:
        content = Content()
    class Root:
        async def get(self): calls.append(("get",)); return {"id":"file"}
        def with_url(self, url): calls.append(("with_url", url)); return self
        content = Content()
    class Drive:
        id = "drive-1"
    class Client:
        def __init__(self): self.users = self; self.drive = self; self.drives = self; self.root = Root()
        def by_user_id(self, _): return self
        async def get(self): return Drive()
        def by_drive_id(self, _): return self
    monkeypatch.setattr(tools, "create_graph_client", lambda _: Client())
    class Context:
        def get_config(self, key, default=None):
            return {"capabilities":{"onedrive":{"read":True, "download_files":True, "upload_files":True}}}.get(key, default)
    for action, expected in (("read", "get"), ("download_files", "get"), ("upload_files", "put")):
        args = {"action":action, "path":"folder/a.txt", "content_base64":"eA=="}
        result = json.loads(asyncio.run(tools._run("onedrive", args, Context())))
        assert result.get("success") is True, result
        assert any(call[0] == expected for call in calls)


def test_teams_search_uses_query_post_body_and_generated_enum(monkeypatch):
    from plugins.microsoft365 import tools
    calls = []
    class Query:
        async def post(self, body): calls.append(body); return {"value": []}
    class Client:
        def __init__(self): self.search = self; self.query = Query()
    monkeypatch.setattr(tools, "create_graph_client", lambda _: Client())
    class Context:
        def get_config(self, key, default=None):
            return {"capabilities":{"teams":{"search_messages":True}}}.get(key, default)
    result = json.loads(asyncio.run(tools._run("teams", {"action":"search_messages", "query":"hello"}, Context())))
    assert result.get("success") is True, result
    assert type(calls[0]).__name__ == "QueryPostRequestBody"
    request = calls[0].requests[0]
    assert request.query.query_string == "hello"
    assert str(request.entity_types[0].value) == "chatMessage"


def test_generated_todo_and_teams_models_construct_with_sdk_1_62():
    from plugins.microsoft365.tools import _body, _model
    task = _model("TodoTask", title="Task", body=_body("Details"))
    message = _model("ChatMessage", body=_body("Hello"))
    assert task.title == "Task"
    assert message.body.content == "Hello"


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


def test_support_resolver_blocks_unsupported_app_only_operation_before_client(monkeypatch):
    from plugins.microsoft365.backend import Microsoft365Settings, operation_support
    from plugins.microsoft365 import tools

    settings = Microsoft365Settings.from_mapping({"capabilities": {"teams": {"send_messages": True}}})
    status = operation_support("application", "teams", "send_messages")
    assert status.code == "unsupported_auth_mode"
    monkeypatch.setattr(tools, "create_graph_client", lambda _: (_ for _ in ()).throw(AssertionError("client created")))
    class Context:
        def get_config(self, key, default=None):
            return {"capabilities": {"teams": {"send_messages": True}}}.get(key, default)
    result = asyncio.run(tools._run("teams", {"action": "send_messages"}, Context()))
    assert "unsupported" in result.lower()


def test_preflight_is_not_ready_for_me_or_unsupported_selection():
    from plugins.microsoft365.backend import Microsoft365Settings, preflight
    result = preflight(Microsoft365Settings.from_mapping({
        "tenant_id": "t", "client_id": "c", "client_secret": "secret",
        "capabilities": {"outlook": {"read": True}, "teams": {"send_messages": True}},
    }), sdk_available=True)
    assert result["ready"] is False
    assert result["locally_ready"] is False
    assert result["authentication"] == "not_tested"
    assert result["permissions"] == "not_tested"
    assert "user_id" in result["missing"]
    assert result["unsupported_operations"] == ["teams.send_messages"]
    assert result["configuration"]["client_secret"] == "[REDACTED]"


def test_preflight_can_be_locally_ready_without_claiming_remote_access():
    from plugins.microsoft365.backend import Microsoft365Settings, preflight
    result = preflight(Microsoft365Settings.from_mapping({
        "tenant_id": "t", "client_id": "c", "client_secret": "secret", "user_id": "u",
        "capabilities": {"sharepoint": {"search": True}},
    }), sdk_available=True)
    assert result["ready"] is True
    assert result["locally_ready"] is True
    assert result["authentication"] == "not_tested"
    assert result["permissions"] == "not_tested"


def test_schema_only_contains_enabled_supported_operations():
    from plugins.microsoft365.tools import schema
    assert schema("outlook", {"search"})["parameters"]["properties"]["action"]["enum"] == ["search"]
    assert "send_messages" not in schema("teams", {"send_messages"})["parameters"]["properties"]["action"]["enum"]


def test_enabled_capability_keeps_one_tool_with_empty_enum_when_nothing_is_supported():
    from plugins.microsoft365 import register
    manager = PluginManager(); context = PluginContext(PluginManifest(name="microsoft365"), manager)
    context.get_config = lambda key, default=None: {"capabilities": {"teams": {"send_messages": True}}}.get(key, default)
    register(context)
    assert "microsoft365_teams" in manager._plugin_tool_names
    from plugins.microsoft365.tools import schema
    assert schema("teams", set())["parameters"]["properties"]["action"]["enum"] == []


def test_pre_tool_call_emits_host_approval_directive_for_enabled_write():
    from plugins.microsoft365 import register
    manager = PluginManager(); context = PluginContext(PluginManifest(name="microsoft365"), manager)
    context.get_config = lambda key, default=None: {"tenant_id": "t", "client_id": "c", "client_secret": "s", "user_id": "u", "capabilities": {"outlook": {"send": True}}}.get(key, default)
    register(context)
    result = manager.invoke_hook("pre_tool_call", tool_name="microsoft365_outlook", args={"action": "send", "to": "a@example.com", "body": "B"})
    assert result == [{"action": "approve", "message": "Microsoft 365 send: external side effect", "rule_key": "microsoft365.outlook.send"}]


def test_pre_tool_call_blocks_invalid_write_args_before_host_approval():
    from plugins.microsoft365 import register
    manager = PluginManager(); context = PluginContext(PluginManifest(name="microsoft365"), manager)
    context.get_config = lambda key, default=None: {"capabilities": {"outlook": {"send": True}}}.get(key, default)
    register(context)
    result = manager.invoke_hook("pre_tool_call", tool_name="microsoft365_outlook", args={"action": "send"})
    assert result[0]["action"] == "block"
    assert "required" in result[0]["message"]


def test_pre_tool_call_ignores_reads_and_other_tools_and_blocks_malformed_args():
    from plugins.microsoft365 import register
    manager = PluginManager(); context = PluginContext(PluginManifest(name="microsoft365"), manager)
    context.get_config = lambda key, default=None: {"capabilities": {"outlook": {"search": True, "send": True}}}.get(key, default)
    register(context)
    assert manager.invoke_hook("pre_tool_call", tool_name="microsoft365_outlook", args={"action": "search", "query": "hello"}) == []
    assert manager.invoke_hook("pre_tool_call", tool_name="other_tool", args={"action": "send"}) == []
    assert manager.invoke_hook("pre_tool_call", tool_name="microsoft365_outlook", args=None)[0]["action"] == "block"


def test_host_gate_denial_is_fail_closed_before_handler_client(monkeypatch):
    import hermes_cli.plugins as plugin_host
    from plugins.microsoft365 import register
    manager = PluginManager(); context = PluginContext(PluginManifest(name="microsoft365"), manager)
    context.get_config = lambda key, default=None: {"capabilities": {"outlook": {"send": True}}}.get(key, default)
    register(context)
    monkeypatch.setattr(plugin_host, "_plugin_manager", manager)
    monkeypatch.setattr("tools.approval.request_tool_approval", lambda *a, **k: {"approved": False, "message": "denied"})
    from hermes_cli.plugins import resolve_pre_tool_block
    assert resolve_pre_tool_block("microsoft365_outlook", {"action": "send", "to": "a@example.com", "body": "B"}) == "denied"
