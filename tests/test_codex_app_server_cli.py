import importlib
import sys
import types
from contextlib import nullcontext
from unittest.mock import MagicMock


def _install_prompt_toolkit_stubs():
    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    class _Condition:
        def __init__(self, func):
            self.func = func

        def __bool__(self):
            return bool(self.func())

    class _ANSI(str):
        pass

    root = types.ModuleType("prompt_toolkit")
    history = types.ModuleType("prompt_toolkit.history")
    styles = types.ModuleType("prompt_toolkit.styles")
    patch_stdout = types.ModuleType("prompt_toolkit.patch_stdout")
    application = types.ModuleType("prompt_toolkit.application")
    layout = types.ModuleType("prompt_toolkit.layout")
    processors = types.ModuleType("prompt_toolkit.layout.processors")
    filters = types.ModuleType("prompt_toolkit.filters")
    dimension = types.ModuleType("prompt_toolkit.layout.dimension")
    menus = types.ModuleType("prompt_toolkit.layout.menus")
    widgets = types.ModuleType("prompt_toolkit.widgets")
    key_binding = types.ModuleType("prompt_toolkit.key_binding")
    completion = types.ModuleType("prompt_toolkit.completion")
    formatted_text = types.ModuleType("prompt_toolkit.formatted_text")

    history.FileHistory = _Dummy
    styles.Style = _Dummy
    patch_stdout.patch_stdout = lambda *args, **kwargs: nullcontext()
    application.Application = _Dummy
    layout.Layout = _Dummy
    layout.HSplit = _Dummy
    layout.Window = _Dummy
    layout.FormattedTextControl = _Dummy
    layout.ConditionalContainer = _Dummy
    processors.Processor = _Dummy
    processors.Transformation = _Dummy
    processors.PasswordProcessor = _Dummy
    processors.ConditionalProcessor = _Dummy
    filters.Condition = _Condition
    dimension.Dimension = _Dummy
    menus.CompletionsMenu = _Dummy
    widgets.TextArea = _Dummy
    key_binding.KeyBindings = _Dummy
    completion.Completer = _Dummy
    completion.Completion = _Dummy
    formatted_text.ANSI = _ANSI
    root.print_formatted_text = lambda *args, **kwargs: None

    sys.modules.setdefault("prompt_toolkit", root)
    sys.modules.setdefault("prompt_toolkit.history", history)
    sys.modules.setdefault("prompt_toolkit.styles", styles)
    sys.modules.setdefault("prompt_toolkit.patch_stdout", patch_stdout)
    sys.modules.setdefault("prompt_toolkit.application", application)
    sys.modules.setdefault("prompt_toolkit.layout", layout)
    sys.modules.setdefault("prompt_toolkit.layout.processors", processors)
    sys.modules.setdefault("prompt_toolkit.filters", filters)
    sys.modules.setdefault("prompt_toolkit.layout.dimension", dimension)
    sys.modules.setdefault("prompt_toolkit.layout.menus", menus)
    sys.modules.setdefault("prompt_toolkit.widgets", widgets)
    sys.modules.setdefault("prompt_toolkit.key_binding", key_binding)
    sys.modules.setdefault("prompt_toolkit.completion", completion)
    sys.modules.setdefault("prompt_toolkit.formatted_text", formatted_text)


def _import_cli():
    # Avoid importing heavyweight tool stacks (web_tools/firecrawl/etc.) in unit tests.
    if "run_agent" not in sys.modules:
        run_agent_stub = types.ModuleType("run_agent")

        class _DummyAgent:
            def __init__(self, *args, **kwargs):
                pass

        run_agent_stub.AIAgent = _DummyAgent
        sys.modules["run_agent"] = run_agent_stub

    if "model_tools" not in sys.modules:
        model_tools_stub = types.ModuleType("model_tools")
        model_tools_stub.get_tool_definitions = lambda *a, **k: []
        model_tools_stub.get_toolset_for_tool = lambda *a, **k: "other"
        model_tools_stub.check_tool_availability = lambda *a, **k: ([], [])
        model_tools_stub.TOOLSET_REQUIREMENTS = {}
        sys.modules["model_tools"] = model_tools_stub

    if "tools" not in sys.modules:
        tools_pkg = types.ModuleType("tools")
        tools_pkg.__path__ = []  # mark as package
        sys.modules["tools"] = tools_pkg

    if "tools.terminal_tool" not in sys.modules:
        terminal_stub = types.ModuleType("tools.terminal_tool")
        terminal_stub.cleanup_all_environments = lambda: None
        terminal_stub.set_sudo_password_callback = lambda cb: None
        terminal_stub.set_approval_callback = lambda cb: None
        sys.modules["tools.terminal_tool"] = terminal_stub

    if "tools.browser_tool" not in sys.modules:
        browser_stub = types.ModuleType("tools.browser_tool")
        browser_stub._emergency_cleanup_all_sessions = lambda: None
        sys.modules["tools.browser_tool"] = browser_stub

    try:
        importlib.import_module("prompt_toolkit")
    except ModuleNotFoundError:
        _install_prompt_toolkit_stubs()
    if "cli" in sys.modules:
        del sys.modules["cli"]
    return importlib.import_module("cli")


class _BaseFakeClient:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.closed = False

    def start(self):
        return None

    def initialize(self, **kwargs):
        self.init_kwargs = kwargs
        return None

    def notify(self, method, params=None):
        self.calls.append(("notify", method, params))

    def close(self):
        self.closed = True


def test_codex_app_server_init_uses_existing_chatgpt_auth(monkeypatch):
    cli = _import_cli()
    fake_client_ref = {}

    class FakeClient(_BaseFakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            fake_client_ref["client"] = self

        def call(self, method, params=None, timeout=30.0):
            self.calls.append(("call", method, params))
            if method == "account/read":
                return {
                    "account": {"type": "chatgpt", "email": "user@example.com"},
                    "requiresOpenaiAuth": True,
                }
            if method == "thread/start":
                return {"thread": {"id": "thr_existing_auth"}}
            raise AssertionError(f"Unexpected call: {method}")

    monkeypatch.setattr("hermes_cli.codex_app_server.CodexAppServerClient", FakeClient)
    monkeypatch.setattr("hermes_cli.codex_app_server.wait_for_notification", lambda *a, **k: {})
    monkeypatch.setattr("cli._cprint", lambda *a, **k: None)

    shell = cli.HermesCLI(model="gpt-5.3-codex", provider="codex-app-server", compact=True, max_turns=1)
    assert shell._init_codex_app_server() is True
    assert shell._codex_thread_id == "thr_existing_auth"

    called_methods = [item[1] for item in fake_client_ref["client"].calls if item[0] == "call"]
    assert "account/read" in called_methods
    assert "thread/start" in called_methods
    assert "account/login/start" not in called_methods


def test_codex_app_server_init_runs_oauth_when_needed(monkeypatch):
    cli = _import_cli()
    fake_client_ref = {}
    wait_calls = {"count": 0}
    opened_urls = []

    class FakeClient(_BaseFakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            fake_client_ref["client"] = self

        def call(self, method, params=None, timeout=30.0):
            self.calls.append(("call", method, params))
            if method == "account/read":
                return {"account": None, "requiresOpenaiAuth": True}
            if method == "account/login/start":
                return {"authUrl": "https://chatgpt.com/login-test", "loginId": "login_123"}
            if method == "thread/start":
                return {"thread": {"id": "thr_after_login"}}
            raise AssertionError(f"Unexpected call: {method}")

    def _wait_for_notification(client, *, method, timeout, login_id=None):
        wait_calls["count"] += 1
        assert method == "account/login/completed"
        assert login_id == "login_123"
        return {"loginId": "login_123", "success": True}

    monkeypatch.setattr("hermes_cli.codex_app_server.CodexAppServerClient", FakeClient)
    monkeypatch.setattr("hermes_cli.codex_app_server.wait_for_notification", _wait_for_notification)
    monkeypatch.setattr("cli._cprint", lambda *a, **k: None)
    monkeypatch.setattr("cli.webbrowser.open", lambda url: opened_urls.append(url))

    shell = cli.HermesCLI(model="gpt-5.3-codex", provider="codex-app-server", compact=True, max_turns=1)
    assert shell._init_codex_app_server() is True
    assert shell._codex_thread_id == "thr_after_login"
    assert wait_calls["count"] == 1
    assert opened_urls == ["https://chatgpt.com/login-test"]

    called_methods = [item[1] for item in fake_client_ref["client"].calls if item[0] == "call"]
    assert called_methods[:2] == ["account/read", "account/login/start"]


def test_codex_turn_lifecycle_parses_final_text(monkeypatch):
    cli = _import_cli()
    monkeypatch.setattr("cli._cprint", lambda *a, **k: None)

    class FakeTurnClient:
        def __init__(self):
            self.notifications = [
                {"method": "item/agentMessage/delta", "params": {"delta": "Hel"}},
                {"method": "item/agentMessage/delta", "params": {"textDelta": "lo"}},
                {"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "Hello final"}}},
                {"method": "turn/completed", "params": {"turn": {"id": "turn_1", "status": "completed"}}},
            ]

        def call(self, method, params=None, timeout=30.0):
            if method == "turn/start":
                return {"turn": {"id": "turn_1"}}
            raise AssertionError(f"Unexpected call: {method}")

        def next_notification(self, timeout=0.2):
            if self.notifications:
                return self.notifications.pop(0)
            return None

    shell = cli.HermesCLI(model="gpt-5.3-codex", provider="codex-app-server", compact=True, max_turns=1)
    shell._codex_app_client = FakeTurnClient()
    shell._codex_thread_id = "thr_1"

    response = shell._chat_via_codex_app_server("hello?")
    assert response == "Hello final"


def test_codex_turn_start_includes_mapped_effort(monkeypatch):
    cli = _import_cli()
    monkeypatch.setattr("cli._cprint", lambda *a, **k: None)

    class FakeTurnClient:
        def __init__(self):
            self.turn_start_params = None
            self.notifications = [
                {"method": "turn/completed", "params": {"turn": {"id": "turn_effort", "status": "completed"}}},
            ]

        def call(self, method, params=None, timeout=30.0):
            if method == "turn/start":
                self.turn_start_params = params
                return {"turn": {"id": "turn_effort"}}
            raise AssertionError(f"Unexpected call: {method}")

        def next_notification(self, timeout=0.2):
            if self.notifications:
                return self.notifications.pop(0)
            return None

    shell = cli.HermesCLI(model="gpt-5.3-codex", provider="codex-app-server", compact=True, max_turns=1)
    shell.reasoning_config = {"enabled": True, "effort": "xhigh"}
    shell._codex_app_client = FakeTurnClient()
    shell._codex_thread_id = "thr_effort"

    shell._chat_via_codex_app_server("test")
    assert shell._codex_app_client.turn_start_params.get("effort") == "high"


def test_codex_app_server_process_failure_and_client_rotation(monkeypatch):
    cli = _import_cli()

    class FailingClient(_BaseFakeClient):
        def start(self):
            raise FileNotFoundError("codex not found")

    monkeypatch.setattr("hermes_cli.codex_app_server.CodexAppServerClient", FailingClient)
    monkeypatch.setattr("hermes_cli.codex_app_server.wait_for_notification", lambda *a, **k: {})
    monkeypatch.setattr("cli._cprint", lambda *a, **k: None)

    shell = cli.HermesCLI(model="gpt-5.3-codex", provider="codex-app-server", compact=True, max_turns=1)
    assert shell._init_codex_app_server() is False

    old_client = MagicMock()
    shell._codex_app_client = old_client
    shell.provider = "codex-app-server"
    shell.api_mode = "codex_app_server"
    shell.base_url = "stdio://codex-app-server"
    shell.api_key = ""

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "source": "env/config",
        },
    )
    monkeypatch.setattr("hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc))

    assert shell._ensure_runtime_credentials() is True
    old_client.close.assert_called_once()
    assert shell._codex_app_client is None
    assert shell.provider == "openrouter"
