from collections import OrderedDict
from types import ModuleType, SimpleNamespace
import sys

import model_tools
import toolsets


if "acp" not in sys.modules:
    fake_acp = ModuleType("acp")
    fake_acp.Agent = type("Agent", (), {})
    fake_acp.Client = type("Client", (), {})
    sys.modules["acp"] = fake_acp

if "acp.schema" not in sys.modules:
    fake_schema = ModuleType("acp.schema")

    def _schema_getattr(name: str):
        value = type(name, (), {})
        setattr(fake_schema, name, value)
        return value

    fake_schema.__getattr__ = _schema_getattr
    for name in [
        "AgentCapabilities",
        "AuthenticateResponse",
        "AvailableCommand",
        "AvailableCommandsUpdate",
        "ClientCapabilities",
        "EmbeddedResourceContentBlock",
        "ForkSessionResponse",
        "ImageContentBlock",
        "AudioContentBlock",
        "Implementation",
        "InitializeResponse",
        "ListSessionsResponse",
        "LoadSessionResponse",
        "McpServerHttp",
        "McpServerSse",
        "McpServerStdio",
        "ModelInfo",
        "NewSessionResponse",
        "PromptResponse",
        "ResumeSessionResponse",
        "SetSessionConfigOptionResponse",
        "SetSessionModelResponse",
        "SetSessionModeResponse",
        "ResourceContentBlock",
        "SessionCapabilities",
        "SessionForkCapabilities",
        "SessionListCapabilities",
        "SessionModelState",
        "SessionResumeCapabilities",
        "SessionInfo",
        "TextContentBlock",
        "UnstructuredCommandInput",
        "Usage",
        "AuthMethodAgent",
        "AuthMethod",
    ]:
        setattr(fake_schema, name, type(name, (), {}))
    sys.modules["acp.schema"] = fake_schema

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionState


def _tool_def(name: str, description: str = "") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Description for {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_state(enabled_toolsets: list[str]) -> SessionState:
    return SessionState(
        session_id="test-session",
        agent=SimpleNamespace(enabled_toolsets=enabled_toolsets),
    )


def _install_toolset_metadata(monkeypatch, all_toolsets: OrderedDict, contracts: dict[str, dict]) -> None:
    monkeypatch.setattr(toolsets, "get_all_toolsets", lambda: all_toolsets)
    monkeypatch.setattr(toolsets, "get_toolset_contract", lambda name: contracts.get(name))


def test_cmd_tools_surfaces_canonical_knowledge_first_when_visible(monkeypatch):
    visible_tools = [
        _tool_def("read_file"),
        _tool_def("search_files"),
        _tool_def("ast_list_defs"),
        _tool_def("ast_find_nodes"),
        _tool_def("lsp_document_symbols"),
        _tool_def("lsp_definition"),
        _tool_def("lsp_diagnostics"),
        _tool_def("web_search"),
        _tool_def("web_extract"),
        _tool_def("browser_navigate"),
        _tool_def("browser_snapshot"),
        _tool_def("browser_click"),
        _tool_def("browser_scroll"),
        _tool_def("browser_back"),
        _tool_def("browser_press"),
        _tool_def("browser_get_images"),
        _tool_def("browser_vision"),
        _tool_def("browser_console"),
        _tool_def("vision_analyze"),
    ]
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: visible_tools)

    all_toolsets = OrderedDict(
        [
            (
                "repo-code-knowledge",
                {"description": "Canonical repo/code surface"},
            ),
            (
                "web-research-knowledge",
                {"description": "Canonical web/research surface"},
            ),
            (
                "document-pdf-diagram-intelligence",
                {"description": "Canonical document surface"},
            ),
        ]
    )
    contracts = {
        "repo-code-knowledge": {
            "name": "repo-code-knowledge",
            "canonical_name": "repo-code-knowledge",
            "source": "builtin",
            "is_builtin": True,
            "is_additive": False,
            "is_alias": False,
            "is_canonical_knowledge_surface": True,
            "tools": [
                "ast_find_nodes",
                "ast_list_defs",
                "lsp_definition",
                "lsp_diagnostics",
                "lsp_document_symbols",
                "read_file",
                "search_files",
            ],
        },
        "web-research-knowledge": {
            "name": "web-research-knowledge",
            "canonical_name": "web-research-knowledge",
            "source": "builtin",
            "is_builtin": True,
            "is_additive": False,
            "is_alias": False,
            "is_canonical_knowledge_surface": True,
            "tools": ["web_extract", "web_search"],
        },
        "document-pdf-diagram-intelligence": {
            "name": "document-pdf-diagram-intelligence",
            "canonical_name": "document-pdf-diagram-intelligence",
            "source": "builtin",
            "is_builtin": True,
            "is_additive": False,
            "is_alias": False,
            "is_canonical_knowledge_surface": True,
            "tools": [
                "browser_back",
                "browser_click",
                "browser_console",
                "browser_get_images",
                "browser_navigate",
                "browser_press",
                "browser_scroll",
                "browser_snapshot",
                "browser_vision",
                "vision_analyze",
            ],
        },
    }
    _install_toolset_metadata(monkeypatch, all_toolsets, contracts)

    output = HermesACPAgent(session_manager=SimpleNamespace())._cmd_tools(
        "",
        _make_state(["hermes-acp"]),
    )

    assert "Canonical built-in knowledge surfaces visible from current tool membership:" in output
    assert "repo-code-knowledge [built-in/canonical; 7 raw tools]" in output
    assert "web-research-knowledge [built-in/canonical; 2 raw tools]" in output
    assert "document-pdf-diagram-intelligence [built-in/canonical; 10 raw tools]" in output
    assert output.index("Canonical built-in knowledge surfaces visible from current tool membership:") < output.index("Raw tools (19):")


def test_cmd_tools_keeps_raw_tool_names_visible(monkeypatch):
    visible_tools = [
        _tool_def("read_file", "Read files from disk"),
        _tool_def("search_files", "Search across the repo"),
    ]
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: visible_tools)
    _install_toolset_metadata(monkeypatch, OrderedDict(), {})

    output = HermesACPAgent(session_manager=SimpleNamespace())._cmd_tools(
        "",
        _make_state(["hermes-acp"]),
    )

    assert "Raw tools (2):" in output
    assert "  read_file: Read files from disk" in output
    assert "  search_files: Search across the repo" in output


def test_cmd_tools_labels_additive_surfaces_without_calling_them_builtin(monkeypatch):
    visible_tools = [
        _tool_def("read_file"),
        _tool_def("search_files"),
        _tool_def("mcp_surface_lookup"),
    ]
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: visible_tools)

    all_toolsets = OrderedDict(
        [
            (
                "repo-code-knowledge",
                {"description": "Canonical repo/code surface"},
            ),
            (
                "test-surface",
                {"description": "Additive lookup surface"},
            ),
        ]
    )
    contracts = {
        "repo-code-knowledge": {
            "name": "repo-code-knowledge",
            "canonical_name": "repo-code-knowledge",
            "source": "builtin",
            "is_builtin": True,
            "is_additive": False,
            "is_alias": False,
            "is_canonical_knowledge_surface": True,
            "tools": ["read_file", "search_files"],
        },
        "test-surface": {
            "name": "test-surface",
            "canonical_name": "mcp-test-surface",
            "source": "mcp",
            "is_builtin": False,
            "is_additive": True,
            "is_alias": True,
            "tools": ["mcp_surface_lookup"],
        },
    }
    _install_toolset_metadata(monkeypatch, all_toolsets, contracts)

    output = HermesACPAgent(session_manager=SimpleNamespace())._cmd_tools(
        "",
        _make_state(["hermes-acp", "test-surface"]),
    )

    additive_line = next(
        line for line in output.splitlines() if "mcp-test-surface" in line
    )
    assert "Additive tool surfaces visible from current tool membership:" in output
    assert "additive/mcp" in additive_line
    assert "built-in" not in additive_line
    assert "canonical" not in additive_line
