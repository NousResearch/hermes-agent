"""Per-session / per-process ACP toolset scoping (issue #45955).

Covers the resolution chain that decides a session's ``enabled_toolsets``:
per-session override > process default (``--toolsets``) > historical
``["hermes-acp"]``.
"""

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import (
    SessionManager,
    _expand_acp_enabled_toolsets,
    _normalize_acp_toolsets,
)


def test_normalize_handles_list_string_and_none():
    assert _normalize_acp_toolsets(None) is None
    assert _normalize_acp_toolsets(["web", "file"]) == ["web", "file"]
    assert _normalize_acp_toolsets("web,file") == ["web", "file"]
    assert _normalize_acp_toolsets("web, file") == ["web", "file"]
    assert _normalize_acp_toolsets("web file") == ["web", "file"]
    # Empty / whitespace-only input falls back to None so the default applies.
    assert _normalize_acp_toolsets("") is None
    assert _normalize_acp_toolsets([" ", ""]) is None


def test_resolved_toolsets_precedence():
    # No default + no override → historical hermes-acp toolset.
    assert SessionManager()._resolved_toolsets(None) == ["hermes-acp"]

    # Process default is normalized and honored.
    mgr = SessionManager(default_toolsets="web,file")
    assert mgr._default_toolsets == ["web", "file"]
    assert mgr._resolved_toolsets(None) == ["web", "file"]

    # Per-session override wins over the process default.
    assert mgr._resolved_toolsets(["memory"]) == ["memory"]


def test_resolved_toolsets_feed_expand_for_enabled_toolsets():
    # The resolved list is exactly what the agent's enabled_toolsets is built
    # from, so a scoped session sees only the requested toolsets (+ MCP).
    mgr = SessionManager(default_toolsets=["web", "file"])
    enabled = _expand_acp_enabled_toolsets(
        mgr._resolved_toolsets(None), mcp_server_names=["github"]
    )
    assert enabled == ["web", "file", "mcp-github"]


def test_new_session_param_extraction():
    extract = HermesACPAgent._toolsets_from_new_session_params
    assert extract({"toolsets": ["web"]}) == ["web"]
    assert extract({"toolsets": "web, file"}) == ["web", "file"]
    assert extract({"_meta": {"toolsets": ["memory"]}}) == ["memory"]
    # Top-level wins over _meta when both are present.
    assert extract({"toolsets": ["web"], "_meta": {"toolsets": ["memory"]}}) == ["web"]
    assert extract({}) is None
