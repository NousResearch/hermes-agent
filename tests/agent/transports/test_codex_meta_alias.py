"""Meta-OAuth attested tool-name aliasing in the Responses transport."""

import pytest


@pytest.fixture
def transport():
    from agent.transports.codex import ResponsesApiTransport
    return ResponsesApiTransport()


_TOOLS = [
    {"type": "function", "function": {
        "name": "search_files", "description": "Search files.",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}}}}},
]


def _names(kw):
    return [t.get("name") for t in kw.get("tools", []) if t.get("type") == "function"]


def test_meta_oauth_aliases_attested_names(transport):
    """Meta subscription attestation 403s read_file/write_file/web_search on the wire."""
    tools = list(_TOOLS) + [
        {"type": "function", "function": {
            "name": "write_file", "description": "Write a file.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}}}},
    ]
    kw = transport.build_kwargs(
        model="muse-spark-1.3",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        provider="meta-oauth",
        base_url="https://api.meta.ai/v1",
    )
    names = _names(kw)
    assert "hermes_read_file" in names
    assert "hermes_write_file" in names
    assert "hermes_web_search" in names
    assert "read_file" not in names
    assert "write_file" not in names
    assert "web_search" not in names
    assert "search_files" in names  # not attested; untouched
    assert transport._last_wire_aliases == {
        "hermes_read_file": "read_file",
        "hermes_write_file": "write_file",
        "hermes_web_search": "web_search",
    }

def test_meta_ai_api_key_keeps_original_names(transport):
    """The pay-as-you-go meta-ai provider is not subscription-attested."""
    kw = transport.build_kwargs(
        model="muse-spark-1.3",
        messages=[{"role": "user", "content": "hi"}],
        tools=list(_TOOLS),
        provider="meta-ai",
        base_url="https://api.meta.ai/v1",
    )
    names = _names(kw)
    assert "read_file" in names
    assert "web_search" in names
    assert "hermes_read_file" not in names
