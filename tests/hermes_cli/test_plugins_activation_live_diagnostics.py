"""A plugin install's MCP row never reports a failure without a diagnostic (#140543).

``_server_error`` used to hand back only a *recorded* error. A server that was still connecting, or
that was never attempted, produced ``None``, the row fell back to a generic sentence and the card's
detail read ``MCP server <name> not connected: unknown error`` — with the child's own output sitting
in ``mcp-stderr.log`` and nothing pointing at it.
"""

import pytest

from hermes_cli.plugins_activation_live import _last_stderr_line, _server_error


@pytest.fixture
def status(monkeypatch):
    """Point ``get_mcp_status`` at a fixed list of rows."""
    def _install(rows):
        monkeypatch.setattr("tools.mcp_tool_discovery.get_mcp_status", lambda *a, **k: list(rows))
    return _install


@pytest.fixture
def stderr_tail(monkeypatch):
    def _install(text):
        monkeypatch.setattr("tools.mcp_tool_config._tail_server_stderr", lambda *a, **k: text)
    return _install


def _note(name="blender"):
    error = _server_error(name)
    assert isinstance(error, str) and error, "the row must carry a diagnostic"
    return error


def test_a_recorded_error_is_reported_verbatim(status, stderr_tail):
    status([{"name": "blender", "connected": False, "status": "failed",
             "error": "missing executable 'uv'"}])
    stderr_tail("")
    assert _server_error("blender") == "missing executable 'uv'"


def test_a_still_connecting_server_names_the_log_instead_of_nothing(status, stderr_tail):
    status([{"name": "blender", "connected": False, "status": "connecting"}])
    stderr_tail("")
    error = _note()
    assert error.startswith("the server is still connecting")
    assert "mcp-stderr.log" in error


def test_a_never_attempted_server_says_so(status, stderr_tail):
    status([{"name": "blender", "connected": False, "status": "configured"}])
    stderr_tail("")
    assert _note().startswith("the server was never attempted")


def test_the_child_s_last_line_travels_with_the_note(status, stderr_tail):
    status([{"name": "blender", "connected": False, "status": "connecting"}])
    stderr_tail("===== starting MCP server 'blender' =====\nBuilding blender-mcp...\n")
    assert _note().endswith("child output ends: Building blender-mcp...)")


def test_the_tee_header_alone_is_not_a_diagnostic(status, stderr_tail):
    status([{"name": "blender", "connected": False, "status": "connecting"}])
    stderr_tail("===== starting MCP server 'blender' =====\n")
    assert _last_stderr_line("blender") == ""
    assert _note().endswith("(see mcp-stderr.log)")


def test_credentials_in_the_child_output_are_scrubbed(status, stderr_tail):
    status([{"name": "blender", "connected": False, "status": "connecting"}])
    stderr_tail("auth failed for key sk-abcdefghijklmnopqrstuvwxyz012345")
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in _note()


def test_a_server_the_status_does_not_know_still_returns_none(status, stderr_tail):
    status([{"name": "other", "connected": True, "status": "connected"}])
    stderr_tail("")
    assert _server_error("blender") is None


def test_a_status_read_that_raises_keeps_the_row_fallible(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("no registry")

    monkeypatch.setattr("tools.mcp_tool_discovery.get_mcp_status", _boom)
    assert _server_error("blender") is None
