"""Regression test for #44640 — session.resume must resolve the compression chain.

When context compression forks a new child session, the TUI gateway's
``session.resume`` handler must call ``resolve_resume_session_id()`` so
that messages are loaded from the descendant session, not the stale parent.

``web_server.py`` and ``cli_commands_mixin.py`` already do this; this test
pins the behavior for ``tui_gateway/server.py``.
"""

import ast
import inspect
import textwrap


def _get_session_resume_source():
    """Return the source of the session.resume handler from tui_gateway/server.py."""
    import tui_gateway.server as srv

    # The handler is registered via @method("session.resume"). Walk module-level
    # AST to find the decorated function.
    source = inspect.getsource(srv)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "method"
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                    and dec.args[0].value == "session.resume"
                ):
                    return ast.get_source_segment(source, node)
    raise AssertionError("session.resume handler not found in tui_gateway/server.py")


class TestTuiResumeResolvesCompressionChain:
    """Ensure the TUI gateway session.resume handler resolves compressed sessions."""

    def test_session_resume_calls_resolve_resume_session_id(self):
        """The handler must call resolve_resume_session_id before reopen_session."""
        handler_src = _get_session_resume_source()

        assert "resolve_resume_session_id" in handler_src, (
            "session.resume handler in tui_gateway/server.py must call "
            "resolve_resume_session_id() to follow the compression chain. "
            "See #44640 and the equivalent calls in web_server.py / cli_commands_mixin.py."
        )

    def test_resolve_before_reopen(self):
        """resolve_resume_session_id must appear before reopen_session."""
        handler_src = _get_session_resume_source()

        resolve_pos = handler_src.find("resolve_resume_session_id")
        reopen_pos = handler_src.find("reopen_session")

        assert resolve_pos != -1, (
            "resolve_resume_session_id() not found in session.resume handler"
        )
        assert reopen_pos != -1, (
            "reopen_session() not found in session.resume handler"
        )
        assert resolve_pos < reopen_pos, (
            "resolve_resume_session_id() must be called BEFORE reopen_session() "
            "so the resolved target is used for loading messages."
        )

    def test_resolve_guards_against_none(self):
        """The resolve call should guard against None return."""
        handler_src = _get_session_resume_source()

        # The guard should check that resolved is truthy before using it
        assert "resolved and" in handler_src or "if resolved" in handler_src, (
            "resolve_resume_session_id() result must be guarded against None "
            "to handle deleted sessions / DB inconsistency."
        )
