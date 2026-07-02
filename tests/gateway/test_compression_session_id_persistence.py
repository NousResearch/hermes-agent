"""Regression tests for #29335 — gateway must persist compression session splits.

When ``_compress_context()`` rolls the agent forward into a new session, the
agent returns the new ``session_id`` in its result dict. The gateway must publish
that child through one durability helper, update the live routing entry, call
``session_store._save()``, and preserve gateway peer metadata so the mapping
survives restarts and background notifications.

Without the durable publish step, the next turn loads the OLD session's
transcript and re-triggers compression forever.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import MagicMock, call

from gateway import run as gateway_run
from gateway.session_context import set_current_session_id, get_session_env


def _session_entry_assignments(source: str) -> list[int]:
    """Return line numbers for direct ``session_entry.session_id = ...`` writes."""
    tree = ast.parse(textwrap.dedent(source))
    results: list[int] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "session_id"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "session_entry"
                ):
                    results.append(node.lineno)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return results


def _helper_persists_entry_assignment(source: str) -> bool:
    """Return True iff the publish helper writes ``entry.session_id`` and saves."""
    tree = ast.parse(textwrap.dedent(source))
    assigned = False
    saved = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "session_id"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "entry"
                ):
                    assigned = True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_save"
        ):
            saved = True
    return assigned and saved


def test_compression_session_id_publishing_is_centralized_and_persisted():
    """Compression route mutation must go through the durable publish helper.

    Regression for #29335 and follow-ons: direct ``session_entry.session_id``
    writes are easy to forget to save or to race with stale generations. The
    helper performs the generation guard, active-route guard, save, peer record,
    and topic binding refresh in one place.
    """
    source = inspect.getsource(gateway_run)
    helper_source = inspect.getsource(
        gateway_run.GatewayRunner._publish_compression_session_split
    )

    assert _helper_persists_entry_assignment(helper_source), (
        "_publish_compression_session_split() must assign the live entry's "
        "session_id and call _save() so compression children survive restarts."
    )

    direct_assignments = _session_entry_assignments(source)
    assert not direct_assignments, (
        "Compression session id publication must be centralized in "
        "_publish_compression_session_split(); direct session_entry.session_id "
        f"assignments found at gateway/run.py AST lines {direct_assignments}."
    )


class TestCompressionSessionPropagation:
    """Behavioral tests for durable compression session publication."""

    def _runner_with_entry(self, session_key: str, session_entry: MagicMock):
        runner = object.__new__(gateway_run.GatewayRunner)
        session_store = MagicMock()
        session_store._entries = {session_key: session_entry}
        session_store._ensure_loaded = MagicMock()
        session_store._save = MagicMock()
        session_store._record_gateway_session_peer = MagicMock()
        runner.session_store = session_store
        runner._is_session_run_current = MagicMock(return_value=True)
        runner._sync_telegram_topic_binding = MagicMock()
        return runner, session_store

    def test_gateway_session_entry_follows_compression_rotation(self) -> None:
        """The publish helper must update the live entry and persist it."""
        old_sid = "20260101_000000_aaaaaa"
        new_sid = "20260101_000001_bbbbbb"
        session_key = "agent:main:telegram:dm:123"

        session_entry = MagicMock()
        session_entry.session_id = old_sid
        session_entry.origin = None
        runner, session_store = self._runner_with_entry(session_key, session_entry)

        published = runner._publish_compression_session_split(
            session_key=session_key,
            source=None,
            previous_session_id=old_sid,
            new_session_id=new_sid,
            reason="test",
            run_generation=1,
        )

        assert published is session_entry
        assert session_entry.session_id == new_sid, (
            "session_entry.session_id was not updated to the compressed session id. "
            "The next turn would load the old transcript and re-trigger compression."
        )
        session_store._save.assert_called_once_with(), (
            "session_store._save() was not called after session_entry update. "
            "The new session mapping would not survive a gateway restart."
        )

    def test_no_update_when_session_id_unchanged(self) -> None:
        """Publishing the same session id is a no-op."""
        same_sid = "20260101_000000_aaaaaa"
        session_key = "agent:main:telegram:dm:123"

        session_entry = MagicMock()
        session_entry.session_id = same_sid
        runner, session_store = self._runner_with_entry(session_key, session_entry)

        published = runner._publish_compression_session_split(
            session_key=session_key,
            source=None,
            previous_session_id=same_sid,
            new_session_id=same_sid,
            reason="test",
            run_generation=1,
        )

        assert published is None
        assert session_entry.session_id == same_sid
        session_store._save.assert_not_called()

    def test_contextvar_and_session_entry_agree_after_compression(self) -> None:
        """After compression, the contextvar and live entry should agree."""
        old_sid = "20260101_000000_cccccc"
        new_sid = "20260101_000002_dddddd"
        session_key = "agent:main:telegram:dm:123"

        set_current_session_id(new_sid)
        session_entry = MagicMock()
        session_entry.session_id = old_sid
        session_entry.origin = None
        runner, _session_store = self._runner_with_entry(session_key, session_entry)

        runner._publish_compression_session_split(
            session_key=session_key,
            source=None,
            previous_session_id=old_sid,
            new_session_id=new_sid,
            reason="test",
            run_generation=1,
        )

        contextvar_sid = get_session_env("HERMES_SESSION_ID", "")
        assert contextvar_sid == new_sid, (
            f"Contextvar still holds old session_id '{contextvar_sid}' after "
            f"set_current_session_id('{new_sid}'). Tool calls in the next turn "
            "will read stale routing state."
        )
        assert session_entry.session_id == new_sid, (
            f"session_entry.session_id is '{session_entry.session_id}' but contextvar "
            f"says '{contextvar_sid}'. The two routing paths disagree after compression."
        )
        assert contextvar_sid == session_entry.session_id, (
            "Contextvar and session_entry disagree on the active session_id "
            "after compression rotation. Exactly one of the two ordering steps "
            "was skipped."
        )
