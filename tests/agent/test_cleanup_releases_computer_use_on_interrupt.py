"""Regression tests for #81220 — interrupted turns release computer use.

An interrupted turn used to leave the session-owned computer-use backend
(and any mapped cua-driver cursor overlay) alive until agent ``close()`` or
process exit. On Linux/X11 the overlay is a fullscreen InputOutput
override-redirect window, so a wedged turn left the desktop unusable until
a manual ``set_agent_cursor_enabled(false)`` — the reporter's exact
symptom. The fix: ``cleanup_task_resources`` releases the computer-use
backend **only when the turn was interrupted**; normal completion keeps
the cached one-backend-per-session behavior (no respawn cost
mid-conversation).

Keying contract under test: backends are cached under the tool
``session_id`` (``agent.session_id`` — see ``agent/tool_executor.py``
dispatch kwargs), NOT the per-turn ``task_id``. The release must use the
same key or it pops a cache entry that never exists.
"""

from unittest.mock import patch

import agent.chat_completion_helpers as helpers
from agent.chat_completion_helpers import cleanup_task_resources


class _StubBudget:
    used = 0
    max_total = 100
    remaining = 100


class _StubAgent:
    """Minimal agent surface that ``cleanup_task_resources`` reads from."""

    def __init__(self, session_id: str | None = "sess-42"):
        self.session_id = session_id
        self.verbose_logging = False
        self.cleaned_vm = []
        self.cleaned_browser = []
        self._interrupt_requested = False

    def cleanup_vm(self, task_id):
        self.cleaned_vm.append(task_id)

    def cleanup_browser(self, task_id):
        self.cleaned_browser.append(task_id)


def _cleanup(monkeypatch, agent, *, interrupted, task_id="turn-uuid-1"):
    monkeypatch.setattr(helpers, "is_persistent_env", lambda tid: False)
    monkeypatch.setattr(helpers, "_ra", lambda: agent)
    agent._interrupt_requested = interrupted
    cleanup_task_resources(agent, task_id)


class TestInterruptedTurnReleasesComputerUse:
    def test_interrupted_turn_releases_backend(self, monkeypatch):
        """An interrupted turn must end the cua-driver session so the
        overlay cannot outlive the turn (#81220). Patched at the real
        seam — the ``tools.computer_use`` free function the production
        code imports at call time — and asserting the SESSION key, not
        the per-turn task id."""
        agent = _StubAgent(session_id="sess-42")
        with patch(
            "tools.computer_use.release_computer_use_session"
        ) as mock_release:
            _cleanup(monkeypatch, agent, interrupted=True, task_id="turn-uuid-1")
        mock_release.assert_called_once_with("sess-42")

    def test_interrupted_release_pops_real_backend_cache(self, monkeypatch):
        """Cross-seam proof against the real cache: a backend registered
        under the agent's session_id is gone after an interrupted turn,
        and a backend under any other key survives untouched."""
        import tools.computer_use.tool as cu_tool

        class _FakeBackend:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        agent = _StubAgent(session_id="sess-42")
        mine = _FakeBackend()
        other = _FakeBackend()
        cu_tool._backends["sess-42"] = mine
        cu_tool._backends["sess-99"] = other
        try:
            _cleanup(monkeypatch, agent, interrupted=True, task_id="turn-uuid-1")
            assert "sess-42" not in cu_tool._backends, (
                "interrupted turn must pop the session-keyed backend"
            )
            assert cu_tool._backends.get("sess-99") is other, (
                "other sessions' backends must be untouched"
            )
            assert mine.stopped and not other.stopped, (
                "exactly the session backend was stopped"
            )
        finally:
            cu_tool._backends.pop("sess-99", None)
            cu_tool._backends.pop("sess-42", None)

    def test_normal_completion_keeps_cached_backend(self, monkeypatch):
        """Normal completion must NOT release: the one-backend-per-session
        cache is load-bearing (no respawn, no re-handshake mid-chat)."""
        agent = _StubAgent(session_id="sess-42")
        with patch(
            "tools.computer_use.release_computer_use_session"
        ) as mock_release:
            _cleanup(monkeypatch, agent, interrupted=False)
        mock_release.assert_not_called()

    def test_interrupted_turn_without_session_id_skips_release(self, monkeypatch):
        """An incomplete agent stub must not target the empty cache key."""
        agent = _StubAgent(session_id=None)
        with patch(
            "tools.computer_use.release_computer_use_session"
        ) as mock_release:
            _cleanup(monkeypatch, agent, interrupted=True)
        mock_release.assert_not_called()

    def test_release_failure_never_raises(self, monkeypatch):
        """Teardown is best-effort: a dead driver socket must not break the
        rest of the cleanup chain (same contract as VM/browser steps)."""
        agent = _StubAgent(session_id="sess-42")
        with patch(
            "tools.computer_use.release_computer_use_session",
            side_effect=RuntimeError("driver socket gone"),
        ):
            _cleanup(monkeypatch, agent, interrupted=True)
        assert agent.cleaned_vm == ["turn-uuid-1"]  # the other steps still ran
