"""gateway.run._resolve_approval_session_is_guest against the concrete Bot
API 10.0 guest-mode adapter contract (plugins/platforms/telegram/adapter.py's
``_is_guest_chat``, ``_pending_guest_queries``, ``_guest_only_chats`` — see
the linked #56476 guest-mode PR), plus a regression test for the smart-
approval bypass sweeper flagged: a guest session must deny BEFORE smart
approval or cached per-session grants get a chance to auto-approve.

Not theorized: no adapter on current main defines ``_is_guest_chat`` yet, so
tests/tools/test_approval.py's own guest-mode tests exercise the approval
short-circuit by calling ``mark_session_guest``/``unmark_session_guest``
directly. The tests below instead build a stub adapter with the REAL shape
#56476 ships (same attribute names, same ``str(chat_id)`` key normalization)
and drive it through the actual ``getattr``-based wiring extracted from
``gateway/run.py``, so the integration between that wiring and a real guest
predicate is exercised — not just the approval module's internal state.
"""
from __future__ import annotations

import os

from unittest.mock import MagicMock

from gateway.run import _resolve_approval_session_is_guest


class _StubGuestAwareAdapter:
    """Mirrors #56476's TelegramAdapter._is_guest_chat exactly:
    True while a guest turn is in flight (_pending_guest_queries) or for
    the remainder of processing after the query was consumed
    (_guest_only_chats). Keys are normalized via str(chat_id).
    """

    def __init__(self):
        self._pending_guest_queries: dict[str, object] = {}
        self._guest_only_chats: set[str] = set()

    def _is_guest_chat(self, chat_id) -> bool:
        cid_str = str(chat_id)
        return self._pending_guest_queries.get(cid_str) is not None or cid_str in self._guest_only_chats


def test_false_when_neither_pending_nor_guest_only():
    adapter = _StubGuestAwareAdapter()
    assert _resolve_approval_session_is_guest(adapter, "12345") is False


def test_true_when_chat_is_pending_guest_query():
    adapter = _StubGuestAwareAdapter()
    adapter._pending_guest_queries["12345"] = object()
    assert _resolve_approval_session_is_guest(adapter, "12345") is True


def test_true_when_chat_is_guest_only():
    adapter = _StubGuestAwareAdapter()
    adapter._guest_only_chats.add("12345")
    assert _resolve_approval_session_is_guest(adapter, "12345") is True


def test_normalizes_non_string_chat_id_like_the_real_predicate_does():
    """gateway/run.py passes chat_id through as-is (may be int); the real
    predicate's own str() conversion must still match a string-keyed entry."""
    adapter = _StubGuestAwareAdapter()
    adapter._guest_only_chats.add("12345")
    assert _resolve_approval_session_is_guest(adapter, 12345) is True


def test_false_for_adapter_without_is_guest_chat():
    """Today's reality on main: no adapter defines _is_guest_chat at all --
    must be a safe no-op, not an AttributeError."""
    adapter = MagicMock(spec=[])  # no attributes at all
    assert _resolve_approval_session_is_guest(adapter, "12345") is False


def test_false_when_is_guest_chat_is_not_callable():
    """Defensive: a plain attribute (not a method) must not be invoked."""
    adapter = MagicMock()
    adapter._is_guest_chat = "not callable"
    assert _resolve_approval_session_is_guest(adapter, "12345") is False


class TestGuestPredicateDrivesRealApprovalDenial:
    """End-to-end: a real-shaped guest adapter resolving True, marked via
    mark_session_guest (exactly as gateway/run.py's turn-dispatch wiring
    does), must cause check_all_command_guards to deny a dangerous command
    instantly and WITHOUT ever attempting to send an approval prompt -- both
    in the default manual-approval flow AND when approvals.mode=smart would
    otherwise auto-approve. The smart case is the regression this test suite
    exists for: the guest check used to run only inside
    _await_gateway_decision, which a smart-approve "approve" verdict never
    reaches.
    """

    SESSION_KEY = "test-guest-predicate-session"

    def setup_method(self):
        from tools import approval as mod
        mod._gateway_queues.clear()
        mod._gateway_notify_cbs.clear()
        mod._gateway_approval_blocked.clear()
        mod._session_approved.clear()
        mod._permanent_approved.clear()

        self._saved_env = {
            k: os.environ.get(k)
            for k in ("HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION",
                      "HERMES_YOLO_MODE", "HERMES_SESSION_KEY", "HERMES_INTERACTIVE")
        }
        os.environ.pop("HERMES_YOLO_MODE", None)
        os.environ.pop("HERMES_INTERACTIVE", None)
        os.environ.pop("HERMES_CRON_SESSION", None)
        os.environ["HERMES_GATEWAY_SESSION"] = "1"
        os.environ["HERMES_SESSION_KEY"] = self.SESSION_KEY

    def teardown_method(self):
        from tools import approval as mod
        mod._gateway_queues.clear()
        mod._gateway_notify_cbs.clear()
        mod._gateway_approval_blocked.clear()
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _mark_guest_via_real_predicate(self) -> None:
        from tools.approval import mark_session_guest

        adapter = _StubGuestAwareAdapter()
        adapter._guest_only_chats.add("999")
        assert _resolve_approval_session_is_guest(adapter, "999") is True
        mark_session_guest(self.SESSION_KEY)

    def test_manual_mode_denies_without_notify(self):
        from tools import approval as mod

        notified = []
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: notified.append(data))
        self._mark_guest_via_real_predicate()

        result = mod.check_all_command_guards("rm -rf .git", "local")

        assert result["approved"] is False
        assert result.get("outcome") == "guest_unsupported"
        assert notified == []

    def test_smart_mode_approve_verdict_still_denies_guest(self, monkeypatch):
        """The bug: smart approval's 'approve' verdict used to return
        approved=True directly, before the guest check (which only lived in
        _await_gateway_decision) ever ran. A guest session must deny even
        when the aux LLM would have auto-approved."""
        from tools import approval as mod

        monkeypatch.setattr(mod, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(mod, "_smart_approve", lambda command, description: "approve")

        notified = []
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: notified.append(data))
        self._mark_guest_via_real_predicate()

        result = mod.check_all_command_guards("rm -rf .git", "local")

        assert result["approved"] is False
        assert result.get("outcome") == "guest_unsupported"
        assert notified == [], "smart-approve must not reach notify_cb for a guest session"

    def test_cached_session_grant_still_denies_guest(self, monkeypatch):
        """The other half of the bug: a pattern already approved earlier in
        the session (is_approved) used to skip straight to approved=True,
        never reaching the guest check either."""
        from tools import approval as mod

        monkeypatch.setattr(mod, "_get_approval_mode", lambda: "manual")

        notified = []
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: notified.append(data))
        # Pre-approve the pattern this command will match, as if the user
        # had approved it earlier in a non-guest turn on the same session.
        _, pattern_key, _ = mod.detect_dangerous_command("rm -rf .git")
        mod.approve_session(self.SESSION_KEY, pattern_key)
        self._mark_guest_via_real_predicate()

        result = mod.check_all_command_guards("rm -rf .git", "local")

        assert result["approved"] is False
        assert result.get("outcome") == "guest_unsupported"
        assert notified == []

    def test_execute_code_smart_mode_approve_verdict_still_denies_guest(self, monkeypatch):
        from tools import approval as mod

        monkeypatch.setattr(mod, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(mod, "_smart_approve", lambda command, description: "approve")

        notified = []
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: notified.append(data))
        self._mark_guest_via_real_predicate()

        result = mod.check_execute_code_guard("import os; os.system('rm -rf /')", "local")

        assert result["approved"] is False
        assert result.get("outcome") == "guest_unsupported"
        assert notified == []
