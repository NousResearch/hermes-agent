from gateway.platforms.api_server import _approval_event_choices
from gateway.run import (
    _build_exec_approval_kwargs,
    _format_exec_approval_reply_options,
)


class _LegacyApprovalAdapter:
    async def send_exec_approval(
        self, chat_id, command, session_key, description="dangerous command", metadata=None,
    ):
        raise AssertionError("not called")


class _PermanentAwareApprovalAdapter:
    async def send_exec_approval(
        self,
        chat_id,
        command,
        session_key,
        description="dangerous command",
        metadata=None,
        allow_permanent=True,
    ):
        raise AssertionError("not called")


def _approval_kwargs(adapter, allow_permanent):
    return _build_exec_approval_kwargs(
        adapter,
        chat_id="chat-1",
        command="rm -rf /tmp/example",
        session_key="session-1",
        description="dangerous command",
        metadata={"thread_id": "thread-1"},
        allow_permanent=allow_permanent,
    )


def test_dispatcher_forwards_allow_permanent_to_aware_adapters():
    kwargs = _approval_kwargs(_PermanentAwareApprovalAdapter(), False)

    assert kwargs["allow_permanent"] is False


def test_dispatcher_preserves_legacy_adapter_contract():
    kwargs = _approval_kwargs(_LegacyApprovalAdapter(), False)

    assert "allow_permanent" not in kwargs


def test_api_choices_hide_permanent_approval_when_disallowed():
    assert _approval_event_choices(False) == ["once", "session", "deny"]
    assert _approval_event_choices(True) == ["once", "session", "always", "deny"]


def test_text_fallback_hides_permanent_approval_when_disallowed():
    restricted = _format_exec_approval_reply_options("!", False)
    unrestricted = _format_exec_approval_reply_options("!", True)

    assert "!approve always" not in restricted
    assert "!approve always" in unrestricted
    assert "!approve session" in restricted
    assert "!deny" in restricted
