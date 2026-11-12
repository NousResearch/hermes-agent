"""Regression for #145473: a Matrix peer bot must not approve its own command.

Use the real pending-approval queue and Matrix reaction handler; only the
mautrix transport is replaced by the existing test fixture.
"""

from unittest.mock import AsyncMock

import hermes_yaml as yaml
import pytest

from gateway.config import PlatformConfig
from gateway.approval_actor import answerable_request_ids
from hermes_constants import get_hermes_home
from tests.gateway.test_approval_requester_binding import pending_from, write_approvers
from tests.gateway.test_matrix_approval_reaction_fail_closed import _make_adapter, _MatrixApprovalPrompt

PEER, HUMAN, GUEST = "@peer:hs", "@human:hs", "@guest:hs"


@pytest.fixture(autouse=True)
def _clean_approvals():
    from tools import approval

    approval._gateway_queues.clear()
    yield
    approval._gateway_queues.clear()


def _configure_independent_approvers(approvers=None):
    write_approvers([HUMAN] if approvers is None else approvers)
    path = get_hermes_home() / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["approvals"]["approvers_only"] = True
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize(("sender", "emoji", "expected"), [
    pytest.param(PEER, "✅", None, id="peer-cannot-approve"),
    pytest.param(HUMAN, "✅", "once", id="listed-human-approves"),
    pytest.param(GUEST, "✅", None, id="unlisted-guest-cannot-approve"),
    pytest.param(GUEST, "❎", "deny", id="any-admitted-member-may-deny"),
])
async def test_matrix_reaction_uses_separate_approver_list(sender, emoji, expected):
    _configure_independent_approvers()
    entry = pending_from(PEER, "Peer bot", "sess-m", platform="matrix")
    entry.card_deny_only = False  # Full command displayed on the approval card.
    adapter = _make_adapter(allowed_user_ids={PEER, HUMAN, GUEST})
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._approval_require_sender = True  # Real Matrix default, not lowered.
    adapter.send = AsyncMock()
    adapter._redact_bot_approval_reactions = AsyncMock()
    prompt = _MatrixApprovalPrompt(
        session_key="sess-m", chat_id="!room:hs", message_id="$p",
        request_id=entry.data["request_id"], requester_user_id=PEER,
        offered=("once", "deny"),
    )
    adapter._approval_prompts_by_event["$p"] = prompt

    assert await adapter._handle_approval_reaction("!room:hs", "$p", emoji, sender)
    assert entry.event.is_set() is (expected is not None)
    assert entry.result == expected
    if expected is None:
        assert adapter._approval_prompts_by_event["$p"] is prompt
        assert not prompt.resolved
        adapter.send.assert_not_awaited()  # Avoid bot-to-bot feedback loops.


def test_typed_and_bare_approval_use_the_same_separate_authority():
    _configure_independent_approvers()
    entry = pending_from(PEER, "Peer bot", "sess-typed", platform="matrix")
    ids, refused = answerable_request_ids("sess-typed", PEER, choice="once")
    assert ids == [] and refused is not None
    ids, refused = answerable_request_ids("sess-typed", HUMAN, choice="once")
    assert ids == [entry.data["request_id"]] and refused is None
    ids, refused = answerable_request_ids("sess-typed", GUEST, choice="deny")
    assert ids == [entry.data["request_id"]] and refused is None


def test_independent_mode_with_wildcard_fails_closed_for_approval():
    _configure_independent_approvers(["*"])
    pending_from(PEER, "Peer bot", "sess-wild", platform="matrix")
    ids, refused = answerable_request_ids("sess-wild", PEER, choice="once")
    assert ids == [] and refused is not None


