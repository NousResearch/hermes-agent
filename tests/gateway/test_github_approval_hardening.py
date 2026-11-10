import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.github_approval_broker import ApprovalProposal, GitHubApprovalBroker
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_clarify_binding import ClarifyPrompt


class BoundTelegram:
    def __init__(self, *, outcome="success"):
        self.outcome = outcome
        self.sent = []
        self.retired = []
        self._clarify_state = {}

    def approval_destination_identity(self, clarify_id=None):
        return "telegram-bot:501"

    async def send_clarify(self, **kwargs):
        self.sent.append(kwargs)
        if self.outcome == "raise":
            raise TimeoutError("unknown send outcome")
        if self.outcome == "failure":
            return SendResult(success=False, error="rejected before send")
        self._clarify_state[kwargs["clarify_id"]] = kwargs["session_key"]
        return SendResult(success=True, message_id="501")

    def approval_callback_binding(self, clarify_id, session_key):
        return self._clarify_state.get(clarify_id) == session_key

    async def retire_clarify_card(self, clarify_id, notice):
        self._clarify_state.pop(clarify_id, None)
        self.retired.append((clarify_id, notice))


def proposal(item_key="owner/repo#7", head="a" * 40, digest="b" * 64):
    return ApprovalProposal(item_key, head, digest)


@pytest.mark.parametrize("item_key", [
    "owner/repo#0", "owner/repo#-1", "owner/repo#1x", "owner/repo#1/2",
    "/repo#1", "owner/#1", "owner/repo# 1", "owner/repo#01", "owner/repo#1#2",
])
def test_item_key_requires_strict_owner_repo_and_positive_canonical_number(item_key):
    with pytest.raises(ValueError, match="owner/repo#number"):
        proposal(item_key=item_key).validate()


@pytest.mark.asyncio
async def test_unknown_provider_send_outcome_is_never_retried(tmp_path):
    path = tmp_path / "approval.db"
    broker = GitHubApprovalBroker(path, instance_id="one", delivery_lease_seconds=1)
    outcome = await broker.request(
        proposal(), adapter=BoundTelegram(outcome="raise"), chat_id="42", question="Approve?",
        session_key="webhook:1", authorized_user_id="7")

    assert outcome.action == "operator_reconciliation"
    assert broker.observe(proposal())["state"] == "delivery_unknown"
    assert await GitHubApprovalBroker(path, instance_id="two").retry_pending(adapter=BoundTelegram()) == []


@pytest.mark.asyncio
async def test_webhook_binding_is_checked_on_delegated_telegram_adapter(tmp_path):
    target = BoundTelegram()
    runner = MagicMock()
    runner._authorization_adapter.return_value = target
    adapter = object.__new__(WebhookAdapter)
    adapter.gateway_runner = runner
    adapter._delivery_info = {"delivery": {"deliver": "telegram", "deliver_extra": {"chat_id": "42"}}}
    broker = GitHubApprovalBroker(tmp_path / "approval.db", instance_id="webhook")

    outcome = await broker.request(
        proposal(), adapter=adapter, chat_id="delivery", question="Approve?",
        session_key="webhook:1", authorized_user_id="7")

    assert outcome.action == "sent"
    assert broker.observe(proposal())["state"] == "active"


@pytest.mark.asyncio
async def test_callback_is_bound_to_user_chat_thread_and_message(tmp_path):
    broker = GitHubApprovalBroker(tmp_path / "approval.db", instance_id="one")
    sent = await broker.request(
        proposal(), adapter=BoundTelegram(), chat_id="42", thread_id="9", question="Approve?",
        session_key="webhook:1", authorized_user_id="7")

    assert broker.resolve_callback(
        sent.clarify_id, 0, approver="telegram:7", user_id="7", chat_id="42",
        thread_id="9", message_id="wrong", destination_id="telegram-bot:501") is None
    assert broker.resolve_callback(
        sent.clarify_id, 0, approver="telegram:8", user_id="8", chat_id="42",
        thread_id="9", message_id="501", destination_id="telegram-bot:501") is None
    decision = broker.resolve_callback(
        sent.clarify_id, 0, approver="telegram:7", user_id="7", chat_id="42",
        thread_id="9", message_id="501", destination_id="telegram-bot:501")
    assert decision is not None and decision.response == "Approve"


def test_replayed_opposite_choice_returns_original_durable_decision(tmp_path):
    broker = GitHubApprovalBroker(tmp_path / "approval.db", instance_id="one")
    sent = asyncio.run(broker.request(
        proposal(), adapter=BoundTelegram(), chat_id="42", question="Approve?",
        session_key="webhook:1", authorized_user_id="7"))
    first = broker.resolve_callback(
        sent.clarify_id, 0, approver="telegram:7", user_id="7", chat_id="42",
        thread_id=None, message_id="501", destination_id="telegram-bot:501")
    replay = broker.resolve_callback(
        sent.clarify_id, 1, approver="telegram:7", user_id="7", chat_id="42",
        thread_id=None, message_id="501", destination_id="telegram-bot:501")

    assert first.response == replay.response == "Approve"
    assert broker.callback_choice(sent.clarify_id, 1, "telegram-bot:501") == "Approve"


class LiveReader:
    def __init__(self, *, head="a" * 40, digest="b" * 64):
        self.head = head
        self.digest = digest

    async def read_contract(self, item_key):
        return ApprovalProposal(item_key, self.head, self.digest)


class ReceiptExecutor:
    def __init__(self):
        self.receipts = {}
        self.calls = []

    async def read_receipt(self, key):
        return self.receipts.get(key)

    async def execute_idempotent(self, claim, *, idempotency_key):
        self.calls.append((claim, idempotency_key))
        receipt = {"id": idempotency_key}
        self.receipts[idempotency_key] = receipt
        return receipt


async def _approved_broker(tmp_path):
    broker = GitHubApprovalBroker(tmp_path / "approval.db", instance_id="one")
    sent = await broker.request(
        proposal(), adapter=BoundTelegram(), chat_id="42", question="Approve?",
        session_key="webhook:1", authorized_user_id="7")
    broker.resolve_callback(
        sent.clarify_id, 0, approver="telegram:7", user_id="7", chat_id="42",
        thread_id=None, message_id="501", destination_id="telegram-bot:501")
    return broker


@pytest.mark.asyncio
async def test_do_not_approve_never_invokes_executor(tmp_path):
    broker = GitHubApprovalBroker(tmp_path / "approval.db", instance_id="one")
    sent = await broker.request(
        proposal(), adapter=BoundTelegram(), chat_id="42", question="Approve?",
        session_key="webhook:1", authorized_user_id="7")
    broker.resolve_callback(
        sent.clarify_id, 1, approver="telegram:7", user_id="7", chat_id="42",
        thread_id=None, message_id="501", destination_id="telegram-bot:501")
    executor = ReceiptExecutor()

    assert await broker.resume_approved(owner="worker", github=LiveReader(), executor=executor) == ["declined"]
    assert executor.calls == []


@pytest.mark.asyncio
async def test_resumer_revalidates_live_head_and_contract_and_retires_stale(tmp_path):
    broker = await _approved_broker(tmp_path)
    adapter = BoundTelegram()
    executor = ReceiptExecutor()

    outcomes = await broker.resume_approved(
        owner="worker", github=LiveReader(head="c" * 40), executor=executor, adapter=adapter)

    assert outcomes == ["stale"]
    assert executor.calls == []
    assert broker.observe(proposal())["state"] == "superseded"


@pytest.mark.asyncio
async def test_generic_executor_is_rejected_before_side_effect(tmp_path):
    broker = await _approved_broker(tmp_path)
    generic = AsyncMock()

    outcomes = await broker.resume_approved(owner="worker", github=LiveReader(), executor=generic)

    assert outcomes == ["blocked_executor"]
    generic.assert_not_awaited()


@pytest.mark.asyncio
async def test_receipt_contract_makes_continuation_replay_safe(tmp_path):
    broker = await _approved_broker(tmp_path)
    executor = ReceiptExecutor()

    assert await broker.resume_approved(owner="worker", github=LiveReader(), executor=executor) == ["executed"]
    assert await broker.resume_approved(owner="worker-2", github=LiveReader(), executor=executor) == []
    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_telegram_retire_clarify_card_removes_binding_and_keyboard():
    adapter = object.__new__(TelegramAdapter)
    adapter._clarify_state = {"gha-id": ClarifyPrompt("session", "42", "501", None)}
    adapter._application = SimpleNamespace(bot=SimpleNamespace(edit_message_text=AsyncMock()))

    await adapter.retire_clarify_card("gha-id", "Superseded")

    assert "gha-id" not in adapter._clarify_state
    adapter._application.bot.edit_message_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_approval_prompt_has_no_free_text_other_choice(monkeypatch):
    adapter = object.__new__(TelegramAdapter)
    adapter._clarify_state = {}
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InlineKeyboardButton",
        lambda text, callback_data: SimpleNamespace(text=text, callback_data=callback_data),
    )
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
        lambda rows: SimpleNamespace(inline_keyboard=rows),
    )

    async def capture(_kind, _chat_id, _metadata, build, **_kwargs):
        _text, keyboard, _remember = build()
        callbacks = [button.callback_data for row in keyboard.inline_keyboard for button in row]
        assert callbacks == ["cl:gha-id:0", "cl:gha-id:1"]
        return SendResult(success=True, message_id="501")

    adapter._send_prompt = capture
    await adapter.send_clarify(
        "42", "Approve?", ["Approve", "Do not approve"], "gha-id", "session",
        {"is_approval_prompt": True})


@pytest.mark.asyncio
async def test_repository_dispatch_producer_creates_card_and_never_admits_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.webhook.web",
        SimpleNamespace(json_response=lambda body, status=200: SimpleNamespace(status=status, body=body)))
    target = BoundTelegram()
    runner = MagicMock()
    runner._authorization_adapter.return_value = target
    adapter = object.__new__(WebhookAdapter)
    adapter.gateway_runner = runner
    adapter._delivery_info = {}
    adapter._delivery_info_created = {}
    adapter._delivery_info_order = []
    route = {
        "deliver": "telegram",
        "deliver_extra": {"chat_id": "42", "message_thread_id": "9"},
        "github_approval": {"authorized_user_id": "7"},
    }
    payload = {
        "action": "hermes_contract_approval",
        "repository": {"full_name": "owner/repo"},
        "client_payload": {
            "pull_request_number": 7,
            "head_sha": "a" * 40,
            "contract_digest_sha256": "b" * 64,
            "executor": "hermes",
        },
    }
    broker = GitHubApprovalBroker(tmp_path / "approval.db", instance_id="producer")
    monkeypatch.setattr("gateway.github_approval_broker.default_broker", lambda **kwargs: broker)
    adapter._admit_agent_run = AsyncMock(side_effect=AssertionError("approval producer executed agent"))

    response = await adapter._handle_github_approval_contract(
        route, payload, "repository_dispatch", "route", None, "delivery-1")

    assert response.status == 202
    assert len(target.sent) == 1
    assert broker.observe(proposal())["state"] == "active"
    adapter._admit_agent_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_approval_route_rejects_unrecognized_github_contract_before_agent(monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.webhook.web",
        SimpleNamespace(json_response=lambda body, status=200: SimpleNamespace(status=status, body=body)))
    adapter = object.__new__(WebhookAdapter)
    response = await adapter._handle_github_approval_contract(
        {"github_approval": {"authorized_user_id": "7"}}, {"action": "opened"},
        "pull_request", "route", None, "delivery-1")
    assert response.status == 400
