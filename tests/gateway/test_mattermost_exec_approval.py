"""Exec-approval button cards (legacy attachment actions) for the Mattermost adapter.

Behavior contracts, live-verified against Mattermost 11.9.0 (see the adapter sibling
``plugins/platforms/mattermost/exec_approval.py`` docstrings for the server-side gates
these tests pin): action IDs must match ``[A-Za-z0-9]+``, the card rides in
``props.attachments`` (mm_blocks is stripped when the server flag is off).

Presses follow the shared card contract (``tools.approval.resolve_gateway_approval_card`` +
``gateway.approval_actor``): a card answers only the request it was rendered for, honours
``approvals.approvers``, and stays Deny-only until its card shows the whole command. The
contract tests at the bottom drive the REAL resolver, not a mock.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.mattermost.adapter import MattermostAdapter
from plugins.platforms.mattermost.exec_approval import (
    DENY_ONLY_NOTICE,
    action_response,
    approval_actions_config,
    build_approval_props,
    build_ntfy_escalation,
    parse_action_payload,
)

_ACTIONS_URL = "http://192.0.2.10:8647"
_ACTIONS_CFG = {"approval_actions_url": _ACTIONS_URL, "approval_actions_secret": "s3cret",
                "approval_actions_port": 0, "approval_escalate_after": 0}  # port 0 = ephemeral bind

# Keys the base adapter passes for a full (non-smart-denied) approval.
_FULL_ACTIONS = [("Approve once", "once", "primary"), ("Approve session", "session", ""),
                 ("Always approve", "always", ""), ("Deny", "deny", "danger")]


class _FakePrompt:
    chat_id, session_key = "chan123", "agent:main:mattermost:dm:abc"
    command, description, smart_denied = "dd if=/dev/zero of=/tmp/x", "destructive", False
    metadata, text = None, "Hermes wants to run a command that needs your OK"
    actions = _FULL_ACTIONS
    request_id = "req-1"


def _make_adapter(extra=None):
    return MattermostAdapter(PlatformConfig(enabled=True, token="tok",
                                           extra={"url": "https://mm.example.com", **(extra or {})}))


# --- config gate ---------------------------------------------------------------

class TestApprovalActionsConfig:
    def test_buttons_disabled_without_secret(self):
        assert approval_actions_config({"approval_actions_url": "http://x:1"}) is None

    def test_buttons_disabled_without_url(self):
        assert approval_actions_config({"approval_actions_secret": "s"}) is None

    def test_values_parsed(self):
        cfg = approval_actions_config(_ACTIONS_CFG)
        assert cfg["secret"] == "s3cret" and cfg["escalate_after"] == 0 and cfg["port"] == 0


# --- card payload ---------------------------------------------------------------

class TestBuildApprovalProps:
    def test_actions_use_legacy_attachment_format_with_legal_ids(self):
        props = build_approval_props(_FakePrompt(), f"{_ACTIONS_URL}/mattermost/approval/s3cret")
        att = props["attachments"][0]
        assert att["text"] == _FakePrompt.text
        # [A-Za-z0-9]+ only: this Mattermost build's action route rejects IDs with
        # underscores/hyphens at the router (404 "could not find the page").
        for action in att["actions"]:
            assert action["type"] == "button"
            assert action["id"] == action["id"].lower()
            assert action["id"].isalnum()
            assert action["integration"]["url"].endswith("/mattermost/approval/s3cret")

    def test_context_carries_session_choice_and_request(self):
        props = build_approval_props(_FakePrompt(), f"{_ACTIONS_URL}/mattermost/approval/s3cret")
        contexts = [a["integration"]["context"] for a in props["attachments"][0]["actions"]]
        assert [c["choice"] for c in contexts] == ["once", "session", "always", "deny"]
        assert all(c["session_key"] == _FakePrompt.session_key for c in contexts)
        # every button names the one request its card answers
        assert all(c["request_id"] == "req-1" for c in contexts)


# --- callback validation ----------------------------------------------------------

def _press(user_id="u1", choice="once", request_id: "str | None" = "req-1", session_key="s", **extra):
    context = {"session_key": session_key, "choice": choice}
    if request_id is not None:
        context["request_id"] = request_id
    return {"user_id": user_id, "user_name": "alice", "post_id": "p1", "context": context, **extra}


class TestParseActionPayload:
    def test_ok_press_carries_request_id(self):
        verdict, fields = parse_action_payload(_press())
        assert verdict == "ok"
        assert fields is not None and fields["choice"] == "once" and fields["request_id"] == "req-1"

    def test_press_without_request_id_is_stale(self):
        # a card from an older build cannot say which request it answers
        verdict, fields = parse_action_payload(_press(request_id=None))
        assert verdict == "stale" and fields is not None

    def test_invalid_choice_rejected(self):
        assert parse_action_payload(_press(choice="rm -rf /"))[0] == "invalid"

    def test_missing_presser_is_invalid(self):
        assert parse_action_payload(_press(user_id=""))[0] == "invalid"

    def test_non_dict_payload_is_invalid(self):
        assert parse_action_payload("nope")[0] == "invalid"


class TestActionResponse:
    def test_update_clears_button_props(self):
        body = action_response("ok", {"user_name": "alice"}, "once")
        assert "Approved by alice" in body["update"]["message"]
        assert body["update"]["props"] == {}  # clears the buttons — no zombie cards

    def test_deny_and_already_wording(self):
        assert "Denied by alice" in action_response("ok", {"user_name": "alice"}, "deny")["update"]["message"]
        assert "already resolved" in action_response("already", None)["update"]["message"]

    def test_refused_and_stale_leave_the_card_live(self):
        # ephemeral reply to the presser only — no "update", so the card keeps its buttons
        assert action_response("refused", None, refusal="Only Alice can answer this approval.") == {
            "ephemeral_text": "Only Alice can answer this approval."}
        assert "update" not in action_response("stale", None)


# --- adapter flow -----------------------------------------------------------------

class TestSendExecApprovalPrompt:
    def test_unconfigured_falls_back_to_text_prompt(self):
        adapter = _make_adapter()
        result = adapter._approval_cfg
        assert result is None  # buttons off unless URL + secret are set

    @pytest.mark.asyncio
    async def test_card_posted_and_tracked(self, monkeypatch):
        adapter = _make_adapter(_ACTIONS_CFG)
        posted = {}

        async def fake_api_post(path, payload):
            posted["payload"] = payload
            return {"id": "post777"}

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        prompt = _FakePrompt()
        result = await adapter.send_exec_approval(
            chat_id=prompt.chat_id, command=prompt.command, session_key=prompt.session_key,
            description=prompt.description, metadata=None)
        assert result.success and result.message_id == "post777"
        payload = posted["payload"]
        assert payload["channel_id"] == prompt.chat_id
        assert payload["props"]["attachments"][0]["actions"]  # card, not plain text
        assert adapter._approval_cards["post777"]["session_key"] == prompt.session_key
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_resolution_cancels_pending_timers(self, monkeypatch):
        adapter = _make_adapter({**_ACTIONS_CFG, "approval_escalate_after": 300})

        async def fake_api_post(path, payload):
            return {"id": "post777"}

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        import gateway.platforms.base_exec_approval as bea
        monkeypatch.setattr(bea, "approval_timeout_seconds", lambda: 600)
        await adapter.send_exec_approval(chat_id="c", command="x",
                                         session_key="agent:main:mattermost:dm:abc")
        card = adapter._approval_cards["post777"]
        assert card["escalate_task"] is not None and card["expire_task"] is not None
        adapter._mark_approval_card_resolved("post777")
        assert card["resolved"]
        await asyncio.sleep(0)  # let cancel() materialize on the tasks
        assert card["escalate_task"].cancelled()
        assert card["expire_task"].cancelled()
        await adapter.disconnect()


# --- ntfy escalation ---------------------------------------------------------------

class TestBuildNtfyEscalation:
    def test_view_action_deep_links_to_card(self):
        message, headers = build_ntfy_escalation("rm -rf /tmp/cache",
                                                 "https://mm.example.com/team/pl/post1")
        assert "rm -rf /tmp/cache" in message
        assert headers["Actions"] == "view, Open in Mattermost, https://mm.example.com/team/pl/post1, clear=true"
        assert headers["Priority"] == "high"


# --- shared card contract, against the REAL resolver --------------------------------
# Same fixtures as tests/gateway/test_approval_requester_binding_adapters.py: Alice's turn raised
# the approval; Bob (admitted to the chat) presses. ``approvers: []`` narrows answering to Alice.

from tests.gateway.test_approval_requester_binding import DEFAULT, pending_from, write_approvers

REFUSAL = "Only Alice can answer this approval."
_SK = "agent:main:mattermost:group:chan123"


@pytest.fixture
def _clean_queues():
    from tools import approval as approval_mod

    approval_mod._gateway_queues.clear()
    yield
    approval_mod._gateway_queues.clear()


async def _click(adapter, payload):
    """Drive the real post-action handler the Mattermost server calls on a button press."""
    request = SimpleNamespace(json=lambda: asyncio.sleep(0, result=payload))
    response = await adapter._handle_approval_action(request)
    return json.loads(response.text)


def _alice_pending():
    return pending_from("u-alice", "Alice", _SK, platform="mattermost")


@pytest.mark.usefixtures("_clean_queues")
class TestCardContract:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("approvers", "allowed"), [(DEFAULT, True), ([], False)])
    async def test_press_honours_approvals_approvers(self, approvers, allowed):
        write_approvers(approvers)
        entry = _alice_pending()
        body = await _click(_make_adapter(_ACTIONS_CFG),
                            _press(user_id="u-bob", session_key=_SK, request_id=entry.data["request_id"]))
        assert entry.event.is_set() is allowed
        if allowed:
            assert entry.result == "once" and body["update"]["props"] == {}
        else:
            assert body == {"ephemeral_text": REFUSAL}  # presser only — Alice's card stays live

    @pytest.mark.asyncio
    async def test_card_of_a_settled_request_never_resolves_the_next_one(self):
        """A session-scoped resolve would let an old card's Approve answer a NEWER request."""
        from tools.approval import resolve_gateway_approval

        write_approvers(DEFAULT)
        first, second = _alice_pending(), _alice_pending()
        resolve_gateway_approval(_SK, "deny", request_id=first.data["request_id"])  # typed /deny
        body = await _click(_make_adapter(_ACTIONS_CFG),
                            _press(user_id="u-alice", session_key=_SK, request_id=first.data["request_id"]))
        assert not second.event.is_set()
        assert "already resolved" in body["update"]["message"]

    @pytest.mark.asyncio
    async def test_press_without_request_id_resolves_nothing(self):
        write_approvers(DEFAULT)
        entry = _alice_pending()
        body = await _click(_make_adapter(_ACTIONS_CFG),
                            _press(user_id="u-alice", session_key=_SK, request_id=None))
        assert not entry.event.is_set()
        assert "update" not in body and "/approve" in body["ephemeral_text"]

    @pytest.mark.asyncio
    async def test_approve_on_a_deny_only_request_resolves_nothing_and_deny_stays_live(self):
        from tools.approval import restrict_gateway_approval_card_to_deny

        write_approvers(DEFAULT)
        entry = _alice_pending()
        rid = entry.data["request_id"]
        restrict_gateway_approval_card_to_deny(_SK, rid)  # card could not show the whole command
        adapter = _make_adapter(_ACTIONS_CFG)
        body = await _click(adapter, _press(user_id="u-alice", session_key=_SK, request_id=rid))
        assert not entry.event.is_set()
        assert body == {"ephemeral_text": DENY_ONLY_NOTICE}  # card NOT wiped: Deny still answers
        body = await _click(adapter, _press(user_id="u-alice", choice="deny", session_key=_SK, request_id=rid))
        assert entry.result == "deny" and body["update"]["props"] == {}

    @pytest.mark.asyncio
    async def test_rendered_card_lifts_deny_only_and_its_approve_resolves(self, monkeypatch):
        """End to end: the real send path renders a card that shows the whole command, which lifts
        the request's Deny-only mark; that card's own Approve button then resolves it."""
        from tools.approval import restrict_gateway_approval_card_to_deny

        write_approvers(DEFAULT)
        entry = _alice_pending()
        rid = entry.data["request_id"]
        restrict_gateway_approval_card_to_deny(_SK, rid)  # every request starts Deny-only
        adapter = _make_adapter(_ACTIONS_CFG)
        posted = {}

        async def fake_api_post(path, payload):
            posted["payload"] = payload
            return {"id": "post777"}

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        result = await adapter.send_exec_approval(chat_id="chan123", command="ls /tmp",
                                                  session_key=_SK, request_id=rid)
        assert result.success
        assert entry.card_deny_only is False  # the card showed the whole command
        actions = posted["payload"]["props"]["attachments"][0]["actions"]
        once = next(a["integration"]["context"] for a in actions if a["integration"]["context"]["choice"] == "once")
        assert once["request_id"] == rid
        body = await _click(adapter, {"user_id": "u-alice", "user_name": "alice", "post_id": "post777",
                                      "context": once})
        assert entry.result == "once" and "Approved by alice" in body["update"]["message"]
        await adapter.disconnect()
