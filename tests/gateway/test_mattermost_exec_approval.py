"""Exec-approval button cards (legacy attachment actions) for the Mattermost adapter.

Behavior contracts, live-verified against Mattermost 11.9.0 (see the adapter sibling
``plugins/platforms/mattermost/exec_approval.py`` docstrings for the server-side gates
these tests pin): action IDs must match ``[A-Za-z0-9]+``, the card rides in
``props.attachments`` (mm_blocks is stripped when the server flag is off), presses
resolve via ``tools.approval.resolve_gateway_approval``.
"""

import asyncio

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.mattermost.adapter import MattermostAdapter
from plugins.platforms.mattermost.exec_approval import (
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

    def test_context_carries_session_and_choice(self):
        props = build_approval_props(_FakePrompt(), f"{_ACTIONS_URL}/mattermost/approval/s3cret")
        contexts = [a["integration"]["context"] for a in props["attachments"][0]["actions"]]
        assert [c["choice"] for c in contexts] == ["once", "session", "always", "deny"]
        assert all(c["session_key"] == _FakePrompt.session_key for c in contexts)


# --- callback validation ----------------------------------------------------------

class TestParseActionPayload:
    def test_ok_press(self):
        verdict, fields = parse_action_payload(
            {"user_id": "u1", "user_name": "gary", "post_id": "p1",
             "context": {"session_key": "s", "choice": "once"}}, {"u1"})
        assert verdict == "ok"
        assert fields is not None and fields["choice"] == "once" and fields["session_key"] == "s"

    def test_unauthorized_presser_rejected(self):
        verdict, _ = parse_action_payload(
            {"user_id": "intruder", "context": {"session_key": "s", "choice": "deny"}},
            {"u1"})
        assert verdict == "unauthorized"

    def test_invalid_choice_rejected(self):
        verdict, _ = parse_action_payload(
            {"user_id": "u1", "context": {"session_key": "s", "choice": "rm -rf /"}},
            {"u1"})
        assert verdict == "invalid"

    def test_empty_allowlist_is_open_access(self):
        verdict, _ = parse_action_payload(
            {"user_id": "anyone", "context": {"session_key": "s", "choice": "once"}}, set())
        assert verdict == "ok"

    def test_non_dict_payload_is_invalid(self):
        assert parse_action_payload("nope", set())[0] == "invalid"


class TestActionResponse:
    def test_update_clears_button_props(self):
        body = action_response("ok", {"user_name": "gary"}, "once")
        assert "Approved by gary" in body["update"]["message"]
        assert body["update"]["props"] == {}  # clears the buttons — no zombie cards

    def test_deny_and_already_wording(self):
        assert "Denied by gary" in action_response("ok", {"user_name": "gary"}, "deny")["update"]["message"]
        assert "already resolved" in action_response("already", None)["update"]["message"]


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
