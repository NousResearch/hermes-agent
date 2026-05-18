"""Tests for the read-only GHL Manager dashboard plugin backend."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PLUGIN_MODULE_PATH = Path(__file__).resolve().parents[2] / "plugins" / "ghl-manager" / "dashboard" / "plugin_api.py"


def _load_plugin_api():
    spec = importlib.util.spec_from_file_location(f"ghl_manager_plugin_api_test_{id(object())}", PLUGIN_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_message_context_enriches_latest_inbound_and_last_outbound_from_local_artifacts(tmp_path):
    plugin_api = _load_plugin_api()
    plugin_api.ARTIFACT_ROOT = tmp_path
    (tmp_path / "t_c317a5f5_classified_inbox_candidates.json").write_text(
        json.dumps(
            [
                {
                    "source_row": "4",
                    "conversationId": "conv_123",
                    "contactName": "Jane Customer",
                    "lastMessageDate": "2026-05-08T00:27:21.736000+00:00",
                    "lastMessageDirection": "inbound",
                    "lastMessageType": "TYPE_FACEBOOK",
                    "preview": "Actual customer message from GHL snapshot.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "t_42bcdc5a_latest_inbound_conversations.json").write_text("[]", encoding="utf-8")
    (tmp_path / "t_bb320de1_refreshed_pending_drafts.md").write_text(
        """# refreshed triage

## Current / still approval-ready, with context brief required

1. Source row 4 / t_demo — Solar Renew, contact contact_123, conversation conv_123.
   Status: still current.
   Latest customer ask: summary from refreshed prose.
   Last outbound: 2026-05-07 follow-up asking panel count and suburb.
""",
        encoding="utf-8",
    )
    packet = plugin_api._base_packet(
        approval_id="approval_1",
        source_type="json_approval_item",
        source_path=tmp_path / "pending.json",
    )
    packet["source"]["source_row"] = 4
    packet["conversation"]["conversation_id"] = "conv_123"

    plugin_api._apply_message_context([packet])

    assert packet["conversation"]["latest_customer_summary"] == "Actual customer message from GHL snapshot."
    assert packet["conversation"]["latest_customer_at"] == "2026-05-08T00:27:21.736000+00:00"
    assert packet["conversation"]["latest_customer_source"] == "t_c317a5f5_classified_inbox_candidates.json"
    assert packet["conversation"]["last_outbound_summary"] == "2026-05-07 follow-up asking panel count and suburb."
    assert packet["conversation"]["last_outbound_at"] == "2026-05-07"
    assert packet["conversation"]["last_outbound_source"] == "t_bb320de1_refreshed_pending_drafts.md"
    assert packet["contact"]["name"] == "Jane Customer"
    assert packet["conversation"]["channel"] == "Facebook"
    assert packet["conversation"]["channel_source"] == "TYPE_FACEBOOK"


def test_base_packet_uses_clear_missing_message_contract(tmp_path):
    plugin_api = _load_plugin_api()
    plugin_api.ARTIFACT_ROOT = tmp_path
    packet = plugin_api._base_packet(
        approval_id="approval_2",
        source_type="json_approval_item",
        source_path=tmp_path / "pending.json",
    )

    assert packet["conversation"]["latest_customer_source"] == "source_packet_omitted_message"
    assert packet["conversation"]["last_outbound_source"] == "source_packet_omitted_outbound"
    assert "omitted" in packet["conversation"]["latest_customer_summary"]
    assert "No outbound context found" in packet["conversation"]["last_outbound_summary"]


def test_mixed_facebook_sms_target_keeps_social_channel_and_name(tmp_path):
    plugin_api = _load_plugin_api()
    plugin_api.ARTIFACT_ROOT = tmp_path
    source = tmp_path / "pending-approval.json"
    item = {
        "approval_id": "approval_fb",
        "brand": "The Blue Crew",
        "contactId": "contact_123",
        "conversationId": "conv_fb",
        "send_target": "Facebook/SMS thread to Tom Cooling/contact contact_123 via The Blue Crew",
        "source_row": 111,
        "evidence": "Fetched messages; latest inbound asks if solar panel cleaning is still offered.",
        "proposed_action": {"customer_facing_send": True, "draft": "Hi Tom, yes we still do solar panel cleaning."},
    }

    packet = plugin_api._normalize_approval_item(item, {"task_id": "t_demo", "approval_items": [item]}, source)

    assert packet["conversation"]["channel"] == "Facebook"
    assert packet["send_target"]["channel"] == "Facebook"
    assert packet["contact"]["name"] == "Tom Cooling"


def test_data_contract_missing_reasons_are_embedded(tmp_path):
    plugin_api = _load_plugin_api()
    plugin_api.ARTIFACT_ROOT = tmp_path
    packet = plugin_api._base_packet(
        approval_id="approval_missing",
        source_type="json_approval_item",
        source_path=tmp_path / "pending.json",
    )

    plugin_api._apply_data_contract_reasons([packet])

    missing = packet["data_contract"]["missing_data"]
    assert {item["field"] for item in missing} >= {
        "contact.name",
        "conversation.latest_customer_at",
        "conversation.last_outbound_at",
    }
    assert packet["ui_indicators"]["data_incomplete"] is True
