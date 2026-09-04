"""WhatsApp DM/group allowlist must resolve phone↔LID aliases at intake.

Regression for #14486: WhatsApp now delivers inbound DM senders in LID form
(``<id>@lid``) while operators configure the allowlist with phone numbers.
The adapter-level gate (``_is_dm_allowed`` / ``_is_group_allowed`` →
``_should_process_message``) did a raw set-membership check with no LID
resolution, so every DM from an allowed user was silently dropped before the
gateway authz layer ever ran.

The fix routes the adapter gate through the shared
``gateway.whatsapp_identity.expand_whatsapp_aliases`` helper, which reads the
bridge's ``lid-mapping-*.json`` session files (the same source the gateway
authz and session-key paths already use).
"""

import json
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from hermes_constants import get_hermes_home


PHONE = "351912345678"
LID = "77214955630717"


def _make_adapter(dm_policy=None, allow_from=None, group_policy=None, group_allow_from=None):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    extra = {}
    if dm_policy is not None:
        extra["dm_policy"] = dm_policy
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if group_policy is not None:
        extra["group_policy"] = group_policy
    if group_allow_from is not None:
        extra["group_allow_from"] = group_allow_from

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = str(extra.get("dm_policy", "open")).strip().lower()
    adapter._allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("allow_from"))
    adapter._group_policy = str(extra.get("group_policy", "open")).strip().lower()
    adapter._group_allow_from = WhatsAppAdapter._coerce_allow_list(
        extra.get("group_allow_from")
    )
    return adapter


def _write_lid_mapping(phone=PHONE, lid=LID):
    """Mirror what the JS bridge writes: phone→lid and lid→phone (reverse)."""
    session_dir = get_hermes_home() / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / f"lid-mapping-{phone}.json").write_text(json.dumps(lid), encoding="utf-8")
    (session_dir / f"lid-mapping-{lid}_reverse.json").write_text(
        json.dumps(phone), encoding="utf-8"
    )


# --------------------------------------------------------------------- DM gate

def test_dm_phone_allowlist_matches_lid_sender():
    """allow_from has the phone number; inbound sender arrives as @lid (the bug)."""
    _write_lid_mapping()
    adapter = _make_adapter(dm_policy="allowlist", allow_from=[PHONE])

    assert adapter._is_dm_allowed(f"{LID}@lid") is True


def test_dm_phone_with_plus_allowlist_matches_lid_sender():
    """A ``+``-prefixed phone allowlist entry still resolves to the LID sender."""
    _write_lid_mapping()
    adapter = _make_adapter(dm_policy="allowlist", allow_from=[f"+{PHONE}"])

    assert adapter._is_dm_allowed(f"{LID}@lid") is True


# ------------------------------------------------------------------ group gate

def test_group_jid_exact_match_still_works():
    """Group allowlists use full ``@g.us`` JIDs — exact match must pass through."""
    adapter = _make_adapter(
        group_policy="allowlist", group_allow_from=["120363001234567890@g.us"]
    )

    assert adapter._is_group_allowed("120363001234567890@g.us") is True


def test_group_unlisted_jid_blocked():
    adapter = _make_adapter(
        group_policy="allowlist", group_allow_from=["120363001234567890@g.us"]
    )

    assert adapter._is_group_allowed("120363009999999999@g.us") is False


# ------------------------------------------------------ end-to-end intake gate

def test_should_process_message_dm_phone_allowlist_lid_sender():
    """Full intake path: a DM from a phone-allowlisted contact arriving as @lid."""
    _write_lid_mapping()
    adapter = _make_adapter(dm_policy="allowlist", allow_from=[PHONE])

    data = {
        "isGroup": False,
        "body": "hello",
        "senderId": f"{LID}@lid",
        "from": f"{LID}@lid",
        "botIds": [],
        "mentionedIds": [],
    }
    assert adapter._should_process_message(data) is True


class TestJsonArrayStringAllowlist:
    """#102329: a JSON-array string (pre-#88163 `config set` shape, or
    hand-written) must authorize exactly like the equivalent YAML list —
    never as bracket-polluted entries that match nobody."""

    def test_coerce_json_array_string(self):
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        assert WhatsAppAdapter._coerce_allow_list('["15551234567", "15557654321"]') == {
            "15551234567",
            "15557654321",
        }

    def test_coerce_plain_shapes_unchanged(self):
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        assert WhatsAppAdapter._coerce_allow_list(["15551234567", "15557654321"]) == {
            "15551234567",
            "15557654321",
        }
        assert WhatsAppAdapter._coerce_allow_list("15551234567,15557654321") == {
            "15551234567",
            "15557654321",
        }
        assert WhatsAppAdapter._coerce_allow_list(None) == set()

    def test_coerce_python_literal_list(self):
        """Hand-written Python-literal shape, mirroring parse_config_string_list."""
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        assert WhatsAppAdapter._coerce_allow_list("['15551234567']") == {"15551234567"}

    def test_malformed_bracket_string_warns(self, caplog):
        """A `[`-leading value that parses as neither JSON nor Python literal
        falls back to comma-split (matching nobody) and must log, so the
        next silent-deaf report starts from the value."""
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        with caplog.at_level("WARNING"):
            assert WhatsAppAdapter._coerce_allow_list("[1555") == {"[1555"}
        assert "parses as neither JSON nor Python literal" in caplog.text

    def test_intake_accepts_json_array_string_allowlist(self):
        """End to end: the reported deaf-bot scenario now processes."""
        _write_lid_mapping()
        adapter = _make_adapter(dm_policy="allowlist", allow_from=f'["{PHONE}"]')

        data = {
            "isGroup": False,
            "body": "hello",
            "senderId": f"{LID}@lid",
            "from": f"{LID}@lid",
            "botIds": [],
            "mentionedIds": [],
        }
        assert adapter._should_process_message(data) is True

    def test_yaml_export_normalizes_json_array_string(self, monkeypatch):
        """The bridge env export must carry clean comma values, so the Node
        bridge parses the same membership the adapter enforces."""
        from plugins.platforms.whatsapp.adapter import _apply_yaml_config

        monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)
        _apply_yaml_config({}, {"allow_from": '["15551234567", "15557654321"]'})
        import os

        assert os.environ["WHATSAPP_ALLOWED_USERS"] == "15551234567,15557654321"
