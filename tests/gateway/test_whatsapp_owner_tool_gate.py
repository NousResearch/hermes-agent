"""Owner-aware WhatsApp toolset gating (``_whatsapp_owner_tool_gate``).

A shared / community WhatsApp line otherwise exposes the full agent toolset
(terminal, code execution, file, ...) to every allowlisted sender. When
``whatsapp.nonowner_disabled_toolsets`` is configured, only owners -- the
``whatsapp.home_channel`` chat_id or ids in ``whatsapp.owner_users`` -- keep the
full toolset; everyone else has the configured tools disabled. The feature is
opt-in (no config -> no behavior change) and fails closed.
"""

from types import SimpleNamespace

from gateway.run import _whatsapp_owner_tool_gate

OWNER = "178172540747977"
DROP = ["terminal", "code_execution", "file"]
CFG = {
    "whatsapp": {
        "home_channel": {"platform": "whatsapp", "chat_id": f"{OWNER}@lid"},
        "owner_users": ["15551234567"],
        "nonowner_disabled_toolsets": DROP,
    }
}


def _src(chat_id):
    return SimpleNamespace(chat_id=chat_id)


def test_noop_when_feature_unconfigured():
    cfg = {"whatsapp": {"home_channel": {"chat_id": f"{OWNER}@lid"}}}
    assert _whatsapp_owner_tool_gate(_src("99999@lid"), "whatsapp", cfg, None) is None


def test_noop_on_other_platforms():
    assert _whatsapp_owner_tool_gate(_src("99999"), "telegram", CFG, None) is None


def test_owner_keeps_full_toolset():
    # home_channel owner, including a device-suffixed LID variant
    assert _whatsapp_owner_tool_gate(_src(f"{OWNER}@lid"), "whatsapp", CFG, None) is None
    assert _whatsapp_owner_tool_gate(_src(f"{OWNER}:12@lid"), "whatsapp", CFG, None) is None
    # explicit owner_users entry (configured as a bare number, sender arrives as a jid)
    assert _whatsapp_owner_tool_gate(_src("15551234567@s.whatsapp.net"), "whatsapp", CFG, None) is None


def test_non_owner_is_restricted():
    out = _whatsapp_owner_tool_gate(_src("19998887777@s.whatsapp.net"), "whatsapp", CFG, None)
    assert set(DROP).issubset(set(out))


def test_non_owner_preserves_existing_disabled():
    out = _whatsapp_owner_tool_gate(_src("19998887777@lid"), "whatsapp", CFG, ["moa"])
    assert "moa" in out
    assert set(DROP).issubset(set(out))


def test_fails_closed_when_sender_unknown():
    # feature active but sender id missing -> restrict, never grant
    out = _whatsapp_owner_tool_gate(SimpleNamespace(), "whatsapp", CFG, None)
    assert set(DROP).issubset(set(out))
