"""agent.autonomy — per-action autonomy levels, the working-hours veto window,
and the pending-action store (education-teammate design, "Autonomy levels")."""
from datetime import datetime, timezone

import pytest

import agent.autonomy as au


def _cfg(**autonomy):
    return {"autonomy": autonomy}


def test_default_level_is_signoff():
    assert au.resolve_level("zendesk_create_draft", {}) == "signoff"
    assert au.resolve_level("zendesk_create_draft", _cfg(actions={})) == "signoff"
    # A non-dict block is a misconfiguration, not a grant: still signoff.
    assert au.resolve_level("zendesk_create_draft", {"autonomy": "auto"}) == "signoff"


def test_reads_configured_level():
    for level in ("off", "signoff", "veto", "auto"):
        cfg = _cfg(actions={"zendesk_create_draft": level})
        assert au.resolve_level("zendesk_create_draft", cfg) == level
    # Case/whitespace tolerant — it's a hand-edited YAML value.
    cfg = _cfg(actions={"zendesk_create_draft": " Veto "})
    assert au.resolve_level("zendesk_create_draft", cfg) == "veto"


def test_paused_caps_at_signoff_but_keeps_off():
    for level, expected in (("auto", "signoff"), ("veto", "signoff"),
                            ("signoff", "signoff"), ("off", "off")):
        cfg = _cfg(paused=True, actions={"zendesk_create_draft": level})
        assert au.resolve_level("zendesk_create_draft", cfg) == expected


def test_unbuilt_level_is_refused():
    cases = [
        ("zendesk_publish", "auto"),
        ("zendesk_publish", "signoff"),
        ("zendesk_create_draft", "autopilot"),
        ("post_release_update", "veto"),  # not built by this change
    ]
    for action, level in cases:
        with pytest.raises(au.AutonomyConfigError) as exc:
            au.resolve_level(action, _cfg(actions={action: level}))
        msg = str(exc.value)
        assert action in msg and level in msg and "built levels" in msg.lower()
    # Even paused, an invalid value is reported rather than silently capped.
    with pytest.raises(au.AutonomyConfigError):
        au.resolve_level("zendesk_publish",
                         _cfg(paused=True, actions={"zendesk_publish": "auto"}))


def test_publish_defaults_off():
    assert au.resolve_level("zendesk_publish", {}) == "off"


_WH = au.WorkingHours(timezone="UTC", start=9, end=18, days=frozenset({0, 1, 2, 3, 4}))


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def test_working_hours_within_day():
    # Wed 10:00 + 4h → Wed 14:00
    assert au.add_working_hours(_utc(2026, 11, 18, 10), 4, _WH) == _utc(2026, 11, 18, 14)
    # Wed 16:00 + 4h → Thu 11:00 (2h today, 2h tomorrow)
    assert au.add_working_hours(_utc(2026, 11, 18, 16), 4, _WH) == _utc(2026, 11, 19, 11)


def test_working_hours_skip_weekend():
    # Fri 2026-11-20 16:00 + 4h → Mon 2026-11-23 11:00
    assert au.add_working_hours(_utc(2026, 11, 20, 16), 4, _WH) == _utc(2026, 11, 23, 11)


def test_working_hours_start_outside_hours():
    # Sat 12:00 → clock starts Mon 09:00
    assert au.add_working_hours(_utc(2026, 11, 21, 12), 4, _WH) == _utc(2026, 11, 23, 13)
    # Tue 06:00 → clock starts Tue 09:00
    assert au.add_working_hours(_utc(2026, 11, 17, 6), 1.5, _WH) == _utc(2026, 11, 17, 10, 30)


def test_working_hours_from_config_and_timezone():
    cfg = _cfg(working_hours={"timezone": "Asia/Jerusalem", "start": 9, "end": 18,
                              "days": ["sun", "mon", "tue", "wed", "thu"]})
    wh = au.working_hours_from_config(cfg)
    assert wh.days == frozenset({6, 0, 1, 2, 3})
    # Thu 2026-11-19 17:00 Jerusalem (15:00 UTC) + 2h → Sun 10:00 Jerusalem (08:00 UTC)
    out = au.add_working_hours(_utc(2026, 11, 19, 15), 2, wh)
    assert out == _utc(2026, 11, 22, 8)


def test_veto_settings_defaults():
    channel, hours, wh = au.veto_settings({})
    assert channel == "" and hours == 4
    assert wh.days == frozenset({0, 1, 2, 3, 4}) and wh.timezone == "UTC"
    channel, hours, _ = au.veto_settings(_cfg(announce_channel="C1", veto_wait_working_hours=2))
    assert channel == "C1" and hours == 2


def test_pending_roundtrip_signed_and_list():
    pid = au.new_pending_id()
    assert len(pid) == 12
    au.save_pending("zendesk_create_draft", pid, {"row": "r1", "not_before": "t"}, title="T")
    rec = au.load_pending("zendesk_create_draft", pid)
    assert rec["signed"] == {"row": "r1", "not_before": "t"} and rec["status"] == "pending"
    assert rec["title"] == "T" and au.verify_pending(rec)
    au.update_pending("zendesk_create_draft", pid, status="done", article_id=5)
    rec = au.load_pending("zendesk_create_draft", pid)
    assert rec["status"] == "done" and rec["article_id"] == 5 and au.verify_pending(rec)
    assert [r["pending_id"] for r in au.list_pending("zendesk_create_draft")] == [pid]
    assert au.list_pending("other_action") == []
    with pytest.raises(ValueError):
        au.update_pending("zendesk_create_draft", pid, signed={"row": "evil"})
    with pytest.raises(ValueError):
        au.save_pending("zendesk_create_draft", pid, {"row": "again"})


def test_tampered_pending_record_fails_verification():
    import json
    pid = au.new_pending_id()
    au.save_pending("zendesk_create_draft", pid, {"row": "r1", "not_before": "t"})
    path = au.autonomy_dir() / "pending" / "zendesk_create_draft" / f"{pid}.json"
    rec = json.loads(path.read_text())
    rec["signed"]["not_before"] = "1970-01-01T00:00:00+00:00"
    path.write_text(json.dumps(rec))
    assert au.verify_pending(au.load_pending("zendesk_create_draft", pid)) is False
    assert au.verify_pending({"pending_id": pid}) is False


def test_hmac_key_is_private_and_stable():
    import stat
    pid = au.new_pending_id()
    au.save_pending("zendesk_create_draft", pid, {"x": 1})
    key = au.autonomy_dir() / ".hmac_key"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    first = key.read_text()
    au.save_pending("zendesk_create_draft", au.new_pending_id(), {"x": 2})
    assert key.read_text() == first


def test_pending_id_rejects_traversal():
    assert au.load_pending("zendesk_create_draft", "../../config") is None
    assert au.load_pending("zendesk_create_draft", "abc") is None
    with pytest.raises(ValueError):
        au.update_pending("zendesk_create_draft", "../x", status="done")
    with pytest.raises(ValueError):
        au.save_pending("../evil", au.new_pending_id(), {})


def test_ledger_roundtrip_and_key_validation():
    assert au.ledger_get("zendesk_create_draft", "abc123") is None
    au.ledger_put("zendesk_create_draft", "abc123", {"article_id": 5})
    assert au.ledger_get("zendesk_create_draft", "abc123") == {"article_id": 5}
    for bad in ("../x", "a/b", "", "x" * 65):
        with pytest.raises(ValueError):
            au.ledger_put("zendesk_create_draft", bad, {})


def test_exclusive_lock_blocks_second_holder_and_releases():
    with au.exclusive("zendesk_create_draft", "row1"):
        with pytest.raises(au.AutonomyBusy):
            with au.exclusive("zendesk_create_draft", "row1"):
                pass
        with au.exclusive("zendesk_create_draft", "row2"):
            pass
    with au.exclusive("zendesk_create_draft", "row1"):
        pass  # released


def test_veto_expiry_hours():
    assert au.veto_expiry_hours({}) == 72
    assert au.veto_expiry_hours(_cfg(veto_expiry_hours=24)) == 24
    for bad in (0, -1, "x"):
        with pytest.raises(au.AutonomyConfigError):
            au.veto_expiry_hours(_cfg(veto_expiry_hours=bad))
