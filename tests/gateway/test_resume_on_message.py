"""Tests for opt-in resume-on-message after idle expiry (issue #43008, proposal #2).

When `session_reset.resume_on_message` is enabled, the first message that lands
within `resume_grace_minutes` of the idle deadline resumes the existing session
(same session_id, no auto-reset) instead of starting a blank one. Beyond the
grace window — or when the option is off — the session hard-resets as before.
Daily resets are never affected.
"""

import datetime

import gateway.session as session_mod
from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionSource, SessionStore


def _store(tmp_path, **policy_kw):
    cfg = GatewayConfig()
    cfg.default_reset_policy = SessionResetPolicy(mode="idle", idle_minutes=1, **policy_kw)
    return SessionStore(sessions_dir=tmp_path, config=cfg)


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1")


def _age_minutes(store, source, minutes):
    key = store._generate_session_key(source)
    store._entries[key].updated_at = session_mod._now() - datetime.timedelta(minutes=minutes)


# --- Policy round-trips the new fields -------------------------------------

def test_policy_roundtrips_resume_fields():
    p = SessionResetPolicy.from_dict(
        {"mode": "idle", "resume_on_message": True, "resume_grace_minutes": 30}
    )
    assert p.resume_on_message is True
    assert p.resume_grace_minutes == 30
    assert p.to_dict()["resume_on_message"] is True
    assert p.to_dict()["resume_grace_minutes"] == 30


def test_policy_defaults_are_off():
    p = SessionResetPolicy.from_dict({"mode": "idle"})
    assert p.resume_on_message is False
    assert p.resume_grace_minutes == 60


# --- Behavior --------------------------------------------------------------

def test_resumes_within_grace_window(tmp_path):
    store = _store(tmp_path, resume_on_message=True, resume_grace_minutes=60)
    src = _source()
    first = store.get_or_create_session(src)
    sid = first.session_id
    _age_minutes(store, src, 5)  # past idle(1m), within grace(60m)
    second = store.get_or_create_session(src)
    assert second.session_id == sid          # resumed, same session
    assert second.was_auto_reset is False


def test_hard_resets_beyond_grace_window(tmp_path):
    store = _store(tmp_path, resume_on_message=True, resume_grace_minutes=60)
    src = _source()
    first = store.get_or_create_session(src)
    sid = first.session_id
    _age_minutes(store, src, 120)  # past idle + grace
    second = store.get_or_create_session(src)
    assert second.session_id != sid          # blank reset
    assert second.was_auto_reset is True


def test_off_by_default_still_hard_resets(tmp_path):
    store = _store(tmp_path)  # resume_on_message defaults False
    src = _source()
    first = store.get_or_create_session(src)
    sid = first.session_id
    _age_minutes(store, src, 5)  # past idle, within would-be grace
    second = store.get_or_create_session(src)
    assert second.session_id != sid          # current behavior preserved
    assert second.was_auto_reset is True


def test_zero_grace_disables_resume(tmp_path):
    store = _store(tmp_path, resume_on_message=True, resume_grace_minutes=0)
    src = _source()
    first = store.get_or_create_session(src)
    sid = first.session_id
    _age_minutes(store, src, 5)
    second = store.get_or_create_session(src)
    assert second.session_id != sid          # grace=0 → never resumes
    assert second.was_auto_reset is True


def test_resume_clears_expiry_finalized(tmp_path):
    store = _store(tmp_path, resume_on_message=True, resume_grace_minutes=60)
    src = _source()
    store.get_or_create_session(src)
    key = store._generate_session_key(src)
    store._entries[key].expiry_finalized = True  # watcher had finalized it
    _age_minutes(store, src, 5)
    resumed = store.get_or_create_session(src)
    assert resumed.expiry_finalized is False     # re-armed for next expiry
