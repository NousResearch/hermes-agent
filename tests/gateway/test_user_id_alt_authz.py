"""Authorization regression tests for alternate platform user IDs."""
from __future__ import annotations

from types import SimpleNamespace

from gateway.session import Platform, SessionSource


def _make_bare_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def test_allowlist_matches_user_id_alt(monkeypatch):
    """Adapters may sanitize user_id for session keys while preserving raw IDs in user_id_alt."""
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "users/ronny")
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="room1",
        chat_type="group",
        user_id="users_ronny",
        user_id_alt="users/ronny",
        user_name="ronny",
    )

    assert runner._is_user_authorized(source) is True


def test_allowlist_still_denies_when_neither_user_id_nor_alt_match(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "users/ronny")
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="room1",
        chat_type="group",
        user_id="users_eve",
        user_id_alt="users/eve",
        user_name="eve",
    )

    assert runner._is_user_authorized(source) is False
