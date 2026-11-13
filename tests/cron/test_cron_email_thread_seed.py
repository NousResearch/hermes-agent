"""An email cron thread seed must key exactly like the reply that continues it.

The email adapter builds every inbound source as a DM from the sender, threaded on the
root Message-ID, so the seed for an email thread lands on the ``dm`` slot. Otherwise
its session key differs from the reply's and the reply starts a fresh session.
"""

from unittest.mock import MagicMock, patch

from cron.scheduler_delivery import _THREAD_REPLY_CHAT_TYPE, _seed_cron_thread_session
from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def test_email_seed_key_matches_inbound_reply():
    store = MagicMock()
    adapter = MagicMock()
    adapter._session_store = store

    with patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
        _seed_cron_thread_session(
            {"id": "j1", "name": "Report"}, adapter, "email",
            "user@example.com", "hermes-abc123@example.com", "report body",
        )

    store.get_or_create_session.assert_called_once()
    seeded_source = store.get_or_create_session.call_args[0][0]
    assert seeded_source.chat_type == "dm"
    assert seeded_source.thread_id == "hermes-abc123@example.com"
    seed_key = build_session_key(seeded_source)

    # What a reply in that thread resolves to: the adapter builds every
    # inbound email source as a DM from the sender, threaded on the root.
    inbound = SessionSource(
        platform=Platform.EMAIL, chat_id="user@example.com", chat_type="dm",
        user_id="user@example.com", thread_id="hermes-abc123@example.com",
    )
    assert seed_key == build_session_key(inbound), (
        f"seed key {seed_key} != inbound reply key {build_session_key(inbound)}"
    )
    mirror_mock.assert_called_once()
    assert mirror_mock.call_args.kwargs.get("thread_id") == "hermes-abc123@example.com"


def test_thread_reply_chat_type_table_has_email_dm():
    """Email seeds the DM lane; the group-slot platforms are unchanged."""
    assert _THREAD_REPLY_CHAT_TYPE["email"] == "dm"
    assert _THREAD_REPLY_CHAT_TYPE["slack"] == "group"
    assert _THREAD_REPLY_CHAT_TYPE["matrix"] == "group"
    assert _THREAD_REPLY_CHAT_TYPE["telegram"] == "group"


def test_email_target_with_capitals_keys_like_the_reply():
    """A deliver target written with capitals resolves lowercase, as the adapter lowercases every inbound
    sender, so the thread seeded for the report and the reply share one session key."""
    from cron.scheduler_delivery import _resolve_delivery_targets
    from plugins.platforms.email.adapter import _extract_email_address

    targets = _resolve_delivery_targets({"id": "j1", "deliver": "email:User@Example.COM"})
    assert [t["chat_id"] for t in targets] == ["user@example.com"]
    with patch("cron.scheduler_delivery._get_home_target_chat_id", return_value="User@Example.COM"):
        home = _resolve_delivery_targets({"id": "j2", "deliver": "email"})
    assert [t["chat_id"] for t in home] == ["user@example.com"]

    store = MagicMock()
    adapter = MagicMock()
    adapter._session_store = store
    with patch("gateway.mirror.mirror_to_session", return_value=True):
        _seed_cron_thread_session(
            {"id": "j1", "name": "Report"}, adapter, "email",
            targets[0]["chat_id"], "hermes-abc123@example.com", "report body",
        )
    sender = _extract_email_address("User <User@Example.COM>")
    inbound = SessionSource(
        platform=Platform.EMAIL, chat_id=sender, chat_type="dm",
        user_id=sender, thread_id="hermes-abc123@example.com",
    )
    assert build_session_key(store.get_or_create_session.call_args[0][0]) == build_session_key(inbound)
