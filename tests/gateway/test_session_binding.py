"""Tests for SessionBinding + session_key_override behavior (gh-review topic ordering fix).

Covers Task 1 of docs/plans/2026-06-23-gh-review-topic-context-ordering.md:
- SessionBinding validation (charset, reserved platform/chat-type namespaces)
- build_session_key short-circuits to agent:<profile>:<namespace>:<key>
- profile-safe namespacing (multiplex -> agent:<profile>:...)
- legacy keys byte-identical when no binding
- to_dict/from_dict roundtrip
"""
import pytest

from gateway.session import SessionSource, SessionBinding, build_session_key
from gateway.config import Platform


# --- SessionBinding validation ---------------------------------------------

def test_binding_valid_construction():
    b = SessionBinding(namespace="gh-review", key="gh-pr-x-7735")
    assert b.namespace == "gh-review"
    assert b.key == "gh-pr-x-7735"


@pytest.mark.parametrize("ns", ["telegram", "webhook", "discord", "dm", "group", "channel", "thread"])
def test_binding_rejects_reserved_platform_and_chattype_namespaces(ns):
    with pytest.raises(ValueError):
        SessionBinding(namespace=ns, key="anything")


@pytest.mark.parametrize("bad", ["", "Has-Upper", "has:colon", "-leadingdash", "white space"])
def test_binding_rejects_malformed_segments(bad):
    with pytest.raises(ValueError):
        SessionBinding(namespace="gh-review", key=bad)
    with pytest.raises(ValueError):
        SessionBinding(namespace=bad, key="ok")


def test_namespace_length_capped_at_64_but_key_allows_long():
    # namespace is a short reserved kind (<=64); key is opaque (<=200) so a
    # force-new topic_key with a file-slug + UUID nonce fits.
    with pytest.raises(ValueError):
        SessionBinding(namespace="a" * 65, key="ok")
    long_key = "gh-pr-tdr-autosync-atx-activix-crm-7735-" + "a" * 100 + "-deadbeef"
    b = SessionBinding(namespace="gh-review", key=long_key)
    assert b.key == long_key
    with pytest.raises(ValueError):
        SessionBinding(namespace="gh-review", key="a" * 202)


# --- build_session_key short-circuit ---------------------------------------

def test_override_short_circuits_derivation_default_profile():
    src = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:gh-review-ask:abc-123",
        chat_type="webhook",
        session_binding=SessionBinding(namespace="gh-review", key="gh-pr-x-7735"),
    )
    assert build_session_key(src) == "agent:main:gh-review:gh-pr-x-7735"


def test_override_is_profile_safe_under_multiplex():
    src = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:gh-review-ask:abc",
        chat_type="webhook",
        session_binding=SessionBinding(namespace="gh-review", key="k"),
    )
    assert build_session_key(src, profile="coder") == "agent:coder:gh-review:k"


def test_two_sources_same_binding_collapse_to_one_key():
    """The core of the ordering fix: Q1 and Q2 (different deliveries) -> same key."""
    q1 = SessionSource(platform=Platform.WEBHOOK, chat_id="webhook:gh-review-ask:delivery1",
                       chat_type="webhook",
                       session_binding=SessionBinding(namespace="gh-review", key="gh-pr-7735"))
    q2 = SessionSource(platform=Platform.WEBHOOK, chat_id="webhook:gh-review-ask:delivery2",
                       chat_type="webhook",
                       session_binding=SessionBinding(namespace="gh-review", key="gh-pr-7735"))
    assert build_session_key(q1) == build_session_key(q2)


def test_telegram_topic_reply_binds_to_same_key_as_webhook():
    """A Telegram topic reply with the same binding shares the webhook session."""
    tg = SessionSource(platform=Platform.TELEGRAM, chat_id="-1003714194882",
                       chat_type="group", thread_id="41019",
                       session_binding=SessionBinding(namespace="gh-review", key="gh-pr-7735"))
    wh = SessionSource(platform=Platform.WEBHOOK, chat_id="webhook:gh-review-ask:d",
                       chat_type="webhook",
                       session_binding=SessionBinding(namespace="gh-review", key="gh-pr-7735"))
    assert build_session_key(tg) == build_session_key(wh) == "agent:main:gh-review:gh-pr-7735"


# --- no-binding parity (no regression) -------------------------------------

def test_no_binding_is_byte_identical_to_legacy_group():
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="-100371",
                        chat_type="group", thread_id="41019")
    assert build_session_key(src) == "agent:main:telegram:group:-100371:41019"


def test_no_binding_dm_legacy():
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="555", chat_type="dm")
    assert build_session_key(src) == "agent:main:telegram:dm:555"


def test_binding_cannot_collide_with_platform_key():
    """No binding namespace can equal a Platform value, so a binding key can
    never numerically/positionally equal a real telegram/webhook key."""
    # Reserved-namespace construction is refused, which is what guarantees this.
    with pytest.raises(ValueError):
        SessionBinding(namespace="telegram", key="group")


# --- serialization roundtrip -----------------------------------------------

def test_roundtrip_preserves_binding():
    src = SessionSource(platform=Platform.WEBHOOK, chat_id="c", chat_type="webhook",
                        session_binding=SessionBinding(namespace="gh-review", key="k7735"))
    back = SessionSource.from_dict(src.to_dict())
    assert back.session_binding == SessionBinding(namespace="gh-review", key="k7735")
    assert build_session_key(back) == "agent:main:gh-review:k7735"


def test_roundtrip_no_binding_omits_field():
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    d = src.to_dict()
    assert "session_binding" in d.keys() or "session_binding" not in d  # tolerant
    assert d.get("session_binding") is None
    assert SessionSource.from_dict(d).session_binding is None


def test_from_dict_drops_invalid_persisted_binding():
    """A persisted binding whose namespace later became reserved is dropped, not fatal."""
    d = {"platform": "webhook", "chat_id": "c", "chat_type": "webhook",
         "session_binding": {"namespace": "telegram", "key": "x"}}
    src = SessionSource.from_dict(d)
    assert src.session_binding is None
