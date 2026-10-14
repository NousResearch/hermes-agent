"""The QQ update-prompt authz key comes from ``build_session_key`` (profile-namespaced), not a
hard-coded ``agent:main:`` literal — a multiplexed secondary bot's clicks were rejected otherwise."""

from __future__ import annotations

from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.qqbot.adapter import QQAdapter
from gateway.platforms.qqbot.keyboards import InteractionEvent
from gateway.session import SessionSource, build_session_key


def _adapter(owner_profile=None):
    adapter = QQAdapter.__new__(QQAdapter)
    adapter.config = SimpleNamespace(extra={})
    adapter.platform = Platform.QQBOT
    if owner_profile:
        adapter._owner_profile = owner_profile
    return adapter


def test_update_prompt_key_is_the_canonical_session_key_per_profile():
    event = InteractionEvent(scene="c2c", user_openid="U1")
    for profile in (None, "ops"):
        adapter = _adapter(profile)
        key = adapter._update_prompt_session_key(event, "U1")
        source = SessionSource(platform=Platform.QQBOT, chat_id="U1", chat_type="c2c", profile=profile)
        assert key == build_session_key(source, profile=profile)
        assert adapter._is_authorized_interaction_for_session(event, key)
    assert _adapter()._update_prompt_session_key(event, "U1") == "agent:main:qqbot:c2c:U1"
