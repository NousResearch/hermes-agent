from __future__ import annotations

from datetime import datetime

from gateway.config import GatewayConfig, SessionResetPolicy, Platform
from gateway.session import SessionSource, SessionStore


def test_get_reset_policy_prefers_profile_override():
    cfg = GatewayConfig.from_dict(
        {
            "default_reset_policy": {"mode": "none"},
            "reset_by_profile": {
                "admin": {
                    "mode": "always",
                    "notify": False,
                    "post_task_new_button": True,
                }
            },
        }
    )

    policy = cfg.get_reset_policy(platform=Platform.TELEGRAM, session_type="forum", profile="admin")

    assert policy.mode == "always"
    assert policy.notify is False
    assert policy.post_task_new_button is True


def test_always_reset_policy_starts_each_turn_in_fresh_session(tmp_path):
    cfg = GatewayConfig(
        reset_by_profile={
            "task-manager": SessionResetPolicy(mode="always", notify=False),
        }
    )
    store = SessionStore(sessions_dir=tmp_path, config=cfg)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="forum",
        thread_id="107",
        user_id="42",
        profile="task-manager",
    )

    first = store.get_or_create_session(source)
    first.last_prompt_tokens = 123
    first.updated_at = datetime.now()
    store._save()

    second = store.get_or_create_session(source)

    assert second.session_id != first.session_id
    assert second.was_auto_reset is True
    assert second.auto_reset_reason == "always"
    assert second.reset_had_activity is True
