from gateway.context_guard import (
    ContextGuardConfig,
    build_context_guard_notice,
    resolve_context_guard_config,
    should_guard_context,
)


def test_context_guard_defaults_off():
    cfg = resolve_context_guard_config({}, "telegram")
    assert cfg.enabled is False
    assert cfg.threshold == 0.60
    assert cfg.auto_reset_next_turn is True


def test_context_guard_platform_override_wins():
    cfg = resolve_context_guard_config(
        {
            "gateway": {
                "context_guard": {
                    "enabled": False,
                    "threshold": 0.80,
                    "platforms": {
                        "telegram": {
                            "enabled": True,
                            "threshold": 55,
                            "append_notice": False,
                        }
                    },
                }
            }
        },
        "telegram",
    )
    assert cfg.enabled is True
    assert cfg.threshold == 0.55
    assert cfg.append_notice is False


def test_should_guard_context_uses_threshold():
    cfg = ContextGuardConfig(enabled=True, threshold=0.60)
    assert should_guard_context(
        prompt_tokens=600,
        context_length=1000,
        config=cfg,
    )
    assert not should_guard_context(
        prompt_tokens=599,
        context_length=1000,
        config=cfg,
    )


def test_should_guard_context_respects_disabled_and_missing_usage():
    assert not should_guard_context(
        prompt_tokens=900,
        context_length=1000,
        config=ContextGuardConfig(enabled=False),
    )
    assert not should_guard_context(
        prompt_tokens=0,
        context_length=1000,
        config=ContextGuardConfig(enabled=True),
    )


def test_notice_is_short_and_mentions_fresh_next_message():
    notice = build_context_guard_notice(ContextGuardConfig(enabled=True, threshold=0.55))
    assert "55%" in notice
    assert "next message" in notice
    assert len(notice) < 180
