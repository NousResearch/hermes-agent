import time

from agent import provider_rate_guard as guard


def test_provider_cooldown_records_and_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "_hermes_home", lambda: str(tmp_path))

    guard.record_provider_cooldown(
        "openai-codex",
        error_context={"reason": "usage_limit_reached", "reset_at": time.time() + 60},
    )

    remaining = guard.provider_cooldown_remaining("openai-codex")
    assert remaining is not None
    assert 0 < remaining <= 60

    guard.clear_provider_cooldown("openai-codex")
    assert guard.provider_cooldown_remaining("openai-codex") is None


def test_usage_limit_context_matches_codex_plus_message():
    assert guard.is_usage_limit_context(
        {"message": "HTTP 429: The usage limit has been reached, plan_type=plus"}
    )


def test_usage_limit_context_rejects_generic_rate_limit():
    assert not guard.is_usage_limit_context({"message": "Too many requests"})
