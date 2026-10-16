"""Cron deliver=origin on a non-push surface (#69304).

An api_server turn binds ``async_delivery=False``: its adapter's ``send()`` is a stub, so a
job that captured ``origin.platform="api_server"`` ran fine (``last_status=ok``) while every
fire recorded ``last_delivery_error`` and nothing reached the creator. Both legs are covered:
a non-push session never stamps an origin (so the creation-time local-only notice fires), and
a job already stamped with such an origin resolves to the configured home channel instead.
"""

from gateway.session_context import clear_session_vars, set_session_vars


def test_non_push_session_stamps_no_origin_and_gets_creation_notice():
    from tools.cronjob_job_args import _local_delivery_notice, _origin_from_env

    tokens = set_session_vars(platform="api_server", chat_id="desk-1", session_key="desk-1",
                              async_delivery=False)
    try:
        origin = _origin_from_env()
        notice = _local_delivery_notice({"id": "j", "deliver": "origin", "origin": origin}, None)
    finally:
        clear_session_vars(tokens)
    assert origin is None
    assert notice

    # Control: a push-capable platform keeps its origin.
    tokens = set_session_vars(platform="telegram", chat_id="777", session_key="tg", async_delivery=True)
    try:
        assert _origin_from_env()["platform"] == "telegram"
    finally:
        clear_session_vars(tokens)


def test_stamped_api_server_origin_falls_back_to_home_channel(monkeypatch):
    from cron import scheduler_delivery as sd

    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "12345")
    job = {"id": "old", "deliver": "origin", "origin": {"platform": "api_server", "chat_id": "desk-1"}}
    targets = sd._resolve_delivery_targets(job)
    assert [(t["platform"], t["chat_id"], t["_resolved_from"]) for t in targets] == [
        ("telegram", "12345", "origin_fallback")]
    # Control: a push-capable origin is delivered as written.
    job["origin"] = {"platform": "discord", "chat_id": "42"}
    assert sd._resolve_delivery_targets(job)[0]["platform"] == "discord"


def test_non_push_session_creation_notice_names_home_channel_fallback(monkeypatch):
    """With a home channel configured the rerouted job must still tell the creating client where
    the report goes (the api_server client never sees it); a push-capable session stays silent."""
    from tools.cronjob_job_args import _local_delivery_notice, _origin_from_env

    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "12345")
    tokens = set_session_vars(platform="api_server", chat_id="desk-1", session_key="desk-1",
                              async_delivery=False)
    try:
        notice = _local_delivery_notice({"id": "j", "deliver": "origin", "origin": _origin_from_env()}, None)
    finally:
        clear_session_vars(tokens)
    assert notice and "telegram:12345" in notice

    tokens = set_session_vars(platform="telegram", chat_id="777", session_key="tg", async_delivery=True)
    try:
        assert _local_delivery_notice(
            {"id": "j", "deliver": "origin", "origin": _origin_from_env()}, None) is None
    finally:
        clear_session_vars(tokens)


def test_suggestions_accept_origin_shares_the_non_push_guard():
    from hermes_cli.suggestions_cmd import _resolve_origin

    tokens = set_session_vars(platform="api_server", chat_id="desk-1", session_key="desk-1",
                              async_delivery=False)
    try:
        assert _resolve_origin() is None
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(platform="telegram", chat_id="777", session_key="tg", async_delivery=True)
    try:
        assert _resolve_origin()["chat_id"] == "777"
    finally:
        clear_session_vars(tokens)
