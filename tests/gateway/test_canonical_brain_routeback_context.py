from gateway.canonical_brain_routeback_context import (
    build_routeback_context_prompt_for_session,
    lookup_routeback_cases_for_thread,
)
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource, build_session_context


class _FakeSock:
    def close(self):
        return None


class _FakeHelper:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def get_secret_value(self):
        return "fake-password"

    def connect(self, password):
        assert password == "fake-password"
        return _FakeSock()

    def sql_quote(self, value):
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    def query(self, sock, sql):
        self.queries.append(sql)
        return {"rows": self.rows}


def _enable_with_rows(monkeypatch, rows):
    import gateway.canonical_brain_routeback_context as ctx

    helper = _FakeHelper(rows)
    monkeypatch.setattr(ctx, "_load_helper", lambda: helper)
    monkeypatch.setattr(ctx, "_helper_available", lambda: True)
    monkeypatch.setattr(ctx, "_routeback_context_enabled", lambda: True)
    return helper


def test_lookup_routeback_cases_requires_current_thread_as_target(monkeypatch):
    rows = [
        {
            "event_id": "evt-source",
            "event_type": "case.note",
            "case_id": "case:mp4",
            "occurred_at": "2026-06-24T08:45:00Z",
            "source": {"source_refs": {"thread_id": "source-thread", "message_id": "m1"}},
            "payload": {"summary": "source requester note"},
        },
        {
            "event_id": "evt-sent",
            "event_type": "route_back.sent",
            "case_id": "case:mp4",
            "occurred_at": "2026-06-24T08:46:00Z",
            "source": {"source_refs": {"thread_id": "source-thread", "message_id": "m2"}},
            "payload": {
                "route_back": {"target_ref": {"id": "owner-thread"}},
                "receipt": {"chat_id": "owner-thread", "message_id": "r1"},
            },
        },
    ]
    helper = _enable_with_rows(monkeypatch, rows)

    contexts = lookup_routeback_cases_for_thread("owner-thread")

    assert len(contexts) == 1
    assert contexts[0].case_id == "case:mp4"
    assert contexts[0].source_thread_id == "source-thread"
    assert "owner-thread" in helper.queries[0]


def test_lookup_routeback_cases_accepts_legacy_delivery_receipt(monkeypatch):
    rows = [
        {
            "event_id": "evt-source",
            "event_type": "handoff.created",
            "case_id": "case:bonus-product-switch",
            "occurred_at": "2026-06-29T07:00:00Z",
            "source": {"source_refs": {"thread_id": "plamenka-thread", "message_id": "m1"}},
            "payload": {"summary": "requester asks for backend resolver handoff"},
        },
        {
            "event_id": "evt-handoff-waiting",
            "event_type": "handoff.waiting",
            "case_id": "case:bonus-product-switch",
            "occurred_at": "2026-06-29T07:10:00Z",
            "source": {"source_refs": {"thread_id": "plamenka-thread", "message_id": "m2"}},
            "payload": {
                "delivery_receipt": {
                    "chat_id": "backend-resolver-thread",
                    "thread_id": "backend-resolver-thread",
                    "message_id": "starter-message",
                },
            },
        },
    ]
    helper = _enable_with_rows(monkeypatch, rows)

    contexts = lookup_routeback_cases_for_thread("backend-resolver-thread")

    assert len(contexts) == 1
    assert contexts[0].case_id == "case:bonus-product-switch"
    assert contexts[0].source_thread_id == "plamenka-thread"
    assert "delivery_receipt" in helper.queries[0]


def test_lookup_routeback_cases_ignores_source_only_match(monkeypatch):
    rows = [
        {
            "event_id": "evt-source",
            "event_type": "case.note",
            "case_id": "case:mp4",
            "occurred_at": "2026-06-24T08:45:00Z",
            "source": {"source_refs": {"thread_id": "source-thread", "message_id": "m1"}},
            "payload": {"summary": "source requester note"},
        },
    ]
    _enable_with_rows(monkeypatch, rows)

    assert lookup_routeback_cases_for_thread("source-thread") == []


def test_prompt_tells_owner_thread_to_continue_case_and_add_next_action(monkeypatch):
    rows = [
        {
            "event_id": "evt-source",
            "event_type": "case.note",
            "case_id": "case:video-mp4",
            "occurred_at": "2026-06-24T08:45:00Z",
            "source": {"source_refs": {"thread_id": "plamenka-thread", "message_id": "m1"}},
            "payload": {"summary": "requester needs MP4"},
        },
        {
            "event_id": "evt-route",
            "event_type": "route_back.sent",
            "case_id": "case:video-mp4",
            "occurred_at": "2026-06-24T08:46:00Z",
            "source": {"source_refs": {"thread_id": "plamenka-thread", "message_id": "m2"}},
            "payload": {
                "route_back": {"target_ref": {"id": "owner-thread"}},
                "receipt": {"chat_id": "owner-thread", "message_id": "r1"},
            },
        },
    ]
    _enable_with_rows(monkeypatch, rows)
    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake")},
    )
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        chat_name="Adventico / control-tower",
        chat_type="thread",
        thread_id="owner-thread",
        user_name="Emil",
    )

    prompt = build_routeback_context_prompt_for_session(build_session_context(source, config))

    assert "Canonical Brain Route-Back Context" in prompt
    assert "`case:video-mp4`" in prompt
    assert "`plamenka-thread`" in prompt
    assert "do not create a new duplicate case" in prompt
    assert "durable case state before any requester closeout" in prompt
    assert "at most once" in prompt
    assert "Do not use cron for immediate route-back delivery" in prompt
    assert "Do not repeat the owner/resolver request" in prompt
    assert "concrete next-action artifact" in prompt
    assert "email subject/body" in prompt
    assert "forward/notify the requester" in prompt
    assert "not a terminal outcome" in prompt
