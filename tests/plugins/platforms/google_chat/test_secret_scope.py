from types import SimpleNamespace

import pytest

from agent import secret_scope
from agent.secret_scope import UnscopedSecretError
from plugins.platforms.google_chat import adapter as google_chat


@pytest.fixture(autouse=True)
def _reset_secret_scope():
    secret_scope.set_multiplex_active(False)
    yield
    secret_scope.set_multiplex_active(False)


def test_env_enablement_uses_profile_secret_scope_not_process_env(monkeypatch):
    monkeypatch.setattr(google_chat, "check_google_chat_requirements", lambda: True)
    monkeypatch.setenv("GOOGLE_CHAT_PROJECT_ID", "foreign-project")
    monkeypatch.setenv(
        "GOOGLE_CHAT_SUBSCRIPTION_NAME",
        "projects/foreign-project/subscriptions/foreign-sub",
    )
    monkeypatch.setenv("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON", '{"project_id":"foreign"}')
    monkeypatch.setenv("GOOGLE_CHAT_HOME_CHANNEL", "spaces/FOREIGN")
    monkeypatch.setenv("GOOGLE_CHAT_HTTP_EVENTS_URL", "https://foreign.example/events")
    monkeypatch.setenv("GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE", "foreign-audience")
    monkeypatch.setenv(
        "GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL",
        "foreign@example.iam.gserviceaccount.com",
    )

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({
        "GOOGLE_CHAT_PROJECT_ID": "scoped-project",
        "GOOGLE_CHAT_SUBSCRIPTION_NAME": "projects/scoped-project/subscriptions/scoped-sub",
        "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON": '{"project_id":"scoped"}',
        "GOOGLE_CHAT_HOME_CHANNEL": "spaces/SCOPED",
        "GOOGLE_CHAT_HOME_CHANNEL_NAME": "Scoped Home",
        "GOOGLE_CHAT_HTTP_EVENTS_URL": "https://scoped.example/events",
        "GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE": "scoped-audience",
        "GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL": "scoped@example.iam.gserviceaccount.com",
    })
    try:
        seed = google_chat._env_enablement()
    finally:
        secret_scope.reset_secret_scope(token)

    assert seed == {
        "project_id": "scoped-project",
        "subscription_name": "projects/scoped-project/subscriptions/scoped-sub",
        "http_events_url": "https://scoped.example/events",
        "http_events_audience": "scoped-audience",
        "http_events_service_account_email": "scoped@example.iam.gserviceaccount.com",
        "service_account_json": '{"project_id":"scoped"}',
        "home_channel": {"chat_id": "spaces/SCOPED", "name": "Scoped Home"},
    }


def test_env_enablement_unscoped_multiplex_read_fails_closed(monkeypatch):
    monkeypatch.setattr(google_chat, "check_google_chat_requirements", lambda: True)
    monkeypatch.setenv("GOOGLE_CHAT_PROJECT_ID", "foreign-project")
    monkeypatch.setenv(
        "GOOGLE_CHAT_SUBSCRIPTION_NAME",
        "projects/foreign-project/subscriptions/foreign-sub",
    )

    secret_scope.set_multiplex_active(True)

    with pytest.raises(UnscopedSecretError):
        google_chat._env_enablement()


def test_registry_check_ignores_foreign_http_events_url(monkeypatch):
    monkeypatch.setattr(google_chat, "check_google_chat_requirements", lambda: True)
    monkeypatch.setenv("GOOGLE_CHAT_HTTP_EVENTS_URL", "https://foreign.example/events")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        enabled = google_chat._check_for_registry()
    finally:
        secret_scope.reset_secret_scope(token)

    assert enabled is False


def test_env_enablement_ignores_foreign_http_events_url_in_empty_scope(monkeypatch):
    monkeypatch.setenv("GOOGLE_CHAT_HTTP_EVENTS_URL", "https://foreign.example/events")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        seed = google_chat._env_enablement()
    finally:
        secret_scope.reset_secret_scope(token)

    assert seed is None


def test_adapter_http_events_fallback_uses_profile_secret_scope(monkeypatch):
    monkeypatch.setattr(google_chat, "_load_google_modules", lambda: True)
    monkeypatch.setenv("GOOGLE_CHAT_HTTP_EVENTS_URL", "https://foreign.example/events")
    monkeypatch.setenv("GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE", "foreign-audience")
    monkeypatch.setenv(
        "GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL",
        "foreign@example.iam.gserviceaccount.com",
    )

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({
        "GOOGLE_CHAT_HTTP_EVENTS_URL": "https://scoped.example/events",
        "GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE": "scoped-audience",
        "GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL": "scoped@example.iam.gserviceaccount.com",
    })
    try:
        adapter = google_chat.GoogleChatAdapter(google_chat.PlatformConfig(enabled=True))
    finally:
        secret_scope.reset_secret_scope(token)

    assert adapter._http_events_url == "https://scoped.example/events"
    assert adapter._http_events_audience == "scoped-audience"
    assert adapter._http_events_service_account_email == "scoped@example.iam.gserviceaccount.com"


def test_load_sa_credentials_uses_profile_secret_scope_not_process_env(monkeypatch):
    class _FakeCredentials:
        @staticmethod
        def from_service_account_info(info, scopes):
            return {"info": info, "scopes": scopes}

    monkeypatch.setattr(
        google_chat,
        "service_account",
        SimpleNamespace(Credentials=_FakeCredentials),
    )
    monkeypatch.setenv("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON", '{"project_id":"foreign"}')

    instance = google_chat.GoogleChatAdapter.__new__(google_chat.GoogleChatAdapter)
    instance.config = SimpleNamespace(extra={})
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({
        "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON": '{"project_id":"scoped"}'
    })
    try:
        credentials = google_chat.GoogleChatAdapter._load_sa_credentials(instance)
    finally:
        secret_scope.reset_secret_scope(token)

    assert credentials["info"] == {"project_id": "scoped"}
    assert credentials["scopes"] == google_chat._CHAT_SCOPES


@pytest.mark.asyncio
async def test_standalone_send_skips_foreign_process_adc_in_multiplex(monkeypatch):
    import google.auth

    calls = []

    def unexpected_default(**kwargs):
        calls.append(kwargs)
        raise AssertionError("foreign process ADC must not be used")

    monkeypatch.setattr(
        google_chat,
        "service_account",
        SimpleNamespace(Credentials=object()),
    )
    monkeypatch.setattr(google.auth, "default", unexpected_default)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/foreign/profile.json")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        result = await google_chat._standalone_send(
            SimpleNamespace(extra={}),
            "spaces/SAFE",
            "hello",
        )
    finally:
        secret_scope.reset_secret_scope(token)

    assert calls == []
    assert "ADC skipped for this profile" in result["error"]


@pytest.mark.asyncio
async def test_standalone_send_allows_workload_identity_adc_in_multiplex(monkeypatch):
    import google.auth

    calls = []

    def marker_default(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("workload identity ADC reached")

    monkeypatch.setattr(
        google_chat,
        "service_account",
        SimpleNamespace(Credentials=object()),
    )
    monkeypatch.setattr(google.auth, "default", marker_default)
    monkeypatch.delenv("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        result = await google_chat._standalone_send(
            SimpleNamespace(extra={}),
            "spaces/SAFE",
            "hello",
        )
    finally:
        secret_scope.reset_secret_scope(token)

    assert calls == [{"scopes": google_chat._CHAT_SCOPES}]
    assert "workload identity ADC reached" in result["error"]
