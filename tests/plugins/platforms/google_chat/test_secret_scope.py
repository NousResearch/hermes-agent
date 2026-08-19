from types import SimpleNamespace

import pytest

from agent import secret_scope
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


def test_env_enablement_unscoped_multiplex_uses_own_process_env(monkeypatch):
    """The DEFAULT profile's config path runs unscoped under multiplexing.

    There ``os.environ`` holds that profile's own values (its ``.env`` was
    loaded into the process env at gateway start), so ``_env_enablement``
    must fall back to it instead of raising ``UnscopedSecretError`` — a
    raise here would leave a configured default profile unserved.
    """
    monkeypatch.setattr(google_chat, "check_google_chat_requirements", lambda: True)
    monkeypatch.setenv("GOOGLE_CHAT_PROJECT_ID", "own-project")
    monkeypatch.setenv(
        "GOOGLE_CHAT_SUBSCRIPTION_NAME",
        "projects/own-project/subscriptions/own-sub",
    )
    monkeypatch.setenv("GOOGLE_CHAT_HTTP_EVENTS_URL", "https://own.example/events")

    secret_scope.set_multiplex_active(True)

    seed = google_chat._env_enablement()

    assert seed == {
        "project_id": "own-project",
        "subscription_name": "projects/own-project/subscriptions/own-sub",
        "http_events_url": "https://own.example/events",
    }


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


@pytest.mark.asyncio
async def test_resolve_bot_user_id_ignores_foreign_bootstrap_spaces_in_empty_scope(
    monkeypatch,
):
    called_spaces = []

    class _FakeMembers:
        def list(self, parent, pageSize):
            called_spaces.append(parent)
            return self

        def execute(self, http=None):
            return {
                "memberships": [
                    {"member": {"type": "BOT", "name": "users/101234567890123456789"}}
                ]
            }

    class _FakeSpaces:
        def members(self):
            return _FakeMembers()

    class _FakeChatApi:
        def spaces(self):
            return _FakeSpaces()

    monkeypatch.setenv("GOOGLE_CHAT_BOOTSTRAP_SPACES", "spaces/FOREIGN")

    monkeypatch.setattr(google_chat, "_load_google_modules", lambda: True)
    instance = google_chat.GoogleChatAdapter(google_chat.PlatformConfig(enabled=True))
    instance._chat_api = _FakeChatApi()
    instance._new_authed_http = lambda: None

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        bot_id = await instance._resolve_bot_user_id()
    finally:
        secret_scope.reset_secret_scope(token)

    assert bot_id is None
    assert called_spaces == []


@pytest.mark.asyncio
async def test_resolve_bot_user_id_uses_scoped_bootstrap_spaces(monkeypatch):
    called_spaces = []

    class _FakeMembers:
        def __init__(self, parent):
            self._parent = parent

        def list(self, parent, pageSize):
            called_spaces.append(parent)
            self._parent = parent
            return self

        def execute(self, http=None):
            if self._parent == "spaces/SCOPED2":
                return {
                    "memberships": [
                        {"member": {"type": "BOT", "name": "users/101234567890123456789"}}
                    ]
                }
            return {"memberships": []}

    class _FakeSpaces:
        def members(self, parent=None):
            return _FakeMembers(parent)

    class _FakeChatApi:
        def spaces(self):
            return _FakeSpaces()

    monkeypatch.setenv("GOOGLE_CHAT_BOOTSTRAP_SPACES", "spaces/FOREIGN")

    monkeypatch.setattr(google_chat, "_load_google_modules", lambda: True)
    instance = google_chat.GoogleChatAdapter(google_chat.PlatformConfig(enabled=True))
    instance._chat_api = _FakeChatApi()
    instance._new_authed_http = lambda: None

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope(
        {"GOOGLE_CHAT_BOOTSTRAP_SPACES": "spaces/SCOPED, spaces/SCOPED2"}
    )
    try:
        bot_id = await instance._resolve_bot_user_id()
    finally:
        secret_scope.reset_secret_scope(token)

    assert bot_id == "users/101234567890123456789"
    assert called_spaces == ["spaces/SCOPED", "spaces/SCOPED2"]


def test_adapter_init_unscoped_multiplex_uses_own_process_env(monkeypatch):
    """The DEFAULT profile constructs its adapter unscoped under multiplexing.

    Its own values live in ``os.environ`` (loaded from its ``.env`` at
    gateway start); a bare ``get_secret`` would raise
    ``UnscopedSecretError`` and crash startup, so the reads must fall back
    to the process env.
    """
    monkeypatch.setattr(google_chat, "_load_google_modules", lambda: True)
    monkeypatch.setenv("GOOGLE_CHAT_HTTP_EVENTS_URL", "https://own.example/events")
    monkeypatch.setenv("GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE", "own-audience")
    monkeypatch.setenv(
        "GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL",
        "own@example.iam.gserviceaccount.com",
    )

    secret_scope.set_multiplex_active(True)

    adapter = google_chat.GoogleChatAdapter(google_chat.PlatformConfig(enabled=True))

    assert adapter._http_events_url == "https://own.example/events"
    assert adapter._http_events_audience == "own-audience"
    assert adapter._http_events_service_account_email == "own@example.iam.gserviceaccount.com"


def test_adapter_init_unscoped_multiplex_empty_optional_scope(monkeypatch):
    """Default-profile multiplex startup with an empty optional secret scope.

    No Google Chat secrets configured anywhere: construction must not raise
    and the optional http-events values must default to empty — never crash
    the default profile's startup over values that are all optional.
    """
    monkeypatch.setattr(google_chat, "_load_google_modules", lambda: True)
    monkeypatch.delenv("GOOGLE_CHAT_HTTP_EVENTS_URL", raising=False)
    monkeypatch.delenv("GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE", raising=False)
    monkeypatch.delenv("GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL", raising=False)

    secret_scope.set_multiplex_active(True)

    adapter = google_chat.GoogleChatAdapter(google_chat.PlatformConfig(enabled=True))

    assert adapter._http_events_url == ""
    assert adapter._http_events_audience == ""
    assert adapter._http_events_service_account_email == ""


def test_load_sa_credentials_unscoped_multiplex_env_fallback(monkeypatch):
    """DEFAULT profile's SA read falls back to its own process env."""

    class _FakeCredentials:
        @staticmethod
        def from_service_account_info(info, scopes):
            return {"info": info, "scopes": scopes}

    monkeypatch.setattr(
        google_chat,
        "service_account",
        SimpleNamespace(Credentials=_FakeCredentials),
    )
    monkeypatch.setenv("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON", '{"project_id":"own"}')

    instance = google_chat.GoogleChatAdapter.__new__(google_chat.GoogleChatAdapter)
    instance.config = SimpleNamespace(extra={})
    secret_scope.set_multiplex_active(True)

    credentials = google_chat.GoogleChatAdapter._load_sa_credentials(instance)

    assert credentials["info"] == {"project_id": "own"}
    assert credentials["scopes"] == google_chat._CHAT_SCOPES


@pytest.mark.asyncio
async def test_resolve_bot_user_id_unscoped_multiplex_uses_own_env_bootstrap_spaces(
    monkeypatch,
):
    """DEFAULT profile's connect-time bootstrap read uses its own env value."""
    called_spaces = []

    class _FakeMembers:
        def list(self, parent, pageSize):
            called_spaces.append(parent)
            return self

        def execute(self, http=None):
            return {
                "memberships": [
                    {"member": {"type": "BOT", "name": "users/101234567890123456789"}}
                ]
            }

    class _FakeSpaces:
        def members(self):
            return _FakeMembers()

    class _FakeChatApi:
        def spaces(self):
            return _FakeSpaces()

    monkeypatch.setenv("GOOGLE_CHAT_BOOTSTRAP_SPACES", "spaces/OWN")

    monkeypatch.setattr(google_chat, "_load_google_modules", lambda: True)
    instance = google_chat.GoogleChatAdapter(google_chat.PlatformConfig(enabled=True))
    instance._chat_api = _FakeChatApi()
    instance._new_authed_http = lambda: None

    secret_scope.set_multiplex_active(True)

    bot_id = await instance._resolve_bot_user_id()

    assert bot_id == "users/101234567890123456789"
    assert called_spaces == ["spaces/OWN"]
