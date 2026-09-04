"""Host-declared channel ownership on the dashboard Channels surface.

A hosting layer stamps ``HERMES_MANAGED_PLATFORMS``; the dashboard then reports
those channels as managed and refuses every write path that could change them.
Without the stamp nothing changes, down to the payload shape.
"""
import pytest
import yaml


VALID_TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_1234"
PORTAL_URL = "https://portal.example.com"


@pytest.fixture
def homes(tmp_path, monkeypatch, _isolate_hermes_home):
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_alpha"
    for home in (default_home, worker_home):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
        (home / ".env").write_text("", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker_alpha": worker_home}


@pytest.fixture
def client(monkeypatch, homes):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


@pytest.fixture
def managed(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED_PLATFORMS", "telegram:native,discord:relay")
    monkeypatch.setenv("HERMES_MANAGED_PLATFORMS_LABEL", "Nous Portal")
    monkeypatch.setenv("HERMES_DASHBOARD_PORTAL_URL", PORTAL_URL)


def _platform(payload, platform_id):
    return next(p for p in payload["platforms"] if p["id"] == platform_id)


def _files_unchanged(home):
    return (
        (home / ".env").read_text(encoding="utf-8") == ""
        and yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) == {}
    )


class TestReads:
    def test_payload_shape_is_unchanged_without_the_stamp(self, client):
        payload = client.get("/api/messaging/platforms").json()
        assert "managed_by" not in payload
        assert all("managed" not in p for p in payload["platforms"])

    def test_declared_platforms_carry_their_record(self, client, managed):
        payload = client.get("/api/messaging/platforms").json()
        assert payload["managed_by"] == {"label": "Nous Portal", "url": PORTAL_URL}
        assert _platform(payload, "telegram")["managed"] == {
            "kind": "native",
            "label": "Nous Portal",
            "url": PORTAL_URL,
        }
        assert _platform(payload, "discord")["managed"]["kind"] == "relay"
        assert _platform(payload, "slack")["managed"] is None


class TestChannelWrites:
    def test_platform_update_is_refused_before_any_write(self, client, managed, homes):
        resp = client.put(
            "/api/messaging/platforms/telegram",
            json={"enabled": True, "env": {"TELEGRAM_BOT_TOKEN": VALID_TOKEN}},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"] == "Telegram is managed by Nous Portal."
        assert _files_unchanged(homes["default"])

    def test_unmanaged_platform_still_writable(self, client, managed, homes):
        resp = client.put("/api/messaging/platforms/slack", json={"enabled": False})
        assert resp.status_code == 200
        cfg = yaml.safe_load((homes["default"] / "config.yaml").read_text())
        assert cfg["platforms"]["slack"]["enabled"] is False

    def test_lock_applies_to_every_profile(self, client, managed, homes):
        resp = client.put(
            "/api/messaging/platforms/telegram",
            params={"profile": "worker_alpha"},
            json={"enabled": True, "env": {"TELEGRAM_BOT_TOKEN": VALID_TOKEN}},
        )
        assert resp.status_code == 409
        assert _files_unchanged(homes["worker_alpha"])

    def test_telegram_onboarding_refused_without_side_effects(
        self, client, managed, monkeypatch
    ):
        import hermes_cli.web_routers.messaging as messaging_router
        from hermes_cli import web_server_messaging

        async def _no_network(*args, **kwargs):
            raise AssertionError("the setup service must not be contacted")

        monkeypatch.setattr(messaging_router, "_telegram_onboarding_request", _no_network)

        start = client.post("/api/messaging/telegram/onboarding/start", json={})
        assert start.status_code == 409
        assert web_server_messaging._telegram_onboarding_pairings == {}

        apply = client.post(
            "/api/messaging/telegram/onboarding/some-id/apply",
            json={"allowed_user_ids": ["123"]},
        )
        assert apply.status_code == 409

    def test_whatsapp_onboarding_refused_without_side_effects(self, client, monkeypatch):
        from hermes_cli import web_server_messaging

        monkeypatch.setenv("HERMES_MANAGED_PLATFORMS", "whatsapp:native")

        start = client.post(
            "/api/messaging/whatsapp/onboarding/start", json={"mode": "bot"}
        )
        assert start.status_code == 409
        assert web_server_messaging._whatsapp_onboarding_sessions == {}

        apply = client.post(
            "/api/messaging/whatsapp/onboarding/some-id/apply", json={}
        )
        assert apply.status_code == 409


class TestConnectionTest:
    def test_native_test_still_runs(self, client, managed):
        resp = client.post("/api/messaging/platforms/telegram/test")
        assert resp.status_code == 200

    def test_relay_test_is_refused(self, client, managed):
        resp = client.post("/api/messaging/platforms/discord/test")
        assert resp.status_code == 409
        assert "Nous Portal" in resp.json()["detail"]


class TestGenericWriters:
    @pytest.mark.parametrize(
        "key",
        [
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_ALLOWED_USERS",
            # Hidden from the setup card but honoured by the gateway.
            "TELEGRAM_ALLOW_ALL_USERS",
            "TELEGRAM_HOME_CHANNEL",
        ],
    )
    def test_env_routes_refuse_the_managed_platform_namespace(
        self, client, managed, homes, key
    ):
        put = client.put("/api/env", json={"key": key, "value": "anything"})
        assert put.status_code == 409
        delete = client.request("DELETE", "/api/env", json={"key": key})
        assert delete.status_code == 409
        assert _files_unchanged(homes["default"])

    def test_env_routes_leave_other_keys_alone(self, client, managed, homes):
        resp = client.put(
            "/api/env", json={"key": "SLACK_BOT_TOKEN", "value": "xoxb-not-managed"}
        )
        assert resp.status_code == 200
        assert "SLACK_BOT_TOKEN=xoxb-not-managed" in (
            homes["default"] / ".env"
        ).read_text(encoding="utf-8")

    def test_direct_platforms_switch_is_owned_only_with_a_relay_platform(
        self, client, monkeypatch, homes
    ):
        monkeypatch.setenv("HERMES_MANAGED_PLATFORMS", "telegram:native,discord:relay")
        refused = client.put(
            "/api/env",
            json={"key": "GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS", "value": "true"},
        )
        assert refused.status_code == 409

        monkeypatch.setenv("HERMES_MANAGED_PLATFORMS", "telegram:native")
        allowed = client.put(
            "/api/env",
            json={"key": "GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS", "value": "true"},
        )
        assert allowed.status_code == 200

    @pytest.mark.parametrize(
        "key", ["HERMES_MANAGED_PLATFORMS", "HERMES_MANAGED_PLATFORMS_LABEL"]
    )
    def test_the_stamps_themselves_cannot_be_set(self, client, homes, key):
        put = client.put("/api/env", json={"key": key, "value": ""})
        assert put.status_code == 400
        assert "denylist" in put.json()["detail"]
        assert _files_unchanged(homes["default"])

    @pytest.mark.parametrize(
        "config",
        [
            {"platforms": {"telegram": {"enabled": False}}},
            {"gateway": {"platforms": {"telegram": {"enabled": False}}}},
            {"gateway": {"telegram": {"enabled": False}}},
        ],
    )
    def test_config_form_refuses_every_managed_platform_location(
        self, client, managed, homes, config
    ):
        refused = client.put("/api/config", json={"config": config})
        assert refused.status_code == 409
        assert _files_unchanged(homes["default"])

    def test_config_form_leaves_other_platforms_alone(self, client, managed):
        allowed = client.put(
            "/api/config",
            json={"config": {"platforms": {"slack": {"enabled": False}}}},
        )
        assert allowed.status_code == 200

    @pytest.mark.parametrize(
        "yaml_text",
        [
            "platforms:\n  telegram:\n    enabled: false\n",
            "gateway:\n  platforms:\n    telegram:\n      enabled: false\n",
            "gateway:\n  telegram:\n    enabled: false\n",
        ],
    )
    def test_raw_editor_refuses_every_managed_platform_location(
        self, client, managed, homes, yaml_text
    ):
        refused = client.put("/api/config/raw", json={"yaml_text": yaml_text})
        assert refused.status_code == 409
        assert _files_unchanged(homes["default"])

    def test_raw_editor_leaves_other_platforms_alone(self, client, managed):
        allowed = client.put(
            "/api/config/raw",
            json={"yaml_text": "platforms:\n  slack:\n    enabled: false\n"},
        )
        assert allowed.status_code == 200
