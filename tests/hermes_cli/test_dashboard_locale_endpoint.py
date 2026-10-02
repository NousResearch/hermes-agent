"""GET/PUT /api/dashboard/locale — the server-side UI language preference.

The dashboard resolves its language from ``dashboard.locale`` in the profile's
config.yaml so the choice follows the user to any browser. These cover the
round-trip and the allow-list guard (a config hand-edited to an unsupported id
must read back as "unset", and the PUT must reject ids the SPA has no catalog
for rather than persisting them).
"""

import pytest


class TestDashboardLocaleEndpoint:
    @pytest.fixture(autouse=True)
    def _home(self, _isolate_hermes_home):
        pass

    def _client(self):
        from starlette.testclient import TestClient

        from hermes_cli import web_server

        client = TestClient(web_server.app)
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        return client

    def test_round_trip_persists_to_config(self):
        pytest.importorskip("fastapi")
        from hermes_cli.config import load_config

        client = self._client()

        # Unset initially.
        resp = client.get("/api/dashboard/locale")
        assert resp.status_code == 200
        assert resp.json() == {"locale": None}

        resp = client.put("/api/dashboard/locale", json={"locale": "zh"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "locale": "zh"}
        assert (load_config().get("dashboard") or {}).get("locale") == "zh"

        resp = client.get("/api/dashboard/locale")
        assert resp.status_code == 200
        assert resp.json() == {"locale": "zh"}

    def test_unsupported_locale_is_rejected(self):
        pytest.importorskip("fastapi")
        from hermes_cli.config import load_config

        client = self._client()

        resp = client.put("/api/dashboard/locale", json={"locale": "nl"})
        assert resp.status_code == 400
        assert (load_config().get("dashboard") or {}).get("locale") is None

    def test_hand_edited_unsupported_value_reads_back_as_unset(self):
        pytest.importorskip("fastapi")
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg.setdefault("dashboard", {})["locale"] = "xx-not-a-locale"
        save_config(cfg)

        resp = self._client().get("/api/dashboard/locale")
        assert resp.status_code == 200
        assert resp.json() == {"locale": None}
