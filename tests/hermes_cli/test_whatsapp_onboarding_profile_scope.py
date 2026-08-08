"""WhatsApp onboarding must start in the profile the dashboard is managing.

``/api/messaging/whatsapp/onboarding`` is in the client's
``PROFILE_SCOPED_PREFIXES``, so the dashboard appends ``?profile=<mgmt>`` to
every call in that family. ``apply`` honoured it (``body.profile or profile or
record.profile``) but ``start`` only ever read ``body.profile`` — which the UI
never sets — so it had no ``profile`` query parameter at all and FastAPI
discarded it.

The consequence was a split onboarding: the Baileys session directory, the
bridge subprocess and the linked ``creds.json`` were created under the LAUNCH
profile, while ``apply`` wrote the WhatsApp config into the MANAGED profile and
restarted that profile's gateway — which then had WhatsApp enabled with no
session on disk.

These assert the observable contract: which home the handler body resolves
against, and which profile is recorded for the follow-up ``apply``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    (root / "profiles" / "coder").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


@pytest.fixture
def client(profile_env):
    import hermes_cli.web_server as ws

    c = TestClient(ws.app)
    c.headers[ws._SESSION_HEADER_NAME] = ws._SESSION_TOKEN
    return c


@pytest.fixture
def captured(monkeypatch):
    """Record the HERMES_HOME the handler body resolves the session path in."""
    import hermes_cli.web_server as ws
    from hermes_constants import get_hermes_home

    seen: dict = {}

    def _session_path():
        seen["home"] = str(get_hermes_home())
        p = get_hermes_home() / "platforms" / "whatsapp" / "session"
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(ws, "_whatsapp_session_path", _session_path)
    # Keep the bridge subprocess out of the test.
    monkeypatch.setattr(ws, "_spawn_whatsapp_pairing", lambda *a, **k: None, raising=False)
    return seen


def _record_for(pairing_id: str):
    import hermes_cli.web_server as ws

    return ws._whatsapp_onboarding_sessions.get(pairing_id)


class TestStartWhatsAppOnboardingProfileScope:
    def test_query_profile_scopes_the_session_directory(
        self, client, profile_env, captured
    ):
        resp = client.post(
            "/api/messaging/whatsapp/onboarding/start?profile=coder",
            json={"mode": "bot"},
        )

        assert resp.status_code == 200
        assert captured["home"] == str(profile_env / "profiles" / "coder")

    def test_query_profile_is_recorded_for_the_follow_up_apply(
        self, client, profile_env, captured
    ):
        resp = client.post(
            "/api/messaging/whatsapp/onboarding/start?profile=coder",
            json={"mode": "bot"},
        )

        pairing_id = resp.json()["pairing_id"]
        record = _record_for(pairing_id)
        assert record is not None
        # apply resolves body.profile or profile or record.profile — a None
        # here is what let start and apply disagree about the target profile.
        assert record.profile == "coder"

    def test_body_profile_still_wins(self, client, profile_env, captured):
        """Back-compat: an explicit body profile keeps precedence."""
        resp = client.post(
            "/api/messaging/whatsapp/onboarding/start?profile=coder",
            json={"mode": "bot", "profile": "default"},
        )

        assert resp.status_code == 200
        assert captured["home"] == str(profile_env)
        assert _record_for(resp.json()["pairing_id"]).profile == "default"

    def test_no_profile_uses_the_launch_home(self, client, profile_env, captured):
        resp = client.post(
            "/api/messaging/whatsapp/onboarding/start",
            json={"mode": "bot"},
        )

        assert resp.status_code == 200
        assert captured["home"] == str(profile_env)
        assert _record_for(resp.json()["pairing_id"]).profile is None
