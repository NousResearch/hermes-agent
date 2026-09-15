"""Dashboard routes that read the SERVING profile's credentials stay inside its scope once this
process is fail-closed for multi-profile hosting (``hermes dashboard`` / ``hermes serve``).

Regression for the class the reported ``UnscopedSecretError`` belongs to: ``_config_profile_scope``
was only entered for a *named* profile, so after any ``?profile=<other>`` request flipped the
process (``tui_gateway/launch_profile_policy.activate_multi_profile_hosting``) every own-profile
handler that reads a secret raised instead of reading the launch profile's frozen env. Those
handlers are fail-soft (``except Exception: _log.exception(...)``), so nothing ever returned 500:
the payload silently lost its credential-derived parts — no gateway platforms in the cron deliver
picker, memory providers reported unavailable, no Portal features, no recommended model default.
"""

from __future__ import annotations

import json
import logging
import os

import pytest

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402

OWN_TOKEN = "telegram-own-profile-token"
FC_KEY = "fc-own-profile-0001"
OTHER_FC_KEY = "fc-other-profile-0002"

# Credentials the serving profile has; every route below reads at least one of them.
OWN_ENV = {
    "TELEGRAM_BOT_TOKEN": OWN_TOKEN,
    "FIRECRAWL_API_KEY": FC_KEY,
    "OPENCODE_GO_BASE_URL": "https://opencode.example/v1",
    "MEM0_MODE": "platform",
    "MEM0_HOST": "https://mem0.example",
}
OTHER_ENV = {"FIRECRAWL_API_KEY": OTHER_FC_KEY}


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes_home"
    other = root / "profiles" / "b"
    other.mkdir(parents=True)
    (root / ".env").write_text("".join(f"{k}={v}\n" for k, v in OWN_ENV.items()), encoding="utf-8")
    (other / ".env").write_text(
        "".join(f"{k}={v}\n" for k, v in OTHER_ENV.items()), encoding="utf-8")
    (root / "config.yaml").write_text(
        "model:\n  provider: opencode-go\n  default: opencode-go/some-model\n"
        "memory:\n  provider: mem0\n"
        "gateway:\n  platforms:\n    telegram:\n      enabled: true\n"
        "      bot_token: ${TELEGRAM_BOT_TOKEN}\n"
        "      home_channel:\n        platform: telegram\n        chat_id: '507817203'\n"
        "        name: Home\n",
        encoding="utf-8")
    (other / "config.yaml").write_text("model:\n  default: openai/gpt-4o-mini\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    # The dashboard process loaded its own .env at boot; the profile dir holds the rest.
    for key, val in OWN_ENV.items():
        monkeypatch.setenv(key, val)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)  # only reachable through the scope
    from agent import secret_scope
    from tui_gateway import launch_profile_policy as lpp
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(lpp, "_snapshot", None)
    from hermes_cli import config as cfg_mod
    for attr in ("_CONFIG_CACHE", "_config_cache"):
        if hasattr(cfg_mod, attr):
            current = getattr(cfg_mod, attr)
            monkeypatch.setattr(cfg_mod, attr, {} if isinstance(current, dict) else None)
    return root, other


@pytest.fixture
def client(two_homes):
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


@pytest.fixture
def flipped(client):
    """A named-profile request, as the dashboard's profile switcher issues — this is what flips the
    process to fail-closed multi-profile hosting."""
    assert client.get("/api/config?profile=b").status_code == 200
    from agent.secret_scope import is_multiplex_active

    assert is_multiplex_active()
    return client


def _unscoped_failures(caplog):
    return [
        rec.getMessage() for rec in caplog.records
        if rec.exc_info and type(rec.exc_info[1]).__name__ == "UnscopedSecretError"
    ]


# Serving-profile read endpoints: every one of these read a secret and swallow the failure.
OWN_PROFILE_ROUTES = [
    "/api/portal",
    "/api/status",
    "/api/cron/blueprints",
    "/api/cron/delivery-targets",
    "/api/memory",
    "/api/dashboard/plugins/hub",
    "/api/credentials/pool",
    "/api/model/recommended-default?provider=opencode-go",
    "/api/plugins/kanban/home-channels",
    "/api/plugins/kanban/model-options",
]


def test_own_profile_routes_do_not_raise_unscoped_secret_error(flipped, caplog):
    payloads = {}
    with caplog.at_level(logging.DEBUG):
        for path in OWN_PROFILE_ROUTES:
            resp = flipped.get(path)
            assert resp.status_code == 200, f"{path}: {resp.status_code} {resp.text[:400]}"
            payloads[path] = resp.json()

    assert _unscoped_failures(caplog) == []

    # The platform list is built from the serving profile's .env token, which is NOT in
    # os.environ: it can only arrive through the profile secret scope.
    assert "TELEGRAM_BOT_TOKEN" not in os.environ
    targets = {t["id"] for t in payloads["/api/cron/delivery-targets"]["targets"]}
    assert "telegram" in targets, payloads["/api/cron/delivery-targets"]

    # Same read through the plugin route (live GatewayConfig, env overlay honored).
    homes = payloads["/api/plugins/kanban/home-channels"]["home_channels"]
    assert [(h["platform"], h["chat_id"]) for h in homes] == [("telegram", "507817203")], homes

    # Credential-derived payloads are populated, not emptied by a swallowed raise.
    assert payloads["/api/plugins/kanban/model-options"]["providers"], payloads[
        "/api/plugins/kanban/model-options"]
    assert payloads["/api/memory"]["providers"], payloads["/api/memory"]
    assert payloads["/api/portal"]["features"], payloads["/api/portal"]

    # Nothing anywhere came from profile B.
    assert OTHER_FC_KEY not in json.dumps(payloads)
    assert os.environ.get("FIRECRAWL_API_KEY") == FC_KEY


def test_named_profile_request_still_bounds_the_next_own_profile_read(client, caplog):
    """The flip is sticky: the request AFTER the named-profile one must still be scoped."""
    with caplog.at_level(logging.DEBUG):
        assert client.get("/api/config?profile=b").status_code == 200
        first = client.get("/api/cron/delivery-targets").json()
        second = client.get("/api/cron/delivery-targets").json()

    assert _unscoped_failures(caplog) == []
    assert first == second
    assert "telegram" in {t["id"] for t in second["targets"]}, second


def test_credential_pool_mutations_are_own_profile_scoped(flipped, caplog):
    """``load_pool()`` re-seeds from the serving profile's env on every call — the exact call path
    of the reported ``OPENCODE_GO_BASE_URL`` traceback (``ops.py`` → ``load_pool`` →
    ``_seed_from_env`` → ``get_env_prefer_dotenv``)."""
    with caplog.at_level(logging.DEBUG):
        added = flipped.post(
            "/api/credentials/pool",
            json={"provider": "opencode-go", "api_key": "sk-test-own-profile"},
        )
        assert added.status_code == 200, added.text
        listed = flipped.get("/api/credentials/pool").json()
        removed = flipped.delete("/api/credentials/pool/opencode-go/1")

    assert _unscoped_failures(caplog) == []
    assert "opencode-go" in {p["provider"] for p in listed["providers"]}, listed
    # The row the mutation created is listed with a usable label — not an empty pool with no
    # credential-derived content (no exact label format: that would be a change-detector).
    entries = listed["providers"][0]["entries"]
    assert entries and all(e.get("label") for e in entries), listed
    assert removed.status_code == 200, removed.text
