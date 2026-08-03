"""GET /api/env surfaces arbitrary/custom .env keys (not just catalogued ones).

The dashboard Keys page previously rendered only keys present in a catalog
(``OPTIONAL_ENV_VARS`` or the provider catalog); any other key the user had set
in ``.env`` was invisible. This asserts the behavior contract that an
unrecognised on-disk key is surfaced as a ``custom`` row — set, redacted, and
password-masked — so the page can list and manage it, while a catalogued key is
NOT mislabelled custom.
"""

from fastapi.testclient import TestClient

import hermes_cli.web_server as web_server
from hermes_cli.web_server import _SESSION_TOKEN, app

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


def _env_rows(monkeypatch, env_on_disk):
    """Drive GET /api/env with a controlled on-disk env mapping."""
    monkeypatch.setattr(web_server, "load_env", lambda: dict(env_on_disk))
    # Channel-managed key detection reads real config; force empty so the test
    # is hermetic and the custom-key path is exercised directly.
    monkeypatch.setattr(web_server, "_channel_managed_env_keys", lambda: set())
    resp = client.get("/api/env", headers=HEADERS)
    assert resp.status_code == 200
    return resp.json()


def test_unknown_env_key_surfaces_as_custom(monkeypatch):
    rows = _env_rows(monkeypatch, {"MY_CUSTOM_THING": "s3cret-value"})
    assert "MY_CUSTOM_THING" in rows, "unknown .env key not surfaced by /api/env"
    row = rows["MY_CUSTOM_THING"]
    assert row["custom"] is True
    assert row["category"] == "custom"
    assert row["is_set"] is True


def test_custom_key_is_password_masked(monkeypatch):
    """A custom key could hold anything → treated as a secret (redacted)."""
    rows = _env_rows(monkeypatch, {"MY_CUSTOM_THING": "s3cret-value"})
    row = rows["MY_CUSTOM_THING"]
    assert row["is_password"] is True
    # The raw value must never ride in the listing payload.
    assert row["redacted_value"] != "s3cret-value"
    assert "s3cret-value" not in str(row)


def test_catalogued_key_is_not_marked_custom(monkeypatch):
    """A key present in OPTIONAL_ENV_VARS keeps its real category, not custom."""
    rows = _env_rows(monkeypatch, {"HONCHO_API_KEY": "abc123"})
    row = rows["HONCHO_API_KEY"]
    assert row.get("custom") is not True
    assert row["category"] == "tool"


def test_every_row_has_custom_flag(monkeypatch):
    """The ``custom`` field is always present so the SPA can branch on it."""
    rows = _env_rows(monkeypatch, {"MY_CUSTOM_THING": "x"})
    assert all("custom" in row for row in rows.values())
