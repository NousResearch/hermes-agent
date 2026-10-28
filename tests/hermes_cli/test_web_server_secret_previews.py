"""Dashboard credential previews are keyed ``***#xxxxxx`` fingerprints that reveal no character
of the secret, and a copied preview can never be saved back as the secret."""

import re

import pytest

from hermes_cli.config import load_env, redact_key, save_env_value

_FP = re.compile(r"\*\*\*#[0-9a-f]{6}")


class TestRedactKey:
    def test_never_shows_key_characters(self):
        result = redact_key("sk-" + "1234567890abcdef")
        assert _FP.fullmatch(result), result
        assert "sk-1" not in result and "cdef" not in result

    def test_short_key_also_fingerprinted(self):
        assert _FP.fullmatch(redact_key("short"))

    def test_doubled_password_not_exposed(self):
        assert "Pass" not in redact_key("Password" * 2)


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def test_env_preview_has_no_secret_characters(client):
    save_env_value("EXA_API_KEY", "Password" * 2)
    preview = client.get("/api/env").json()["EXA_API_KEY"]["redacted_value"]
    assert re.fullmatch(r"«redacted:\*\*\*#[0-9a-f]{6}»", preview), preview
    bare = preview[len("«redacted:"):-1]
    assert client.put("/api/env", json={"key": "EXA_API_KEY", "value": bare}).status_code == 400
    assert load_env()["EXA_API_KEY"] == "Password" * 2
