import json
import os

import pytest

from agent import secret_audit as sa

posix_only = pytest.mark.skipif(os.name != "posix", reason="permission bits")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_audit_never_contains_a_secret_value(home):
    (home / ".env").write_text("OPENAI_API_KEY=sk-SUPERSECRET\nHERMES_PORT=8000\nEMPTY_TOKEN=\n")
    (home / "google_token.json").write_text(json.dumps({"refresh_token": "RT-SECRET"}))

    report = sa.audit()

    assert report["env_secrets"] == ["OPENAI_API_KEY"]
    assert "SUPERSECRET" not in json.dumps(report) and "RT-SECRET" not in json.dumps(report)


@posix_only
def test_fix_makes_world_readable_credentials_owner_only(home):
    token = home / "google_token.json"
    token.write_text("{}")
    token.chmod(0o644)
    profile = home / "chrome-debug"
    profile.mkdir()
    profile.chmod(0o755)
    assert sa.audit()["needs_fix"] is True

    after = sa.fix_permissions()

    assert after["needs_fix"] is False and oct(token.stat().st_mode & 0o777) == "0o600"
    assert oct(profile.stat().st_mode & 0o777) == "0o700"
