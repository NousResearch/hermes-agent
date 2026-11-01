"""A fully benched Anthropic pool is reported as a cooldown when a session starts, not as a logout.

When every pooled Anthropic credential sits in a 429 cooldown, ``resolve_anthropic_token`` finds
nothing usable. The session-start error used to say "No Anthropic credentials found. Run 'hermes
auth add anthropic' to sign in" — wrong for valid OAuth grants that are merely cooling down, and it
sends the user chasing a re-login. The auxiliary path already names the cooldown
(``pool_cooldown_message``, #56810); session start must say the same.
"""

import json
import time
from pathlib import Path

import pytest

from agent.credential_pool import STATUS_EXHAUSTED
from hermes_cli.auth import AuthError


def _oauth_entry(cred_id: str, label: str, *, status=STATUS_EXHAUSTED, priority=0) -> dict:
    return {
        "provider": "anthropic", "id": cred_id, "label": label, "auth_type": "oauth",
        "priority": priority, "source": "manual:hermes_pkce",
        "access_token": "sk-ant-oat01-" + cred_id * 8, "refresh_token": "sk-ant-ort01-" + cred_id * 8,
        "expires_at_ms": int((time.time() + 3600) * 1000),
        "last_status": status, "last_status_at": time.time(), "last_error_code": 429,
        "last_error_message": "Error code: 429 - rate_limit_error",
    }


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"  # not Path.home()/.hermes: the auth-store guard refuses that
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    (home / "config.yaml").write_text(
        "model:\n  provider: anthropic\n  default: claude-opus-5-5\n"
        "auth:\n  adopt_external_logins: false\n", encoding="utf-8")
    # No Claude Code keychain/file credentials: only the Hermes pool counts.
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda *a, **k: None,
                        raising=False)
    return home


def _write_pool(home: Path, entries: list[dict]) -> None:
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": entries}}), encoding="utf-8")


def test_benched_pool_names_the_cooldown_not_a_missing_login(hermes_home):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _write_pool(hermes_home, [_oauth_entry("a", "personal"), _oauth_entry("b", "work", priority=1)])
    with pytest.raises(AuthError) as excinfo:
        resolve_runtime_provider(requested="anthropic")
    message = str(excinfo.value)
    assert "cooling down" in message
    assert "No Anthropic credentials found" not in message


def test_empty_pool_still_asks_to_sign_in(hermes_home):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _write_pool(hermes_home, [])
    with pytest.raises(AuthError) as excinfo:
        resolve_runtime_provider(requested="anthropic")
    assert "hermes auth add anthropic" in str(excinfo.value)
