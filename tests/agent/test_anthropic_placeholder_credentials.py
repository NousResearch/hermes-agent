"""A placeholder Anthropic credential must not shadow a valid lower-priority source.

``resolve_anthropic_token()`` took the first non-blank env value, so a profile ``.env``
carrying a copied console example (``sk-ant...gQAA``) won over the real key stored in a
later source: every call 401'd and the runtime provider silently fell back to a cheap
model.  Placeholder detection is the shared ``hermes_cli.auth.has_usable_secret`` family —
three consecutive dots cannot occur in a base64url API key or in any JWT segment, so the
ellipsis shape is provably not a credential.

Real imports and the production resolution path throughout; ``HOME``/``CLAUDE_CONFIG_DIR``/
``HERMES_HOME`` point into ``tmp_path`` so the borrowed Claude Code login and the credential
pool start genuinely empty instead of being mocked away.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from agent import anthropic_credentials as AA
from agent.anthropic_credentials import resolve_anthropic_token

_PLACEHOLDER = "sk-ant...gQAA"
_REAL_API_KEY = "sk-ant-api03-b7Qw2mZ8kLpR4tYvN1xHcJdFsAeGiOuP3Wa"
_REAL_OAUTH = "sk-ant-oat01-Zx9Kq2Wm7VtRb4HnYc6LdPgSfAe1Uo3I"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Temp HOME/HERMES_HOME + an empty Claude config dir: no borrowed login, no pool."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setattr(AA, "_read_claude_code_credentials_from_keychain", lambda: None)
    for var in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_placeholder_oauth_env_does_not_shadow_api_key(monkeypatch):
    """The reported shape: a copied ``sk-ant...gQAA`` in ANTHROPIC_TOKEN, real key later."""
    monkeypatch.setenv("ANTHROPIC_TOKEN", _PLACEHOLDER)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _REAL_API_KEY)

    assert resolve_anthropic_token() == _REAL_API_KEY


def test_placeholder_anthropic_token_falls_through_to_valid_oauth_env(monkeypatch):
    """Same entry point, second name: a placeholder must not hide CLAUDE_CODE_OAUTH_TOKEN."""
    monkeypatch.setenv("ANTHROPIC_TOKEN", _PLACEHOLDER)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _REAL_OAUTH)

    assert resolve_anthropic_token() == _REAL_OAUTH


def test_placeholder_env_does_not_shadow_a_valid_pool_credential(monkeypatch):
    """Lower-priority source: a valid pooled OAuth row is reached past the env placeholder."""
    from agent.credential_pool import AUTH_TYPE_OAUTH, SOURCE_MANUAL, PooledCredential, load_pool

    load_pool("anthropic").add_entry(PooledCredential(
        provider="anthropic", id=uuid.uuid4().hex[:6], label="real-row",
        auth_type=AUTH_TYPE_OAUTH, priority=0, source=SOURCE_MANUAL,
        access_token=_REAL_OAUTH,
    ))
    monkeypatch.setenv("ANTHROPIC_TOKEN", _PLACEHOLDER)

    assert resolve_anthropic_token() == _REAL_OAUTH


def test_real_looking_env_token_still_resolves(monkeypatch):
    """Control: a real OAuth-shaped value is untouched by the placeholder guard."""
    monkeypatch.setenv("ANTHROPIC_TOKEN", _REAL_OAUTH)

    assert resolve_anthropic_token() == _REAL_OAUTH


def test_all_placeholder_sources_resolve_as_unconfigured(monkeypatch):
    """Every source a placeholder: resolution reports "none", never the placeholder."""
    monkeypatch.setenv("ANTHROPIC_TOKEN", _PLACEHOLDER)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")

    assert resolve_anthropic_token() is None
