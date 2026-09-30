"""External credential presence is observable without borrowing/refreshing (#20675)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def dump_fixture(tmp_path, monkeypatch, capsys):
    from agent import anthropic_credentials as credentials
    from hermes_cli import auth, dump

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    external = tmp_path / "custom-claude"
    external.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(external))
    monkeypatch.setattr(credentials, "_read_claude_code_credentials_from_keychain", lambda: None)
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: {})
    for key, _ in dump._API_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-openrouter")
    monkeypatch.setattr(dump, "load_hermes_dotenv", lambda **kw: None)
    monkeypatch.setattr(dump, "load_config", lambda: {})
    monkeypatch.setattr(dump, "_gateway_status", lambda: "fixture")
    monkeypatch.setattr(dump, "_version_line", lambda root: "fixture")
    monkeypatch.setattr(dump, "_openai_version", lambda: "fixture")
    monkeypatch.setattr(credentials, "refresh_anthropic_oauth_pure", lambda *a, **kw: pytest.fail("display refreshed credentials"))

    def render():
        dump.run_dump(SimpleNamespace(show_keys=True))
        return capsys.readouterr().out

    return home, external / ".credentials.json", render


def test_dump_external_claude_presence_is_readonly_and_obeys_adoption(dump_fixture, monkeypatch):
    from agent import anthropic_credentials as credentials

    home, path, render = dump_fixture
    path.write_text(json.dumps({"claudeAiOauth": {
        "accessToken": "fixture-access-secret", "refreshToken": "fixture-refresh-secret", "expiresAt": 1,
    }}), encoding="utf-8")
    before = path.read_bytes()
    output = render()
    assert "claude-code-cli" in output
    assert "set (oauth; presence only)" in output
    assert "fixture-access-secret" not in output
    assert "fixture-refresh-secret" not in output
    assert path.read_bytes() == before
    assert not (home / "auth.json").exists()

    # The shared reader owns opt-out; displaying presence must not bypass it.
    (home / "config.yaml").write_text("auth:\n  adopt_external_logins: false\n", encoding="utf-8")
    monkeypatch.setattr(credentials, "_read_claude_code_credentials_from_file", lambda: pytest.fail("opted-out file read"))
    monkeypatch.setattr(credentials, "_read_claude_code_credentials_from_keychain", lambda: pytest.fail("opted-out keychain read"))
    assert "claude-code-cli" not in render()
    assert path.read_bytes() == before


@pytest.mark.parametrize("payload", [None, "{broken", "[]", '{"claudeAiOauth": {"accessToken": ""}}', '{"claudeAiOauth": {"accessToken": 7}}', "raised"])
def test_dump_external_claude_failure_preserves_other_rows(dump_fixture, monkeypatch, payload):
    from agent import anthropic_credentials as credentials

    _, path, render = dump_fixture
    if payload == "raised":
        def fail():
            raise OSError("fixture-access-secret")
        monkeypatch.setattr(credentials, "read_claude_code_credentials", fail)
    elif payload is not None:
        path.write_text(payload, encoding="utf-8")
    output = render()
    assert "claude-code-cli" not in output
    assert "fixture-access-secret" not in output
    assert "--- end dump ---" in output
    assert "openrouter" in output
    if payload not in (None, "raised"):
        assert path.read_text(encoding="utf-8") == payload
