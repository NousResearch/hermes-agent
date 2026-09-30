"""Public diagnostic parity with the read-only Codex singleton resolver (#20675)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def dump_fixture(tmp_path, monkeypatch, capsys):
    from hermes_cli import auth, dump
    import agent.credential_pool as pools

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: {})
    for key, _ in dump._API_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-openrouter")
    for name, value in [("load_config", {}), ("_gateway_status", "fixture"),
                        ("_openai_version", "fixture")]:
        monkeypatch.setattr(dump, name, lambda value=value: value)
    monkeypatch.setattr(dump, "load_hermes_dotenv", lambda **kw: None)
    monkeypatch.setattr(dump, "_version_line", lambda root: "fixture")

    def forbidden(*args, **kwargs):
        pytest.fail("diagnostic tried to refresh, adopt, save or seed credentials")

    monkeypatch.setattr(auth, "_save_auth_store", forbidden)
    from hermes_cli import auth_codex
    monkeypatch.setattr(auth_codex, "_recover_codex_tokens_from_cli", forbidden)
    monkeypatch.setattr(auth_codex, "_refresh_codex_auth_tokens", forbidden)
    monkeypatch.setattr(pools, "load_pool", forbidden)

    def render():
        dump.run_dump(SimpleNamespace(show_keys=True))
        return capsys.readouterr().out

    return home / "auth.json", render


def test_dump_sees_codex_singleton_without_refresh_or_disclosure(dump_fixture):
    from hermes_cli.auth import resolve_codex_runtime_credentials

    path, render = dump_fixture
    path.write_text(json.dumps({"providers": {"openai-codex": {"tokens": {
        "access_token": "fixture-codex-access", "refresh_token": "fixture-codex-refresh",
    }}}, "credential_pool": {}}), encoding="utf-8")
    before = path.read_bytes()
    # The already-working sibling proves this fixture represents a recognized login.
    assert resolve_codex_runtime_credentials(read_only=True)["api_key"] == "fixture-codex-access"
    output = render()
    assert "openai-codex" in output
    assert "set (oauth; presence only)" in output
    assert "fixture-codex-access" not in output
    assert "fixture-codex-refresh" not in output
    assert path.read_bytes() == before
    assert not (path.parent / "auth.lock").exists()


@pytest.mark.parametrize("tokens", [None, [], {}, {"access_token": ""},
                                    {"access_token": 7}, {"access_token": "no-refresh"}])
def test_invalid_codex_singleton_preserves_dump(dump_fixture, tokens):
    path, render = dump_fixture
    path.write_text(json.dumps({"providers": {"openai-codex": {"tokens": tokens}}}), encoding="utf-8")
    before = path.read_bytes()
    output = render()
    assert "auth:openai-codex" not in output
    assert "--- end dump ---" in output
    assert "no-refresh" not in output
    assert path.read_bytes() == before


def test_codex_reader_exception_preserves_other_rows(dump_fixture, monkeypatch):
    from hermes_cli import auth

    path, render = dump_fixture
    def fail(**kwargs):
        assert kwargs == {"read_only": True}
        raise OSError("fixture-codex-secret")
    monkeypatch.setattr(auth, "resolve_codex_runtime_credentials", fail)
    output = render()
    assert "auth:openai-codex" not in output
    assert "fixture-codex-secret" not in output
    assert "openrouter" in output
    assert "--- end dump ---" in output
    assert not path.exists()


def test_codex_pool_preserves_type_and_skips_singleton_lookup(dump_fixture, monkeypatch):
    from hermes_cli import auth

    path, render = dump_fixture
    path.write_text(json.dumps({"providers": {}, "credential_pool": {
        "openai-codex": [{"access_token": "fixture-pool-key", "auth_type": "api_key"}],
    }}), encoding="utf-8")
    monkeypatch.setattr(auth, "resolve_codex_runtime_credentials",
                        lambda **kw: pytest.fail("pool row was already represented"))
    output = render()
    row = next(line for line in output.splitlines() if "pool:openai-codex" in line)
    assert "set (auth pool)" in row
    assert "auth:openai-codex" not in output
    assert "fixture-pool-key" not in output


def test_codex_singleton_follows_selected_home_without_cached_presence(dump_fixture, monkeypatch):
    path, render = dump_fixture
    path.write_text(json.dumps({"providers": {"openai-codex": {"tokens": {
        "access_token": "fixture-scope-a", "refresh_token": "fixture-refresh-a",
    }}}}), encoding="utf-8")
    before = path.read_bytes()
    other = path.parent.parent / "other"
    other.mkdir()
    for home, expected in [(path.parent, True), (other, False), (path.parent, True)]:
        monkeypatch.setenv("HERMES_HOME", str(home))
        output = render()
        assert ("auth:openai-codex" in output) is expected
        assert "fixture-scope-a" not in output
    assert path.read_bytes() == before
    assert not (other / "auth.json").exists()
