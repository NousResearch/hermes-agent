"""Behavior tests for label-scoped shared Hermes PKCE grants."""

import json
from pathlib import Path

import pytest

from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential


def _entry(label, access="old-access", refresh="old-refresh"):
    return PooledCredential(
        provider="anthropic", id=f"id-{label}", label=label,
        auth_type=AUTH_TYPE_OAUTH, priority=0, source="hermes_pkce_shared",
        access_token=access, refresh_token=refresh,
    )


@pytest.fixture
def shared_home(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("agent.anthropic_adapter.get_default_hermes_root", lambda: root)
    monkeypatch.setattr("agent.credential_pool.get_default_hermes_root", lambda: root)
    monkeypatch.setattr("agent.anthropic_adapter.read_hermes_oauth_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", lambda: None)
    return root


def test_refresh_publication_is_visible_to_another_profile(shared_home, monkeypatch):
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    write_shared_hermes_oauth_credentials({"accessToken": "old-access", "refreshToken": "old-refresh"}, label="team")
    first = CredentialPool("anthropic", [_entry("team")])
    second = CredentialPool("anthropic", [_entry("team")])
    monkeypatch.setattr("agent.anthropic_adapter.refresh_anthropic_oauth_pure", lambda *_a, **_k: {
        "access_token": "new-access", "refresh_token": "new-refresh", "expires_at_ms": 123
    })
    first._refresh_entry(first.entries()[0], force=True)
    selected = second.select()
    assert selected is not None and selected.access_token == "new-access"


def test_failed_refresh_syncs_shared_grant_then_retries(shared_home, monkeypatch):
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    write_shared_hermes_oauth_credentials({"accessToken": "old-access", "refreshToken": "old-refresh"}, label="team")
    calls = []

    def refresh(_refresh, **_kwargs):
        calls.append(_refresh)
        if len(calls) == 1:
            write_shared_hermes_oauth_credentials({"accessToken": "winner-access", "refreshToken": "winner-refresh"}, label="team")
            raise RuntimeError("invalid_grant")
        return {"access_token": "final-access", "refresh_token": "final-refresh", "expires_at_ms": 456}

    monkeypatch.setattr("agent.anthropic_adapter.refresh_anthropic_oauth_pure", refresh)
    pool = CredentialPool("anthropic", [_entry("team")])
    result = pool._refresh_entry(pool.entries()[0], force=True)
    assert result is not None and result.access_token == "final-access"
    assert calls == ["old-refresh", "winner-refresh"]


def test_second_refresh_winner_does_not_post_again(shared_home, monkeypatch):
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    write_shared_hermes_oauth_credentials({"accessToken": "old-access", "refreshToken": "old-refresh"}, label="team")
    calls = []
    monkeypatch.setattr("agent.anthropic_adapter.refresh_anthropic_oauth_pure", lambda refresh, **_k: (
        calls.append(refresh) or {"access_token": "new-access", "refresh_token": "new-refresh", "expires_at_ms": 1}
    ))
    first = CredentialPool("anthropic", [_entry("team")])
    second = CredentialPool("anthropic", [_entry("team")])
    first._refresh_entry(first.entries()[0], force=True)
    second._refresh_entry(second.entries()[0], force=True)
    assert calls == ["old-refresh"]


def test_copy_and_shared_rows_select_only_shared(shared_home, monkeypatch):
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials
    from agent.credential_pool import _migrate_legacy_hermes_pkce_entries

    write_shared_hermes_oauth_credentials({"accessToken": "shared-access", "refreshToken": "shared-refresh"}, label="account")
    entries = [_entry("account", access="stale", refresh="stale-refresh")]
    entries[0].source = "manual:hermes_pkce"
    assert _migrate_legacy_hermes_pkce_entries(entries)
    assert [entry.source for entry in entries] == ["hermes_pkce_shared"]
    assert CredentialPool("anthropic", entries).select().access_token == "shared-access"


def test_labels_never_cross_hydrate(shared_home):
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    write_shared_hermes_oauth_credentials({"accessToken": "a-access", "refreshToken": "a-refresh"}, label="a")
    write_shared_hermes_oauth_credentials({"accessToken": "b-access", "refreshToken": "b-refresh"}, label="b")
    pool = CredentialPool("anthropic", [_entry("a"), _entry("b")])
    selected = {pool._sync_anthropic_entry_from_shared_file(entry).label: pool._sync_anthropic_entry_from_shared_file(entry).access_token for entry in pool.entries()}
    assert selected == {"a": "a-access", "b": "b-access"}


def test_missing_shared_source_is_skipped_with_warning(shared_home, caplog):
    pool = CredentialPool("anthropic", [_entry("missing", access="")])
    with caplog.at_level("WARNING"):
        assert pool.select() is None
    assert "shared OAuth source is missing" in caplog.text


def test_shared_grants_use_independent_label_files_and_safe_names(shared_home):
    from agent.anthropic_adapter import (
        _shared_pkce_label_filename,
        read_shared_hermes_oauth_credentials,
        write_shared_hermes_oauth_credentials,
    )

    write_shared_hermes_oauth_credentials({"accessToken": "a-access"}, label="team a")
    write_shared_hermes_oauth_credentials({"accessToken": "b-access"}, label="team-b")
    files = sorted((shared_home / ".anthropic_oauth").glob("*.json"))
    assert [path.name for path in files] == ["team-b.json", "team_a.json"]
    first = read_shared_hermes_oauth_credentials("team a")
    second = read_shared_hermes_oauth_credentials("team-b")
    assert first and first["accessToken"] == "a-access"
    assert second and second["accessToken"] == "b-access"
    assert _shared_pkce_label_filename("") == "default.json"
    for unsafe in ("../escape", ".hidden", "a/b", "a\\b"):
        with pytest.raises(ValueError):
            _shared_pkce_label_filename(unsafe)


def test_legacy_grant_map_migrates_only_matching_labels(shared_home):
    from agent.anthropic_adapter import _get_shared_hermes_oauth_file
    from agent.credential_pool import _migrate_legacy_hermes_pkce_entries

    legacy = _get_shared_hermes_oauth_file()
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({"grants": {
        "one": {"accessToken": "one-access", "refreshToken": "one-refresh"},
        "two": {"accessToken": "two-access", "refreshToken": "two-refresh"},
    }}))
    entries = [_entry("one"), _entry("two")]
    for entry in entries:
        entry.source = "manual:hermes_pkce"
    assert _migrate_legacy_hermes_pkce_entries(entries)
    assert (shared_home / ".anthropic_oauth" / "one.json").exists()
    assert (shared_home / ".anthropic_oauth" / "two.json").exists()
    assert {entry.access_token for entry in entries} == {"one-access", "two-access"}
