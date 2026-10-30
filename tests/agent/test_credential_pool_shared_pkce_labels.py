"""Regression tests for label-scoped shared Hermes PKCE migration and loading."""

import json
from pathlib import Path

import pytest

FUTURE_MS = 4_102_444_800_000


def _entry(label, *, source="manual:hermes_pkce", access="access", refresh=None):
    return {
        "id": f"id-{label}-{source}",
        "label": label,
        "auth_type": "oauth",
        "priority": 0,
        "source": source,
        "access_token": access,
        "refresh_token": refresh,
        "expires_at_ms": FUTURE_MS,
    }


def _write_auth(home: Path, entries: list[dict]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"anthropic": entries}}),
        encoding="utf-8",
    )


def _read_entries(home: Path) -> list[dict]:
    return json.loads((home / "auth.json").read_text(encoding="utf-8"))["credential_pool"]["anthropic"]


@pytest.fixture
def homes(tmp_path, monkeypatch):
    default_root = tmp_path / "default"
    profile_root = default_root / "profiles" / "worker"
    default_root.mkdir(parents=True)
    profile_root.mkdir(parents=True)
    monkeypatch.setattr("agent.anthropic_adapter.get_default_hermes_root", lambda: default_root)
    monkeypatch.setattr("agent.credential_pool.get_default_hermes_root", lambda: default_root)
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda _provider: True)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("agent.anthropic_adapter.refresh_anthropic_oauth_pure", lambda *_args, **_kwargs: pytest.fail("refresh must not be called"))
    return default_root, profile_root


def _load(monkeypatch, home: Path):
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent.credential_pool import load_pool

    return load_pool("anthropic")


def _shared_file(default_root: Path, label: str) -> Path:
    from agent.anthropic_adapter import _shared_pkce_label_filename

    return default_root / ".anthropic_oauth" / _shared_pkce_label_filename(label)


def test_auth_json_with_copy_and_shared_entries_selects_shared_only(homes, monkeypatch):
    default_root, profile_root = homes
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "shared-access", "refreshToken": "shared-refresh", "expiresAt": FUTURE_MS},
        label="work",
    )
    _write_auth(
        profile_root,
        [
            _entry("work", access="stale-access", refresh="stale-refresh"),
            _entry("work", source="hermes_pkce_shared", access="", refresh=None),
        ],
    )

    pool = _load(monkeypatch, profile_root)
    entries = pool.entries()

    assert len(entries) == 1
    assert entries[0].source == "hermes_pkce_shared"
    selected = pool.select()
    assert selected is not None
    assert selected.access_token == "shared-access"
    assert _read_entries(profile_root)[0]["source"] == "hermes_pkce_shared"


def test_two_labels_migrate_to_separate_shared_files_without_cross_contamination(homes, monkeypatch):
    default_root, profile_root = homes
    _write_auth(
        default_root,
        [
            _entry("work", access="work-access", refresh="work-refresh"),
            _entry("home", access="home-access", refresh="home-refresh"),
        ],
    )
    _write_auth(
        profile_root,
        [
            _entry("work", access="stale-work", refresh="stale-work-refresh"),
            _entry("home", access="stale-home", refresh="stale-home-refresh"),
        ],
    )

    _load(monkeypatch, default_root)
    pool = _load(monkeypatch, profile_root)

    work_file = _shared_file(default_root, "work")
    home_file = _shared_file(default_root, "home")
    assert work_file.exists() and home_file.exists() and work_file != home_file
    assert json.loads(work_file.read_text(encoding="utf-8"))["refreshToken"] == "work-refresh"
    assert json.loads(home_file.read_text(encoding="utf-8"))["refreshToken"] == "home-refresh"
    assert "home-refresh" not in work_file.read_text(encoding="utf-8")
    assert "work-refresh" not in home_file.read_text(encoding="utf-8")

    assert {entry.label for entry in pool.entries()} == {"work", "home"}
    assert all(entry.source == "hermes_pkce_shared" for entry in pool.entries())
    hydrated = {entry.label: pool._sync_anthropic_entry_from_shared_file(entry) for entry in pool.entries()}
    assert hydrated["work"].refresh_token == "work-refresh"
    assert hydrated["home"].refresh_token == "home-refresh"

    from agent.anthropic_adapter import _shared_pkce_label_filename

    assert _shared_pkce_label_filename("../x")
    assert _shared_pkce_label_filename("a/b")
    assert _shared_pkce_label_filename("a\\b")
    assert _shared_pkce_label_filename("")
    for label in ("../x", "a/b", "a\\b"):
        with pytest.raises(ValueError):
            _shared_pkce_label_filename(label)


def test_missing_shared_source_skips_entry_with_single_warning(homes, monkeypatch, caplog):
    _default_root, profile_root = homes
    _write_auth(
        profile_root,
        [_entry("missing", source="hermes_pkce_shared", access="", refresh=None)],
    )

    with caplog.at_level("WARNING"):
        pool = _load(monkeypatch, profile_root)
        assert pool.select() is None
        assert pool.select() is None

    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "shared OAuth source is missing" in warnings[0].message
    assert "missing" in warnings[0].message
    assert "refresh" not in warnings[0].message.lower()
    assert "access" not in warnings[0].message.lower()
