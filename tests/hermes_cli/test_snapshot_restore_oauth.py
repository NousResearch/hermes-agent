"""Quick snapshot auth restore must not roll back rotated single-use OAuth grants."""

import json
import os

import pytest

from hermes_cli.auth_oauth_grants import (
    SINGLE_USE_REFRESH_POOL_PROVIDERS,
    merge_snapshot_auth_preserving_live_single_use_grants,
)


def _oauth_row(provider, token, *, row_id="shared", priority=0):
    access = f"sk-ant-oat01-{token}" if provider == "anthropic" else f"access-{token}"
    return {
        "id": row_id,
        "label": row_id,
        "auth_type": "oauth",
        "priority": priority,
        "source": "manual",
        "access_token": access,
        "refresh_token": f"refresh-{token}",
    }


@pytest.mark.parametrize("provider", sorted(SINGLE_USE_REFRESH_POOL_PROVIDERS))
def test_merge_preserves_rotated_pool_grant_by_stable_id(provider):
    snapshot = {
        "version": 1,
        "providers": {},
        "credential_pool": {
            provider: [
                {**_oauth_row(provider, "spent", priority=4), "label": "snapshot-label"},
                _oauth_row(provider, "independent", row_id="snapshot-only", priority=2),
                {"id": "api", "auth_type": "api_key", "priority": 3, "api_key": "snapshot-key"},
            ]
        },
    }
    live = {
        "version": 1,
        "providers": {},
        "credential_pool": {provider: [_oauth_row(provider, "rotated")]},
    }

    merged = merge_snapshot_auth_preserving_live_single_use_grants(snapshot, live)
    rows = merged["credential_pool"][provider]

    shared = next(row for row in rows if row.get("id") == "shared")
    assert shared["refresh_token"] == "refresh-rotated"
    assert shared["priority"] == 4
    assert shared["label"] == "snapshot-label"
    assert not any(
        row.get("id") == "shared" and row.get("refresh_token") == "refresh-spent"
        for row in rows
    )
    assert not any(row.get("id") == "snapshot-only" for row in rows)
    assert any(row.get("id") == "api" and row.get("api_key") == "snapshot-key" for row in rows)


def test_live_grant_replaces_any_historical_row_with_the_same_stable_id():
    """One credential id cannot survive as both the rotated grant and an older row shape."""
    snapshot = {
        "providers": {},
        "credential_pool": {
            "openai-codex": [
                {"id": "shared", "auth_type": "api_key", "priority": 0, "api_key": "historical"},
                _oauth_row("openai-codex", "spent-duplicate", row_id="shared", priority=1),
                _oauth_row("openai-codex", "independent", row_id="snapshot-only", priority=2),
            ]
        },
    }
    live = {
        "providers": {},
        "credential_pool": {
            "openai-codex": [_oauth_row("openai-codex", "rotated", row_id="shared")]
        },
    }

    merged = merge_snapshot_auth_preserving_live_single_use_grants(snapshot, live)
    rows = merged["credential_pool"]["openai-codex"]

    assert [row.get("id") for row in rows].count("shared") == 1
    assert next(row for row in rows if row.get("id") == "shared")["refresh_token"] == "refresh-rotated"
    assert not any(row.get("id") == "snapshot-only" for row in rows)


def test_merge_legacy_oauth_without_id_keeps_only_live_generation():
    """Rows predating stable ids must not leave the spent snapshot generation beside live."""
    spent = _oauth_row("openai-codex", "spent")
    rotated = _oauth_row("openai-codex", "rotated")
    spent.pop("id")
    rotated.pop("id")
    snapshot = {
        "providers": {},
        "credential_pool": {
            "openai-codex": [
                spent,
                {"id": "api", "auth_type": "api_key", "api_key": "snapshot-key"},
            ]
        },
    }
    live = {
        "providers": {},
        "credential_pool": {"openai-codex": [rotated]},
    }

    rows = merge_snapshot_auth_preserving_live_single_use_grants(
        snapshot, live
    )["credential_pool"]["openai-codex"]
    oauth_rows = [row for row in rows if row.get("refresh_token")]

    assert len(oauth_rows) == 1
    assert oauth_rows[0]["refresh_token"] == "refresh-rotated"
    assert any(row.get("id") == "api" and row.get("api_key") == "snapshot-key" for row in rows)


@pytest.mark.parametrize("provider", ["openai-codex", "xai-oauth", "nous"])
def test_merge_preserves_live_device_code_refresh_block(provider):
    def block(token):
        if provider == "nous":
            return {"refresh_token": f"refresh-{token}", "access_token": f"access-{token}"}
        return {
            "tokens": {
                "refresh_token": f"refresh-{token}",
                "access_token": f"access-{token}",
            }
        }

    snapshot = {"providers": {provider: block("spent")}}
    live = {"providers": {provider: block("rotated")}}

    merged = merge_snapshot_auth_preserving_live_single_use_grants(snapshot, live)

    assert merged["providers"][provider] == live["providers"][provider]

    without_live = merge_snapshot_auth_preserving_live_single_use_grants(snapshot, {"providers": {}})
    assert provider not in without_live.get("providers", {})


def test_quick_snapshot_restore_keeps_live_oauth_and_restores_static_auth(
    tmp_path, monkeypatch
):
    from hermes_cli.backup import restore_quick_snapshot

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    live = {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "access-rotated",
                    "refresh_token": "refresh-rotated",
                }
            }
        },
        "credential_pool": {
            "openai-codex": [_oauth_row("openai-codex", "rotated")],
            "openrouter": [
                {"id": "static", "auth_type": "api_key", "priority": 0, "api_key": "live-static"}
            ],
        },
    }
    (home / "auth.json").write_text(json.dumps(live), encoding="utf-8")

    snapshot = {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "access-spent",
                    "refresh_token": "refresh-spent",
                }
            }
        },
        "credential_pool": {
            "openai-codex": [_oauth_row("openai-codex", "spent")],
            "openrouter": [
                {"id": "static", "auth_type": "api_key", "priority": 0, "api_key": "snapshot-static"}
            ],
        },
    }
    snap_dir = home / "state-snapshots" / "20260928-before-rotation"
    snap_dir.mkdir(parents=True)
    snap_auth = snap_dir / "auth.json"
    snap_auth.write_text(json.dumps(snapshot), encoding="utf-8")
    (snap_dir / "manifest.json").write_text(
        json.dumps({"files": {"auth.json": snap_auth.stat().st_size}}),
        encoding="utf-8",
    )

    assert restore_quick_snapshot(
        "20260928-before-rotation", hermes_home=home
    ) is True

    restored = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    codex = restored["credential_pool"]["openai-codex"][0]
    assert codex["refresh_token"] == "refresh-rotated"
    assert (
        restored["providers"]["openai-codex"]["tokens"]["refresh_token"]
        == "refresh-rotated"
    )
    assert restored["credential_pool"]["openrouter"][0]["api_key"] == "snapshot-static"
    if os.name != "nt":
        assert (home / "auth.json").stat().st_mode & 0o777 == 0o600



def test_current_non_oauth_identity_survives_historical_oauth_row():
    snapshot = {
        "providers": {},
        "credential_pool": {
            "openai-codex": [_oauth_row("openai-codex", "spent", row_id="shared")]
        },
    }
    live = {
        "providers": {},
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "shared",
                    "auth_type": "api_key",
                    "priority": 0,
                    "api_key": "current-static",
                }
            ]
        },
    }

    rows = merge_snapshot_auth_preserving_live_single_use_grants(
        snapshot, live
    )["credential_pool"]["openai-codex"]

    assert rows == [
        {
            "id": "shared",
            "auth_type": "api_key",
            "priority": 0,
            "api_key": "current-static",
        }
    ]


def test_quick_snapshot_restore_does_not_resurrect_snapshot_only_oauth(
    tmp_path, monkeypatch
):
    from hermes_cli.backup import restore_quick_snapshot

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": {}}),
        encoding="utf-8",
    )

    snapshot = {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "access-spent",
                    "refresh_token": "refresh-spent",
                }
            }
        },
        "credential_pool": {
            "openai-codex": [_oauth_row("openai-codex", "spent", row_id="old")]
        },
    }
    snap_dir = home / "state-snapshots" / "before-refresh"
    snap_dir.mkdir(parents=True)
    snap_auth = snap_dir / "auth.json"
    snap_auth.write_text(json.dumps(snapshot), encoding="utf-8")
    (snap_dir / "manifest.json").write_text(
        json.dumps({"files": {"auth.json": snap_auth.stat().st_size}}),
        encoding="utf-8",
    )

    assert restore_quick_snapshot("before-refresh", hermes_home=home) is True

    restored = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    assert not restored.get("credential_pool", {}).get("openai-codex")
    assert "openai-codex" not in restored.get("providers", {})


def test_auth_restore_resolves_symlink_before_lock_and_publish(tmp_path, monkeypatch):
    from hermes_cli.backup_restore import _restore_auth_json

    home = tmp_path / "root"
    profile = tmp_path / "profile"
    home.mkdir()
    profile.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    target = home / "auth.json"
    target.write_text(
        json.dumps(
            {
                "providers": {},
                "credential_pool": {
                    "openai-codex": [_oauth_row("openai-codex", "rotated")]
                },
            }
        ),
        encoding="utf-8",
    )
    alias = profile / "auth.json"
    try:
        alias.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    snapshot = tmp_path / "snapshot-auth.json"
    snapshot.write_text(
        json.dumps(
            {
                "providers": {},
                "credential_pool": {
                    "openai-codex": [_oauth_row("openai-codex", "spent")],
                    "openrouter": [
                        {
                            "id": "static",
                            "auth_type": "api_key",
                            "api_key": "snapshot-static",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    assert _restore_auth_json(snapshot, alias) is True
    assert alias.is_symlink()
    restored = json.loads(target.read_text(encoding="utf-8"))
    assert restored["credential_pool"]["openai-codex"][0]["refresh_token"] == "refresh-rotated"
    assert restored["credential_pool"]["openrouter"][0]["api_key"] == "snapshot-static"
    assert (home / "auth.lock").exists()
    assert not (profile / "auth.lock").exists()


def test_quick_snapshot_restore_allows_named_profile_auth_alias_to_root(
    tmp_path, monkeypatch
):
    from hermes_cli.backup import restore_quick_snapshot

    root = tmp_path / "root"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    target = root / "auth.json"
    target.write_text(
        json.dumps(
            {
                "providers": {},
                "credential_pool": {
                    "openai-codex": [_oauth_row("openai-codex", "rotated")],
                    "openrouter": [
                        {
                            "id": "static",
                            "auth_type": "api_key",
                            "api_key": "live-static",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    alias = profile / "auth.json"
    try:
        alias.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    snapshot = {
        "providers": {},
        "credential_pool": {
            "openai-codex": [_oauth_row("openai-codex", "spent")],
            "openrouter": [
                {
                    "id": "static",
                    "auth_type": "api_key",
                    "api_key": "snapshot-static",
                }
            ],
        },
    }
    snap_dir = profile / "state-snapshots" / "named-profile"
    snap_dir.mkdir(parents=True)
    snap_auth = snap_dir / "auth.json"
    snap_auth.write_text(json.dumps(snapshot), encoding="utf-8")
    (snap_dir / "manifest.json").write_text(
        json.dumps({"files": {"auth.json": snap_auth.stat().st_size}}),
        encoding="utf-8",
    )

    assert restore_quick_snapshot("named-profile", hermes_home=profile) is True
    assert alias.is_symlink()
    restored = json.loads(target.read_text(encoding="utf-8"))
    assert restored["credential_pool"]["openai-codex"][0]["refresh_token"] == "refresh-rotated"
    assert restored["credential_pool"]["openrouter"][0]["api_key"] == "snapshot-static"
    assert (root / "auth.lock").exists()
    assert not (profile / "auth.lock").exists()


def test_quick_snapshot_restore_rejects_untrusted_auth_alias(
    tmp_path, monkeypatch
):
    from hermes_cli.backup import restore_quick_snapshot

    root = tmp_path / "root"
    profile = root / "profiles" / "work"
    external = tmp_path / "external"
    profile.mkdir(parents=True)
    external.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile))

    target = external / "auth.json"
    target.write_text(json.dumps({"providers": {}, "sentinel": "outside"}), encoding="utf-8")
    before = target.read_bytes()
    alias = profile / "auth.json"
    try:
        alias.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    snap_dir = profile / "state-snapshots" / "untrusted-alias"
    snap_dir.mkdir(parents=True)
    snap_auth = snap_dir / "auth.json"
    snap_auth.write_text(
        json.dumps(
            {
                "providers": {},
                "credential_pool": {
                    "openrouter": [
                        {"id": "static", "auth_type": "api_key", "api_key": "snapshot-static"}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (snap_dir / "manifest.json").write_text(
        json.dumps({"files": {"auth.json": snap_auth.stat().st_size}}),
        encoding="utf-8",
    )

    assert restore_quick_snapshot("untrusted-alias", hermes_home=profile) is False
    assert alias.is_symlink()
    assert target.read_bytes() == before


def test_auth_restore_refuses_hardlinked_store_instead_of_splitting_it(
    tmp_path, monkeypatch
):
    from hermes_cli.backup_restore import _restore_auth_json

    home = tmp_path / "root"
    twin = tmp_path / "twin"
    home.mkdir()
    twin.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    target = home / "auth.json"
    target.write_text(
        json.dumps(
            {
                "providers": {},
                "credential_pool": {
                    "openai-codex": [_oauth_row("openai-codex", "rotated")]
                },
            }
        ),
        encoding="utf-8",
    )
    alias = twin / "auth.json"
    try:
        os.link(target, alias)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"hardlink unavailable: {exc}")

    snapshot = tmp_path / "snapshot-auth.json"
    snapshot.write_text(
        json.dumps(
            {
                "providers": {},
                "credential_pool": {
                    "openai-codex": [_oauth_row("openai-codex", "spent")]
                },
            }
        ),
        encoding="utf-8",
    )
    before = target.read_bytes()

    assert _restore_auth_json(snapshot, alias) is False
    assert alias.samefile(target)
    assert target.read_bytes() == before
    assert alias.read_bytes() == before


def test_invalid_snapshot_auth_never_replaces_live_store(tmp_path, monkeypatch):
    from hermes_cli.backup import restore_quick_snapshot

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    live_path = home / "auth.json"
    live_path.write_text(json.dumps({"providers": {}, "sentinel": "live"}), encoding="utf-8")
    before = live_path.read_bytes()

    snap_dir = home / "state-snapshots" / "bad"
    snap_dir.mkdir(parents=True)
    (snap_dir / "auth.json").write_bytes(b"{not-json")
    (snap_dir / "manifest.json").write_text(
        json.dumps({"files": {"auth.json": 9}}), encoding="utf-8"
    )

    assert restore_quick_snapshot("bad", hermes_home=home) is False
    assert live_path.read_bytes() == before


@pytest.mark.parametrize("refusal", ["invalid-json", "hardlinked-store"])
def test_auth_refusal_is_not_hidden_by_another_restored_file(tmp_path, monkeypatch, refusal):
    """A partial restore must not report success after refusing the auth store (#127010)."""
    from hermes_cli.backup import restore_quick_snapshot

    home = tmp_path / "home"
    snap_dir = home / "state-snapshots" / "partial"
    snap_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    live_path = home / "auth.json"
    live_path.write_text(json.dumps({"providers": {}, "sentinel": "live"}), encoding="utf-8")
    before = live_path.read_bytes()
    if refusal == "hardlinked-store":
        try:
            os.link(live_path, home / "shared-auth.json")
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"hardlink unavailable: {exc}")

    (home / "config.yaml").write_text("model: live\n", encoding="utf-8")
    (snap_dir / "config.yaml").write_text("model: snapshot\n", encoding="utf-8")
    snapshot_auth = "{invalid-json" if refusal == "invalid-json" else json.dumps({"providers": {}})
    (snap_dir / "auth.json").write_text(snapshot_auth, encoding="utf-8")
    files = {name: (snap_dir / name).stat().st_size for name in ("auth.json", "config.yaml")}
    (snap_dir / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")

    result = restore_quick_snapshot("partial", hermes_home=home)

    assert live_path.read_bytes() == before
    assert (home / "config.yaml").read_text(encoding="utf-8") == "model: snapshot\n"
    assert result is False
