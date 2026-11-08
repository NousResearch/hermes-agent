"""Quick snapshot auth restore must not roll back rotated single-use OAuth grants."""

import json
import os


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


def test_auth_refusal_is_not_hidden_by_another_restored_file(tmp_path, monkeypatch, capsys):
    """A partial restore must not report success after refusing the auth store (#127010)."""
    from hermes_cli.backup import restore_quick_snapshot

    snap_id = "20260101-000000-pre-update"
    home = tmp_path / "home"
    snap_dir = home / "state-snapshots" / snap_id
    snap_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    live_path = home / "auth.json"
    live_path.write_text(json.dumps({"providers": {}, "sentinel": "live"}), encoding="utf-8")
    before = live_path.read_bytes()

    (home / "config.yaml").write_text("model: live\n", encoding="utf-8")
    (snap_dir / "config.yaml").write_text("model: snapshot\n", encoding="utf-8")
    (snap_dir / "auth.json").write_text("{invalid-json", encoding="utf-8")
    files = {name: (snap_dir / name).stat().st_size for name in ("auth.json", "config.yaml")}
    (snap_dir / "manifest.json").write_text(json.dumps({"id": snap_id, "files": files}), encoding="utf-8")

    result = restore_quick_snapshot(snap_id, hermes_home=home)

    assert live_path.read_bytes() == before
    assert (home / "config.yaml").read_text(encoding="utf-8") == "model: snapshot\n"
    assert result is False

    # The /snapshot restore caller must not report an existing snapshot as missing.
    from types import SimpleNamespace

    from hermes_cli.cli_commands_mixin import CLICommandsMixin, _t

    capsys.readouterr()
    CLICommandsMixin._snapshot_restore(SimpleNamespace(), ["/snapshot", "restore", snap_id])
    out = capsys.readouterr().out
    assert out.strip() == _t("snapshot.restore_incomplete", snapshot_id=snap_id)
    assert live_path.read_bytes() == before

    # A manifest-listed auth.json missing from the snapshot dir also fails closed.
    (snap_dir / "auth.json").unlink()
    (home / "config.yaml").write_text("model: live\n", encoding="utf-8")

    assert restore_quick_snapshot(snap_id, hermes_home=home) is False
    assert (home / "config.yaml").read_text(encoding="utf-8") == "model: snapshot\n"
    assert live_path.read_bytes() == before
