"""Quick snapshots are file-loss recovery points — never a credential archive.

``hermes update`` snapshots every profile into ``state-snapshots/<id>/`` (also ``/snapshot``,
``hermes backup --quick``, ``hermes sessions repair-profiles``). That set used to include ``.env``
and ``auth.json``, so each update left a second, plaintext copy of every profile secret on disk:

* no exclusion policy sees that copy — every backup/export exclusion is written as ``.env``, so it
  sails past ``**/.env`` and the ``state-snapshots/`` subtree;
* it survives a rotation, so rolling a snapshot back silently re-arms a credential that was
  revoked;
* it contradicts the policy the harness already declares for exports ("Profile exports should
  NEVER include auth.json or .env").

Nothing in the update flow needs them: the code swap never rewrites ``.env``
(``_backfill_profile_env`` only fills a *missing* one) and ``auth.json`` is rebuilt by the auth
store on the next sign-in. The snapshot still has to restore old captures that carry them, so the
exclusion is producer-side only.

Regression for #77470.
"""

import json


def _seed_home(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-synthetic-not-a-key\n", encoding="utf-8")
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": {}}),
        encoding="utf-8",
    )
    return home


def test_quick_snapshot_leaves_credentials_out(tmp_path, monkeypatch):
    from hermes_cli.backup import create_quick_snapshot

    home = _seed_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    snap_id = create_quick_snapshot(hermes_home=home)
    assert snap_id

    snap_dir = home / "state-snapshots" / snap_id
    assert (snap_dir / "config.yaml").exists(), "config rollback must still be captured"
    assert not (snap_dir / ".env").exists(), (
        "the snapshot must not carry a second plaintext copy of the profile secrets"
    )
    assert not (snap_dir / "auth.json").exists(), (
        "auth.json holds the credential pool; the auth store owns its own recovery"
    )

    manifest = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "config.yaml" in manifest["files"]
    assert not {".env", "auth.json"} & set(manifest["files"])


def test_pre_update_snapshot_all_profiles_leaves_credentials_out(tmp_path, monkeypatch):
    """The fleet pre-update path snapshots every sibling profile with the same set."""
    from hermes_cli.backup import create_pre_update_snapshots_all_profiles

    root = tmp_path / ".hermes"
    root.mkdir()
    (root / "config.yaml").write_text("model: test\n", encoding="utf-8")
    profiles = root / "profiles"
    for name in ("alpha", "beta"):
        p = profiles / name
        p.mkdir(parents=True)
        (p / "config.yaml").write_text(f"model: {name}\n", encoding="utf-8")
        (p / ".env").write_text(f"{name.upper()}_API_KEY=sk-{name}\n", encoding="utf-8")
        (p / "auth.json").write_text(
            json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
        )
    monkeypatch.setenv("HERMES_HOME", str(root))

    snaps = create_pre_update_snapshots_all_profiles()
    # Fail loudly if HERMES_HOME resolved back to the real ~/.hermes: this test would then
    # snapshot the operator's whole fleet instead of the two seeded profiles.
    assert set(snaps) == {"alpha", "beta"}

    for name in ("alpha", "beta"):
        snap_dir = profiles / name / "state-snapshots" / snaps[name]
        assert (snap_dir / "config.yaml").exists()
        assert not (snap_dir / ".env").exists(), (
            f"{name}: .env leaked into the pre-update snapshot"
        )
        assert not (snap_dir / "auth.json").exists(), (
            f"{name}: auth.json leaked into the snapshot"
        )


def test_restore_still_accepts_an_older_snapshot_carrying_credentials(
    tmp_path, monkeypatch
):
    """Snapshots taken before this policy keep restoring — including their .env."""
    from hermes_cli.backup import restore_quick_snapshot

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-current\n", encoding="utf-8")
    (home / "config.yaml").write_text("model: current\n", encoding="utf-8")

    snap_id = "20261029-154300-pre-update"
    snap_dir = home / "state-snapshots" / snap_id
    snap_dir.mkdir(parents=True)
    (snap_dir / ".env").write_text("OPENROUTER_API_KEY=sk-captured\n", encoding="utf-8")
    (snap_dir / "config.yaml").write_text("model: captured\n", encoding="utf-8")
    files = {name: (snap_dir / name).stat().st_size for name in (".env", "config.yaml")}
    (snap_dir / "manifest.json").write_text(
        json.dumps({"id": snap_id, "files": files}), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert restore_quick_snapshot(snap_id, hermes_home=home) is True
    assert (home / "config.yaml").read_text(encoding="utf-8") == "model: captured\n"
    assert (home / ".env").read_text(
        encoding="utf-8"
    ) == "OPENROUTER_API_KEY=sk-captured\n"
