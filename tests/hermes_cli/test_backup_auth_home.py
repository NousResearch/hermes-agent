"""Backup/import coverage for a distinct provider-auth residence."""

from __future__ import annotations

import os
import zipfile
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import pytest


_CREDENTIALS = {
    "auth.json": b'{"store":"root"}\n',
    ".anthropic_oauth.json": b'{"store":"root-anthropic"}\n',
    "profiles/coder/auth.json": b'{"store":"coder"}\n',
    "profiles/coder/.anthropic_oauth.json": b'{"store":"coder-anthropic"}\n',
    "shared/nous_auth.json": b'{"store":"shared"}\n',
}


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    for rel, payload in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    runtime_home: Path,
    auth_home: Path | None,
    operator_home: Path,
) -> None:
    operator_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: operator_home)
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.delenv("HERMES_SHARED_AUTH_DIR", raising=False)
    if auth_home is None:
        monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)
    else:
        monkeypatch.setenv("HERMES_AUTH_HOME", str(auth_home))


def _create_full_archive(kind: str, runtime_home: Path, out_path: Path) -> Path:
    from hermes_cli.backup import (
        create_pre_migration_backup,
        create_pre_update_backup,
        run_backup,
    )

    if kind == "full":
        run_backup(Namespace(output=str(out_path)))
        return out_path
    if kind == "pre-update":
        result = create_pre_update_backup(hermes_home=runtime_home)
    else:
        result = create_pre_migration_backup(hermes_home=runtime_home)
    assert result is not None
    return result


@pytest.mark.parametrize("kind", ["full", "pre-update", "pre-migration"])
def test_distinct_residence_full_backups_round_trip(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_import

    source_runtime = tmp_path / "source-runtime"
    source_auth = tmp_path / "source-auth"
    source_runtime.mkdir()
    (source_runtime / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _write_files(source_auth, _CREDENTIALS)

    stale_runtime = {
        "auth.json": b'{"store":"stale-runtime"}\n',
        "profiles/coder/auth.json": b'{"store":"stale-profile"}\n',
        "shared/nous_auth.json": b'{"store":"stale-shared"}\n',
    }
    _write_files(source_runtime, stale_runtime)
    _write_files(
        source_auth,
        {
            "auth.lock": b"lock",
            ".anthropic_oauth.lock": b"lock",
            "auth.json.corrupt": b"corrupt",
            "auth.json.tmp.1.dead": b"temp",
            "profiles/coder/auth.json.corrupt": b"corrupt",
            "shared/nous_auth.lock": b"lock",
            "unknown.json": b'{"not":"owned"}',
        },
    )
    _configure(
        monkeypatch,
        source_runtime,
        source_auth,
        tmp_path / "source-operator",
    )

    archive = _create_full_archive(kind, source_runtime, tmp_path / f"{kind}.zip")
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        archived_credentials = {
            name
            for name in names
            if name.startswith("_auth-residence/")
        }
        assert archived_credentials == {
            f"_auth-residence/{rel}" for rel in _CREDENTIALS
        }
        for rel in stale_runtime:
            assert rel not in names

    destination_runtime = tmp_path / f"{kind}-destination-runtime"
    destination_auth = tmp_path / f"{kind}-destination-auth"
    destination_runtime.mkdir()
    _configure(
        monkeypatch,
        destination_runtime,
        destination_auth,
        tmp_path / f"{kind}-destination-operator",
    )
    run_import(Namespace(zipfile=str(archive), force=True))

    assert (destination_runtime / "config.yaml").read_text(encoding="utf-8") == (
        "model: {}\n"
    )
    for rel, payload in _CREDENTIALS.items():
        restored = destination_auth / rel
        assert restored.read_bytes() == payload
        if os.name == "posix":
            assert restored.stat().st_mode & 0o777 == 0o600
        assert not (destination_runtime / rel).exists()


def test_legacy_flat_credentials_import_into_residence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_import

    archive = tmp_path / "legacy.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("config.yaml", "model: {}\n")
        for rel, payload in _CREDENTIALS.items():
            zf.writestr(rel, payload)
        zf.writestr("auth.lock", "lock")
        zf.writestr("profiles/coder/auth.json.corrupt", "corrupt")
        zf.writestr("shared/nous_auth.json.tmp.1.dead", "temp")

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    run_import(Namespace(zipfile=str(archive), force=True))

    for rel, payload in _CREDENTIALS.items():
        assert (auth_home / rel).read_bytes() == payload
        assert not (runtime_home / rel).exists()
    assert not (runtime_home / "auth.lock").exists()
    assert not (
        runtime_home / "profiles" / "coder" / "auth.json.corrupt"
    ).exists()
    assert not (
        runtime_home / "shared" / "nous_auth.json.tmp.1.dead"
    ).exists()


def test_shared_auth_override_round_trips_the_store_runtime_uses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_backup, run_import

    source_runtime = tmp_path / "source-runtime"
    source_auth = tmp_path / "source-auth"
    source_shared_dir = tmp_path / "source-shared"
    source_runtime.mkdir()
    source_auth.mkdir()
    source_shared_dir.mkdir()
    (source_runtime / "config.yaml").write_text(
        "model: {}\n",
        encoding="utf-8",
    )
    _write_files(
        source_auth,
        {"shared/nous_auth.json": b'{"value":"decoy"}\n'},
    )
    actual_shared = source_shared_dir / "nous_auth.json"
    actual_shared.write_bytes(b'{"value":"actual"}\n')
    _configure(
        monkeypatch,
        source_runtime,
        source_auth,
        tmp_path / "source-operator",
    )
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(source_shared_dir))

    archive = tmp_path / "shared-override.zip"
    run_backup(Namespace(output=str(archive)))
    with zipfile.ZipFile(archive) as zf:
        assert zf.read("_auth-residence/shared/nous_auth.json") == (
            b'{"value":"actual"}\n'
        )

    destination_runtime = tmp_path / "destination-runtime"
    destination_auth = tmp_path / "destination-auth"
    destination_shared_dir = tmp_path / "destination-shared"
    destination_runtime.mkdir()
    _configure(
        monkeypatch,
        destination_runtime,
        destination_auth,
        tmp_path / "destination-operator",
    )
    monkeypatch.setenv(
        "HERMES_SHARED_AUTH_DIR",
        str(destination_shared_dir),
    )
    run_import(Namespace(zipfile=str(archive), force=True))

    assert (destination_shared_dir / "nous_auth.json").read_bytes() == (
        b'{"value":"actual"}\n'
    )
    assert not (destination_auth / "shared" / "nous_auth.json").exists()


def test_shared_override_keeps_flat_layout_without_auth_residence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_backup, run_import

    source_runtime = tmp_path / "source-runtime"
    source_shared = tmp_path / "source-shared"
    source_runtime.mkdir()
    source_shared.mkdir()
    (source_runtime / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (source_shared / "nous_auth.json").write_bytes(b'{"value":"shared"}\n')
    _configure(monkeypatch, source_runtime, None, tmp_path / "source-operator")
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(source_shared))

    archive = tmp_path / "flat-shared.zip"
    run_backup(Namespace(output=str(archive)))
    with zipfile.ZipFile(archive) as zf:
        assert zf.read("shared/nous_auth.json") == b'{"value":"shared"}\n'
        assert not any(
            name.startswith("_auth-residence/")
            for name in zf.namelist()
        )

    destination_runtime = tmp_path / "destination-runtime"
    destination_shared = tmp_path / "destination-shared"
    destination_runtime.mkdir()
    _configure(
        monkeypatch,
        destination_runtime,
        None,
        tmp_path / "destination-operator",
    )
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(destination_shared))
    run_import(Namespace(zipfile=str(archive), force=True))
    assert (destination_shared / "nous_auth.json").read_bytes() == (
        b'{"value":"shared"}\n'
    )
    assert not (
        destination_runtime / "shared" / "nous_auth.json"
    ).exists()


def test_profile_pre_migration_backup_keeps_complete_residence_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import create_pre_migration_backup

    runtime_home = tmp_path / "runtime" / "profiles" / "coder"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir(parents=True)
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _write_files(auth_home, _CREDENTIALS)
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    archive = create_pre_migration_backup(hermes_home=runtime_home)
    assert archive is not None
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert {
        f"_auth-residence/{rel}" for rel in _CREDENTIALS
    }.issubset(names)


def test_full_helper_maps_explicit_home_independently_of_ambient_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import create_pre_migration_backup

    explicit_runtime = tmp_path / "explicit-runtime"
    ambient_home = tmp_path / "auth"
    explicit_runtime.mkdir()
    ambient_home.mkdir()
    (explicit_runtime / "config.yaml").write_text(
        "model: {}\n",
        encoding="utf-8",
    )
    _write_files(ambient_home, _CREDENTIALS)
    _configure(monkeypatch, ambient_home, ambient_home, tmp_path / "operator")

    archive = create_pre_migration_backup(hermes_home=explicit_runtime)
    assert archive is not None
    with zipfile.ZipFile(archive) as zf:
        assert "_auth-residence/auth.json" in zf.namelist()


def test_symlinked_credential_file_is_included_from_residence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_backup

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    outside_store = tmp_path / "mounted-auth.json"
    runtime_home.mkdir()
    auth_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    outside_store.write_bytes(b'{"value":"mounted"}\n')
    try:
        (auth_home / "auth.json").symlink_to(outside_store)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    archive = tmp_path / "symlink.zip"
    run_backup(Namespace(output=str(archive)))
    with zipfile.ZipFile(archive) as zf:
        assert zf.read("_auth-residence/auth.json") == b'{"value":"mounted"}\n'


def test_runtime_nested_beneath_residence_is_rejected_before_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli.backup import run_backup

    auth_home = tmp_path / "auth"
    runtime_home = auth_home / "runtime"
    runtime_home.mkdir(parents=True)
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (runtime_home / "notes.txt").write_text("keep me\n", encoding="utf-8")
    _write_files(auth_home, {"auth.json": b'{"value":"credential"}\n'})
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    archive = tmp_path / "nested.zip"
    run_backup(Namespace(output=str(archive)))

    assert not archive.exists()
    assert "HERMES_AUTH_HOME must not contain HERMES_HOME" in (
        capsys.readouterr().out
    )


def test_no_override_backup_keeps_flat_credential_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_backup, run_import

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _write_files(source, _CREDENTIALS)
    _write_files(
        source,
        {
            "auth.lock": b"lock",
            "auth.json.corrupt": b"corrupt",
            "auth.json.rebootstrap.tmp": b"legacy rebootstrap temp",
            ".anthropic_oauth.tmp.1.dead": b"temp",
            "profiles/coder/auth.json.tmp.1.dead": b"temp",
            "shared/nous_auth.lock": b"lock",
        },
    )
    _configure(monkeypatch, source, None, tmp_path / "source-operator")

    archive = tmp_path / "flat.zip"
    run_backup(Namespace(output=str(archive)))
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert not any(name.startswith("_auth-residence/") for name in names)
    assert set(_CREDENTIALS).issubset(names)
    assert "auth.lock" not in names
    assert "auth.json.corrupt" not in names
    assert "auth.json.rebootstrap.tmp" not in names
    assert ".anthropic_oauth.tmp.1.dead" not in names
    assert "profiles/coder/auth.json.tmp.1.dead" not in names
    assert "shared/nous_auth.lock" not in names

    destination = tmp_path / "destination"
    destination.mkdir()
    _configure(monkeypatch, destination, None, tmp_path / "destination-operator")
    run_import(Namespace(zipfile=str(archive), force=True))
    for rel, payload in _CREDENTIALS.items():
        assert (destination / rel).read_bytes() == payload


def test_no_override_backup_follows_only_known_credential_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_backup

    runtime_home = tmp_path / "runtime"
    mounted_auth = tmp_path / "mounted-auth.json"
    unrelated = tmp_path / "unrelated.txt"
    runtime_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    mounted_auth.write_bytes(b'{"value":"mounted"}\n')
    unrelated.write_text("do not include\n", encoding="utf-8")
    try:
        (runtime_home / "auth.json").symlink_to(mounted_auth)
        (runtime_home / "notes-link.txt").symlink_to(unrelated)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    _configure(monkeypatch, runtime_home, None, tmp_path / "operator")

    archive = tmp_path / "flat-symlink.zip"
    run_backup(Namespace(output=str(archive)))
    with zipfile.ZipFile(archive) as zf:
        assert zf.read("auth.json") == b'{"value":"mounted"}\n'
        assert "notes-link.txt" not in zf.namelist()


def test_active_profile_path_equal_override_keeps_flat_root_and_shared_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_backup, run_import

    source_root = tmp_path / "source"
    source_active_home = source_root / "profiles" / "coder"
    source_active_home.mkdir(parents=True)
    (source_root / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    credentials = {
        **_CREDENTIALS,
        "profiles/sibling/auth.json": b'{"store":"sibling"}\n',
        "profiles/sibling/.anthropic_oauth.json": (
            b'{"store":"sibling-anthropic"}\n'
        ),
    }
    _write_files(source_root, credentials)
    _configure(
        monkeypatch,
        source_active_home,
        source_active_home,
        tmp_path / "source-operator",
    )

    archive = tmp_path / "path-equal.zip"
    run_backup(Namespace(output=str(archive)))
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        assert set(credentials).issubset(names)
        assert not any(name.startswith("_auth-residence/") for name in names)

    destination_root = tmp_path / "destination"
    destination_active_home = destination_root / "profiles" / "coder"
    destination_active_home.mkdir(parents=True)
    _configure(
        monkeypatch,
        destination_active_home,
        destination_active_home,
        tmp_path / "destination-operator",
    )
    run_import(Namespace(zipfile=str(archive), force=True))

    for rel, payload in credentials.items():
        assert (destination_root / rel).read_bytes() == payload


def test_quick_snapshot_uses_explicit_profile_home_and_restores_old_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.auth as auth_module
    from hermes_cli.backup import (
        QuickSnapshotStatus,
        create_quick_snapshot,
        restore_quick_snapshot,
    )

    runtime_root = tmp_path / "runtime"
    explicit_home = runtime_root / "profiles" / "coder"
    ambient_home = runtime_root / "profiles" / "ambient"
    auth_root = tmp_path / "auth"
    explicit_auth = auth_root / "profiles" / "coder"
    explicit_home.mkdir(parents=True)
    ambient_home.mkdir(parents=True)
    (explicit_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    old_primary = b'{"updated_at":"2000-01-01T00:00:00Z","value":"old"}\n'
    old_anthropic = b'{"expiresAt":1,"value":"old"}\n'
    _write_files(
        explicit_auth,
        {
            "auth.json": old_primary,
            ".anthropic_oauth.json": old_anthropic,
        },
    )
    _write_files(
        auth_root,
        {
            "auth.json": b'{"value":"root"}\n',
            "shared/nous_auth.json": b'{"value":"shared"}\n',
        },
    )
    _configure(monkeypatch, ambient_home, auth_root, tmp_path / "operator")

    locked_targets: list[Path] = []
    real_primary_lock = auth_module._auth_store_lock
    real_anthropic_lock = auth_module.anthropic_oauth_store_lock

    @contextmanager
    def primary_lock(*args, **kwargs):
        with real_primary_lock(*args, **kwargs) as locked_path:
            locked_targets.append(locked_path)
            yield locked_path

    @contextmanager
    def anthropic_lock(*args, **kwargs):
        with real_anthropic_lock(*args, **kwargs) as locked_path:
            locked_targets.append(locked_path)
            yield locked_path

    monkeypatch.setattr(auth_module, "_auth_store_lock", primary_lock)
    monkeypatch.setattr(
        auth_module,
        "anthropic_oauth_store_lock",
        anthropic_lock,
    )
    snapshot_id = create_quick_snapshot(hermes_home=explicit_home)
    assert snapshot_id is not None
    assert set(locked_targets) == {
        (explicit_auth / "auth.json").resolve(),
        (explicit_auth / ".anthropic_oauth.json").resolve(),
    }
    snapshot_dir = explicit_home / "state-snapshots" / snapshot_id
    assert (snapshot_dir / "auth.json").read_bytes() == old_primary
    assert (snapshot_dir / ".anthropic_oauth.json").read_bytes() == old_anthropic
    assert not (snapshot_dir / "shared" / "nous_auth.json").exists()

    (explicit_auth / "auth.json").write_bytes(
        b'{"updated_at":"2099-01-01T00:00:00Z","value":"new"}\n'
    )
    (explicit_auth / ".anthropic_oauth.json").write_bytes(
        b'{"expiresAt":9999999999999,"value":"new"}\n'
    )
    assert restore_quick_snapshot(snapshot_id, hermes_home=explicit_home)
    assert (explicit_auth / "auth.json").read_bytes() == old_primary
    assert (explicit_auth / ".anthropic_oauth.json").read_bytes() == old_anthropic
    if os.name == "posix":
        assert (explicit_auth / "auth.json").stat().st_mode & 0o777 == 0o600
        assert (
            explicit_auth / ".anthropic_oauth.json"
        ).stat().st_mode & 0o777 == 0o600


def test_quick_snapshot_round_trips_runtime_credential_suppressions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import (
        QuickSnapshotStatus,
        create_quick_snapshot,
        restore_quick_snapshot,
    )

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    auth_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    suppression_path = runtime_home / ".credential_suppressions.json"
    original = (
        b'{"version":1,"suppressed_sources":'
        b'{"custom:foo":["config:Foo"]}}\n'
    )
    suppression_path.write_bytes(original)
    if os.name != "nt":
        suppression_path.chmod(0o600)
    residence_sentinel = auth_home / ".credential_suppressions.json"
    residence_sentinel.write_bytes(b'{"residence":"untouched"}\n')
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    snapshot_id = create_quick_snapshot(hermes_home=runtime_home)
    assert snapshot_id is not None
    snapshot_file = (
        runtime_home
        / "state-snapshots"
        / snapshot_id
        / ".credential_suppressions.json"
    )
    assert snapshot_file.read_bytes() == original

    suppression_path.write_bytes(b'{"version":1}\n')
    result = restore_quick_snapshot(snapshot_id, hermes_home=runtime_home)

    assert result.status is QuickSnapshotStatus.COMPLETE
    assert suppression_path.read_bytes() == original
    assert residence_sentinel.read_bytes() == b'{"residence":"untouched"}\n'


def test_quick_restore_preserves_live_change_during_lock_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import hermes_cli.auth as auth_module
    from hermes_cli.backup import (
        QuickSnapshotStatus,
        create_quick_snapshot,
        restore_quick_snapshot,
    )

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    auth_home.mkdir()
    target = auth_home / "auth.json"
    target.write_bytes(b'{"value":"snapshot"}\n')
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    snapshot_id = create_quick_snapshot(hermes_home=runtime_home)
    assert snapshot_id is not None

    target.write_bytes(b'{"value":"prepared"}\n')
    real_lock = auth_module._auth_store_lock

    @contextmanager
    def racing_lock(*args, **kwargs):
        target.write_bytes(b'{"value":"live"}\n')
        with real_lock(*args, **kwargs) as locked_path:
            yield locked_path

    monkeypatch.setattr(auth_module, "_auth_store_lock", racing_lock)
    result = restore_quick_snapshot(snapshot_id, hermes_home=runtime_home)
    assert result.status is QuickSnapshotStatus.PARTIAL
    assert target.read_bytes() == b'{"value":"live"}\n'
    assert "live credential changed" in caplog.text


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_quick_credential_snapshot_is_private_under_permissive_umask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import create_quick_snapshot

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    auth_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (auth_home / "auth.json").write_bytes(b'{"value":"secret"}\n')
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    previous_umask = os.umask(0)
    try:
        snapshot_id = create_quick_snapshot(hermes_home=runtime_home)
    finally:
        os.umask(previous_umask)
    assert snapshot_id is not None
    snapshot_auth = (
        runtime_home / "state-snapshots" / snapshot_id / "auth.json"
    )
    assert snapshot_auth.stat().st_mode & 0o777 == 0o600


def test_import_preserves_live_change_during_archive_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_import

    archive = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("_auth-residence/auth.json", b'{"value":"backup"}\n')

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    auth_home.mkdir()
    target = auth_home / "auth.json"
    target.write_bytes(b'{"value":"prepared"}\n')
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    real_read = zipfile.ZipFile.read

    def racing_read(self, name, *args, **kwargs):
        payload = real_read(self, name, *args, **kwargs)
        if str(name).endswith("/auth.json"):
            target.write_bytes(b'{"value":"live"}\n')
        return payload

    monkeypatch.setattr(zipfile.ZipFile, "read", racing_read)
    run_import(Namespace(zipfile=str(archive), force=True))
    assert target.read_bytes() == b'{"value":"live"}\n'


def test_import_existing_credential_requires_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli.backup import run_import

    archive = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("_auth-residence/auth.json", b'{"value":"backup"}\n')

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    auth_home.mkdir()
    live = auth_home / "auth.json"
    live.write_bytes(b'{"value":"live"}\n')
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    run_import(Namespace(zipfile=str(archive), force=False))
    assert live.read_bytes() == b'{"value":"live"}\n'
    assert not (runtime_home / "config.yaml").exists()
    assert "configuration or provider credentials" in capsys.readouterr().out


def test_prefixed_namespaced_member_wins_and_unknown_reserved_member_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli.backup import run_import

    archive = tmp_path / "prefixed.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(".hermes/config.yaml", "model: {}\n")
        zf.writestr(".hermes/auth.json", b'{"value":"legacy"}\n')
        zf.writestr(
            ".hermes/_auth-residence/auth.json",
            b'{"value":"namespaced"}\n',
        )
        zf.writestr(
            ".hermes/_auth-residence/unknown.json",
            b'{"value":"unknown"}\n',
        )

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    run_import(Namespace(zipfile=str(archive), force=True))

    assert (auth_home / "auth.json").read_bytes() == b'{"value":"namespaced"}\n'
    assert not (auth_home / "unknown.json").exists()
    assert not (runtime_home / "_auth-residence").exists()
    assert "unknown reserved auth-residence member" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("lock_name", "filename"),
    [
        ("_auth_store_lock", "auth.json"),
        ("anthropic_oauth_store_lock", ".anthropic_oauth.json"),
        ("_nous_shared_store_lock", "nous_auth.json"),
    ],
)
def test_explicit_pinned_lock_target_is_not_resolved_again(
    lock_name: str,
    filename: str,
    tmp_path: Path,
) -> None:
    import hermes_cli.auth as auth_module

    target = (tmp_path / "store" / filename).absolute()
    target.parent.mkdir()
    outside = tmp_path / "outside" / filename
    outside.parent.mkdir()
    target.symlink_to(outside)

    lock = getattr(auth_module, lock_name)
    with lock(target_path=target, target_is_pinned=True) as locked:
        assert locked == target

    with pytest.raises(ValueError):
        with lock(target_is_pinned=True):
            pass
    with pytest.raises(ValueError):
        with lock(
            target_path=Path("relative") / filename,
            target_is_pinned=True,
        ):
            pass


@pytest.mark.parametrize(
    ("credential_rel", "lock_name"),
    [
        ("auth.json", "_auth_store_lock"),
        (".anthropic_oauth.json", "anthropic_oauth_store_lock"),
        ("shared/nous_auth.json", "_nous_shared_store_lock"),
    ],
)
def test_import_rejects_absent_to_symlink_race_for_all_stores(
    credential_rel: str,
    lock_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.auth as auth_module
    from hermes_cli.backup import run_import

    archive = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            f"_auth-residence/{credential_rel}",
            b'{"value":"backup"}\n',
        )

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    target = auth_home / credential_rel
    outside = tmp_path / "outside" / target.name
    outside.parent.mkdir()
    real_lock = getattr(auth_module, lock_name)

    @contextmanager
    def racing_lock(*args, **kwargs):
        target.symlink_to(outside)
        with real_lock(*args, **kwargs) as locked_path:
            yield locked_path

    monkeypatch.setattr(auth_module, lock_name, racing_lock)
    run_import(Namespace(zipfile=str(archive), force=True))

    assert target.is_symlink()
    assert not outside.exists()


def test_import_rejects_existing_symlink_retarget_but_preserves_stable_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.auth as auth_module
    from hermes_cli.backup import run_import

    archive = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "_auth-residence/.anthropic_oauth.json",
            b'{"value":"backup"}\n',
        )

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    stores = tmp_path / "stores"
    runtime_home.mkdir()
    auth_home.mkdir()
    stores.mkdir()
    target = auth_home / ".anthropic_oauth.json"
    original = stores / "original.json"
    replacement = stores / "replacement.json"
    original.write_bytes(b'{"value":"original"}\n')
    replacement.write_bytes(b'{"value":"replacement"}\n')
    target.symlink_to(original)
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    run_import(Namespace(zipfile=str(archive), force=True))
    assert target.is_symlink()
    assert original.read_bytes() == b'{"value":"backup"}\n'

    original.write_bytes(b'{"value":"original-again"}\n')
    real_lock = auth_module.anthropic_oauth_store_lock

    @contextmanager
    def racing_lock(*args, **kwargs):
        target.unlink()
        target.symlink_to(replacement)
        with real_lock(*args, **kwargs) as locked_path:
            yield locked_path

    monkeypatch.setattr(
        auth_module,
        "anthropic_oauth_store_lock",
        racing_lock,
    )
    run_import(Namespace(zipfile=str(archive), force=True))
    assert original.read_bytes() == b'{"value":"original-again"}\n'
    assert replacement.read_bytes() == b'{"value":"replacement"}\n'


def test_quick_restore_rejects_absent_to_symlink_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.auth as auth_module
    from hermes_cli.backup import (
        QuickSnapshotStatus,
        create_quick_snapshot,
        restore_quick_snapshot,
    )

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    auth_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    target = auth_home / "auth.json"
    target.write_bytes(b'{"value":"snapshot"}\n')
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    snapshot_id = create_quick_snapshot(hermes_home=runtime_home)
    assert snapshot_id is not None
    target.unlink()

    outside = tmp_path / "outside" / "auth.json"
    outside.parent.mkdir()
    real_lock = auth_module._auth_store_lock

    @contextmanager
    def racing_lock(*args, **kwargs):
        target.symlink_to(outside)
        with real_lock(*args, **kwargs) as locked_path:
            yield locked_path

    monkeypatch.setattr(auth_module, "_auth_store_lock", racing_lock)
    result = restore_quick_snapshot(
        snapshot_id,
        hermes_home=runtime_home,
    )
    assert result.status is QuickSnapshotStatus.PARTIAL
    assert target.is_symlink()
    assert not outside.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory identity")
def test_import_rejects_parent_swap_during_pinned_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.auth as auth_module
    from hermes_cli.backup import run_import

    archive = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "_auth-residence/auth.json",
            b'{"value":"backup"}\n',
        )
    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    moved_auth_home = tmp_path / "auth-moved"
    runtime_home.mkdir()
    auth_home.mkdir()
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    real_lock = auth_module._auth_store_lock

    @contextmanager
    def racing_lock(*args, **kwargs):
        auth_home.rename(moved_auth_home)
        auth_home.mkdir()
        with real_lock(*args, **kwargs) as locked_path:
            yield locked_path

    monkeypatch.setattr(auth_module, "_auth_store_lock", racing_lock)
    run_import(Namespace(zipfile=str(archive), force=True))
    assert not (auth_home / "auth.json").exists()
    assert not (moved_auth_home / "auth.json").exists()


@pytest.mark.parametrize("operation", ["import", "quick-restore"])
def test_restore_fallback_does_not_open_credential_parent_directory(
    operation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.backup as backup_module

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    auth_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    target = auth_home / "auth.json"
    target.write_bytes(b'{"value":"old"}\n')
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    if operation == "import":
        archive = tmp_path / "backup.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(
                "_auth-residence/auth.json",
                b'{"value":"restored"}\n',
            )
    else:
        snapshot_id = backup_module.create_quick_snapshot(
            hermes_home=runtime_home
        )
        assert snapshot_id is not None
        target.write_bytes(b'{"value":"changed"}\n')

    monkeypatch.setattr(backup_module.os, "supports_dir_fd", set())
    real_open = backup_module.os.open

    def reject_directory_open(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)):
            if Path(path) == auth_home:
                raise PermissionError("directory open unavailable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(backup_module.os, "open", reject_directory_open)
    if operation == "import":
        backup_module.run_import(
            Namespace(zipfile=str(archive), force=True)
        )
    else:
        result = backup_module.restore_quick_snapshot(
            snapshot_id,
            hermes_home=runtime_home,
        )
        assert result.status is backup_module.QuickSnapshotStatus.COMPLETE
    expected = (
        b'{"value":"restored"}\n'
        if operation == "import"
        else b'{"value":"old"}\n'
    )
    assert target.read_bytes() == expected


def test_import_baseline_is_captured_before_overwrite_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_import

    archive = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "_auth-residence/auth.json",
            b'{"value":"backup"}\n',
        )

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    target = auth_home / "auth.json"
    real_exists = Path.exists
    injected = False

    def racing_exists(path: Path) -> bool:
        nonlocal injected
        if path == runtime_home / "config.yaml" and not injected:
            injected = True
            target.parent.mkdir(parents=True)
            target.write_bytes(b'{"value":"concurrent-login"}\n')
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", racing_exists)
    run_import(Namespace(zipfile=str(archive), force=False))
    assert target.read_bytes() == b'{"value":"concurrent-login"}\n'


@pytest.mark.parametrize("kind", ["full", "pre-update", "pre-migration"])
def test_full_backup_reads_all_store_kinds_under_pinned_locks(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.auth as auth_module

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _write_files(
        auth_home,
        {
            "auth.json": b"primary",
            ".anthropic_oauth.json": b"anthropic",
            "shared/nous_auth.json": b"shared",
        },
    )
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    calls: list[tuple[str, Path, bool]] = []
    for lock_name in (
        "_auth_store_lock",
        "anthropic_oauth_store_lock",
        "_nous_shared_store_lock",
    ):
        real_lock = getattr(auth_module, lock_name)

        @contextmanager
        def spy_lock(*args, __name=lock_name, __real=real_lock, **kwargs):
            calls.append(
                (
                    __name,
                    Path(kwargs["target_path"]),
                    kwargs.get("target_is_pinned", False),
                )
            )
            with __real(*args, **kwargs) as locked_path:
                yield locked_path

        monkeypatch.setattr(auth_module, lock_name, spy_lock)

    archive = _create_full_archive(kind, runtime_home, tmp_path / f"{kind}.zip")
    assert archive.is_file()
    assert {name for name, _path, _pinned in calls} == {
        "_auth_store_lock",
        "anthropic_oauth_store_lock",
        "_nous_shared_store_lock",
    }
    assert all(path.is_absolute() and pinned for _name, path, pinned in calls)


@pytest.mark.parametrize(
    ("rel", "kind"),
    [
        ("auth.json", "primary"),
        (".anthropic_oauth.json", "anthropic"),
        ("shared/nous_auth.json", "shared"),
    ],
)
@pytest.mark.parametrize("force_fallback", [False, True])
def test_pinned_handle_read_accepts_unchanged_store_for_all_store_kinds(
    rel: str,
    kind: str,
    force_fallback: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.backup as backup_module

    target = tmp_path / "auth" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"expected")
    if force_fallback:
        monkeypatch.setattr(backup_module.os, "supports_dir_fd", set())

    payload = backup_module._read_backup_credential(
        backup_module._BackupCredential(
            path=target,
            archive_path=Path(rel),
            kind=kind,
        )
    )
    assert payload == b"expected"


@pytest.mark.parametrize(
    ("rel", "archive_rel"),
    [
        ("auth.json", "_auth-residence/auth.json"),
        (
            ".anthropic_oauth.json",
            "_auth-residence/.anthropic_oauth.json",
        ),
        (
            "shared/nous_auth.json",
            "_auth-residence/shared/nous_auth.json",
        ),
    ],
)
@pytest.mark.parametrize("force_fallback", [False, True])
def test_full_backup_rejects_regular_to_symlink_swap_at_actual_store_read(
    rel: str,
    archive_rel: str,
    force_fallback: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import hermes_cli.backup as backup_module

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    target = auth_home / rel
    outside = tmp_path / "outside.json"
    runtime_home.mkdir()
    target.parent.mkdir(parents=True, exist_ok=True)
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    target.write_bytes(b"authorized")
    outside.write_bytes(b"outside-secret")
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    if force_fallback:
        monkeypatch.setattr(backup_module.os, "supports_dir_fd", set())
        # Exercise the handle-identity fallback itself rather than relying on
        # POSIX O_NOFOLLOW to reject the swapped symlink before open.
        monkeypatch.setattr(backup_module.os, "O_NOFOLLOW", 0, raising=False)
    real_open = backup_module.os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        is_target_open = (
            dir_fd is not None and os.fspath(path) == target.name
        ) or (
            dir_fd is None
            and Path(path) == target
        )
        if is_target_open and not swapped:
            swapped = True
            target.unlink()
            target.symlink_to(outside)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(backup_module.os, "open", racing_open)
    archive = tmp_path / "backup.zip"
    backup_module.run_backup(Namespace(output=str(archive)))

    assert swapped
    assert "Backup incomplete" in capsys.readouterr().out
    with zipfile.ZipFile(archive) as zf:
        assert archive_rel not in zf.namelist()
        assert b"outside-secret" not in (
            zf.read(name) for name in zf.namelist()
        )


def test_full_backup_rejects_store_symlink_retargets_under_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import hermes_cli.auth as auth_module
    from hermes_cli.backup import run_backup

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    stores = tmp_path / "stores"
    runtime_home.mkdir()
    auth_home.mkdir()
    stores.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    specs = [
        ("auth.json", "_auth_store_lock"),
        (".anthropic_oauth.json", "anthropic_oauth_store_lock"),
        ("shared/nous_auth.json", "_nous_shared_store_lock"),
    ]
    for rel, lock_name in specs:
        original = stores / f"{lock_name}-original"
        replacement = stores / f"{lock_name}-replacement"
        original.write_bytes(b"original")
        replacement.write_bytes(b"replacement")
        link = auth_home / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(original)
        real_lock = getattr(auth_module, lock_name)

        @contextmanager
        def racing_lock(
            *args,
            __link=link,
            __replacement=replacement,
            __real=real_lock,
            **kwargs,
        ):
            __link.unlink()
            __link.symlink_to(__replacement)
            with __real(*args, **kwargs) as locked_path:
                yield locked_path

        monkeypatch.setattr(auth_module, lock_name, racing_lock)

    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    archive = tmp_path / "backup.zip"
    run_backup(Namespace(output=str(archive)))
    assert "Backup incomplete" in capsys.readouterr().out
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert not any(name.startswith("_auth-residence/") for name in names)


def test_backup_enumeration_failure_is_not_reported_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli.backup import run_backup

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    profiles = auth_home / "profiles"
    runtime_home.mkdir()
    profiles.mkdir(parents=True)
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    real_iterdir = Path.iterdir

    def denied_iterdir(path: Path):
        if path == profiles:
            raise PermissionError("profile enumeration denied")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", denied_iterdir)
    archive = tmp_path / "backup.zip"
    run_backup(Namespace(output=str(archive)))
    output = capsys.readouterr().out
    assert "Could not inspect provider credentials" in output
    assert "Backup complete" not in output
    assert not archive.exists()


def test_broken_profile_entry_does_not_hide_valid_profile_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_backup

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _write_files(
        auth_home,
        {"profiles/good/auth.json": b'{"value":"good"}\n'},
    )
    broken = auth_home / "profiles" / "broken"
    broken.symlink_to(tmp_path / "missing-profile", target_is_directory=True)
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    archive = tmp_path / "backup.zip"
    run_backup(Namespace(output=str(archive)))
    with zipfile.ZipFile(archive) as zf:
        assert zf.read(
            "_auth-residence/profiles/good/auth.json"
        ) == b'{"value":"good"}\n'


@pytest.mark.parametrize("kind", ["pre-update", "pre-migration"])
@pytest.mark.parametrize("failure_mode", ["read", "write"])
def test_lifecycle_full_backup_credential_failure_removes_partial_archive(
    kind: str,
    failure_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.backup as backup_module

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    auth_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (auth_home / "auth.json").write_bytes(b"credential")
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    if failure_mode == "read":
        def denied_read(_credential):
            raise PermissionError("credential read denied")

        monkeypatch.setattr(
            backup_module,
            "_read_backup_credential",
            denied_read,
        )
    else:
        real_writestr = zipfile.ZipFile.writestr

        def denied_write(self, name, data, *args, **kwargs):
            if str(name).endswith("auth.json"):
                raise PermissionError("credential archive write denied")
            return real_writestr(self, name, data, *args, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "writestr", denied_write)
    if kind == "pre-update":
        result = backup_module.create_pre_update_backup(
            hermes_home=runtime_home
        )
    else:
        result = backup_module.create_pre_migration_backup(
            hermes_home=runtime_home
        )
    assert result is None
    backups_dir = runtime_home / "backups"
    assert not list(backups_dir.glob("*.zip"))


@pytest.mark.parametrize(
    "artifact",
    [
        "auth.json.corrupt.20260731",
        ".anthropic_oauth.json.corrupt.20260731",
        "profiles/coder/auth.json.corrupt.20260731",
        "profiles/coder/.anthropic_oauth.json.corrupt.20260731",
        "shared/nous_auth.json.corrupt.20260731",
    ],
)
def test_suffixed_quarantine_is_neither_archived_nor_imported(
    artifact: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_backup, run_import

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _write_files(source, {artifact: b"quarantined-secret"})
    _configure(monkeypatch, source, None, tmp_path / "source-operator")
    archive = tmp_path / "backup.zip"
    run_backup(Namespace(output=str(archive)))
    with zipfile.ZipFile(archive) as zf:
        assert artifact not in zf.namelist()

    legacy = tmp_path / "legacy.zip"
    with zipfile.ZipFile(legacy, "w") as zf:
        zf.writestr("config.yaml", "model: {}\n")
        zf.writestr(artifact, b"quarantined-secret")
    destination = tmp_path / "destination"
    destination.mkdir()
    _configure(
        monkeypatch,
        destination,
        None,
        tmp_path / "destination-operator",
    )
    run_import(Namespace(zipfile=str(legacy), force=True))
    assert not (destination / artifact).exists()


@pytest.mark.parametrize(
    "operation",
    ["pre-update", "pre-migration", "quick"],
)
def test_invalid_residence_causes_zero_backup_or_snapshot_mutation(
    operation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_constants import HermesAuthHomeError
    from hermes_cli.backup import (
        create_pre_migration_backup,
        create_pre_update_backup,
        create_quick_snapshot,
    )

    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _configure(monkeypatch, runtime_home, None, tmp_path / "operator")
    monkeypatch.setenv("HERMES_AUTH_HOME", "relative/not-allowed")

    if operation == "quick":
        with pytest.raises(HermesAuthHomeError):
            create_quick_snapshot(hermes_home=runtime_home)
    elif operation == "pre-update":
        assert create_pre_update_backup(hermes_home=runtime_home) is None
    else:
        assert create_pre_migration_backup(hermes_home=runtime_home) is None
    assert not (runtime_home / "backups").exists()
    assert not (runtime_home / "state-snapshots").exists()


def test_import_parent_setup_failure_removes_created_credential_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_import

    archive = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "_auth-residence/auth.json",
            b'{"value":"root"}\n',
        )
        zf.writestr(
            "_auth-residence/profiles/coder/auth.json",
            b'{"value":"profile"}\n',
        )
    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    denied_parent = auth_home / "profiles" / "coder"
    real_mkdir = Path.mkdir

    def denied_mkdir(path: Path, *args, **kwargs):
        if path == denied_parent:
            raise PermissionError("profile directory denied")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied_mkdir)
    run_import(Namespace(zipfile=str(archive), force=True))
    assert not auth_home.exists()


def test_quick_capture_failure_and_restore_cas_are_reported_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import hermes_cli.auth as auth_module
    from hermes_cli.backup import (
        QuickSnapshotStatus,
        create_quick_snapshot,
        restore_quick_snapshot,
    )
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    runtime_home = tmp_path / "runtime"
    auth_home = tmp_path / "auth"
    runtime_home.mkdir()
    auth_home.mkdir()
    (runtime_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    target = auth_home / "auth.json"
    target.write_bytes(b'{"value":"snapshot"}\n')
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")

    def denied_lock(*_args, **_kwargs):
        @contextmanager
        def fail():
            raise TimeoutError("credential lock denied")
            yield

        return fail()

    monkeypatch.setattr(auth_module, "_auth_store_lock", denied_lock)
    partial = create_quick_snapshot(hermes_home=runtime_home)
    assert partial is not None
    assert partial.status is QuickSnapshotStatus.PARTIAL
    CLICommandsMixin()._handle_snapshot_command("/snapshot create denied")
    create_output = capsys.readouterr().out
    assert "Snapshot incomplete" in create_output
    assert "Snapshot created:" not in create_output

    monkeypatch.undo()
    _configure(monkeypatch, runtime_home, auth_home, tmp_path / "operator")
    partial_restore = restore_quick_snapshot(
        partial,
        hermes_home=runtime_home,
    )
    assert partial_restore.status is QuickSnapshotStatus.PARTIAL
    complete = create_quick_snapshot(
        label="complete",
        hermes_home=runtime_home,
    )
    assert complete is not None
    target.write_bytes(b'{"value":"prepared"}\n')
    real_lock = auth_module._auth_store_lock

    @contextmanager
    def racing_lock(*args, **kwargs):
        target.write_bytes(b'{"value":"live"}\n')
        with real_lock(*args, **kwargs) as locked_path:
            yield locked_path

    monkeypatch.setattr(auth_module, "_auth_store_lock", racing_lock)
    CLICommandsMixin()._handle_snapshot_command(
        f"/snapshot restore {complete}"
    )
    output = capsys.readouterr().out
    assert "Restore incomplete" in output
    assert "Restored state from" not in output
    assert target.read_bytes() == b'{"value":"live"}\n'


def test_exact_reserved_auth_residence_member_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli.backup import run_import

    archive = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("config.yaml", "model: {}\n")
        zf.writestr("_auth-residence", "reserved payload")
    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir()
    _configure(monkeypatch, runtime_home, None, tmp_path / "operator")
    run_import(Namespace(zipfile=str(archive), force=True))
    assert not (runtime_home / "_auth-residence").exists()
