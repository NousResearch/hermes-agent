"""The restore preflight's epoch floor and the integrity check of an archived copy, through the admin door."""

from __future__ import annotations

from persistence.core.admin import PHYSICAL_ADMIN, AdminUnsupported

_SALVAGE = ("; refusing in-place restore. Use `hermes sessions recover --source <snapshot> --output <separate-path>` "
            "to salvage transcripts into a separate output.")


def _admin(backend, backend_name, target):
    admin = backend.open_store_admin(target)
    if not backend.capabilities.physical_admin:
        # CONTRACT: a backend without physical admin answers the door; neither check runs there.
        assert admin == AdminUnsupported(PHYSICAL_ADMIN, backend_name)
        return None
    return admin


def test_the_epoch_floor_is_the_destination_runtime_epoch(backend, backend_name, scope_a, tmp_path):
    admin = _admin(backend, backend_name, scope_a)
    if admin is None:
        return
    backend.open(scope_a)
    epoch = backend.begin_epoch(scope_a, "instance-a")
    # CONTRACT: a store's floor is the epoch its runtime last began.
    assert admin.restore_epoch_floor() == epoch
    missing = _admin(backend, backend_name, tmp_path / "absent" / "state.db")
    empty = tmp_path / "empty" / "state.db"
    empty.parent.mkdir()
    empty.write_bytes(b"")
    # CONTRACT: no file, or a database without the epoch table, has floor 0.
    assert missing.restore_epoch_floor() == 0
    assert _admin(backend, backend_name, empty).restore_epoch_floor() == 0


def test_an_unreadable_destination_epoch_refuses_the_restore(backend, backend_name, tmp_path):
    damaged = tmp_path / "damaged" / "state.db"
    damaged.parent.mkdir()
    damaged.write_bytes(b"not a database" * 64)
    admin = _admin(backend, backend_name, damaged)
    if admin is None:
        return
    try:
        admin.restore_epoch_floor()
    except OSError as exc:
        # CONTRACT: the refusal is an OSError naming the file and the transcript salvage.
        assert str(exc) == f"Cannot read the canonical runtime epoch at {damaged}{_SALVAGE}"
    else:
        raise AssertionError("an unreadable destination epoch must refuse")


def test_an_archived_copy_is_verified_offline(backend, backend_name, scope_a, tmp_path):
    admin = _admin(backend, backend_name, scope_a)
    if admin is None:
        return
    backend.open(scope_a)
    backend.begin_epoch(scope_a, "instance-a")
    copy = tmp_path / "copy.db"
    assert admin.backup_to(copy)
    size = copy.stat().st_size
    small, garbage, missing = tmp_path / "small.db", tmp_path / "garbage.db", tmp_path / "missing.db"
    small.write_bytes(b"SQLite")
    garbage.write_bytes(b"\x01" * 200)
    header_only = tmp_path / "header-only.db"
    header_only.write_bytes(b"SQLite format 3\x00" + b"\xff" * 4080)
    # CONTRACT: each answer is the {valid, message, size} the integrity policy gives.
    assert admin.verify_file(copy) == {"valid": True, "message": "integrity check passed", "size": size}
    assert admin.verify_file(copy, max_bytes=1) == {
        "valid": True, "size": size,
        "message": f"size {size:,} bytes exceeds max_bytes 1; skipped PRAGMA integrity_check (header + schema probe passed)"}
    assert admin.verify_file(missing) == {"valid": False, "message": f"not found: {missing}", "size": None}
    assert admin.verify_file(small) == {
        "valid": False, "message": "too small (6 bytes) to be a valid SQLite database", "size": 6}
    head = (b"\x01" * 16).hex()
    assert admin.verify_file(garbage) == {
        "valid": False, "message": f"missing SQLite header magic (got {head!r})", "size": 200}
    damaged = admin.verify_file(header_only)
    assert damaged["valid"] is False and damaged["size"] == 4096
    assert damaged["message"].startswith("cannot open database: ")
    probe = admin.verify_file(header_only, max_bytes=1)
    assert probe["valid"] is False and probe["message"].startswith("schema probe failed: ")
