"""Contract tests for quick-snapshot secret file permissions (#77470).

These assert a RELATION — "a secret-named file inside a snapshot is never
readable by group or other, whatever the source mode was" — rather than
freezing an exact mode literal for every file. Non-secret entries are
asserted to keep their source mode so the hardening cannot silently become
a blanket chmod.
"""

import os
import sqlite3
import stat
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="POSIX file permission bits only"
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def permissive_umask():
    """Run with umask 0 so a passing test proves an explicit chmod.

    With a masking umask the snapshot can look hardened purely by accident of
    process state; clearing it removes that false negative.
    """
    old = os.umask(0)
    try:
        yield
    finally:
        os.umask(old)


def _make_home(tmp_path: Path, secret_mode: int) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("OPENROUTER_API_KEY=test-key-123\n")
    (home / "auth.json").write_text('{"providers": {"nous": "token"}}\n')
    (home / "config.yaml").write_text("model:\n  provider: openrouter\n")

    conn = sqlite3.connect(str(home / "state.db"))
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, data TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('s1', 'hello world')")
    conn.commit()
    conn.close()

    for name in (".env", "auth.json", "state.db"):
        os.chmod(home / name, secret_mode)
    os.chmod(home / "config.yaml", 0o644)
    return home


class TestQuickSnapshotSecretPermissions:
    """#77470: snapshot copies must ENFORCE 0600, not inherit the source mode."""

    @pytest.mark.parametrize("source_mode", [0o600, 0o644, 0o664, 0o666])
    @pytest.mark.parametrize("secret_name", [".env", "auth.json", "state.db"])
    def test_secret_copy_never_group_or_world_accessible(
        self, tmp_path, permissive_umask, source_mode, secret_name
    ):
        from hermes_cli.backup import create_quick_snapshot

        home = _make_home(tmp_path, source_mode)
        snap_id = create_quick_snapshot(label="perm", hermes_home=home)
        assert snap_id is not None

        copy = home / "state-snapshots" / snap_id / secret_name
        assert copy.exists(), f"{secret_name} missing from snapshot"

        mode = _mode(copy)
        assert not (mode & 0o077), (
            f"{secret_name} snapshotted from a {source_mode:04o} source landed at "
            f"{mode:04o} — readable by group/other"
        )

    @pytest.mark.parametrize("source_mode", [0o600, 0o644])
    def test_non_secret_files_keep_their_source_mode(
        self, tmp_path, permissive_umask, source_mode
    ):
        """The hardening must be scoped to secrets, not a blanket chmod."""
        from hermes_cli.backup import create_quick_snapshot

        home = _make_home(tmp_path, source_mode)
        snap_id = create_quick_snapshot(label="perm", hermes_home=home)
        snap_dir = home / "state-snapshots" / snap_id

        assert _mode(snap_dir / "config.yaml") == _mode(home / "config.yaml")

    def test_snapshot_directories_not_group_or_world_accessible(
        self, tmp_path, permissive_umask
    ):
        """A hardened file under a traversable dir is still a listing leak."""
        from hermes_cli.backup import create_quick_snapshot

        home = _make_home(tmp_path, 0o600)
        snap_id = create_quick_snapshot(label="perm", hermes_home=home)

        root = home / "state-snapshots"
        for directory in (root, root / snap_id):
            mode = _mode(directory)
            assert not (mode & 0o077), (
                f"{directory.name}/ is {mode:04o} — traversable by group/other"
            )

    def test_hardening_does_not_break_restore(self, tmp_path, permissive_umask):
        """False-positive guard: the snapshot must still be usable."""
        from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot

        home = _make_home(tmp_path, 0o644)
        snap_id = create_quick_snapshot(label="perm", hermes_home=home)

        (home / ".env").write_text("CLOBBERED\n")
        (home / "state.db").unlink()

        assert restore_quick_snapshot(snap_id, hermes_home=home) is True
        assert (home / ".env").read_text() == "OPENROUTER_API_KEY=test-key-123\n"

        conn = sqlite3.connect(str(home / "state.db"))
        rows = conn.execute("SELECT * FROM sessions").fetchall()
        conn.close()
        assert rows == [("s1", "hello world")]

    @pytest.mark.parametrize("secret_name", [".env", "auth.json", "state.db"])
    def test_restore_does_not_loosen_live_secrets(
        self, tmp_path, permissive_umask, secret_name
    ):
        """Restoring a legacy loose snapshot must not downgrade live secrets.

        The zip-import path already chmods 0600 after extract; the quick-snapshot
        restore path is the same contract.
        """
        from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot

        home = _make_home(tmp_path, 0o644)
        snap_id = create_quick_snapshot(label="perm", hermes_home=home)

        # Simulate a snapshot written by an older, unhardened build.
        snap_dir = home / "state-snapshots" / snap_id
        for name in (".env", "auth.json", "state.db"):
            os.chmod(snap_dir / name, 0o644)

        for name in (".env", "auth.json", "state.db"):
            os.chmod(home / name, 0o600)

        assert restore_quick_snapshot(snap_id, hermes_home=home) is True

        mode = _mode(home / secret_name)
        assert not (mode & 0o077), (
            f"restore left live {secret_name} at {mode:04o} — readable by group/other"
        )
