"""Tests for the one-shot legacy ``~/.hermes`` -> ``%LOCALAPPDATA%\\hermes``
fill-gap migration (``hermes_cli/legacy_home_migration.py``).

Regression context: the Windows default home moved to ``%LOCALAPPDATA%\\hermes``
with no data migration at all.  A template-seeded new home came up without
``auth.json`` (provider credentials) and ``cron/jobs.json`` (all scheduled
jobs), so cron jobs died silently with "No Codex credentials stored".  These
tests pin the fill-gap semantics: missing state is copied, existing state is
never overwritten, junk (caches/locks/tooling) is excluded, and the pass is
one-shot via a marker file.
"""

import json

import pytest

from hermes_cli import legacy_home_migration as mig


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A populated legacy home and an (initially empty) new home.

    The module is pinned so ``new_home`` is treated as the platform default
    and the platform check passes regardless of the host OS.
    """
    legacy = tmp_path / "dot_hermes"
    new = tmp_path / "localappdata_hermes"
    legacy.mkdir()

    # State the incident lost:
    (legacy / "auth.json").write_text('{"providers": {"codex": "tok"}}')
    (legacy / "cron").mkdir()
    (legacy / "cron" / "jobs.json").write_text('{"jobs": [{"id": "j1"}]}')

    # Other state that must carry over:
    (legacy / "config.yaml").write_text("model: real-config\n")
    (legacy / ".env").write_text("SOME_KEY=1\n")
    (legacy / "processes.json").write_text("{}")
    (legacy / "gateway_state.json").write_text('{"desired": "running"}')
    (legacy / "channel_directory.json").write_text("{}")
    (legacy / "state.db").write_bytes(b"sqlite-main")
    (legacy / "state.db-wal").write_bytes(b"sqlite-wal")
    (legacy / "state.db-shm").write_bytes(b"sqlite-shm")
    (legacy / "sessions").mkdir()
    (legacy / "sessions" / "abc.jsonl").write_text("{}\n")
    (legacy / "skills").mkdir()
    (legacy / "skills" / "myskill").mkdir()
    (legacy / "skills" / "myskill" / "SKILL.md").write_text("# skill")

    # Junk that must NOT carry over:
    (legacy / "logs").mkdir()
    (legacy / "logs" / "gateway.log").write_text("old logs")
    (legacy / "cache").mkdir()
    (legacy / "cache" / "blob").write_text("x")
    (legacy / "image_cache").mkdir()
    (legacy / "gateway.pid").write_text("123")
    (legacy / "gateway.lock").write_text("")
    (legacy / "auth.lock").write_text("")
    (legacy / "node").mkdir()
    (legacy / "node" / "bin").mkdir(parents=True, exist_ok=True)
    (legacy / "hermes-agent").mkdir()
    (legacy / "hermes-agent" / "cli.py").write_text("# checkout")

    monkeypatch.setattr(
        mig, "_get_platform_default_hermes_home", lambda: new
    )
    return legacy, new


def _run(legacy, new):
    return mig.maybe_migrate_legacy_windows_home(
        new_home=new, legacy_home=legacy, platform="win32"
    )


class TestFillGapCopy:
    def test_auth_and_cron_jobs_are_copied(self, homes):
        """The exact files the incident lost must be carried over."""
        legacy, new = homes
        summary = _run(legacy, new)
        assert summary is not None
        assert (new / "auth.json").read_text() == '{"providers": {"codex": "tok"}}'
        assert (new / "cron" / "jobs.json").read_text() == '{"jobs": [{"id": "j1"}]}'

    def test_all_state_files_are_copied(self, homes):
        legacy, new = homes
        _run(legacy, new)
        for rel in (
            "config.yaml",
            ".env",
            "processes.json",
            "gateway_state.json",
            "channel_directory.json",
            "state.db",
            "sessions/abc.jsonl",
            "skills/myskill/SKILL.md",
        ):
            assert (new / rel).exists(), f"missing {rel}"

    def test_sqlite_wal_copied_with_db_but_shm_is_not(self, homes):
        legacy, new = homes
        _run(legacy, new)
        assert (new / "state.db-wal").exists()
        assert not (new / "state.db-shm").exists()

    def test_junk_is_excluded(self, homes):
        legacy, new = homes
        _run(legacy, new)
        for rel in (
            "logs",
            "cache",
            "image_cache",
            "gateway.pid",
            "gateway.lock",
            "auth.lock",
            "node",
            "hermes-agent",
        ):
            assert not (new / rel).exists(), f"should not have copied {rel}"

    def test_existing_files_are_never_overwritten(self, homes):
        """The new home may hold newer state — it always wins."""
        legacy, new = homes
        new.mkdir()
        (new / "config.yaml").write_text("model: newer-config\n")
        (new / "cron").mkdir()
        (new / "cron" / "jobs.json").write_text('{"jobs": ["newer"]}')

        summary = _run(legacy, new)

        assert (new / "config.yaml").read_text() == "model: newer-config\n"
        assert (new / "cron" / "jobs.json").read_text() == '{"jobs": ["newer"]}'
        # Gaps beside the existing files are still filled.
        assert (new / "auth.json").exists()
        assert summary["skipped_existing_count"] >= 2

    def test_directories_merge_instead_of_being_skipped(self, homes):
        """An existing dir in the new home must not block missing children."""
        legacy, new = homes
        new.mkdir()
        (new / "sessions").mkdir()
        (new / "sessions" / "zzz.jsonl").write_text("{}\n")

        _run(legacy, new)

        assert (new / "sessions" / "abc.jsonl").exists()
        assert (new / "sessions" / "zzz.jsonl").exists()


class TestOneShotAndGuards:
    def test_marker_written_and_second_run_is_noop(self, homes):
        legacy, new = homes
        first = _run(legacy, new)
        assert first is not None
        marker = new / mig.MARKER_NAME
        assert marker.exists()
        payload = json.loads(marker.read_text())
        assert payload["copied_count"] == first["copied_count"] > 0

        # A file added to legacy afterwards must NOT be copied anymore.
        (legacy / "late.json").write_text("{}")
        second = _run(legacy, new)
        assert second is None
        assert not (new / "late.json").exists()

    def test_noop_off_windows(self, homes):
        legacy, new = homes
        assert (
            mig.maybe_migrate_legacy_windows_home(
                new_home=new, legacy_home=legacy, platform="linux"
            )
            is None
        )
        assert not new.exists()

    def test_noop_when_home_is_not_platform_default(self, homes, tmp_path, monkeypatch):
        """A custom HERMES_HOME is respected and left alone."""
        legacy, new = homes
        monkeypatch.setattr(
            mig, "_get_platform_default_hermes_home", lambda: tmp_path / "elsewhere"
        )
        assert _run(legacy, new) is None
        assert not new.exists()

    def test_noop_when_legacy_is_new_home(self, homes):
        legacy, new = homes
        assert (
            mig.maybe_migrate_legacy_windows_home(
                new_home=new, legacy_home=new, platform="win32"
            )
            is None
        )

    def test_noop_when_legacy_home_missing_or_empty(self, homes, tmp_path):
        legacy, new = homes
        missing = tmp_path / "nope"
        assert (
            mig.maybe_migrate_legacy_windows_home(
                new_home=new, legacy_home=missing, platform="win32"
            )
            is None
        )
        empty = tmp_path / "empty"
        empty.mkdir()
        assert (
            mig.maybe_migrate_legacy_windows_home(
                new_home=new, legacy_home=empty, platform="win32"
            )
            is None
        )
        assert not new.exists()

    def test_symlinked_legacy_entries_are_skipped(self, homes):
        legacy, new = homes
        try:
            (legacy / "linked.json").symlink_to(legacy / "auth.json")
        except OSError:
            pytest.skip("symlinks unavailable (Windows without Developer Mode)")
        _run(legacy, new)
        assert not (new / "linked.json").exists()

    def test_never_raises_on_unreadable_source(self, homes, monkeypatch):
        """Copy errors are collected; no marker so the next start retries."""
        legacy, new = homes

        real_copy2 = mig.shutil.copy2

        def flaky_copy2(src, dst, *a, **kw):
            if str(src).endswith("auth.json"):
                raise OSError("locked")
            return real_copy2(src, dst, *a, **kw)

        monkeypatch.setattr(mig.shutil, "copy2", flaky_copy2)
        summary = _run(legacy, new)
        assert summary is not None
        assert summary["error_count"] == 1
        assert not (new / mig.MARKER_NAME).exists()

        # Retry with the error gone completes and writes the marker.
        monkeypatch.setattr(mig.shutil, "copy2", real_copy2)
        retry = _run(legacy, new)
        assert retry is not None
        assert (new / "auth.json").exists()
        assert (new / mig.MARKER_NAME).exists()
