"""Opt-in strict durability of the cron job store (``cron.store.strict_durability``).

Regression for Harness issue #255: jobs.json reads and publication must be lock-guarded (logical
and physical lock, verified inodes), fsynced (file and directory), refuse corrupt/vanished
stores with an exact forensic snapshot, keep a validated last-good copy and the store's
owner/mode, latch an opted-in store so config loss can never downgrade it, type every
interruption as unchanged/uncertain — per profile (A→B→A), while a never-opted-in store keeps
the historical default behavior and liveness.
"""

import errno
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import cron.jobs as jobs
from cron import jobs_store
from cron.jobs_store import CronStoreError, CronStoreUncertainError

_STRICT_ON = "cron:\n  store:\n    strict_durability: true\n"
_BROKEN_YAML = "cron: [broken YAML"
_REPO = Path(jobs.__file__).resolve().parents[1]
_SECOND_UID = 65534  # "nobody": a real, distinct gateway user inside a disposable root container
_NONCANONICAL = {
    "duplicate_json_keys": b'{"jobs":[{"id":"seed","prompt":"retained"}],"jobs":[]}',
    "nested_duplicate_key": b'{"jobs":[{"id":"seed","id":"other"}]}',
    "nonfinite_json": b'{"jobs":[{"id":"seed","value":NaN}]}',
    "overflow_json": b'{"jobs":[{"id":"seed","value":1e999}]}',
    "bom_json": b'\xef\xbb\xbf{"jobs":[{"id":"seed"}]}',
    "invalid_utf8": b'{"jobs":[{"id":"seed","prompt":"\xff\xfe"}]}',
    "lone_surrogate": b'{"jobs":[{"id":"seed","prompt":"\\ud800"}]}',
    # An exhausted job's count must never be coerced (to 0) and silently regranted.
    "repeat_count_string": b'{"jobs":[{"id":"seed","repeat":{"times":3,"completed":"3"}}]}',
    "repeat_count_null": b'{"jobs":[{"id":"seed","repeat":{"times":3,"completed":null}}]}',
}
_LATCHED_CONFIG_LOSS = {  # latched store, then the opt-in disappears from the effective config
    "strict_config_removed": None,
    "strict_config_null": "cron:\n  store:\n    strict_durability: null\n",
    "strict_config_false": "cron:\n  store:\n    strict_durability: false\n",
    "strict_config_key_omitted": "cron: {}\n",
    "strict_config_unparseable": _BROKEN_YAML,
    "managed_overlay_broken": _BROKEN_YAML,
    "managed_overlay_disables": "cron:\n  store:\n    strict_durability: false\n",
}
_STARTUP_CASES = ("startup_corrupt_json", "startup_missing_primary", "managed_mode_retention",
                  "forensic_copy", "malformed_ids", "mode_noop", "nonfinite_serialize")
_DAMAGED = {
    "startup_corrupt_json": b'{"jobs": [broken JSON',
    "forensic_copy": b'{"jobs": ["FORENSIC-ONLY-FIXTURE"]}',
    "malformed_ids": (b'{"jobs": [{"id": "dup", "prompt": "a"}, {"id": "dup", "prompt": "b"}, '
                      b'{"prompt": "FORENSIC-ONLY-FIXTURE"}]}'),
}
_FORENSIC_CASES = ("forensic_fifo", "forensic_permissions", "forensic_link", "forensic_reuse_fsync",
                   "interrupt_forensic_link")
_FIRST_WRITE_CASES = ("strict_config_unparseable_first", "managed_broken_first_opt_in",
                      "interrupt_latch_rename", "interrupt_latch_dir_fsync",
                      "first_backup_fail_lost_primary", "default_unparseable_config",
                      "default_config_liveness", "unknown_store_key", "durable_backup_stale_id",
                      "tool_backup_stale_warning", "tool_uncertain_no_retry", "env_bool_string",
                      "managed_pin_merge_fail", "managed_pin_mismatch", "trace_privacy_user",
                      "trace_privacy_managed", "trace_privacy_effective", "managed_pin_broken_user", "external_registration_recovery", "fdopen_fail_primary", "fdopen_fail_forensic")
_PRIVATE = "PRIVATE-CONFIG-FIXTURE"
_PRIMARY_KIND_CASES = ("fifo_primary", "symlink_loop_primary", "tampered_latch")
_DIR_SWAP_CASES = ("dir_swap_precommit", "dir_swap_postcommit")
_PRIMARY_REPLACE_FAULTS = {
    "fail_primary_replace": lambda: OSError("injected rename failure"),
    "exdev_primary_replace": lambda: OSError(errno.EXDEV, "injected cross-device rename"),
    "interrupt_before_rename": KeyboardInterrupt,
}
# Interrupted second save: (primitive, which call during the save, primary committed?). fchmod and
# fsync count only staged ``.jobs_*.tmp`` fds (1 = primary, 2 = last-good); replace is matched by
# its destination name.
_BOUNDARIES = {
    "interrupt_primary_mode": ("fchmod", 1, False),
    "interrupt_primary_fsync": ("fsync", 1, False),
    "interrupt_after_rename": ("fsync_dir", 1, True),
    "interrupt_backup_mode": ("fchmod", 2, True),
    "interrupt_backup_fsync": ("fsync", 2, True),
    "interrupt_backup_rename": ("replace", "jobs.json" + jobs_store.LAST_GOOD_SUFFIX, True),
    "interrupt_backup_dir_fsync": ("fsync_dir", 2, True),
}
# KeyboardInterrupt still runs every finally/except; these SIGKILL a real writer process at one
# publication primitive instead, so the disk holds exactly what the kernel had at that instant.
# Process crash only: completed writes and renames stay in the page cache (no power-loss claim).
_LAST_GOOD = "jobs.json" + jobs_store.LAST_GOOD_SUFFIX
_LATCH_BYTES = {"armed": jobs_store._LATCH_ARMED, "published": jobs_store._LATCH_PUBLISHED}
# Second save of a seeded store: (primitive, selector, before/after the call, primary new?,
# last-good new?, leftover staging). fchmod/fsync count staged ``.jobs_*.tmp`` fds (1 = primary,
# 2 = last-good), fsync_dir counts calls, replace matches the destination name.
_KILL_SAVE = {
    "sigkill_before_primary_staging_create": ("open", "1", "before", False, False, None),
    "sigkill_before_backup_staging_create": ("open", "2", "before", True, False, None),
    "sigkill_before_primary_dir_fsync": ("fsync_dir", "1", "before", True, False, None),
    "sigkill_before_backup_dir_fsync": ("fsync_dir", "2", "before", True, True, None),
    "sigkill_after_primary_staging_create": ("open", "1", "after", False, False, "empty"),
    "sigkill_after_primary_staging_write": ("fchmod", "1", "before", False, False, "store"),
    "sigkill_after_primary_mode": ("fchmod", "1", "after", False, False, "store"),
    "sigkill_after_primary_file_fsync": ("fsync", "1", "after", False, False, "store"),
    "sigkill_before_primary_replace": ("replace", "jobs.json", "before", False, False, "store"),
    "sigkill_after_primary_replace": ("replace", "jobs.json", "after", True, False, None),
    "sigkill_after_primary_dir_fsync": ("fsync_dir", "1", "after", True, False, None),
    "sigkill_after_backup_staging_create": ("open", "2", "after", True, False, "empty"),
    "sigkill_after_backup_mode": ("fchmod", "2", "after", True, False, "store"),
    "sigkill_before_backup_file_fsync": ("fsync", "2", "before", True, False, "store"),
    "sigkill_after_backup_file_fsync": ("fsync", "2", "after", True, False, "store"),
    "sigkill_before_primary_file_fsync": ("fsync", "1", "before", False, False, "store"),
    "sigkill_after_backup_staging_write": ("fchmod", "2", "before", True, False, "store"),
    "sigkill_before_backup_replace": ("replace", _LAST_GOOD, "before", True, False, "store"),
    "sigkill_after_backup_replace": ("replace", _LAST_GOOD, "after", True, True, None),
    "sigkill_after_backup_dir_fsync": ("fsync_dir", "2", "after", True, True, None),
}
# First create with no jobs.json: (primitive, selector, when, latch after the kill, primary?,
# last-good?, leftover staging). The latch is armed first, then marked published before the rename.
_KILL_FIRST = {
    'sigkill_first_before_latch_staging_create': ('open', '1', 'before', None, False, False, None),
    'sigkill_first_after_latch_staging_create': ('open', '1', 'after', None, False, False, 'empty'),
    'sigkill_first_after_latch_staging_write': ('fchmod', '1', 'before', None, False, False, 'armed'),
    'sigkill_first_after_latch_mode': ('fchmod', '1', 'after', None, False, False, 'armed'),
    'sigkill_first_before_latch_file_fsync': ('fsync', '1', 'before', None, False, False, 'armed'),
    'sigkill_first_after_latch_file_fsync': ('fsync', '1', 'after', None, False, False, 'armed'),
    'sigkill_first_before_latch_dir_fsync': ('fsync_dir', '1', 'before', 'armed', False, False, None),
    'sigkill_first_before_mark_file_fsync': ('fsync', '2', 'before', 'armed', False, False, 'published'),
    'sigkill_first_after_mark_file_fsync': ('fsync', '2', 'after', 'armed', False, False, 'published'),
    'sigkill_first_before_mark_dir_fsync': ('fsync_dir', '2', 'before', 'published', False, False, None),
    'sigkill_first_before_primary_file_fsync': ('fsync', '3', 'before', 'published', False, False, 'store'),
    'sigkill_first_after_primary_file_fsync': ('fsync', '3', 'after', 'published', False, False, 'store'),
    'sigkill_first_before_primary_dir_fsync': ('fsync_dir', '3', 'before', 'published', True, False, None),
    'sigkill_first_before_backup_file_fsync': ('fsync', '4', 'before', 'published', True, False, 'store'),
    'sigkill_first_after_backup_file_fsync': ('fsync', '4', 'after', 'published', True, False, 'store'),
    'sigkill_first_before_backup_dir_fsync': ('fsync_dir', '4', 'before', 'published', True, True, None),
    "sigkill_first_rearm_before_replace": ("replace", "latch-rearmed", "before", "published", False, False, "armed"),
    "sigkill_first_rearm_after_replace": ("replace", "latch-rearmed", "after", "armed", False, False, None),
    "sigkill_first_rearm_before_dir_fsync": ("fsync_dir", "3", "before", "armed", False, False, None),
    "sigkill_first_rearm_after_dir_fsync": ("fsync_dir", "3", "after", "armed", False, False, None),
    "sigkill_first_before_latch_arm": ("replace", "latch-armed", "before", None, False, False, "armed"),
    "sigkill_first_after_latch_arm": ("replace", "latch-armed", "after", "armed", False, False, None),
    "sigkill_first_after_latch_arm_dir_fsync": ("fsync_dir", "1", "after", "armed", False, False, None),
    "sigkill_first_before_latch_mark": ("replace", "latch-published", "before", "armed", False, False,
                                        "published"),
    "sigkill_first_after_latch_mark": ("replace", "latch-published", "after", "published", False, False,
                                       None),
    "sigkill_first_before_primary_replace": ("replace", "jobs.json", "before", "published", False, False,
                                             "store"),
    "sigkill_first_after_primary_replace": ("replace", "jobs.json", "after", "published", True, False, None),
    "sigkill_first_after_backup_replace": ("replace", _LAST_GOOD, "after", "published", True, True, None),
}
# Forensic copy of a damaged store: (os.link before/after, what the next refusal finds).
_KILL_FORENSIC = {
    "sigkill_forensic_after_staging_write": ("staging", "fchmod_before"),
    "sigkill_forensic_after_staging_mode": ("staging", "fchmod_after"),
    "sigkill_forensic_before_file_fsync": ("staging", "fsync_before"),
    "sigkill_forensic_after_file_fsync": ("staging", "fsync_after"),
    "sigkill_forensic_before_unlink": ("cleanup", "unlink_before"),
    "sigkill_forensic_after_unlink": ("cleanup", "unlink_after"),
    "sigkill_forensic_before_dir_fsync": ("cleanup", "fsync_dir_before"),
    "sigkill_forensic_after_dir_fsync": ("cleanup", "fsync_dir_after"),
    "sigkill_forensic_before_link": ("before", "own"),
    "sigkill_forensic_after_link": ("after", "own"),
    "sigkill_forensic_before_link_foreign": ("before", "foreign"),
}


def _ids(path: Path) -> set:
    return {j["id"] for j in json.loads(path.read_bytes())["jobs"]}


def _private(path: Path) -> bool:
    return path.stat().st_mode & 0o077 == 0


def _create(name):
    return jobs.create_job(prompt=name, schedule="every 1h")["id"]


def _store(home: Path):
    cron = home / "cron"
    primary = cron / "jobs.json"
    return (cron, primary, cron / ("jobs.json" + jobs_store.LAST_GOOD_SUFFIX),
            cron / ("jobs.json" + jobs_store.LATCH_SUFFIX))


def _snapshot(primary: Path, raw: bytes) -> Path:
    return primary.with_name("jobs.json.corrupt-" + hashlib.sha256(raw).hexdigest())


def _no_staging_left(cron: Path) -> bool:
    return not list(cron.glob(".jobs_*.tmp")) and not list(cron.glob(".jobs.json.*.staging"))


def _arm_interrupt(monkeypatch, cron: Path, primitive: str, which, armed: dict) -> None:
    seen = {"n": 0}

    def hit() -> bool:
        seen["n"] += 1
        return seen["n"] == which

    def staged(fd) -> bool:
        return os.fstat(fd).st_ino in {p.stat().st_ino for p in cron.glob(".jobs_*.tmp")}

    if primitive in ("fchmod", "fsync"):
        real = getattr(os, primitive)

        def fd_boundary(fd, *args):
            if armed["on"] and staged(fd) and hit():
                raise KeyboardInterrupt
            return real(fd, *args)
        monkeypatch.setattr(os, primitive, fd_boundary)
    elif primitive == "replace":
        real_replace = os.replace

        def replace_boundary(src, dst, *args, **kwargs):
            if armed["on"] and Path(dst).name == which:
                raise KeyboardInterrupt
            return real_replace(src, dst, *args, **kwargs)
        monkeypatch.setattr(os, "replace", replace_boundary)
    else:
        real_fsync_dir = jobs_store._fsync_dir

        def dir_boundary(path):
            if armed["on"] and hit():
                raise KeyboardInterrupt
            return real_fsync_dir(path)
        monkeypatch.setattr(jobs_store, "_fsync_dir", dir_boundary)


def _noncanonical_case(case, home, monkeypatch):
    cron, primary, _backup, latch = _store(home)
    _create("seed")
    raw = _NONCANONICAL[case]
    primary.write_bytes(raw)
    with pytest.raises(CronStoreError):
        jobs.load_jobs()
    with pytest.raises(CronStoreError):
        _create("never merged over collapsed keys")
    assert primary.read_bytes() == raw
    snapshots = list(cron.glob("jobs.json.corrupt-*"))
    assert snapshots == [_snapshot(primary, raw)] and snapshots[0].read_bytes() == raw
    assert _private(snapshots[0]) and snapshots[0].stat().st_nlink == 1 and latch.exists()


def _config_loss_case(case, home, tmp_path, monkeypatch):
    _cron, primary, backup, latch = _store(home)
    _create("seed")
    before, old_backup = primary.read_bytes(), backup.read_bytes()
    change = _LATCHED_CONFIG_LOSS[case]
    if case.startswith("managed_"):
        managed = tmp_path / "managed"
        managed.mkdir()
        (managed / "config.yaml").write_text(change)
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    elif change is None:
        (home / "config.yaml").unlink()
    else:
        (home / "config.yaml").write_text(change)
    for attempt in (lambda: _create("must not downgrade"), jobs.load_jobs):
        with pytest.raises(CronStoreError) as info:
            attempt()
        if case == "strict_config_false":
            assert "downgrade" in str(info.value)
    assert primary.read_bytes() == before and backup.read_bytes() == old_backup and latch.exists()


def _first_write_case(case, home, tmp_path, monkeypatch):
    cron, primary, backup, latch = _store(home)
    if case == "unknown_store_key":
        with pytest.raises(CronStoreError, match="strict_durability"):
            _create("typo never downgrades")
        assert not primary.exists() and not latch.exists()
        return
    if case == "managed_pin_broken_user":
        managed = tmp_path / "managed"
        managed.mkdir()
        (managed / "config.yaml").write_text(_STRICT_ON)
        (home / "config.yaml").write_text(_BROKEN_YAML)
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        with pytest.raises(CronStoreError):
            _create("managed pin must refuse")
        assert not primary.exists() and not latch.exists()
        return
    if case == "external_registration_recovery":
        import tools.cronjob_tools
        import cron.scheduler_provider as providers
        from tools.registry import registry
        registered = set()
        changes = []
        class ExternalProvider:
            def register_job(self, job):
                registered.add(job["id"])
            def on_jobs_changed(self):
                current = jobs.load_jobs()
                changes.append({j["id"] for j in current if j.get("enabled")})
                registered.clear()
                for job in current:
                    if job.get("enabled"):
                        self.register_job(job)
        provider = ExternalProvider()
        monkeypatch.setattr(providers, "resolve_cron_scheduler", lambda: provider)
        handler = registry.get_entry("cronjob_manage").handler
        seed = _create("seed")
        real_fsync = jobs_store._fsync_dir
        def failing(fd):
            raise OSError(errno.EIO, "injected directory fsync failure")
        monkeypatch.setattr(jobs_store, "_fsync_dir", failing)
        result = json.loads(handler({"action":"create","prompt":"watch-end","schedule":"every 1h"}))
        monkeypatch.setattr(jobs_store, "_fsync_dir", real_fsync)
        watcher = result["job_id"]
        assert result["success"] is False and result["scheduler_registered"] is False
        assert result["retry_create"] is False and not registered
        assert _ids(primary) == {seed, watcher}
        paused = json.loads(handler({"action":"pause","job_id":watcher}))
        resumed = json.loads(handler({"action":"resume","job_id":watcher}))
        assert paused["success"] and resumed["success"]
        assert watcher in registered and len(changes) == 2
        assert _ids(primary) == {seed, watcher} == {j["id"] for j in jobs.load_jobs()}
        return
    if case in ("fdopen_fail_primary", "fdopen_fail_forensic"):
        seed = _create("seed")
        before = primary.read_bytes()
        original = os.fdopen
        captured = []
        def fail_fdopen(fd, *args, **kwargs):
            captured.append(fd)
            raise OSError(errno.EMFILE, "injected file object creation failure")
        if case == "fdopen_fail_forensic":
            before = b'{"jobs": [broken JSON'
            primary.write_bytes(before)
        monkeypatch.setattr(os, "fdopen", fail_fdopen)
        with pytest.raises(CronStoreError):
            jobs.load_jobs() if case == "fdopen_fail_forensic" else _create("must not save")
        assert captured
        for fd in captured:
            with pytest.raises(OSError) as closed:
                os.fstat(fd)
            assert closed.value.errno == errno.EBADF
        assert primary.read_bytes() == before
        assert _ids(backup) == {seed}
        assert _no_staging_left(cron)
        monkeypatch.setattr(os, "fdopen", original)
        if case == "fdopen_fail_forensic":
            with pytest.raises(CronStoreError):
                jobs.load_jobs()
            assert _snapshot(primary, before).read_bytes() == before
        else:
            assert {j["id"] for j in jobs.load_jobs()} == {seed}
            after = _create("after descriptor failure")
            assert _ids(primary) == {seed, after}
        return
    if case == "strict_config_unparseable_first":
        # The valid opt-in is observed under the load_jobs lock (latched) BEFORE the config is
        # damaged and before any mutation; the first mutation then refuses.
        assert jobs.load_jobs() == [] and latch.exists() and not primary.exists()
        (home / "config.yaml").write_text(_BROKEN_YAML)
        with pytest.raises(CronStoreError):
            _create("must not downgrade")
        assert not primary.exists()
        return
    if case == "managed_broken_first_opt_in":
        managed = tmp_path / "managed"
        managed.mkdir()
        (managed / "config.yaml").write_text(_BROKEN_YAML)
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        with pytest.raises(CronStoreError):
            _create("overlay unreadable")
        assert not primary.exists() and not latch.exists()
        return
    if case in ("default_unparseable_config", "default_config_liveness"):
        # Never opted in: a broken config.yaml keeps the historical default-path liveness.
        (home / "config.yaml").unlink()
        if case == "default_unparseable_config":
            (home / "config.yaml").write_text(_BROKEN_YAML)
        seed = _create("seed")
        (home / "config.yaml").write_text(_BROKEN_YAML)
        second = _create("default remains live")
        assert _ids(primary) == {seed, second} == {j["id"] for j in jobs.load_jobs()}
        assert not backup.exists() and not latch.exists()
        return
    if case == "env_bool_string":
        # An env-expanded or quoted "true" is text, not a bool: refused, never guessed.
        (home / "config.yaml").write_text("cron:\n  store:\n    strict_durability: ${CRON_STRICT_TEST}\n")
        monkeypatch.setenv("CRON_STRICT_TEST", "true")
        with pytest.raises(CronStoreError, match="true or false"):
            _create("string is not a bool")
        assert not primary.exists() and not latch.exists()
        return
    if case in ("managed_pin_merge_fail", "managed_pin_mismatch"):
        # Only the managed overlay opts in; the canonical loader either fails or drops it.
        import hermes_cli.config_effective as effective
        (home / "config.yaml").write_text("cron: {}\n")
        managed = tmp_path / "managed"
        managed.mkdir()
        (managed / "config.yaml").write_text(_STRICT_ON)
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))

        def merge_fails(*_a, **_k):
            raise RuntimeError("injected merge failure")
        monkeypatch.setattr(effective, "load_user_config_effective",
                            merge_fails if case == "managed_pin_merge_fail" else lambda *_a, **_k: {})
        with pytest.raises(CronStoreError, match="managed" if case == "managed_pin_mismatch" else None):
            _create("a managed pin never fails open")
        assert not primary.exists() and not latch.exists()
        return
    if case.startswith("trace_privacy_"):
        # A latched store whose config breaks refuses naming the file and exception type, and no
        # printed or logged traceback of that refusal (chained causes included) quotes config text.
        import logging
        import traceback
        seed = _create("seed")
        before = primary.read_bytes()
        if case == "trace_privacy_user":
            (home / "config.yaml").write_text(f"cron: [{_PRIVATE}")
            named = f"{home / 'config.yaml'} cannot be parsed ("
        elif case == "trace_privacy_managed":
            managed = tmp_path / "managed"
            managed.mkdir()
            (managed / "config.yaml").write_text(f"cron: [{_PRIVATE}")
            monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
            named = f"{managed / 'config.yaml'} cannot be parsed ("
        else:
            import hermes_cli.config_effective as effective

            def loader_quotes_config(*_a, **_k):
                raise RuntimeError(f"line 1: cron: [{_PRIVATE}")
            (home / "config.yaml").write_text(_STRICT_ON + "# edited\n")  # new signature: no cache hit
            monkeypatch.setattr(effective, "load_user_config_effective", loader_quotes_config)
            named = "cannot be built (RuntimeError)"
        for attempt in (jobs.load_jobs, lambda: _create("refused")):
            with pytest.raises(CronStoreError) as info:
                attempt()
            exc = info.value
            printed = "".join(traceback.format_exception(exc))
            logged = logging.Formatter().formatException((type(exc), exc, exc.__traceback__))
            assert _PRIVATE not in printed and _PRIVATE not in logged
            assert named in str(exc) and "opted in to strict durability" in str(exc)
        assert primary.read_bytes() == before and _ids(primary) == {seed} and latch.exists()
        return
    armed = {"on": True}
    real_replace = os.replace

    def failing_backup(src, dst, *args, **kwargs):
        if armed["on"] and Path(dst).name == backup.name:
            raise OSError("injected backup failure")
        return real_replace(src, dst, *args, **kwargs)
    if case in ("durable_backup_stale_id", "tool_backup_stale_warning"):
        seed = _create("seed")
        before = backup.read_bytes()
        monkeypatch.setattr(os, "replace", failing_backup)
        if case == "tool_backup_stale_warning":
            # The real registry handler: the model gets the id AND the warning, not a failure
            # that would make a watcher creator retry into a duplicate.
            import tools.cronjob_tools  # noqa: F401  (registers cronjob_manage)
            from tools.registry import registry
            handler = registry.get_entry("cronjob_manage").handler
            result = json.loads(handler({"action": "create", "prompt": "watch-end", "schedule": "every 1h"}))
            assert result["success"] is True and backup.name in result["storage_warning"]
            assert "Storage warning" in result["message"]
            watcher, warning = result["job_id"], result["storage_warning"]
        else:
            created = jobs.create_job(prompt="watch-end-like creation", schedule="every 1h")
            watcher, warning = created["id"], created["storage_warning"]
        assert backup.name in warning and "Do not re-create" in warning
        assert watcher != seed and _ids(primary) == {seed, watcher}
        assert backup.read_bytes() == before  # stale last-good kept, never deleted or restored
        assert {j["id"] for j in jobs.load_jobs()} == {seed, watcher}
        assert all("storage_warning" not in j for j in json.loads(primary.read_bytes())["jobs"])
        armed["on"] = False
        after = _create("next write refreshes last-good")
        assert backup.read_bytes() == primary.read_bytes() and _ids(backup) == {seed, watcher, after}
        return
    if case == "tool_uncertain_no_retry":
        import tools.cronjob_tools  # noqa: F401  (registers cronjob_manage)
        from tools.registry import registry
        seed = _create("seed")
        real_fsync_dir = jobs_store._fsync_dir

        def failing_fsync_dir(fd):
            if armed["on"]:
                raise OSError(errno.EIO, "injected directory fsync failure")
            return real_fsync_dir(fd)
        monkeypatch.setattr(jobs_store, "_fsync_dir", failing_fsync_dir)
        handler = registry.get_entry("cronjob_manage").handler
        result = json.loads(handler({"action": "create", "prompt": "watch-end", "schedule": "every 1h"}))
        armed["on"] = False
        assert result["success"] is False and result["retry_create"] is False
        assert result["job_saved"] is None and "list" in result["error"]
        assert result["job_id"] in _ids(primary) and seed in _ids(primary)
        assert result.get("scheduler_registered") is False
        assert "resume" in result["error"]
        return
    if case == "first_backup_fail_lost_primary":
        monkeypatch.setattr(os, "replace", failing_backup)
        first = jobs.create_job(prompt="first", schedule="every 1h")
        armed["on"] = False
        assert backup.name in first["storage_warning"] and "first save" in first["storage_warning"]
        assert _ids(primary) == {first["id"]} == {j["id"] for j in jobs.load_jobs()}
        assert not backup.exists() and latch.exists()
        primary.unlink()  # the only copy is lost; the latch must not allow an empty restart
        for attempt in (jobs.load_jobs, lambda: _create("over a lost store")):
            with pytest.raises(CronStoreError):
                attempt()
        assert not primary.exists() and not backup.exists()
        return
    primitive, outcome = (("replace", "unchanged") if case == "interrupt_latch_rename"
                          else ("fsync_dir", "uncertain"))
    _arm_interrupt(monkeypatch, cron, primitive, 1 if primitive == "fsync_dir" else latch.name, armed)
    with pytest.raises(KeyboardInterrupt) as info:
        jobs.load_jobs()
    assert getattr(info.value, jobs_store.OUTCOME_ATTR) == outcome
    assert latch.exists() == (outcome == "uncertain") and not primary.exists()
    assert _no_staging_left(cron)
    armed["on"] = False
    seed = _create("after the interrupted latch")
    assert _ids(primary) == {seed} and latch.exists() and backup.read_bytes() == primary.read_bytes()


def _forensic_case(case, home, monkeypatch):
    cron, primary, _backup, _latch = _store(home)
    _create("seed")
    damaged = b'{"jobs": [broken, "FORENSIC-ONLY-FIXTURE"'
    primary.write_bytes(damaged)
    snap = _snapshot(primary, damaged)
    if case == "interrupt_forensic_link":
        real_link = os.link
        armed = {"on": True}

        def interrupted_link(src, dst, *args, **kwargs):
            if armed["on"]:
                raise KeyboardInterrupt
            return real_link(src, dst, *args, **kwargs)
        monkeypatch.setattr(os, "link", interrupted_link)
        with pytest.raises(KeyboardInterrupt) as info:
            jobs.load_jobs()
        assert getattr(info.value, jobs_store.OUTCOME_ATTR) == "unchanged"
        assert not snap.exists() and _no_staging_left(cron) and primary.read_bytes() == damaged
        armed["on"] = False
        with pytest.raises(CronStoreError, match=snap.name):
            jobs.load_jobs()
        assert snap.read_bytes() == damaged and _private(snap) and snap.stat().st_nlink == 1
        return
    if case == "forensic_reuse_fsync":
        with pytest.raises(CronStoreError, match=snap.name):
            jobs.load_jobs()
        synced, dirs = set(), []
        real_fsync, real_fsync_dir = os.fsync, jobs_store._fsync_dir

        def recording_fsync(fd):
            synced.add(os.fstat(fd).st_ino)
            return real_fsync(fd)

        def recording_fsync_dir(dir_fd):
            dirs.append(os.fstat(dir_fd).st_ino)  # the pinned directory fd, not a path
            return real_fsync_dir(dir_fd)
        monkeypatch.setattr(os, "fsync", recording_fsync)
        monkeypatch.setattr(jobs_store, "_fsync_dir", recording_fsync_dir)
        with pytest.raises(CronStoreError, match=snap.name):
            jobs.load_jobs()
        assert snap.stat().st_ino in synced and snap.parent.stat().st_ino in dirs
        assert list(cron.glob("jobs.json.corrupt-*")) == [snap]
        return
    other = cron / "elsewhere"
    if case == "forensic_fifo":
        os.mkfifo(snap, 0o600)
    else:
        snap.write_bytes(damaged)
        snap.chmod(0o644 if case == "forensic_permissions" else 0o600)
        if case == "forensic_link":
            os.link(snap, other)
    before = os.lstat(snap)
    with pytest.raises(CronStoreError, match="could NOT be preserved"):
        jobs.load_jobs()  # never blocks on the FIFO, never clobbers or "fixes" the evidence
    after = os.lstat(snap)
    assert (after.st_ino, after.st_mode, after.st_nlink) == (before.st_ino, before.st_mode, before.st_nlink)
    assert primary.read_bytes() == damaged
    assert stat.S_ISFIFO(after.st_mode) if case == "forensic_fifo" else snap.read_bytes() == damaged


def _primary_kind_case(case, home):
    cron, primary, backup, latch = _store(home)
    _create("seed")
    old_backup, old_latch = backup.read_bytes(), latch.read_bytes()
    if case == "tampered_latch":
        latch.write_bytes(b"not a latch this module wrote\n")  # same owner, same 0600 mode
        old_latch = latch.read_bytes()
    else:
        primary.unlink()
        if case == "fifo_primary":
            os.mkfifo(primary, 0o600)
        else:
            primary.symlink_to(primary.name)  # a symlink loop where the store should be
    before = os.lstat(primary)
    for attempt in (jobs.load_jobs, lambda: _create("refused")):
        with pytest.raises(CronStoreError):
            attempt()  # never blocks on the FIFO, never follows the loop, never trusts the latch
    after = os.lstat(primary)
    assert (after.st_ino, after.st_mode) == (before.st_ino, before.st_mode)
    assert backup.read_bytes() == old_backup and latch.read_bytes() == old_latch
    assert not list(cron.glob("jobs.json.corrupt-*")) and _no_staging_left(cron)


def _dir_swap_case(case, home, tmp_path, monkeypatch):
    """The store's directory is replaced INSIDE the publication boundary: before the rename the
    commit is refused (nothing lands anywhere); after it, the commit is reported uncertain."""
    cron, primary, _backup, _latch = _store(home)
    shared, moved = tmp_path / "shared", tmp_path / "moved"
    shared.mkdir()
    cron.mkdir()
    primary.symlink_to(shared / "jobs.json")
    seed = _create("seed")
    before = (shared / "jobs.json").read_bytes()
    armed = {"on": True}

    def swap():
        armed["on"] = False
        shared.rename(moved)
        shared.mkdir()

    if case == "dir_swap_precommit":
        real_fsync = os.fsync

        def swapping_fsync(fd):
            staged = {p.stat().st_ino for p in shared.glob(".jobs_*.tmp")} if armed["on"] else set()
            result = real_fsync(fd)
            if os.fstat(fd).st_ino in staged:
                swap()  # staged + fsynced, rename not yet done
            return result
        monkeypatch.setattr(os, "fsync", swapping_fsync)
        with pytest.raises(CronStoreError) as info:
            _create("never commits into a swapped directory")
        assert type(info.value) is CronStoreError  # unchanged, not uncertain
        assert (moved / "jobs.json").read_bytes() == before
    else:
        real_replace = os.replace

        def swapping_replace(src, dst, *args, **kwargs):
            result = real_replace(src, dst, *args, **kwargs)
            if armed["on"] and Path(dst).name == "jobs.json":
                swap()  # the rename landed; readers no longer see that directory
            return result
        monkeypatch.setattr(os, "replace", swapping_replace)
        with pytest.raises(CronStoreUncertainError):
            _create("committed where readers no longer look")
        assert seed in _ids(moved / "jobs.json") and len(_ids(moved / "jobs.json")) == 2
    assert not list(shared.iterdir())  # nothing staged or committed into the replacement
    assert not list(moved.glob(".jobs_*.tmp"))  # own staging cleaned through the pinned fd


def _in_section_case(case, home, tmp_path):
    cron, primary, _backup, _latch = _store(home)
    if case == "symlink_retarget":
        first, second = tmp_path / "first", tmp_path / "second"
        first.mkdir()
        second.mkdir()
        cron.mkdir()
        primary.symlink_to(first / "jobs.json")
    _create("seed")
    watched = first / "jobs.json" if case == "symlink_retarget" else primary
    before = watched.read_bytes()
    with jobs._jobs_lock():
        current = jobs.load_jobs()
        if case == "lock_inode_swap":
            (cron / ".jobs.lock").unlink()
            (cron / ".jobs.lock").touch()  # another process now locks a different inode
        else:
            (cron / "retarget").symlink_to(second / "jobs.json")
            os.replace(cron / "retarget", primary)
        with pytest.raises(CronStoreError, match="replaced" if case == "lock_inode_swap" else "retargeted"):
            jobs.save_jobs(current + [dict(current[0], id="intruder")])
    assert watched.read_bytes() == before
    if case == "symlink_retarget":
        assert not (second / "jobs.json").exists()
    else:
        assert "intruder" not in {j["id"] for j in jobs.load_jobs()}


def _startup_case(case, home, caplog, monkeypatch):
    cron, primary, backup, _latch = _store(home)
    _create("seed")
    if case in ("managed_mode_retention", "mode_noop"):
        import hermes_cli.config as config
        monkeypatch.setattr(config, "is_managed", lambda: True)
        primary.chmod(0o640)
        if case == "mode_noop":
            before = primary.read_bytes()
            monkeypatch.setattr(os, "fchmod", lambda fd, mode: None)  # a chmod that silently no-ops
            with pytest.raises(CronStoreError, match="staged"):
                _create("never published with the wrong mode")
            assert primary.read_bytes() == before and _no_staging_left(cron)
            return
        _create("second")
        assert primary.stat().st_mode & 0o777 == 0o640
        assert backup.stat().st_mode & 0o777 == 0o640
        return
    if case == "nonfinite_serialize":
        before = primary.read_bytes()
        with pytest.raises(CronStoreError, match="strict JSON"):
            jobs.save_jobs(jobs.load_jobs() + [{"id": "nan-job", "prompt": "x", "weight": float("nan")}])
        assert primary.read_bytes() == before
        return
    if case == "startup_missing_primary":
        old_backup = backup.read_bytes()
        primary.unlink()
        with pytest.raises(CronStoreError):
            jobs.load_jobs()
        assert not primary.exists()
        assert backup.read_bytes() == old_backup  # never restored, never deleted
        return
    damaged = _DAMAGED[case]
    primary.write_bytes(damaged)
    for _attempt in range(2):  # idempotent: a repeated refusal never adds or rewrites a snapshot
        with pytest.raises(CronStoreError):
            _create("refused") if case == "forensic_copy" else jobs.load_jobs()
        assert primary.read_bytes() == damaged
        assert list(cron.glob("jobs.json.corrupt-*")) == [_snapshot(primary, damaged)]
        assert _snapshot(primary, damaged).read_bytes() == damaged
    assert _private(_snapshot(primary, damaged))
    assert "FORENSIC-ONLY-FIXTURE" not in caplog.text


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case", [
    "default", "strict", "invalid_value", "lock_unavailable", "corrupt_primary",
    "missing_primary_with_backup", "fail_primary_replace", "fail_dir_fsync", "fail_backup_replace",
    "unsupported_flock", "fail_file_fsync", "exdev_primary_replace", "interrupt_before_rename",
    *_BOUNDARIES, *_STARTUP_CASES, *_NONCANONICAL, *_LATCHED_CONFIG_LOSS, *_FIRST_WRITE_CASES,
    *_FORENSIC_CASES, "lock_inode_swap", "symlink_retarget", *_PRIMARY_KIND_CASES, *_DIR_SWAP_CASES,
])
def test_strict_store_publication_contract(case, tmp_path, monkeypatch, caplog):
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    if case == "invalid_value":
        (home_a / "config.yaml").write_text('cron:\n  store:\n    strict_durability: "yes"\n')
    elif case == "unknown_store_key":
        (home_a / "config.yaml").write_text("cron:\n  store:\n    strict_durabilty: true\n")
    elif case != "default":
        (home_a / "config.yaml").write_text(_STRICT_ON)
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    groups = (
        (_NONCANONICAL, lambda: _noncanonical_case(case, home_a, monkeypatch)),
        (_LATCHED_CONFIG_LOSS, lambda: _config_loss_case(case, home_a, tmp_path, monkeypatch)),
        (_FIRST_WRITE_CASES, lambda: _first_write_case(case, home_a, tmp_path, monkeypatch)),
        (_FORENSIC_CASES, lambda: _forensic_case(case, home_a, monkeypatch)),
        (("lock_inode_swap", "symlink_retarget"), lambda: _in_section_case(case, home_a, tmp_path)),
        (_STARTUP_CASES, lambda: _startup_case(case, home_a, caplog, monkeypatch)),
        (_PRIMARY_KIND_CASES, lambda: _primary_kind_case(case, home_a)),
        (_DIR_SWAP_CASES, lambda: _dir_swap_case(case, home_a, tmp_path, monkeypatch)),
    )
    for members, run in groups:
        if case in members:
            run()
            return

    cron, primary, backup, latch = _store(home_a)
    # A: first write.
    if case == "invalid_value":
        with pytest.raises(CronStoreError):
            _create("never stored")
        assert not primary.exists()
    else:
        kept = {_create("seed")}
        if case == "default":
            assert not backup.exists() and not latch.exists()
        else:
            assert backup.read_bytes() == primary.read_bytes() and latch.exists()
            assert _private(primary) and _private(backup) and _private(latch)

    # B: another profile's default policy is unaffected by A's config or latch.
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    b_id = _create("b job")
    _b_cron, b_primary, b_backup, b_latch = _store(home_b)
    assert _ids(b_primary) == {b_id} and not b_backup.exists() and not b_latch.exists()

    # Back to A: the case's boundary.
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    if case == "invalid_value":
        with pytest.raises(CronStoreError):
            _create("still refused")
        assert not primary.exists()
        return
    if case in ("default", "strict"):
        kept.add(_create("second"))
        assert _ids(primary) == kept
        assert case == "default" or backup.read_bytes() == primary.read_bytes()
        return

    old_primary, old_backup = primary.read_bytes(), backup.read_bytes()
    if case == "corrupt_primary":
        forensic = b'{"jobs": ["SECRET-PAYLOAD-7f3a"], "updated_at": null}'
        primary.write_bytes(forensic)
        with pytest.raises(CronStoreError):
            _create("refused")
        assert primary.read_bytes() == forensic
        assert backup.read_bytes() == old_backup
        assert "SECRET-PAYLOAD-7f3a" not in caplog.text
        return
    if case == "missing_primary_with_backup":
        primary.unlink()
        with pytest.raises(CronStoreError):
            _create("refused")
        assert not primary.exists()
        assert backup.read_bytes() == old_backup
        return

    armed = {"on": True}
    real_backends = (jobs.fcntl, jobs.msvcrt)
    if case == "lock_unavailable":
        lock = cron / ".jobs.lock"
        lock.unlink()
        lock.mkdir()  # unopenable lock file: no cross-process lock can be held
        expected, committed = CronStoreError, False
    elif case == "unsupported_flock":
        monkeypatch.setattr(jobs, "fcntl", None)
        monkeypatch.setattr(jobs, "msvcrt", None)
        expected, committed = CronStoreError, False
    elif case in _BOUNDARIES:
        primitive, which, committed = _BOUNDARIES[case]
        _arm_interrupt(monkeypatch, cron, primitive, which, armed)
        expected = KeyboardInterrupt
    elif case == "fail_dir_fsync":
        real_fsync_dir = jobs_store._fsync_dir

        def failing_fsync_dir(path):
            if armed["on"]:
                raise OSError("injected directory fsync failure")
            real_fsync_dir(path)
        monkeypatch.setattr(jobs_store, "_fsync_dir", failing_fsync_dir)
        expected, committed = CronStoreUncertainError, True
    elif case == "fail_file_fsync":
        real_fsync = os.fsync

        def failing_fsync(fd):
            if armed["on"]:
                staged = {p.stat().st_ino for p in cron.glob(".jobs_*.tmp")}
                if os.fstat(fd).st_ino in staged:
                    raise OSError(errno.EIO, "injected file fsync failure")
            return real_fsync(fd)
        monkeypatch.setattr(os, "fsync", failing_fsync)
        expected, committed = CronStoreError, False
    else:
        real_replace = os.replace
        on_primary = case in _PRIMARY_REPLACE_FAULTS
        victim = (primary if on_primary else backup).name  # renames are relative to the pinned dir
        fault = _PRIMARY_REPLACE_FAULTS.get(case, lambda: OSError("injected rename failure"))

        def failing_replace(src, dst, *a, **kw):
            if armed["on"] and Path(dst).name == victim:
                raise fault()
            return real_replace(src, dst, *a, **kw)
        monkeypatch.setattr(os, "replace", failing_replace)
        expected = KeyboardInterrupt if case == "interrupt_before_rename" else CronStoreError
        committed = case == "fail_backup_replace"

    if case == "fail_backup_replace":
        # The primary is durable; only last-good is stale. That is the created id plus an
        # actionable warning, never a failure a caller would retry into a duplicate.
        created = jobs.create_job(prompt="backup stale", schedule="every 1h")
        assert backup.name in created["storage_warning"]
        assert {j["id"] for j in jobs.load_jobs()} == _ids(primary) == kept | {created["id"]}
    else:
        with pytest.raises(expected) as info:
            _create("interrupted")
        # Typed outcome: CronStoreError(unchanged)/CronStoreUncertainError, or the interrupt itself
        # (never swallowed) carrying cron_store_outcome.
        assert getattr(info.value, jobs_store.OUTCOME_ATTR) == ("uncertain" if committed else "unchanged")
        if issubclass(expected, CronStoreError):
            assert (type(info.value) is CronStoreUncertainError) == committed
    if committed:
        assert _ids(primary) > kept  # the rename happened; reported as uncertain, not rolled back
        kept = _ids(primary)
        # last-good is never newer than a primary that may be lost
        assert backup.read_bytes() in (old_backup, primary.read_bytes())
    else:
        assert primary.read_bytes() == old_primary and backup.read_bytes() == old_backup
    assert _no_staging_left(cron)  # own staging never leaks

    # Liveness: once the fault clears, the next write succeeds and refreshes last-good.
    armed["on"] = False
    monkeypatch.setattr(jobs, "fcntl", real_backends[0])
    monkeypatch.setattr(jobs, "msvcrt", real_backends[1])
    if case == "lock_unavailable":
        (cron / ".jobs.lock").rmdir()
    kept.add(_create("after recovery"))
    assert _ids(primary) == kept
    assert backup.read_bytes() == primary.read_bytes()
    assert _private(primary) and _private(backup)


def _spawn(code: str, *args, home: Path, env=None, **popen) -> subprocess.Popen:
    child_env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(_REPO),
                 "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code), *map(str, args)], cwd=str(_REPO),
        env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **popen)


def _await_signal(path: Path, proc: subprocess.Popen, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert proc.poll() is None, f"child exited before signalling: {proc.communicate()}"
        assert time.monotonic() < deadline, f"child never signalled {path.name}"
        time.sleep(0.02)


def _reap(proc: subprocess.Popen) -> None:
    """Only ever this test's own child."""
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=30)


_CREATE_AFTER_GO = """
    import pathlib, sys, time
    go, ready = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    from cron.jobs import create_job
    ready.touch()
    deadline = time.monotonic() + 60
    while not go.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    for n in range(count):
        print(create_job(prompt=f"watch {n}", schedule="every 1h")["id"], flush=True)
"""

_HOLD_LOCK = """
    import fcntl, pathlib, sys, time
    lock, ready, go = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    fd = open(lock, "a+")
    fcntl.flock(fd, fcntl.LOCK_EX)
    ready.touch()
    deadline = time.monotonic() + 120
    while not go.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
"""

_TRY_CREATE_WITH_SHORT_TIMEOUT = """
    import cron.jobs as jobs
    from cron.jobs_store import CronStoreError
    jobs._JOBS_LOCK_TIMEOUT_SECONDS = 1.0
    try:
        jobs.create_job(prompt="alias writer", schedule="every 1h")
        print("stored")
    except CronStoreError:
        print("refused")
"""

# Only the refusal's logger.exception reaches stdout; other modules' logs stay on stderr.
_LOG_REFUSAL = """
    import logging, sys
    import cron.jobs as jobs
    from cron.jobs_store import CronStoreError
    log = logging.getLogger("cron.refusal")
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.propagate = False
    try:
        jobs.load_jobs()
        print("loaded")
    except CronStoreError:
        log.exception("cron store refused")
        print("refused")
"""

_GATEWAY_USER_ROUND_TRIP = """
    import json
    from cron.jobs import create_job, load_jobs
    create_job(prompt="gateway write", schedule="every 1h")
    print(json.dumps(sorted(j["id"] for j in load_jobs())))
"""

# One strict store operation that SIGKILLs ITSELF (never another process) at one real filesystem
# primitive: no finally, except, atexit or lock release in Python runs after the boundary.
_KILL_AT_BOUNDARY = """
    import os, pathlib, signal, sys
    store_dir = pathlib.Path(sys.argv[1])
    primitive, selector, when, action = sys.argv[2:6]
    import cron.jobs as jobs
    from cron import jobs_store
    latch_name = "jobs.json" + jobs_store.LATCH_SUFFIX
    seen = [0]

    if action == "rearm":
        original_replace = os.replace
        def refuse_primary(src, dst, *args, **kwargs):
            if os.path.basename(os.fspath(dst)) == "jobs.json":
                raise OSError("fixture: definitive first-primary rename failure")
            return original_replace(src, dst, *args, **kwargs)
        os.replace = refuse_primary

    def nth():
        seen[0] += 1
        return seen[0] == int(selector)

    def staged(fd):
        return os.fstat(fd).st_ino in {p.stat().st_ino for p in store_dir.glob(".jobs_*.tmp")}

    def forensic_staged(fd):
        return os.fstat(fd).st_ino in {p.stat().st_ino for p in store_dir.glob(".jobs.json.corrupt-*.staging")}

    def replaced(src, dst, *args, **kwargs):
        name = os.path.basename(os.fspath(dst))
        if not selector.startswith("latch-"):
            return name == selector
        if name != latch_name:
            return False
        fd = os.open(src, os.O_RDONLY, dir_fd=kwargs.get("src_dir_fd"))
        try:
            published = os.read(fd, 4096) == jobs_store._LATCH_PUBLISHED
        finally:
            os.close(fd)
        if selector == "latch-rearmed":
            if published:
                return False
            seen[0] += 1
            return seen[0] == 2
        return selector == ("latch-published" if published else "latch-armed")

    matchers = {
        "open": lambda path, *a, **k: (os.path.basename(os.fspath(path)).startswith(".jobs_")
                                       and os.fspath(path).endswith(".tmp") and nth()),
        "fchmod": lambda fd, *a, **k: staged(fd) and nth(),
        "fsync": lambda fd, *a, **k: staged(fd) and nth(),
        "forensic_fchmod": lambda fd, *a, **k: forensic_staged(fd) and nth(),
        "forensic_fsync": lambda fd, *a, **k: forensic_staged(fd) and nth(),
        "replace": replaced,
        "link": lambda *a, **k: nth(),
        "unlink": lambda *a, **k: nth(),
        "fsync_dir": lambda *a, **k: nth(),
    }
    owner, attr = (jobs_store, "_fsync_dir") if primitive == "fsync_dir" else (os, primitive.removeprefix("forensic_"))
    real = getattr(owner, attr)

    def boundary(*args, **kwargs):
        hit = matchers[primitive](*args, **kwargs)
        if hit and when == "before":
            os.kill(os.getpid(), signal.SIGKILL)
        result = real(*args, **kwargs)
        if hit:
            os.kill(os.getpid(), signal.SIGKILL)
        return result
    setattr(owner, attr, boundary)
    if action in ("create", "rearm"):
        print(jobs.create_job(prompt="killed writer", schedule="every 1h")["id"])
    else:
        jobs.load_jobs()
    print("survived the boundary")
"""


def _kill_writer(home: Path, cron: Path, primitive: str, selector: str, when: str, action: str) -> str:
    proc = _spawn(_KILL_AT_BOUNDARY, cron, primitive, selector, when, action, home=home)
    try:
        out, err = proc.communicate(timeout=120)
    finally:
        _reap(proc)
    assert proc.returncode == -signal.SIGKILL, (proc.returncode, out, err)
    assert out == ""  # died inside the boundary: nothing was ever reported to a caller
    return err


def _staging(cron: Path) -> dict:
    return {p: (p.stat().st_ino, p.read_bytes()) for p in cron.glob(".jobs_*.tmp")}


def _sigkill_forensic_case(case, home):
    cron, primary, backup, _latch = _store(home)
    seed = _create("seed")
    old_backup = backup.read_bytes()
    damaged = b'{"jobs": [broken, "FORENSIC-ONLY-FIXTURE"'
    primary.write_bytes(damaged)
    snap = _snapshot(primary, damaged)
    staging = cron / f".{snap.name}.staging"
    when, then = _KILL_FORENSIC[case]
    if when == "staging":
        primitive, kill_when = then.rsplit("_", 1)
        err = _kill_writer(home, cron, "forensic_" + primitive, "1", kill_when, "load")
        assert "FORENSIC-ONLY-FIXTURE" not in err
        assert primary.read_bytes() == damaged and backup.read_bytes() == old_backup
        assert staging.read_bytes() == damaged and _private(staging)
        assert staging.stat().st_nlink == 1 and not snap.exists()
        for _attempt in range(2):
            with pytest.raises(CronStoreError, match=snap.name):
                jobs.load_jobs()
            assert not staging.exists() and snap.stat().st_nlink == 1
            assert snap.read_bytes() == damaged and _private(snap)
        with pytest.raises(CronStoreError):
            _create("never overwrite corruption after killed staging")
        assert primary.read_bytes() == damaged and backup.read_bytes() == old_backup
        assert _ids(backup) == {seed}
        return
    if when == "cleanup":
        primitive, kill_when = then.rsplit("_", 1)
        err = _kill_writer(home, cron, primitive, "1", kill_when, "load")
        assert "FORENSIC-ONLY-FIXTURE" not in err
        assert primary.read_bytes() == damaged and backup.read_bytes() == old_backup
        assert snap.read_bytes() == damaged and _private(snap)
        if then == "unlink_before":
            assert staging.exists() and staging.stat().st_ino == snap.stat().st_ino
            assert snap.stat().st_nlink == 2 and _private(staging)
        else:
            assert not staging.exists() and snap.stat().st_nlink == 1
        for _attempt in range(2):
            with pytest.raises(CronStoreError, match=snap.name):
                jobs.load_jobs()
            assert not staging.exists() and snap.stat().st_nlink == 1
            assert snap.read_bytes() == damaged and _private(snap)
        with pytest.raises(CronStoreError):
            _create("never overwrite corruption after killed cleanup")
        assert primary.read_bytes() == damaged and backup.read_bytes() == old_backup
        assert _ids(backup) == {seed}
        return
    err = _kill_writer(home, cron, "link", "1", when, "load")
    assert "FORENSIC-ONLY-FIXTURE" not in err
    assert primary.read_bytes() == damaged and backup.read_bytes() == old_backup
    st = staging.stat()
    assert staging.read_bytes() == damaged and _private(staging)
    if when == "after":
        assert snap.stat().st_ino == st.st_ino and st.st_nlink == 2
    else:
        assert not snap.exists() and st.st_nlink == 1
    if then == "foreign":
        # No longer provably the dead writer's file: refused, never removed or used as evidence.
        staging.write_bytes(b"not the damaged bytes")
        foreign = os.lstat(staging)
        with pytest.raises(CronStoreError, match="could NOT be preserved"):
            jobs.load_jobs()
        assert staging.read_bytes() == b"not the damaged bytes" and os.lstat(staging).st_ino == foreign.st_ino
        assert not snap.exists() and primary.read_bytes() == damaged
        return
    for _attempt in range(2):  # the proven own staging is cleared once; the snapshot is then reused
        with pytest.raises(CronStoreError, match=snap.name):
            jobs.load_jobs()
        assert not staging.exists() and list(cron.glob("jobs.json.corrupt-*")) == [snap]
        assert snap.read_bytes() == damaged and _private(snap) and snap.stat().st_nlink == 1
    with pytest.raises(CronStoreError):
        _create("never merged into a damaged store")
    assert primary.read_bytes() == damaged and backup.read_bytes() == old_backup and _ids(backup) == {seed}


def _sigkill_case(case, home):
    """The writer dies at the boundary: disk holds the old or the new complete store, the seed is
    kept, a dead writer's staging is never swept or merged, and the next process continues where
    the latch allows — or refuses with the documented recovery, never an empty restart."""
    if case in _KILL_FORENSIC:
        _sigkill_forensic_case(case, home)
        return
    cron, primary, backup, latch = _store(home)
    if case in _KILL_FIRST:
        primitive, selector, when, latch_state, new_primary, new_backup, leftover = _KILL_FIRST[case]
        seeds, old_primary, old_backup = set(), None, None
    else:
        primitive, selector, when, new_primary, new_backup, leftover = _KILL_SAVE[case]
        latch_state = "published"
        seeds = {_create("seed")}
        old_primary, old_backup = primary.read_bytes(), backup.read_bytes()
    _kill_writer(home, cron, primitive, selector, when, "rearm" if case.startswith("sigkill_first_rearm_") else "create")

    def killed_writer_store(raw: bytes) -> bool:
        assert jobs_store.noncanonical_reason(raw) is None
        by_id = {j["id"]: j for j in json.loads(raw)["jobs"]}
        added = set(by_id) - seeds
        return set(by_id) >= seeds and len(added) == 1 and by_id[added.pop()]["prompt"] == "killed writer"

    if new_primary:
        assert killed_writer_store(primary.read_bytes())
    else:
        assert primary.read_bytes() == old_primary if old_primary is not None else not primary.exists()
    if new_backup:
        assert backup.read_bytes() == primary.read_bytes()
    else:
        assert backup.read_bytes() == old_backup if old_backup is not None else not backup.exists()
    assert (latch.read_bytes() if latch.exists() else None) == _LATCH_BYTES.get(latch_state)
    left = _staging(cron)
    assert len(left) == (0 if leftover is None else 1)
    if leftover is not None:
        (path, (_ino, raw)), = left.items()
        assert _private(path)
        if leftover == "empty":
            assert raw == b""
        elif leftover == "store":
            assert killed_writer_store(raw) and (not new_primary or raw == primary.read_bytes())
        else:
            assert raw == _LATCH_BYTES[leftover]

    refused = old_primary is None and latch_state == "published" and not new_primary
    if refused:
        # The latch recorded a first publication that never became visible: an actionable refusal
        # naming the latch and the recovery, never a silent empty store.
        for attempt in (jobs.load_jobs, lambda: _create("over an unpublished first store")):
            with pytest.raises(CronStoreError, match="Nothing is restored automatically") as info:
                attempt()
            assert latch.name in str(info.value) and "Strict store durability" in str(info.value)
        assert not primary.exists() and not backup.exists()
        assert latch.read_bytes() == _LATCH_BYTES["published"]
        latch.rename(cron / "latch.aside")  # the documented deliberate empty start
    else:
        assert {j["id"] for j in jobs.load_jobs()} == (_ids(primary) if primary.exists() else set())
    before = _ids(primary) if primary.exists() else set()
    after = _create("after the killed writer")
    assert _ids(primary) == before | {after} == {j["id"] for j in jobs.load_jobs()}
    assert seeds <= _ids(primary)
    assert backup.read_bytes() == primary.read_bytes() and latch.read_bytes() == _LATCH_BYTES["published"]
    assert _private(primary) and _private(backup) and _private(latch)
    assert _staging(cron) == left  # never swept, read or merged; the next writer's own is gone
    if refused:
        assert (cron / "latch.aside").read_bytes() == _LATCH_BYTES["published"]

_ROOT_ONLY = pytest.mark.skipif(getattr(os, "geteuid", lambda: -1)() != 0,
                                reason="needs root in a disposable container for a second real UID")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case", [
    "stale_remove", "concurrent_watchers", "lock_timeout", "symlink_physical_lock", "crossed_alias",
    "trace_privacy_logged", *_KILL_SAVE, *_KILL_FIRST, *_KILL_FORENSIC,
    pytest.param("second_uid", marks=_ROOT_ONLY),
    pytest.param("foreign_nonroot_uid", marks=_ROOT_ONLY),
    pytest.param("second_uid_absent_cron", marks=_ROOT_ONLY),
    pytest.param("owner_noop", marks=_ROOT_ONLY),
    pytest.param("root_symlink_lock", marks=_ROOT_ONLY),
])
def test_strict_store_cross_process_contract(case, tmp_path, monkeypatch, caplog):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(_STRICT_ON)
    monkeypatch.setenv("HERMES_HOME", str(home))
    cron, primary, backup, latch = _store(home)
    go, ready = tmp_path / "go", tmp_path / "ready"

    if case in (*_KILL_SAVE, *_KILL_FIRST, *_KILL_FORENSIC):
        _sigkill_case(case, home)
        return

    if case == "stale_remove":
        doomed = _create("doomed")
        stale = jobs.load_jobs()  # snapshot taken before the other process writes
        proc = _spawn(_CREATE_AFTER_GO, go, ready, home=home)
        try:
            # Event-synchronized: the child's create lands while the parent holds its stale
            # snapshot, and must wait for the parent's cross-process lock before it can write.
            _await_signal(ready, proc)
            with jobs._jobs_lock():
                go.touch()
            out, err = proc.communicate(timeout=120)
        finally:
            _reap(proc)
        assert proc.returncode == 0, err
        fresh = out.strip().splitlines()[-1]

        jobs.save_jobs([j for j in stale if j["id"] != doomed], removed_ids={doomed})

        assert _ids(primary) == {fresh}
        assert backup.read_bytes() == primary.read_bytes()
        return

    if case == "concurrent_watchers":
        # Watcher-like creators in parallel processes plus a removal in this one: every reported
        # id is durably readable afterwards and the removal is not undone.
        doomed = _create("doomed")
        procs = [_spawn(_CREATE_AFTER_GO, go, tmp_path / f"ready{i}", 3, home=home) for i in range(4)]
        try:
            for i, proc in enumerate(procs):
                _await_signal(tmp_path / f"ready{i}", proc)
            go.touch()
            with jobs._jobs_lock():
                jobs.save_jobs([j for j in jobs.load_jobs() if j["id"] != doomed], removed_ids={doomed})
            results = [proc.communicate(timeout=120) for proc in procs]
        finally:
            for proc in procs:
                _reap(proc)
        assert all(proc.returncode == 0 for proc in procs), [err for _out, err in results]
        created = {line for out, _err in results for line in out.split()
                   if len(line) == 12 and set(line) <= set("0123456789abcdef")}
        assert len(created) == 12
        assert _ids(primary) == created == {j["id"] for j in jobs.load_jobs()}
        assert backup.read_bytes() == primary.read_bytes()
        return

    if case == "lock_timeout":
        kept = {_create("seed")}
        proc = _spawn(_HOLD_LOCK, jobs_store.lock_path(primary), ready, go, home=home)
        try:
            _await_signal(ready, proc)
            monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.5)
            before = primary.read_bytes()
            with pytest.raises(CronStoreError):
                _create("refused while another process holds the lock")
            assert primary.read_bytes() == before
            # Turning strict off on a latched store fails loudly instead of degrading...
            (home / "config.yaml").write_text("cron:\n  store:\n    strict_durability: false\n")
            with pytest.raises(CronStoreError, match="downgrade"):
                _create("explicit false never downgrades a latched store")
            # ...until the operator follows the documented steps (latch + backup moved aside).
            latch.rename(cron / "latch.aside")
            backup.rename(cron / "last-good.aside")
            kept.add(_create("degraded default write"))  # the #60703 degradation contract
            assert "Timed out" in caplog.text
            go.touch()
            proc.communicate(timeout=120)
        finally:
            _reap(proc)
        assert _ids(primary) == kept
        assert (cron / "latch.aside").exists() and (cron / "last-good.aside").exists()
        return

    if case == "symlink_physical_lock":
        shared, other = tmp_path / "shared", tmp_path / "other"
        shared.mkdir()
        for h in (home, other):
            (h / "cron").mkdir(parents=True)
            (h / "cron" / "jobs.json").symlink_to(shared / "jobs.json")
            (h / "config.yaml").write_text(_STRICT_ON)
        seed = _create("seed")
        assert primary.is_symlink() and _ids(shared / "jobs.json") == {seed}
        assert (shared / ("jobs.json" + jobs_store.LAST_GOOD_SUFFIX)).read_bytes() == (shared / "jobs.json").read_bytes()
        assert (shared / ("jobs.json" + jobs_store.LATCH_SUFFIX)).exists()
        assert jobs_store.lock_path(primary) == jobs_store.lock_path(other / "cron" / "jobs.json")
        with jobs._jobs_lock():  # home's section holds the PHYSICAL lock; the alias must see it
            proc = _spawn(_TRY_CREATE_WITH_SHORT_TIMEOUT, home=other)
            try:
                out, err = proc.communicate(timeout=120)
            finally:
                _reap(proc)
        assert proc.returncode == 0, err
        assert out.strip().splitlines()[-1] == "refused"
        assert _ids(shared / "jobs.json") == {seed}
        monkeypatch.setenv("HERMES_HOME", str(other))
        _create("alias after release")
        assert len(_ids(shared / "jobs.json")) == 2 and primary.is_symlink()
        return

    if case == "crossed_alias":
        # home's store lives in other's cron dir and vice versa, so the two sections take the
        # same two lock files in opposite orders. The second wait is bounded: a refusal, no hang.
        import fcntl
        other = tmp_path / "other"
        (other / "cron").mkdir(parents=True)
        cron.mkdir()
        (other / "config.yaml").write_text(_STRICT_ON)
        primary.symlink_to(other / "cron" / "store-home.json")
        (other / "cron" / "jobs.json").symlink_to(cron / "store-other.json")
        home_seed = _create("home seed")
        monkeypatch.setenv("HERMES_HOME", str(other))
        other_seed = _create("other seed")
        monkeypatch.setenv("HERMES_HOME", str(home))
        with open(cron / ".jobs.lock", "a+") as holder:  # home's legacy == other's physical lock
            fcntl.flock(holder, fcntl.LOCK_EX)
            started = time.monotonic()
            proc = _spawn(_TRY_CREATE_WITH_SHORT_TIMEOUT, home=other)
            try:
                out, err = proc.communicate(timeout=120)
            finally:
                _reap(proc)
            waited = time.monotonic() - started
        assert proc.returncode == 0, err
        assert out.strip().splitlines()[-1] == "refused" and waited < 60
        assert _ids(cron / "store-other.json") == {other_seed}
        assert _ids(other / "cron" / "store-home.json") == {home_seed}
        proc = _spawn(_TRY_CREATE_WITH_SHORT_TIMEOUT, home=other)
        try:
            out, err = proc.communicate(timeout=120)
        finally:
            _reap(proc)
        assert out.strip().splitlines()[-1] == "stored", err
        assert len(_ids(cron / "store-other.json")) == 2
        return

    if case == "trace_privacy_logged":
        # A fresh process logging the refusal with logger.exception (the scheduler's shape) prints
        # the file and exception type, never the malformed config's text.
        seed = _create("seed")
        (home / "config.yaml").write_text(f"cron: [{_PRIVATE}")
        proc = _spawn(_LOG_REFUSAL, home=home)
        try:
            out, err = proc.communicate(timeout=120)
        finally:
            _reap(proc)
        assert proc.returncode == 0, err
        assert out.strip().splitlines()[-1] == "refused"
        assert "Traceback" in out and "CronStoreError" in out
        assert "config.yaml cannot be parsed (" in out and _PRIVATE not in out
        assert _ids(primary) == {seed} and latch.exists()
        return

    # Root-only cases: the gateway user owns the profile home; root writes first.
    uid = gid = _SECOND_UID
    opened = []
    try:
        for ancestor in (tmp_path, *tmp_path.parents):  # let the gateway user reach its own home
            mode = ancestor.stat().st_mode & 0o7777
            if not mode & 0o001:
                opened.append((ancestor, mode))
                ancestor.chmod(mode | 0o001)
        owned = [home, home / "backups", home / "backups" / "config", home / "tmp"]
        if case != "second_uid_absent_cron":
            owned += [cron, cron / "output"]  # second_uid: pre-created; absent: root creates them
        for d in owned[1:]:
            d.mkdir(parents=True, exist_ok=True)
        for p in (*owned, home / "config.yaml"):
            os.chown(p, uid, gid)
        if case == "root_symlink_lock":
            unrelated = tmp_path / "unrelated-root-file"
            unrelated.write_bytes(b"UNRELATED-ROOT-FILE")
            unrelated.chmod(0o600)
            before = (unrelated.stat().st_uid, unrelated.stat().st_gid, unrelated.read_bytes())
            (cron / ".jobs.lock").symlink_to(unrelated)
            with pytest.raises(CronStoreError):
                _create("refuse symlink lock without side effects")
            assert (unrelated.stat().st_uid, unrelated.stat().st_gid, unrelated.read_bytes()) == before
            assert not primary.exists()
            return
        if case == "owner_noop":
            _create("seed as root")
            for p in (cron, cron / ".jobs.lock", primary, backup, latch):
                os.chown(p, uid, gid)
            before = primary.read_bytes()
            monkeypatch.setattr(os, "fchown", lambda fd, u, g: None)  # a chown that silently no-ops
            with pytest.raises(CronStoreError, match="staged"):
                _create("never published with the wrong owner")
            assert primary.read_bytes() == before and _no_staging_left(cron)
            return
        root_id = _create("root first write")
        for p in (cron, cron / "output", primary, backup, latch, cron / ".jobs.lock"):
            assert (p.stat().st_uid, p.stat().st_gid) == (uid, gid), p.name
        assert _private(primary) and _private(backup) and _private(latch)

        if case == "foreign_nonroot_uid":
            # A non-root user who does not own the store cannot lock, read or publish it: a
            # refusal with every file exactly as it was.
            before = {p: (p.read_bytes(), p.stat().st_uid) for p in (primary, backup, latch)}
            proc = _spawn(_TRY_CREATE_WITH_SHORT_TIMEOUT, home=home,
                          env={"HOME": str(tmp_path), "TMPDIR": str(tmp_path)},
                          user=uid - 1, group=gid - 1, extra_groups=[])
            try:
                out, err = proc.communicate(timeout=120)
            finally:
                _reap(proc)
            assert proc.returncode == 0, err
            assert out.strip().splitlines()[-1] == "refused"
            assert {p: (p.read_bytes(), p.stat().st_uid) for p in before} == before
            return

        proc = _spawn(_GATEWAY_USER_ROUND_TRIP, home=home,
                      env={"HOME": str(home), "TMPDIR": str(home / "tmp")},
                      user=uid, group=gid, extra_groups=[])
        try:
            out, err = proc.communicate(timeout=120)
        finally:
            _reap(proc)
        assert proc.returncode == 0, err
        seen = set(json.loads(out.strip().splitlines()[-1]))
        assert root_id in seen and len(seen) == 2 and _ids(primary) == seen
        assert (primary.stat().st_uid, backup.read_bytes()) == (uid, primary.read_bytes())
    finally:
        for ancestor, mode in reversed(opened):
            ancestor.chmod(mode)
