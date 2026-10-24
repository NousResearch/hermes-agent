"""``jobs.json`` must stay readable by BOTH identities of a shared HERMES_HOME.

The gateway runs as one UID and the desktop app/CLI as another, sharing the tree through
a common group (``HERMES_SKIP_CHMOD=1``, ``HERMES_HOME_MODE=2770``). ``mkstemp`` stages
``jobs.json`` in 0600 and, with chmod skipped, nothing ever widened it: the peer's
``GET /api/cron/jobs`` answered 500 with ``Permission denied`` while the ticker itself
was healthy. The store therefore takes its mode from the directory (group-writable +
setgid = shared) and heals a file a previous release left owner-only.
"""

from __future__ import annotations

import contextlib
import os
import stat

import pytest

from cron import jobs as cronjobs


@pytest.fixture(autouse=True)
def _no_operator_sharing_flags(monkeypatch):
    """Decide from the directory's shape, not from this host's exported flags."""
    monkeypatch.delenv("HERMES_SKIP_CHMOD", raising=False)
    monkeypatch.delenv("HERMES_CONTAINER", raising=False)


def _job(jid: str = "job") -> dict:
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "interval", "minutes": 60},
        "next_run_at": "2026-10-24T00:00:00+00:00",
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": None, "completed": 0},
        "deliver": "local",
    }


@contextlib.contextmanager
def _store(tmp_path, *, shared: bool):
    """A cron store whose directory is (or is not) shaped like a shared HERMES_HOME."""
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    os.chmod(cron_dir, 0o2770 if shared else 0o0755)
    if shared and not stat.S_IMODE(cron_dir.stat().st_mode) & stat.S_ISGID:
        pytest.skip("filesystem drops the setgid bit, cannot model a shared home")
    with cronjobs.use_cron_store(tmp_path):
        yield cron_dir


def test_shared_cron_dir_writes_group_readable_jobs_json(tmp_path):
    with _store(tmp_path, shared=True) as cron_dir:
        cronjobs.save_jobs([_job()])

    mode = stat.S_IMODE((cron_dir / "jobs.json").stat().st_mode)
    assert mode & 0o660 == 0o660, oct(mode)
    assert not mode & 0o007, oct(mode)


def test_private_cron_dir_keeps_owner_only(tmp_path):
    with _store(tmp_path, shared=False) as cron_dir:
        cronjobs.save_jobs([_job()])

    mode = stat.S_IMODE((cron_dir / "jobs.json").stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_load_widens_jobs_json_a_previous_release_left_owner_only(tmp_path):
    """The gateway (the owner) heals the file on its next read, so the peer stops 500ing."""
    with _store(tmp_path, shared=True) as cron_dir:
        target = cron_dir / "jobs.json"
        target.write_text('{"jobs": []}', encoding="utf-8")
        os.chmod(target, 0o600)

        assert cronjobs.load_jobs() == []

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode & 0o660 == 0o660, oct(mode)


def test_load_survives_a_peer_owned_file_it_cannot_chmod(tmp_path, monkeypatch):
    """Only the owner may chmod: the peer's EPERM is expected and must not fail the read."""
    with _store(tmp_path, shared=True) as cron_dir:
        target = cron_dir / "jobs.json"
        target.write_text('{"jobs": []}', encoding="utf-8")

        def _deny(*_a, **_k):
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(os, "chmod", _deny)
        assert cronjobs.load_jobs() == []
