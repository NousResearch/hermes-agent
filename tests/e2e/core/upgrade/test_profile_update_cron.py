"""`hermes profile install` / `hermes profile update` against a real distribution with cron jobs.

A distribution author schedules jobs with the real CLI (``hermes -p author cron create``) and
publishes the profile directory; a user installs it, schedules a job of their own, and later
pulls the author's next version with ``hermes profile update``. Every step is a fresh sandboxed
CLI process over a fake HOME; the assertions read the cron store the scheduler itself reads
(``cron/jobs.json``) and the CLI's own ``cron list``.

Contract (documented in the command help and profile-distributions.md): shipped jobs are
installed but NOT auto-scheduled; an update refreshes the jobs the distribution ships, removes
the ones the author retired, and leaves the jobs the user added in place.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I

pytestmark = [
    pytest.mark.linux_only,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
]


class Gap(Exception):
    """A contract breach tracked in KNOWN. Not an AssertionError, so a strict xfail on it cannot
    swallow an unrelated failure (a crashed CLI, a missing job store) in the same test."""


def gap(ok: bool, msg: str) -> None:
    if not ok:
        raise Gap(msg)


KNOWN: dict[str, str] = {
    "keeps_user_jobs": "#120823 profile update replaces cron/jobs.json wholesale, deleting the user's own jobs",
    "install_paused": "#120823 profile install leaves shipped cron jobs enabled and scheduled",
}


def known(key: str):
    return pytest.mark.xfail(strict=True, raises=Gap, reason=KNOWN[key]) if key in KNOWN else ()


PY = str(H.WORKTREE / ".venv" / "bin" / "python")


class World:
    def __init__(self, root: Path):
        self.root = root
        self.env = H.isolated_env(root, pythonpath=H.WORKTREE)
        (root / "tmp").mkdir(exist_ok=True)
        self.env["TMPDIR"] = str(root / "tmp")
        self.hermes_home = Path(self.env["HERMES_HOME"])
        self.src = root / "dist-src"
        self.log: list[str] = []
        self.after_install: list[dict] = []
        self.before_update: list[dict] = []
        self.after_update: list[dict] = []

    def cli(self, *args: str):
        cp = H.run([PY, "-m", "hermes_cli.main", *args], env=self.env, cwd=self.root, writable=[self.root], timeout=300)
        self.log.append(H.describe(cp, 1500))
        assert cp.returncode == 0 and I.TRACEBACK not in cp.stdout + cp.stderr, H.describe(cp)
        return cp

    @property
    def installed(self) -> Path:
        return self.hermes_home / "profiles" / "shipped-bot"


def _by_name(jobs: list[dict]) -> dict[str, dict]:
    return {str(j.get("name")): j for j in jobs}


def _is_scheduled(job: dict) -> bool:
    return bool(job.get("enabled", True)) and job.get("state") not in ("paused", "disabled")


def _publish(world: World, version: str, soul: str, jobs: list[dict]) -> None:
    (world.src / "distribution.yaml").write_text(
        f"name: shipped-bot\nversion: {version}\ndescription: e2e distribution\n", encoding="utf-8")
    (world.src / "SOUL.md").write_text(soul, encoding="utf-8")
    store = world.src / "cron" / "jobs.json"
    data = json.loads(store.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data["jobs"] = jobs
    else:
        data = jobs
    store.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> World:
    w = World(tmp_path_factory.mktemp("profile-dist"))
    # The author builds the agent with the real CLI.
    w.cli("profile", "create", "author", "--no-alias")
    w.cli("-p", "author", "cron", "create", "--name", "weekly-digest", "0 9 * * 1", "write the weekly digest v1")
    w.cli("-p", "author", "cron", "create", "--name", "retired-job", "0 8 * * *", "a job the author retires in v2")
    author = w.hermes_home / "profiles" / "author"
    assert set(_by_name(I.cron_jobs(author))) == {"weekly-digest", "retired-job"}
    w.src.mkdir()
    shutil.copytree(author / "cron", w.src / "cron", ignore=shutil.ignore_patterns("*.lock", "output", "*.db*"))
    _publish(w, "1.0.0", "You are shipped-bot v1.\n", I.cron_jobs(author))
    # The user installs it.
    w.cli("profile", "install", str(w.src), "-y")
    w.after_install = I.cron_jobs(w.installed)
    # The user schedules a job of their own and pauses one shipped job they do not want.
    w.cli("-p", "shipped-bot", "cron", "create", "--name", "my-own-reminder", "0 7 * * *", "the user's own reminder")
    w.before_update = I.cron_jobs(w.installed)
    assert set(_by_name(w.before_update)) == {"weekly-digest", "retired-job", "my-own-reminder"}, w.before_update
    # The author ships v2: the digest prompt changes, retired-job is gone.
    v2 = [dict(j) for j in I.cron_jobs(author) if j.get("name") == "weekly-digest"]
    v2[0]["prompt"] = "write the weekly digest v2"
    _publish(w, "1.1.0", "You are shipped-bot v2.\n", v2)
    w.cli("profile", "update", "shipped-bot", "-y")
    w.after_update = I.cron_jobs(w.installed)
    return w


def test_profile_update_refreshes_shipped_jobs_and_drops_retired_ones(world):
    jobs = _by_name(world.after_update)
    assert "weekly-digest" in jobs, f"shipped job missing after update: {world.after_update}"
    assert jobs["weekly-digest"].get("prompt") == "write the weekly digest v2", "shipped job definition not refreshed"
    assert "retired-job" not in jobs, "a job the author retired is still scheduled after `profile update`"
    assert sum(1 for j in world.after_update if j.get("name") == "weekly-digest") == 1, "shipped job duplicated"
    assert (world.installed / "SOUL.md").read_text(encoding="utf-8") == "You are shipped-bot v2.\n"
    listed = world.cli("-p", "shipped-bot", "cron", "list").stdout
    assert "weekly-digest" in listed and "retired-job" not in listed, listed


@pytest.mark.parametrize("key", [pytest.param("keeps_user_jobs", marks=known("keeps_user_jobs"))])
def test_profile_update_keeps_the_users_own_cron_jobs(world, key):
    jobs = _by_name(world.after_update)
    gap("my-own-reminder" in jobs, f"`profile update` deleted the user's own cron job; jobs now: {sorted(jobs)}")
    assert jobs["my-own-reminder"].get("prompt") == "the user's own reminder"
    assert _is_scheduled(jobs["my-own-reminder"]), "the user's own job was paused by the update"


@pytest.mark.parametrize("key", [pytest.param("install_paused", marks=known("install_paused"))])
def test_profile_install_does_not_auto_schedule_shipped_jobs(world, key):
    jobs = _by_name(world.after_install)
    assert set(jobs) == {"weekly-digest", "retired-job"}, world.after_install
    running = sorted(n for n, j in jobs.items() if _is_scheduled(j))
    gap(not running, f"shipped jobs are live right after `profile install` (documented: not auto-scheduled): {running}")
