"""Per-lane Chromium user-data dirs must never enter a backup archive.

`hermes backup` excludes the singular ``browser-profile/`` and the underscored
``browser_profiles/``, but a machine running several automation lanes names its Chromium user-data
dirs per lane — ``headless-profiles/``, ``agent-chrome-profile/``, ``acme-headless-profile/``,
``site-cdp-profile/`` — and none of those matched. Every ``hermes backup`` run therefore shipped
``Default/Login Data`` + ``Default/Cookies`` (a Chromium credential store) plus gigabytes of Service
Worker / Code Cache into the archive.

The fix matches a root entry on a conjunction (\"profile\" AND a Chromium/automation hint) so a user
directory such as ``profiles-archive/`` is still archived.

NOTE: ``hermes_cli.backup`` is imported INSIDE each test helper, matching tests/hermes_cli/test_backup.py.
A module-level import resolves the Hermes home before conftest sandboxes it and leaks real-home
state into the rest of the session (observed: 7 -> 20 failures across this file).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _excluded(rel: str) -> bool:
    from hermes_cli import backup as B

    return B._in_excluded_root_dir(Path(rel)) or B._should_exclude(Path(rel))


def _root_dir_excluded(rel: str) -> bool:
    from hermes_cli import backup as B

    return B._in_excluded_root_dir(Path(rel))


def _should_exclude(rel: str) -> bool:
    from hermes_cli import backup as B

    return B._should_exclude(Path(rel))


@pytest.mark.parametrize(
    "rel",
    [
        # Per-lane dirs with the hyphenated names the old rule missed.
        "headless-profiles/billing/Default/Login Data",
        "headless-profiles/intranet/Default/Cookies",
        "headless-profiles/portal/Default/Web Data",
        "agent-chrome-profile/Default/Login Data",
        "agent-chrome-profile/Default/Service Worker/CacheStorage/abc",
        "agent-chrome-profile/Default/Code Cache/js",
        "acme-headless-profile/Default/Cookies",
        "site-cdp-profile/Default/Cookies",
        "chrome-for-testing/Default/Login Data",
        # Pre-existing names must keep working.
        "browser-profile/Default/Login Data",
        "browser_profiles/x/Default/Cookies",
        # A named profile's own browser dir is a root entry under profiles/.
        "profiles/worker/headless-profiles/Default/Login Data",
    ],
)
def test_chromium_userdata_roots_are_excluded(rel: str) -> None:
    assert _root_dir_excluded(rel), rel


@pytest.mark.parametrize(
    "rel",
    [
        # Durable state the backup exists to protect.
        "state.db",
        "config.yaml",
        ".env",
        "auth.json",
        "cron/jobs.json",
        "skills/devops/hermes-encrypted-backup/SKILL.md",
        "sessions/abc.json",
        "logs/gateway.log",
        "kanban/board.json",
        "profiles/worker/config.yaml",
        "profiles/worker/skills/finance/SKILL.md",
        # \"profile\" alone is user data, not a browser tree — the conjunction must hold.
        "profiles-archive/notes.md",
        "profile-exports/report.md",
        # A skill-owned dir that merely CONTAINS a browser dir is not a root entry.
        "profiles/browser-profiles-mirror/x/Cookies",
    ],
)
def test_durable_and_user_paths_are_kept(rel: str) -> None:
    assert not _root_dir_excluded(rel), rel
    assert not _should_exclude(rel), rel


@pytest.mark.parametrize(
    "rel",
    [
        # Already-excluded runtime roots must STAY excluded (regression guard for the new branch).
        "models/foo.gguf",
        "hermes-agent/run_agent.py",
        "tools/thing",
        "installs/thing",
        "runtimes/thing",
    ],
)
def test_runtime_roots_remain_excluded(rel: str) -> None:
    assert _excluded(rel), rel


def test_named_profile_root_matching_is_scoped_to_the_profile() -> None:
    """A browser dir under a NAMED profile is excluded; the same name at a skill path is not."""
    assert _root_dir_excluded("profiles/nimbus/headless-profiles/Default/Cookies")
    # …but a browser-ish dir buried inside a skill's own data is user data.
    assert not _root_dir_excluded("profiles/nimbus/skills/x/headless-profiles/keep.md")