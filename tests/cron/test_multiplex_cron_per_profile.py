"""One cron job per profile in a multiplexed gateway (acceptance + 4f).

Each profile's job runs the way the multiplex ticker runs it
(``set_hermes_home_override`` + ``use_cron_store`` + the profile's secret
scope, as ``_run_one_job_body`` installs it) and must:

  * hand its pre-run script the owning profile's credentials — never the
    default profile's (4f);
  * keep its notepad in its own home (module-level home capture sweep);
  * get its RBAC ceiling from its own config.yaml (fix 1 + ownership registry).

With multiplex off the script env is exactly today's ``build_subprocess_env()``.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager

import pytest

from agent import secret_scope as ss

SCRIPT = (
    "import os, json\n"
    "print(json.dumps({k: os.environ.get(k) for k in "
    "('JIRA_API_TOKEN', 'NOTION_API_KEY', 'EDU_ONLY_KEY', 'HERMES_HOME')}))\n"
)


@pytest.fixture()
def profiles(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    edu = root / "profiles" / "educator"
    for home, token, roles in (
        (root, "jira-default", "admin"),
        (edu, "jira-edu", "readonly"),
    ):
        (home / "scripts").mkdir(parents=True)
        (home / "cron").mkdir(parents=True)
        (home / "scripts" / "dump_env.py").write_text(SCRIPT, encoding="utf-8")
        env = f"JIRA_API_TOKEN={token}\n"
        if home == root:
            env += "NOTION_API_KEY=notion-default\n"
        else:
            env += "EDU_ONLY_KEY=edu-only\n"
        (home / ".env").write_text(env, encoding="utf-8")
        (home / "config.yaml").write_text(
            f"slack:\n  user_roles:\n    UOWNER: {roles}\n", encoding="utf-8"
        )
        (home / "ownership").mkdir()
        (home / "ownership" / "registry.json").write_text(
            json.dumps({"automations": {
                "cron:job1": {
                    "owner": {"user_id": "UOWNER", "platform": "slack"},
                    "collaborators": [],
                    "source": "test",
                }
            }}),
            encoding="utf-8",
        )
    monkeypatch.setenv("HERMES_HOME", str(root))
    # The gateway process loaded the DEFAULT profile's .env at startup.
    monkeypatch.setenv("JIRA_API_TOKEN", "jira-default")
    monkeypatch.setenv("NOTION_API_KEY", "notion-default")
    from gateway import tool_access

    tool_access._config_cache.clear()
    ss.set_multiplex_active(False)
    yield root, edu
    ss.set_multiplex_active(False)
    tool_access._config_cache.clear()


@contextmanager
def as_cron_of(home):
    """The multiplex ticker's per-profile scope (+ the job body's secret scope)."""
    from cron.jobs import use_cron_store
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    htok = set_hermes_home_override(str(home))
    stok = ss.set_secret_scope(ss.build_profile_secret_scope(home))
    try:
        with use_cron_store(home):
            yield
    finally:
        ss.reset_secret_scope(stok)
        reset_hermes_home_override(htok)


def _script_env(home):
    from cron.scheduler import _run_job_script

    with as_cron_of(home):
        ok, out = _run_job_script("dump_env.py")
    assert ok, out
    return json.loads(out.strip().splitlines()[-1])


def test_script_env_is_the_owning_profiles(profiles):
    root, edu = profiles
    ss.set_multiplex_active(True)
    edu_env = _script_env(edu)
    assert edu_env["JIRA_API_TOKEN"] == "jira-edu"
    assert edu_env["EDU_ONLY_KEY"] == "edu-only"
    assert edu_env["NOTION_API_KEY"] is None  # default's secret, not educator's
    default_env = _script_env(root)
    assert default_env["JIRA_API_TOKEN"] == "jira-default"
    assert default_env["NOTION_API_KEY"] == "notion-default"
    assert default_env["EDU_ONLY_KEY"] is None


def test_notepad_lands_in_each_profiles_home(profiles):
    root, edu = profiles
    ss.set_multiplex_active(True)
    from cron import notepad

    for home, value in ((root, "default-cursor"), (edu, "edu-cursor")):
        with as_cron_of(home):
            notepad.set_note("job1", "cursor", value)
    for home, value in ((root, "default-cursor"), (edu, "edu-cursor")):
        assert (home / "cron" / "notepad.db").exists()
        with as_cron_of(home):
            assert notepad.get_note("job1", "cursor") == value


def test_rbac_ceiling_from_each_profiles_config(profiles):
    root, edu = profiles
    ss.set_multiplex_active(True)
    from cron.rbac_ceiling import cron_owner_grant

    with as_cron_of(root):
        assert "*" in cron_owner_grant({"id": "job1"})
    with as_cron_of(edu):
        grant = cron_owner_grant({"id": "job1"})
        assert grant is not None and "*" not in grant and "terminal" not in grant


def test_multiplex_off_script_env_unchanged(profiles):
    root, _ = profiles
    from cron.scheduler import _run_job_script
    from tools.environments.local import build_subprocess_env

    expected = build_subprocess_env()
    ok, out = _run_job_script("dump_env.py")
    assert ok, out
    got = json.loads(out.strip().splitlines()[-1])
    assert got["JIRA_API_TOKEN"] == expected.get("JIRA_API_TOKEN") == "jira-default"
    assert got["NOTION_API_KEY"] == expected.get("NOTION_API_KEY")
