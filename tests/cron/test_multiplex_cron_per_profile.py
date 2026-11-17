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


# ── The per-run .env reload must not rewrite the shared os.environ ─────────
#
# cron/scheduler.py reloaded <profile home>/.env into os.environ on every run
# (agent path: override=True + clearing known keys). Under multiplex that
# handed a secondary profile's credentials to the default bot.


class _StopAfterReload(Exception):
    pass


def _run_no_agent(home):
    from cron.scheduler import run_job

    with as_cron_of(home):
        return run_job({"id": "job1", "name": "dump", "no_agent": True, "script": "dump_env.py"})


def _run_agent_path(home, monkeypatch):
    """Run the agent path up to the line right after the .env reload."""
    import cron.scheduler as sched

    def stop(job):
        raise _StopAfterReload()

    monkeypatch.setattr(sched, "_resolve_delivery_target", stop)
    with as_cron_of(home):
        try:
            sched.run_job({"id": "job1", "name": "agent", "prompt": "hi"})
        except _StopAfterReload:
            pass


def _unscoped_default_reads():
    from gateway.config import _getenv

    return {
        "os.environ": {k: os.environ.get(k) for k in ("JIRA_API_TOKEN", "NOTION_API_KEY", "EDU_ONLY_KEY")},
        "gateway._getenv": _getenv("JIRA_API_TOKEN"),
    }


def test_educator_runs_leave_the_default_environment_alone(profiles, monkeypatch):
    root, edu = profiles
    ss.set_multiplex_active(True)
    env_before = dict(os.environ)
    reads_before = _unscoped_default_reads()

    ok, output, final, err = _run_no_agent(edu)
    assert ok, err
    assert "jira-edu" in output
    _run_agent_path(edu, monkeypatch)

    assert dict(os.environ) == env_before
    assert _unscoped_default_reads() == reads_before
    assert reads_before["os.environ"]["JIRA_API_TOKEN"] == "jira-default"
    assert reads_before["os.environ"]["EDU_ONLY_KEY"] is None


def test_default_runs_leak_nothing_into_an_educator_turn(profiles, monkeypatch):
    root, edu = profiles
    ss.set_multiplex_active(True)
    ok, output, _final, err = _run_no_agent(root)
    assert ok, err and "jira-default" in output
    _run_agent_path(root, monkeypatch)

    # The educator's script and its in-turn reads see only educator values.
    edu_env = _script_env(edu)
    assert edu_env["JIRA_API_TOKEN"] == "jira-edu"
    assert edu_env["NOTION_API_KEY"] is None
    from agent.fork_secrets import read_secret

    with as_cron_of(edu):
        assert read_secret("JIRA_API_TOKEN", tool="t") == "jira-edu"
        assert read_secret("NOTION_API_KEY", tool="t") == ""


def test_multiplex_off_still_reloads_dotenv_on_both_paths(profiles, monkeypatch):
    """Regression: single-profile cron keeps its per-run .env reload."""
    root, _ = profiles
    import hermes_cli.env_loader as el

    calls = []
    monkeypatch.setattr(el, "load_hermes_dotenv", lambda **kw: calls.append(kw.get("hermes_home")))
    ok, _out, _final, err = _run_no_agent(root)
    assert ok, err
    _run_agent_path(root, monkeypatch)
    assert len(calls) == 2
    assert all(str(c) == str(root) for c in calls)


def test_multiplex_on_never_calls_the_reload(profiles, monkeypatch):
    root, edu = profiles
    import hermes_cli.env_loader as el

    calls = []
    monkeypatch.setattr(el, "load_hermes_dotenv", lambda **kw: calls.append(kw))
    ss.set_multiplex_active(True)
    for home in (edu, root):
        _run_no_agent(home)
        _run_agent_path(home, monkeypatch)
    assert calls == []


def test_run_agent_import_loads_the_process_home_under_multiplex(profiles):
    """run_agent loads a .env into os.environ at import; a first import inside
    a secondary profile's scope must load the process (default) home."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    import run_agent

    root, edu = profiles
    tok = set_hermes_home_override(str(edu))
    try:
        assert run_agent._import_time_env_home() == edu  # multiplex off: unchanged
        ss.set_multiplex_active(True)
        assert run_agent._import_time_env_home() == root
    finally:
        reset_hermes_home_override(tok)


# ── Bot Chat delivery runs the child as the job's own profile ──────────────


def _capture_bot_chat(monkeypatch):
    import types

    import cron.scheduler as sched

    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = list(argv)
        seen["env"] = dict(kw.get("env") or {})
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sched.subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", lambda _n: "/usr/local/bin/hermes")
    return sched, seen


def test_bot_chat_delivery_runs_as_the_owning_profile(profiles, monkeypatch):
    root, edu = profiles
    sched, seen = _capture_bot_chat(monkeypatch)
    ss.set_multiplex_active(True)

    with as_cron_of(edu):
        assert sched._deliver_to_bot_chat({"id": "job1", "name": "n"}, "out", "") is None
    argv, env = seen["argv"], seen["env"]
    assert argv[argv.index("-p") + 1] == "educator"
    assert "HERMES_HOME" not in env
    assert env.get("JIRA_API_TOKEN") == "jira-edu"
    assert "NOTION_API_KEY" not in env  # the default's secret

    with as_cron_of(root):
        assert sched._deliver_to_bot_chat({"id": "job1", "name": "n"}, "out", "") is None
    argv, env = seen["argv"], seen["env"]
    assert "-p" not in argv
    assert env.get("JIRA_API_TOKEN") == "jira-default"


def test_bot_chat_delivery_off_is_unchanged(profiles, monkeypatch):
    root, _ = profiles
    sched, seen = _capture_bot_chat(monkeypatch)
    assert sched._deliver_to_bot_chat({"id": "job1", "name": "n"}, "out", "") is None
    assert "-p" not in seen["argv"]
    assert seen["env"] == dict(os.environ)
