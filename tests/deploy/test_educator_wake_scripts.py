"""Wake-gate pre-run scripts of the educator profile (deploy/educator/scripts).

Every external call goes through ``educator_wake.http_json`` / ``run_git`` /
``list_executions`` / ``find_job_id``, which these tests replace — no network,
no git, no real cron ledger.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[2] / "deploy" / "educator" / "scripts"

PRD_DS = "2daff849-6e1b-4e4f-960a-a07acfbc2a1b"
RELEASES_DS = "6f1611b4-c204-4944-a0c7-026e67604c77"
ASSETS_DS = "c7436811-d26d-4a73-8c60-69437f3fcb16"
IDEAS_DS = "11111111-2222-3333-4444-555555555555"
BOT = "bot-user-1"
HUMAN = "human-user-1"

PRD_A = "aaaaaaaa-0000-0000-0000-000000000001"
PRD_B = "bbbbbbbb-0000-0000-0000-000000000002"
ROW_1 = "cccccccc-0000-0000-0000-000000000003"


# ── fixtures ─────────────────────────────────────────────────────────────

def clean_git_config(repo):
    """What a plain `git clone` writes."""
    return ("[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = false\n"
            "\tlogallrefupdates = true\n"
            f'[remote "origin"]\n\turl = https://github.com/everafterai/{repo}.git\n'
            "\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
            '[branch "main"]\n\tremote = origin\n\tmerge = refs/heads/main\n')


def fake_git(calls):
    def run(args, cwd, env):
        calls.append((args, cwd, env))
        if "ls-remote" in args:
            return 0, "ref: refs/heads/main\tHEAD\n0123abcd\tHEAD\n"
        return 0, ""
    return run


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "profiles" / "educator"
    for repo in ("product-skills", "ea-monorepo"):
        (h / "workspace" / "checkouts" / repo / ".git").mkdir(parents=True)
        (h / "workspace" / "checkouts" / repo / ".git" / "config").write_text(clean_git_config(repo))
    (h / "workspace" / "notebook.md").write_text("# notebook\n")
    (h / "config.yaml").write_text(yaml.safe_dump({
        "notion": {"write_guard": {
            "enabled": True,
            "release_kits": {"data_sources": [RELEASES_DS, ASSETS_DS]},
            "prds": {"data_sources": [PRD_DS]},
            "ideas": {"data_sources": [IDEAS_DS]},
        }},
        "release_updates": {"release_assets_data_source_id": ASSETS_DS,
                            "prd_jira_property": "JIRA"},
    }))
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setenv("NOTION_API_KEY", "ntn_test")
    monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "bot@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "jira-token")
    monkeypatch.setenv("GITHUB_PR_TOKEN", "ghp_test")
    return h


@pytest.fixture
def mods(home, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    for name in ("educator_wake", "educator_release_watch", "educator_explore"):
        sys.modules.pop(name, None)
    wake = importlib.import_module("educator_wake")
    watch = importlib.import_module("educator_release_watch")
    explore = importlib.import_module("educator_explore")
    git_calls: list = []
    monkeypatch.setattr(wake, "run_git", fake_git(git_calls))
    monkeypatch.setattr(wake, "find_job_id", lambda script: "job-" + script.split(".")[0])
    monkeypatch.setattr(wake, "list_executions", lambda job_id: [])
    monkeypatch.setattr(wake, "due_pending_ids", lambda action: [])
    wake._test_git_calls = git_calls
    return wake, watch, explore


def title(text):
    return {"type": "title", "title": [{"plain_text": text}]}


def prd(page_id, name, status, jira_url, assets=("Slack post",), owner="Dana"):
    return {
        "object": "page", "id": page_id, "url": f"https://www.notion.so/{page_id.replace('-', '')}",
        "properties": {
            "Item": title(name),
            "Release Comms Status": {"type": "status", "status": {"name": status}},
            "JIRA": {"type": "url", "url": jira_url},
            "Release Comms Assets": {"type": "multi_select",
                                     "multi_select": [{"name": a} for a in assets]},
            "Comm Owner": {"type": "people", "people": [{"id": "u-" + owner, "name": owner}]},
        },
    }


def asset(row_id, name, status, edited_by=HUMAN, edited="2026-11-17T10:00:00.000Z",
          asset_type="Slack post"):
    return {
        "object": "page", "id": row_id, "url": f"https://www.notion.so/{row_id.replace('-', '')}",
        "last_edited_time": edited, "last_edited_by": {"object": "user", "id": edited_by},
        "properties": {
            "Name": title(name),
            "Status": {"type": "select", "select": {"name": status}},
            "Asset type": {"type": "select", "select": {"name": asset_type}},
            "Destination": {"type": "select", "select": {"name": "#release-updates"}},
        },
    }


class FakeHTTP:
    """Routes Notion + Jira calls; records every request."""

    def __init__(self):
        self.prds: list = []
        self.assets: list = []
        self.jira: dict = {}          # key -> status name (missing key = 404)
        self.jira_search_400 = False
        self.notion_down = False
        self.calls: list = []

    def __call__(self, method, url, headers, body=None, timeout=30):
        self.calls.append((method, url, body))
        u = urlparse(url)
        if u.netloc == "api.notion.com":
            if self.notion_down:
                return 503, {"message": "down"}
            assert headers["Authorization"] == "Bearer ntn_test"
            if u.path == "/v1/users/me":
                return 200, {"object": "user", "id": BOT, "type": "bot"}
            if u.path == f"/v1/data_sources/{PRD_DS}" and method == "GET":
                return 200, {"properties": {"Release Comms Status": {"type": "status"},
                                            "JIRA": {"type": "url"}}}
            if u.path == f"/v1/data_sources/{PRD_DS}/query":
                assert method == "POST"
                wanted = {f["status"]["equals"] for f in body["filter"]["or"]}
                rows = [p for p in self.prds
                        if p["properties"]["Release Comms Status"]["status"]["name"] in wanted]
                return 200, {"results": rows, "has_more": False}
            if u.path == f"/v1/data_sources/{ASSETS_DS}/query":
                assert method == "POST"
                since = body["filter"]["last_edited_time"]["on_or_after"]
                rows = [a for a in self.assets if a["last_edited_time"] >= since]
                return 200, {"results": rows, "has_more": False}
            raise AssertionError(f"unexpected Notion call {method} {url}")
        if u.netloc == "example.atlassian.net":
            assert method == "GET"
            if u.path == "/rest/api/3/search/jql":
                if self.jira_search_400:
                    return 400, {"errorMessages": ["An issue with key 'BA-404' does not exist"]}
                jql = parse_qs(u.query)["jql"][0]
                keys = [k.strip() for k in jql[jql.index("(") + 1:jql.index(")")].split(",")]
                issues = [{"key": k, "fields": {"status": {"name": self.jira[k]}}}
                          for k in keys if k in self.jira]
                return 200, {"issues": issues}
            if u.path.startswith("/rest/api/3/issue/"):
                key = u.path.rsplit("/", 1)[1]
                if key not in self.jira:
                    return 404, {"errorMessages": ["Issue does not exist"]}
                return 200, {"key": key, "fields": {"status": {"name": self.jira[key]}}}
            raise AssertionError(f"unexpected Jira call {url}")
        raise AssertionError(f"unexpected host {url}")


@pytest.fixture
def http(mods, monkeypatch):
    wake, _watch, _explore = mods
    fake = FakeHTTP()
    monkeypatch.setattr(wake, "http_json", fake)
    return fake


T0 = datetime(2026, 11, 17, 9, 0, tzinfo=timezone.utc)


def run_watch(mods, now, capsys=None):
    _wake, watch, _ = mods
    payload = watch.main(now=now)
    return payload


def events_of(payload, kind):
    return [e for e in payload.get("events", []) if e["type"] == kind]


# ── release-watch ────────────────────────────────────────────────────────

def test_first_run_with_nothing_eligible_is_silent(mods, http):
    http.prds = [prd(PRD_A, "Audio items", "Not started", "https://x.atlassian.net/browse/BA-1")]
    http.jira = {"BA-1": "In Progress"}
    assert run_watch(mods, T0) == {"wakeAgent": False}


def test_first_run_is_a_baseline_listing_eligible_prds_without_drafting(mods, http):
    http.prds = [prd(PRD_A, "Audio items", "Not started", "https://x.atlassian.net/browse/BA-1"),
                 prd(PRD_B, "Journeys", "Not started", "https://x.atlassian.net/browse/BA-2")]
    http.jira = {"BA-1": "Ready for staging", "BA-2": "In Progress"}
    p = run_watch(mods, T0)
    assert p["wakeAgent"] is True
    assert [e["type"] for e in p["events"]] == ["baseline"]
    assert [x["prd_id"] for x in p["events"][0]["eligible_prds"]] == [PRD_A]
    assert not events_of(p, "kit_start")


def test_quiet_second_run_prints_wake_false(mods, http, capsys):
    http.prds = [prd(PRD_A, "Audio items", "Not started", "https://x.atlassian.net/browse/BA-1")]
    http.jira = {"BA-1": "In Progress"}
    wake, watch, _ = mods
    watch.main(now=T0)
    capsys.readouterr()
    p = watch.main(now=T0 + timedelta(minutes=20))
    assert p == {"wakeAgent": False}
    out = capsys.readouterr().out.strip().splitlines()
    assert json.loads(out[-1]) == {"wakeAgent": False}


def test_kit_start_when_first_ticket_reaches_ready_for_staging(mods, http):
    http.prds = [prd(PRD_A, "Audio items", "Not started", "https://x.atlassian.net/browse/BA-1",
                     assets=("Slack post", "Help article"))]
    http.jira = {"BA-1": "In Progress"}
    run_watch(mods, T0)
    http.jira = {"BA-1": "Ready for staging"}
    p = run_watch(mods, T0 + timedelta(minutes=20))
    [ev] = events_of(p, "kit_start")
    assert ev["prd_id"] == PRD_A and ev["prd_title"] == "Audio items"
    assert ev["assets"] == ["Slack post", "Help article"]
    assert ev["comm_owner"] == [{"id": "u-Dana", "name": "Dana"}]
    assert ev["jira"] == {"BA-1": "Ready for staging"}
    assert ev["retry"] is False
    [change] = events_of(p, "ticket_status")
    assert (change["key"], change["from"], change["to"]) == ("BA-1", "In Progress", "Ready for staging")
    # not repeated next run while the agent works on it
    p3 = run_watch(mods, T0 + timedelta(minutes=40))
    assert p3 == {"wakeAgent": False}


def test_kit_start_retries_after_a_day_if_prd_still_not_started(mods, http):
    http.prds = [prd(PRD_A, "Audio items", "Not started", "https://x.atlassian.net/browse/BA-1")]
    http.jira = {"BA-1": "In Progress"}
    run_watch(mods, T0)
    http.jira = {"BA-1": "In Staging"}
    run_watch(mods, T0 + timedelta(minutes=20))
    p = run_watch(mods, T0 + timedelta(hours=25))
    [ev] = events_of(p, "kit_start")
    assert ev["retry"] is True


def test_new_prd_after_baseline_with_ticket_already_in_staging_starts_a_kit(mods, http):
    http.prds = []
    run_watch(mods, T0)
    http.prds = [prd(PRD_B, "Journeys", "Not started", "https://x.atlassian.net/browse/BA-2")]
    http.jira = {"BA-2": "In Staging"}
    p = run_watch(mods, T0 + timedelta(minutes=20))
    assert [e["prd_id"] for e in events_of(p, "kit_start")] == [PRD_B]


def test_kit_starts_are_capped_per_run_and_the_rest_wait(mods, http, monkeypatch):
    wake, watch, _ = mods
    monkeypatch.setattr(watch, "MAX_KIT_STARTS_PER_RUN", 1)
    run_watch(mods, T0)
    http.prds = [prd(PRD_A, "A", "Not started", "https://x/browse/BA-1"),
                 prd(PRD_B, "B", "Not started", "https://x/browse/BA-2")]
    http.jira = {"BA-1": "Ready for staging", "BA-2": "Ready for staging"}
    p1 = run_watch(mods, T0 + timedelta(minutes=20))
    p2 = run_watch(mods, T0 + timedelta(minutes=40))
    started = [e["prd_id"] for p in (p1, p2) for e in events_of(p, "kit_start")]
    assert sorted(started) == sorted([PRD_A, PRD_B])
    assert p1["deferred_kit_starts"] == 1


def test_status_changes_are_case_insensitive_and_unwatched_ones_are_ignored(mods, http):
    http.prds = [prd(PRD_A, "A", "In progress", "https://x/browse/BA-1")]
    http.jira = {"BA-1": "In Progress"}
    run_watch(mods, T0)
    http.jira = {"BA-1": "Code Review"}
    assert run_watch(mods, T0 + timedelta(minutes=20)) == {"wakeAgent": False}
    http.jira = {"BA-1": "READY FOR STAGING"}
    p = run_watch(mods, T0 + timedelta(minutes=40))
    assert events_of(p, "ticket_status")[0]["to"] == "READY FOR STAGING"


def test_all_done_fires_once_and_rearms_when_a_ticket_reopens(mods, http):
    http.prds = [prd(PRD_A, "A", "In progress", "https://x/browse/BA-1 https://x/browse/BA-3")]
    http.jira = {"BA-1": "In Staging", "BA-3": "In Staging"}
    run_watch(mods, T0)
    http.jira = {"BA-1": "Done", "BA-3": "Done"}
    p = run_watch(mods, T0 + timedelta(minutes=20))
    [ev] = events_of(p, "all_done")
    assert ev["keys"] == ["BA-1", "BA-3"]
    assert run_watch(mods, T0 + timedelta(minutes=40)) == {"wakeAgent": False}
    http.jira = {"BA-1": "In Staging", "BA-3": "Done"}
    run_watch(mods, T0 + timedelta(minutes=60))
    http.jira = {"BA-1": "Done", "BA-3": "Done"}
    assert events_of(run_watch(mods, T0 + timedelta(minutes=80)), "all_done")


def test_jira_search_400_falls_back_to_per_key_reads(mods, http):
    http.prds = [prd(PRD_A, "A", "In progress", "https://x/browse/BA-1 https://x/browse/BA-404")]
    http.jira = {"BA-1": "In Staging"}
    http.jira_search_400 = True
    run_watch(mods, T0)
    http.jira = {"BA-1": "Done"}
    p = run_watch(mods, T0 + timedelta(minutes=20))
    assert events_of(p, "ticket_status")[0]["to"] == "Done"
    assert p["missing_jira_keys"] == ["BA-404"]


def test_asset_moved_to_approved_wakes_once(mods, http):
    run_watch(mods, T0)
    http.assets = [asset(ROW_1, "Slack post: audio", "In review", edited="2026-11-17T09:10:00.000Z")]
    run_watch(mods, T0 + timedelta(minutes=20))
    http.assets = [asset(ROW_1, "Slack post: audio", "Approved", edited="2026-11-17T09:30:00.000Z")]
    p = run_watch(mods, T0 + timedelta(minutes=40))
    [ev] = events_of(p, "asset_approved")
    assert ev["row_id"] == ROW_1 and ev["asset_type"] == "Slack post"
    assert run_watch(mods, T0 + timedelta(minutes=60)) == {"wakeAgent": False}


def test_human_edit_after_approval_wakes_but_bot_edit_does_not(mods, http):
    run_watch(mods, T0)
    http.assets = [asset(ROW_1, "Help: audio", "Approved", edited="2026-11-17T09:10:00.000Z")]
    run_watch(mods, T0 + timedelta(minutes=20))
    http.assets = [asset(ROW_1, "Help: audio", "Approved", edited_by=BOT,
                         edited="2026-11-17T09:30:00.000Z")]
    assert run_watch(mods, T0 + timedelta(minutes=40)) == {"wakeAgent": False}
    http.assets = [asset(ROW_1, "Help: audio", "Approved", edited_by=HUMAN,
                         edited="2026-11-17T09:50:00.000Z")]
    p = run_watch(mods, T0 + timedelta(minutes=60))
    assert events_of(p, "asset_edited_after_approval")[0]["row_id"] == ROW_1


def test_due_veto_actions_wake_the_agent_to_call_run_due(mods, http, monkeypatch):
    wake, _, _ = mods
    run_watch(mods, T0)
    monkeypatch.setattr(wake, "due_pending_ids",
                        lambda action: ["0123456789ab"] if action == "post_release_update" else [])
    p = run_watch(mods, T0 + timedelta(minutes=20))
    [ev] = events_of(p, "veto_due")
    assert ev == {"type": "veto_due", "action": "post_release_update",
                  "pending_ids": ["0123456789ab"], "tool_call": "slack_post_thread(run_due=true)"}


def test_checkouts_are_refreshed_with_the_token_kept_out_of_argv(mods, http, home):
    wake, _, _ = mods
    run_watch(mods, T0)
    calls = wake._test_git_calls
    repos = {Path(cwd).name for _a, cwd, _e in calls}
    assert repos == {"product-skills", "ea-monorepo"}
    for args, _cwd, env in calls:
        assert "ghp_test" not in " ".join(args)
        assert env["GIT_TERMINAL_PROMPT"] == "0" and env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert any("ghp_test" not in v and "Authorization" in v
                   for k, v in env.items() if k.startswith("GIT_CONFIG_VALUE"))
    by_verb = {a[0]: a for a, cwd, _e in calls if Path(cwd).name == "product-skills"}
    assert set(by_verb) == {"ls-remote", "fetch", "reset"}
    # fetch names the expected URL + an explicit refspec, never the configured remote
    assert "https://github.com/everafterai/product-skills.git" in by_verb["fetch"]
    assert "+refs/heads/main:refs/remotes/origin/main" in by_verb["fetch"]
    assert "origin" not in by_verb["fetch"]
    assert by_verb["reset"][-1] == "refs/remotes/origin/main"


def test_run_git_forces_dangerous_config_off(mods, monkeypatch):
    wake, _, _ = mods
    seen = {}

    class Done:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return Done()

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)
    importlib.reload(wake)  # restore the real run_git
    wake.run_git(["fetch"], "/tmp", {})
    argv = seen["argv"]
    for setting in ("core.fsmonitor=false", "core.hooksPath=/dev/null", "credential.helper=",
                    "protocol.file.allow=never", "protocol.ext.allow=never"):
        assert setting in argv
    assert argv.index("fetch") > argv.index("core.fsmonitor=false")


@pytest.mark.parametrize("tamper", [
    '[core]\n\tfsmonitor = /tmp/evil.sh\n',
    '[includeIf "gitdir:/"]\n\tpath = /tmp/evil.cfg\n',
    '[include]\n\tpath = /tmp/evil.cfg\n',
    '[credential]\n\thelper = !/tmp/evil.sh\n',
    '[core]\n\thooksPath = /tmp/hooks\n',
    '[filter "x"]\n\tsmudge = /tmp/evil.sh\n',
    '[branch "main"]\n\tmerge = refs/heads/educator/own-change\n\tpushRemote = evil\n',
])
def test_tampered_git_config_is_refused_before_any_git_runs(mods, http, home, tamper):
    wake, _, _ = mods
    cfg = home / "workspace" / "checkouts" / "product-skills" / ".git" / "config"
    cfg.write_text(cfg.read_text() + tamper)
    p = run_watch(mods, T0)
    [ev] = [e for e in events_of(p, "checkout_error") if e["repo"] == "product-skills"]
    assert "refused, nothing run" in ev["error"]
    assert not [c for c in wake._test_git_calls if Path(c[1]).name == "product-skills"]


def test_repointed_remote_is_refused(mods, http, home):
    wake, _, _ = mods
    cfg = home / "workspace" / "checkouts" / "ea-monorepo" / ".git" / "config"
    cfg.write_text(cfg.read_text().replace("everafterai/ea-monorepo", "attacker/ea-monorepo"))
    p = run_watch(mods, T0)
    [ev] = events_of(p, "checkout_error")
    assert ev["repo"] == "ea-monorepo" and "remote.origin.url" in ev["error"]
    assert not [c for c in wake._test_git_calls if Path(c[1]).name == "ea-monorepo"]


def test_symlinked_git_dir_is_refused(mods, http, home, tmp_path):
    import shutil
    repo = home / "workspace" / "checkouts" / "product-skills"
    shutil.move(str(repo / ".git"), str(tmp_path / "elsewhere"))
    (repo / ".git").symlink_to(tmp_path / "elsewhere")
    p = run_watch(mods, T0)
    assert "not a plain directory" in events_of(p, "checkout_error")[0]["error"]


def test_checkout_failure_is_reported_once_per_error(mods, http, monkeypatch, home):
    wake, _, _ = mods
    run_watch(mods, T0)
    monkeypatch.setattr(wake, "run_git", lambda args, cwd, env: (128, "fatal: auth failed"))
    p = run_watch(mods, T0 + timedelta(minutes=20))
    assert {e["repo"] for e in events_of(p, "checkout_error")} == {"product-skills", "ea-monorepo"}
    assert run_watch(mods, T0 + timedelta(minutes=40)) == {"wakeAgent": False}


def test_missing_checkout_is_reported(mods, http, home):
    import shutil
    shutil.rmtree(home / "workspace" / "checkouts" / "ea-monorepo")
    p = run_watch(mods, T0)
    [ev] = events_of(p, "checkout_error")
    assert ev["repo"] == "ea-monorepo" and "missing" in ev["error"]


def test_notion_outage_reports_once_and_keeps_state(mods, http):
    http.prds = [prd(PRD_A, "A", "Not started", "https://x/browse/BA-1")]
    http.jira = {"BA-1": "In Progress"}
    run_watch(mods, T0)
    http.notion_down = True
    p = run_watch(mods, T0 + timedelta(minutes=20))
    assert events_of(p, "source_error")[0]["source"] == "notion"
    assert run_watch(mods, T0 + timedelta(minutes=40)) == {"wakeAgent": False}
    http.notion_down = False
    http.jira = {"BA-1": "Ready for staging"}
    p = run_watch(mods, T0 + timedelta(minutes=60))
    assert events_of(p, "kit_start")


def test_only_reads_are_sent(mods, http):
    http.prds = [prd(PRD_A, "A", "Not started", "https://x/browse/BA-1")]
    http.jira = {"BA-1": "In Staging"}
    http.assets = [asset(ROW_1, "x", "Approved", edited="2026-11-17T09:10:00.000Z")]
    run_watch(mods, T0)
    run_watch(mods, T0 + timedelta(minutes=20))
    for method, url, _body in http.calls:
        assert method == "GET" or (method == "POST" and url.endswith("/query")), (method, url)


# ── notebook + previous run (both scripts) ───────────────────────────────

def _exec(claimed, status, started=None):
    return {"claimed_at": claimed.isoformat(), "started_at": (started or claimed).isoformat(),
            "finished_at": None, "status": status}


def test_notebook_not_updated_after_a_woken_run(mods, http, home, monkeypatch):
    wake, watch, _ = mods
    http.prds = [prd(PRD_A, "A", "Not started", "https://x/browse/BA-1")]
    http.jira = {"BA-1": "Ready for staging"}
    assert run_watch(mods, T0)["wakeAgent"] is True  # baseline lists PRD_A
    nb = home / "workspace" / "notebook.md"
    old = (T0 - timedelta(hours=1)).timestamp()
    os.utime(nb, (old, old))
    t1 = T0 + timedelta(minutes=20)
    monkeypatch.setattr(wake, "list_executions", lambda job_id: [
        _exec(t1, "running"), _exec(T0 - timedelta(seconds=5), "completed")])
    p = watch.main(now=t1)
    assert p["wakeAgent"] is True
    assert events_of(p, "notebook_not_updated")[0]["job"] == "release-watch"


def test_notebook_updated_during_woken_run_is_fine(mods, http, home, monkeypatch):
    wake, watch, _ = mods
    http.prds = [prd(PRD_A, "A", "Not started", "https://x/browse/BA-1")]
    http.jira = {"BA-1": "Ready for staging"}
    assert run_watch(mods, T0)["wakeAgent"] is True
    nb = home / "workspace" / "notebook.md"
    os.utime(nb, (T0.timestamp() + 60, T0.timestamp() + 60))
    t1 = T0 + timedelta(minutes=20)
    monkeypatch.setattr(wake, "list_executions", lambda job_id: [
        _exec(t1, "running"), _exec(T0 - timedelta(seconds=5), "completed")])
    assert watch.main(now=t1) == {"wakeAgent": False}


def test_failed_woken_run_carries_its_events_over(mods, http, monkeypatch):
    wake, watch, _ = mods
    run_watch(mods, T0)
    http.assets = [asset(ROW_1, "x", "Approved", edited="2026-11-17T09:10:00.000Z")]
    t1 = T0 + timedelta(minutes=20)
    p1 = watch.main(now=t1)
    assert events_of(p1, "asset_approved")
    t2 = T0 + timedelta(minutes=40)
    monkeypatch.setattr(wake, "list_executions", lambda job_id: [
        _exec(t2, "running"), _exec(t1 - timedelta(seconds=5), "failed")])
    p2 = watch.main(now=t2)
    [carried] = events_of(p2, "asset_approved")
    assert carried["carried_over"] is True
    assert p2["previous_run_failed"] is True


# ── explore ──────────────────────────────────────────────────────────────

def test_explore_always_wakes_and_reports_notebook_state(mods, home, monkeypatch):
    wake, _, explore = mods
    p = explore.main(now=T0)
    assert p["wakeAgent"] is True and p["job"] == "explore"
    assert p["notebook_not_updated"] is False and p["previous_run_failed"] is False
    nb = home / "workspace" / "notebook.md"
    old = (T0 - timedelta(hours=1)).timestamp()
    os.utime(nb, (old, old))
    t1 = T0 + timedelta(hours=3)
    monkeypatch.setattr(wake, "list_executions", lambda job_id: [
        _exec(t1, "running"), _exec(T0 - timedelta(seconds=5), "completed")])
    p = explore.main(now=t1)
    assert p["notebook_not_updated"] is True
    assert p["notebook_bytes"] == len("# notebook\n")


def test_explore_makes_no_network_calls(mods, monkeypatch):
    wake, _, explore = mods
    monkeypatch.setattr(wake, "http_json", lambda *a, **k: pytest.fail("explore must not call out"))
    explore.main(now=T0)


def test_state_lives_under_cron_not_the_agent_writable_workspace(mods, http, home):
    run_watch(mods, T0)
    assert (home / "cron" / "educator-state" / "release-watch.json").is_file()
    assert not list((home / "workspace").glob("*.json"))


# ── approval measurement (contract: product-skills learn-from-approval) ──

def _naive_lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@pytest.mark.parametrize("a,b", [("", ""), ("abc", ""), ("", "xyz"), ("kitten", "sitting"),
                                 ("flaw", "lawn"), ("same text", "same text"),
                                 ("abcdefghij" * 7, "abXdefghij" * 6 + "zz"),
                                 ("the quick brown fox", "a quick brown dog jumps")])
def test_edit_distance_is_exact_levenshtein(mods, a, b):
    wake, _, _ = mods
    dist, exact = wake.edit_distance(a, b)
    assert exact is True and dist == _naive_lev(a, b)


def test_edit_distance_is_fast_for_a_long_lightly_edited_document(mods):
    import random
    import time
    wake, _, _ = mods
    rng = random.Random(7)
    a = "".join(rng.choice("abcdefghij klmnop") for _ in range(12000))
    b = list(a)
    for _ in range(150):
        b[rng.randrange(len(b))] = "#"
    b = "".join(b)
    t = time.monotonic()
    dist, exact = wake.edit_distance(a, b)
    assert exact and dist <= 150 and time.monotonic() - t < 10


def test_edit_distance_falls_back_to_an_estimate_for_a_full_rewrite(mods, monkeypatch):
    wake, _, _ = mods
    monkeypatch.setattr(wake, "EDIT_DISTANCE_CELL_BUDGET", 10_000)
    dist, exact = wake.edit_distance("a" * 3000, "b" * 2500)
    assert exact is False and dist >= 2500


def test_normalise_strips_the_snapshot_header_and_collapses_whitespace(mods):
    wake, _, _ = mods
    text = "<!-- snapshot row=x asset_type=Slack post taken=2026-11-17T09:00:00Z -->\n# Title\n\n  body   text\n"
    assert wake.normalise_markdown(text) == "# Title body text"


def _approve_flow(mods, http, home, monkeypatch, *, draft=None, redraft=None, approved_md="# T approved body"):
    wake, watch, _ = mods
    drafts = home / "workspace" / "drafts"
    drafts.mkdir(exist_ok=True)
    if draft is not None:
        (drafts / f"{ROW_1}.md").write_text(draft)
    if redraft is not None:
        (drafts / f"{ROW_1}.redraft.md").write_text(redraft)
    real = http.__call__

    def with_markdown(method, url, headers, body=None, timeout=30):
        if url.endswith(f"/v1/pages/{ROW_1}/markdown"):
            http.calls.append((method, url, body))
            return 200, {"object": "page_markdown", "markdown": approved_md, "truncated": False}
        return real(method, url, headers, body, timeout)

    monkeypatch.setattr(wake, "http_json", with_markdown)
    run_watch(mods, T0)
    http.assets = [asset(ROW_1, "Slack post: audio", "Approved", edited="2026-11-17T09:10:00.000Z")]
    return events_of(run_watch(mods, T0 + timedelta(minutes=20)), "asset_approved")[0]


def test_approved_row_is_measured_against_its_snapshot(mods, http, home, monkeypatch):
    ev = _approve_flow(mods, http, home, monkeypatch,
                       draft="<!-- snapshot row=r asset_type=Slack post taken=t -->\n# T draft body\n")
    assert ev["snapshot_missing"] is False and ev["baseline"] == "draft"
    assert ev["draft_chars"] == len("# T draft body")
    assert ev["approved_chars"] == len("# T approved body")
    assert ev["chars_changed"] == _naive_lev("# T draft body", "# T approved body")
    assert ev["chars_changed_exact"] is True
    saved = (home / "workspace" / "drafts" / f"{ROW_1}.approved.md").read_text()
    assert saved.startswith(f"<!-- approved row={ROW_1} asset_type=Slack post taken=")
    assert saved.rstrip().endswith("# T approved body")


def test_redraft_splits_assumption_chars_from_writing_edits(mods, http, home, monkeypatch):
    ev = _approve_flow(mods, http, home, monkeypatch, draft="# T old", redraft="# T approved body")
    total = _naive_lev("# T old", "# T approved body")
    assert ev["baseline"] == "draft" and ev["chars_changed"] == total
    assert ev["assumption_chars"] == total  # the reviewer changed nothing after the re-draft


def test_redraft_alone_is_the_baseline(mods, http, home, monkeypatch):
    ev = _approve_flow(mods, http, home, monkeypatch, redraft="# T approved bodY")
    assert ev["baseline"] == "redraft" and ev["chars_changed"] == 1 and "assumption_chars" not in ev


def test_missing_snapshot_still_reports_the_approved_size(mods, http, home, monkeypatch):
    ev = _approve_flow(mods, http, home, monkeypatch)
    assert ev["snapshot_missing"] is True
    assert ev["approved_chars"] == len("# T approved body")
    assert "draft_chars" not in ev and "chars_changed" not in ev


def test_markdown_fetch_failure_is_recorded_on_the_event_not_fatal(mods, http, home):
    run_watch(mods, T0)
    http.assets = [asset(ROW_1, "x", "Approved", edited="2026-11-17T09:10:00.000Z")]
    p = run_watch(mods, T0 + timedelta(minutes=20))   # FakeHTTP has no markdown route -> AssertionError
    [ev] = events_of(p, "asset_approved")
    assert "measure_error" in ev and "chars_changed" not in ev
