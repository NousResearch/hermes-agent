"""The rest of deploy/educator: Slack manifest, cron job definitions and
their applier, SOUL.md, .env.example."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest
import yaml

DEPLOY = Path(__file__).resolve().parents[2] / "deploy" / "educator"
FORBIDDEN = {"terminal", "code_execution", "jira_write", "messaging", "cronjob", "delegation"}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"educator_{name}", DEPLOY / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _values(**over) -> dict:
    return {**yaml.safe_load((DEPLOY / "values.example.yaml").read_text()), **over}


def _rendered_config(values: dict) -> dict:
    render = _load("render_config")
    return yaml.safe_load(render.render((DEPLOY / "config.yaml.template").read_text(), values))


# ── Slack manifest ───────────────────────────────────────────────────────

def test_manifest_covers_the_adapter_and_the_fork_tools():
    from hermes_cli.slack_cli import _build_full_manifest
    render = _load("render_config")
    manifest = yaml.safe_load(render.render((DEPLOY / "slack-app-manifest.yaml").read_text(), _values()))
    base = _build_full_manifest("x", "y", messaging_experience="agent")
    scopes = set(manifest["oauth_config"]["scopes"]["bot"])
    assert scopes >= set(base["oauth_config"]["scopes"]["bot"]) - {"commands"}
    assert {"reactions:write", "reactions:read", "chat:write", "users:read.email"} <= scopes
    assert "commands" not in scopes and "slash_commands" not in manifest["features"]
    events = set(manifest["settings"]["event_subscriptions"]["bot_events"])
    assert events >= set(base["settings"]["event_subscriptions"]["bot_events"])
    assert manifest["settings"]["socket_mode_enabled"] is True
    assert manifest["display_information"]["name"] == _values()["PERSONA_NAME"]


# ── cron jobs ────────────────────────────────────────────────────────────

def test_job_definitions_match_the_design():
    apply = _load("apply_cron_jobs")
    values = _values()
    jobs = {j["name"]: j for j in apply.load_jobs(values)}
    assert set(jobs) == {"release-watch", "explore"}
    rw, ex = jobs["release-watch"], jobs["explore"]
    assert rw["schedule"] == "every 20m" and rw["discretionary"] is False
    assert ex["discretionary"] is True
    assert len(ex["schedule"].split()[1].split(",")) == 3  # 3x per working day
    offered = set(_rendered_config(values)["platform_toolsets"]["slack"])
    home = values["EDUCATOR_HOME"]
    for job in jobs.values():
        assert (DEPLOY / "scripts" / job["script"]).is_file()
        assert set(job["enabled_toolsets"]) <= offered
        assert not set(job["enabled_toolsets"]) & FORBIDDEN
        assert set(job["required_toolsets"]) <= set(job["enabled_toolsets"])
        assert job["workdir"] == f"{home}/workspace"
        assert job["deliver"] == f"slack:{values['HOME_CHANNEL_ID']}"
        assert "{{" not in job["prompt"] and f"{home}/workspace" in job["prompt"]
        assert "notebook-keeping" in job["prompt"] and "[SILENT]" in job["prompt"]
    for skill in ("release-comms", "learn-from-approval", "zendesk-draft", "hermes-runtime.md"):
        assert skill in rw["prompt"]
    assert "education-explore" in ex["prompt"]
    assert values["RELEASE_UPDATES_CHANNEL_ID"] in rw["prompt"]


def test_jobs_pass_the_schedulers_capability_preflight():
    from cron.capability_preflight import evaluate_job_capabilities
    from gateway.tool_access import policy_from_extra
    apply = _load("apply_cron_jobs")
    values = _values()
    cfg = _rendered_config(values)
    # worst case: owned by a product-team member, not an admin
    grant = policy_from_extra(cfg["slack"]).grant_for(values["PRODUCT_TEAM_SLACK_USER_ID"])
    for job in apply.load_jobs(values):
        report = evaluate_job_capabilities(job, cfg, grant, include_enabled=True,
                                           check_availability=False)
        assert report.problems == [], (job["name"], report.problems)
        assert not set(report.effective or []) & FORBIDDEN


@pytest.fixture
def educator_home(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "educator"
    (home / "scripts").mkdir(parents=True)
    (home / "workspace").mkdir()
    (home / "cron" / "output").mkdir(parents=True)
    for f in (DEPLOY / "scripts").glob("*.py"):
        shutil.copy(f, home / "scripts" / f.name)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", home / "cron" / "output")
    values = _values(EDUCATOR_HOME=str(home))
    render = _load("render_config")
    (home / "config.yaml").write_text(render.render((DEPLOY / "config.yaml.template").read_text(), values))
    vfile = tmp_path / "values.yaml"
    vfile.write_text(yaml.safe_dump(values))
    return home, vfile


def test_apply_creates_then_updates_the_jobs(educator_home, capsys):
    home, vfile = educator_home
    apply = _load("apply_cron_jobs")
    assert apply.main(["apply", str(vfile)]) == 0
    from cron.jobs import list_jobs
    jobs = {j["name"]: j for j in list_jobs(include_disabled=True)}
    assert set(jobs) == {"release-watch", "explore"}
    ex = jobs["explore"]
    assert ex["discretionary"] is True and ex["script"] == "educator_explore.py"
    assert ex["required_toolsets"] == ["notion", "file", "skills", "web"]
    assert jobs["release-watch"]["discretionary"] is False
    assert jobs["release-watch"]["workdir"] == str((home / "workspace").resolve())
    out = capsys.readouterr().out
    assert f"own claim cron:{jobs['explore']['id']}" in out
    # second apply updates in place
    assert apply.main(["apply", str(vfile)]) == 0
    again = {j["name"]: j["id"] for j in list_jobs(include_disabled=True)}
    assert again == {n: j["id"] for n, j in jobs.items()}
    # the installed script finds its own job (it is not told its id)
    import sys
    sys.path.insert(0, str(home / "scripts"))
    try:
        import educator_wake
        assert educator_wake.find_job_id("educator_explore.py") == jobs["explore"]["id"]
    finally:
        sys.path.remove(str(home / "scripts"))
        sys.modules.pop("educator_wake", None)


def test_apply_refuses_a_stale_script_copy_or_the_wrong_home(educator_home, capsys):
    home, vfile = educator_home
    apply = _load("apply_cron_jobs")
    (home / "scripts" / "educator_explore.py").write_text("print('{}')\n")
    assert apply.main(["apply", str(vfile)]) == 1
    assert "differs" in capsys.readouterr().err
    shutil.copy(DEPLOY / "scripts" / "educator_explore.py", home / "scripts" / "educator_explore.py")
    vfile.write_text(yaml.safe_dump(_values(EDUCATOR_HOME="/somewhere/else")))
    assert apply.main(["apply", str(vfile)]) == 1
    assert "is not EDUCATOR_HOME" in capsys.readouterr().err
    from cron.jobs import list_jobs
    assert list_jobs(include_disabled=True) == []


# ── SOUL.md, .env.example ────────────────────────────────────────────────

def test_soul_renders_and_names_only_real_tools():
    import re

    import model_tools  # noqa: F401
    from tools.registry import registry
    render = _load("render_config")
    soul = render.render((DEPLOY / "SOUL.md").read_text(), _values(), is_yaml=False)
    assert _values()["PERSONA_NAME"] in soul and "{{" not in soul
    for phrase in ("explore and suggest, never deploy", "you have no shell", "every skill change is a pull request"):
        assert phrase in " ".join(soul.lower().split())
    assert f"{_values()['EDUCATOR_HOME']}/workspace/notebook.md" in soul
    tools = re.findall(r"`([a-z]+(?:_[a-z]+)+)`", soul)  # every `snake_case` name is a tool
    assert {"notion_api", "jira_api", "zendesk_create_draft", "slack_post_thread"} <= set(tools)
    for tool in tools:
        assert tool in registry._tools, tool


def test_env_example_lists_every_credential_the_profile_reads():
    names = {line.split("=", 1)[0].strip() for line in (DEPLOY / ".env.example").read_text().splitlines()
             if "=" in line and not line.lstrip().startswith("#")}
    assert {"SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_API_KEY", "NOTION_API_KEY", "NOTION_KEYRING",
            "JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "ZENDESK_SUBDOMAIN",
            "ZENDESK_OAUTH_CLIENT_ID", "ZENDESK_OAUTH_CLIENT_SECRET", "GITHUB_PR_TOKEN"} <= names
    for line in (DEPLOY / ".env.example").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            value = line.split("=", 1)[1].split("#", 1)[0].strip()
            assert value in ("", "0"), line  # names only, no secrets
    assert "ZENDESK_API_TOKEN" not in names and "ZENDESK_EMAIL" not in names
