"""Cron capability preflight (cron/capability_preflight.py) and its three
surfaces: cronjob create/update, the scheduler's pre-dispatch preflight, and
the local-only failure path (agent-declared [FAILED] + owner DM).

Regression anchor: the Ready-for-Staging auto-merger (2026-09-16) was created
with enabled_toolsets=[jira, messaging, file] — messaging is stripped from
every cron run and the job lacked jira_write — and ran "ok" for a week while
never completing its work."""

import json

import pytest

import cron.capability_preflight as cp
import cron.scheduler as sched
import tools.cronjob_tools as cj

_KNOWN = {"jira", "jira_write", "file", "slack_post", "messaging", "clarify",
          "cronjob", "web", "github_rw", "terminal", "code_execution"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Deterministic toolset universe: every known toolset exposes one tool,
    github_rw is an enabled MCP server, nothing is approval-gated."""
    monkeypatch.setattr(cp, "_mcp_server_names", lambda cfg: frozenset({"github_rw"}))
    monkeypatch.setattr("toolsets.validate_toolset", lambda name: name in _KNOWN)
    monkeypatch.setattr(cp, "_toolset_tool_names", lambda name: [f"{name}_tool"])
    monkeypatch.setattr("tools.approval.tool_requires_approval", lambda name: False)
    monkeypatch.setattr(
        "hermes_cli.tools_config.enabled_mcp_server_names", lambda cfg: {"github_rw"}
    )


def _eval(job, grant=None, *, include_enabled=True, check_availability=True):
    return cp.evaluate_job_capabilities(
        job, {}, grant, include_enabled=include_enabled,
        check_availability=check_availability,
    )


def test_stripped_messaging_is_a_problem_with_slack_post_hint():
    report = _eval({"enabled_toolsets": ["jira", "messaging", "file"]})
    assert report.missing == ["messaging"]
    assert "slack_post" in report.problems[0]


def test_toolset_outside_owner_role_is_a_problem():
    grant = frozenset({"jira", "file", "slack_post"})
    report = _eval({"enabled_toolsets": ["jira", "file"],
                    "required_toolsets": ["jira_write"]}, grant)
    assert report.missing == ["jira_write"]
    assert "role" in report.problems[0]


def test_required_toolset_missing_from_explicit_list():
    report = _eval({"enabled_toolsets": ["jira", "file"],
                    "required_toolsets": ["slack_post"]})
    assert report.missing == ["slack_post"]
    assert "enabled_toolsets" in report.problems[0]


def test_unknown_toolset_is_a_problem():
    report = _eval({"enabled_toolsets": ["jiraa"]})
    assert report.missing == ["jiraa"]


def test_mcp_prefixed_spelling_matches_bare_server_name():
    report = _eval({"enabled_toolsets": ["github_rw", "file"],
                    "required_toolsets": ["mcp-github_rw"]},
                   frozenset({"github_rw", "mcp-github_rw", "file"}))
    assert report.ok, report.problems
    assert "github_rw" in report.effective


def test_unavailable_toolset_only_checked_when_asked(monkeypatch):
    monkeypatch.setattr(cp, "_toolset_tool_names", lambda name: [] if name == "web" else ["x"])
    job = {"enabled_toolsets": ["web"]}
    assert _eval(job).missing == ["web"]
    assert _eval(job, check_availability=False).ok


def test_correctly_configured_job_passes():
    grant = frozenset({"jira", "jira_write", "file", "slack_post", "github_rw"})
    job = {"enabled_toolsets": ["jira", "jira_write", "github_rw", "slack_post", "file"]}
    report = _eval({**job, "required_toolsets": job["enabled_toolsets"]}, grant)
    assert report.ok, report.problems


def test_gated_unacked_tools_reported(monkeypatch):
    monkeypatch.setattr("tools.approval.tool_requires_approval",
                        lambda name: name == "github_rw_tool")
    report = _eval({"enabled_toolsets": ["github_rw"]})
    assert report.gated_unacked == ["github_rw_tool"]
    acked = _eval({"enabled_toolsets": ["github_rw"],
                   "unattended_approved_tools": ["github_rw_tool"]})
    assert acked.gated_unacked == []


def test_automation_manifest_requirements(tmp_path):
    (tmp_path / "automation.yaml").write_text(
        "name: x\nrequires_toolsets: [jira_write, slack_post]\n", encoding="utf-8"
    )
    assert cp.automation_required_toolsets(str(tmp_path)) == ["jira_write", "slack_post"]
    report = _eval({"enabled_toolsets": ["jira"], "workdir": str(tmp_path)},
                   include_enabled=False)
    assert set(report.missing) == {"jira_write", "slack_post"}


# --- scheduler runtime preflight ------------------------------------------------

def test_runtime_check_ignores_jobs_that_name_nothing():
    assert sched._preflight_check_capabilities({"id": "j1"}, {}) is None


def test_runtime_check_flags_explicitly_listed_stripped_toolset():
    # Naming a toolset the job can never receive is a misconfiguration: the
    # auto-merger listed messaging and never posted a single blocker notice.
    job = {"id": "j1", "enabled_toolsets": ["jira", "messaging"]}
    reason = sched._preflight_check_capabilities(job, {})
    assert reason and "messaging" in reason


def test_runtime_check_blocks_when_declared_requirement_lost(monkeypatch):
    monkeypatch.setattr("cron.rbac_ceiling.cron_owner_grant",
                        lambda job: frozenset({"jira", "file"}))
    job = {"id": "j1", "enabled_toolsets": ["jira", "jira_write", "file"],
           "required_toolsets": ["jira_write"]}
    reason = sched._preflight_check_capabilities(job, {})
    assert reason and "jira_write" in reason


def test_runtime_check_passes_when_requirements_resolve(monkeypatch):
    monkeypatch.setattr("cron.rbac_ceiling.cron_owner_grant",
                        lambda job: frozenset({"jira", "jira_write", "file"}))
    job = {"id": "j1", "enabled_toolsets": ["jira", "jira_write", "file"],
           "required_toolsets": ["jira_write"]}
    assert sched._preflight_check_capabilities(job, {}) is None


# --- cronjob tool ---------------------------------------------------------------

def test_create_rejects_stripped_toolset(monkeypatch):
    monkeypatch.setattr(cj, "_creator_grant", lambda: None)
    out = json.loads(cj.cronjob(action="create", schedule="every 5m", prompt="p",
                                enabled_toolsets=["jira", "messaging", "file"]))
    assert out.get("success") is False
    assert "messaging" in out["error"] and "slack_post" in out["error"]


def test_create_rejects_required_toolset_outside_creator_role(monkeypatch):
    monkeypatch.setattr(cj, "_creator_grant", lambda: frozenset({"jira", "file"}))
    monkeypatch.setattr(cj, "_rbac_creation_error", lambda **kw: None)
    out = json.loads(cj.cronjob(action="create", schedule="every 5m", prompt="p",
                                enabled_toolsets=["jira", "file"],
                                required_toolsets=["jira_write"]))
    assert out.get("success") is False and "jira_write" in out["error"]


def test_create_stores_requirements_and_echoes_capabilities(monkeypatch):
    monkeypatch.setattr(cj, "_creator_grant", lambda: None)
    out = json.loads(cj.cronjob(action="create", schedule="every 5m", prompt="p",
                                enabled_toolsets=["jira", "slack_post"],
                                required_toolsets=["slack_post"]))
    assert out["success"] is True, out
    assert sorted(out["capabilities"]["required_toolsets"]) == ["jira", "slack_post"]
    from cron.jobs import get_job
    assert get_job(out["job_id"])["required_toolsets"] == ["slack_post"]


def test_update_unrelated_field_warns_instead_of_rejecting(monkeypatch):
    monkeypatch.setattr(cj, "_creator_grant", lambda: None)
    from cron.jobs import create_job
    job = create_job(prompt="p", schedule="every 5m",
                     enabled_toolsets=["jira", "messaging"])
    out = json.loads(cj.cronjob(action="update", job_id=job["id"], name="renamed"))
    assert out["success"] is True, out
    assert any("messaging" in w for w in out["capabilities"]["warnings"])
    rejected = json.loads(cj.cronjob(action="update", job_id=job["id"],
                                     enabled_toolsets=["jira", "messaging", "file"]))
    assert rejected.get("success") is False


# --- agent-declared failure + owner DM -----------------------------------------

def test_agent_reported_failure_marker():
    assert sched._agent_reported_failure("[FAILED] slack_post missing") == "slack_post missing"
    assert sched._agent_reported_failure("  [failed]\nno tool") == "no tool"
    assert sched._agent_reported_failure("All good; nothing [FAILED] here") is None
    assert sched._agent_reported_failure("") is None


def test_failure_summary_passes_agent_reason_through():
    msg = sched._summarize_cron_failure_for_delivery(
        {"name": "merger"}, "Agent reported failure: Jira returned 401 and timed out"
    )
    assert "reported a failure" in msg and "401" in msg
    assert "provider" not in msg


class _Recorder:
    def __init__(self):
        self.dms = []

    def __call__(self, platform, user_id, message):
        self.dms.append((platform, user_id, message))
        return True


@pytest.fixture
def owned_local_job(monkeypatch):
    from agent import automation_ownership as ao
    rec = _Recorder()
    monkeypatch.setattr(ao, "_send_dm", rec)
    monkeypatch.setattr(ao, "get_record", lambda key: {
        "owner": {"platform": "slack", "user_id": "U_OWNER"}} if key == "cron:j1" else None)
    monkeypatch.setattr(sched, "_resolve_delivery_targets", lambda job: [])
    return {"id": "j1", "name": "merger", "deliver": "local"}, rec


def test_local_job_failure_dms_owner_once_per_incident(monkeypatch, owned_local_job):
    job, rec = owned_local_job
    state = {}
    monkeypatch.setattr("cron.incidents.get_incident",
                        lambda iid: {"state": state.get(iid, "detected")})
    monkeypatch.setattr(sched, "_mark_incident_alerted",
                        lambda iid: state.__setitem__(iid, "alerted"))
    sched._alert_owner_of_undelivered_failure(job, "⚠️ boom", "inc-1")
    sched._alert_owner_of_undelivered_failure(job, "⚠️ boom", "inc-1")
    assert len(rec.dms) == 1 and rec.dms[0][1] == "U_OWNER"
    sched._alert_owner_of_undelivered_failure(job, "⚠️ other", "inc-2")
    assert len(rec.dms) == 2


def test_owner_dm_skipped_when_job_has_delivery_target(monkeypatch, owned_local_job):
    job, rec = owned_local_job
    monkeypatch.setattr(sched, "_resolve_delivery_targets", lambda job: [{"platform": "slack"}])
    sched._alert_owner_of_undelivered_failure(job, "⚠️ boom", None)
    assert rec.dms == []


def test_owner_dm_skipped_without_owner(monkeypatch, owned_local_job):
    job, rec = owned_local_job
    sched._alert_owner_of_undelivered_failure({**job, "id": "other"}, "⚠️ boom", None)
    assert rec.dms == []


# --- no shell in cron agents ----------------------------------------------------

def test_cron_strips_shell_toolsets_by_default():
    disabled = sched._resolve_cron_disabled_toolsets({})
    assert "terminal" in disabled and "code_execution" in disabled
    opted_in = sched._resolve_cron_disabled_toolsets({"cron": {"allow_agent_shell": True}})
    assert "terminal" not in opted_in and "code_execution" not in opted_in


def test_listed_terminal_is_flagged_with_script_hint():
    report = _eval({"enabled_toolsets": ["terminal", "file"]})
    assert report.missing == ["terminal"]
    assert "post_script" in report.problems[0]


# --- script import preflight ----------------------------------------------------

@pytest.fixture
def scripts_dir():
    from hermes_constants import get_hermes_home
    d = get_hermes_home() / "scripts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_script_import_problems_names_missing_module(scripts_dir):
    (scripts_dir / "collector.py").write_text(
        "import json\nfrom definitely_missing_mod_xyz import thing\n"
        "try:\n    import optional_missing_mod_xyz\nexcept ImportError:\n    pass\n"
        "from sibling_helper import x\n",
        encoding="utf-8",
    )
    (scripts_dir / "sibling_helper.py").write_text("x = 1\n", encoding="utf-8")
    msg = cp.script_import_problems("collector.py")
    assert msg and "definitely_missing_mod_xyz" in msg
    assert "optional_missing_mod_xyz" not in msg   # try-guarded = optional
    assert "sibling_helper" not in msg             # local module
    assert "uv pip install --python" in msg


def test_script_import_problems_clean_and_non_python(scripts_dir):
    (scripts_dir / "ok.py").write_text("import json, os\n", encoding="utf-8")
    (scripts_dir / "w.sh").write_text("import nothing\n", encoding="utf-8")
    assert cp.script_import_problems("ok.py") is None
    assert cp.script_import_problems("w.sh") is None
    assert cp.script_import_problems("absent.py") is None
    assert cp.script_import_problems(None) is None


def test_run_job_blocks_before_script_runs_on_missing_import(scripts_dir):
    marker = scripts_dir / "ran.txt"
    (scripts_dir / "bad.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n"
        "import definitely_missing_mod_xyz\n",
        encoding="utf-8",
    )
    ok, doc, final, err = sched.run_job(
        {"id": "imp1", "name": "imp", "prompt": "p", "script": "bad.py",
         "schedule": {"kind": "interval", "minutes": 5}}
    )
    assert ok is False and final == ""
    assert sched.BLOCKED_CONFIG_MARKER in err and "definitely_missing_mod_xyz" in err
    assert not marker.exists()   # the script never executed


def test_create_rejects_script_with_missing_import(monkeypatch, scripts_dir):
    monkeypatch.setattr(cj, "_creator_grant", lambda: None)
    (scripts_dir / "bad2.py").write_text("import definitely_missing_mod_xyz\n", encoding="utf-8")
    out = json.loads(cj.cronjob(action="create", schedule="every 5m", prompt="p",
                                script="bad2.py"))
    assert out.get("success") is False and "definitely_missing_mod_xyz" in out["error"]


# --- post_script ----------------------------------------------------------------

def test_post_script_receives_response_and_workdir(scripts_dir, tmp_path):
    (scripts_dir / "apply.py").write_text(
        "import os, pathlib\n"
        "resp = pathlib.Path(os.environ['HERMES_CRON_RESPONSE_FILE']).read_text()\n"
        "print(f\"applied {resp} for {os.environ['HERMES_CRON_JOB_ID']} in {pathlib.Path.cwd().name}\")\n",
        encoding="utf-8",
    )
    ok, out = sched._run_job_post_script(
        {"id": "pj", "post_script": "apply.py", "workdir": str(tmp_path)}, "PLAN-1"
    )
    assert ok and out == f"applied PLAN-1 for pj in {tmp_path.name}"


def test_post_script_failure_reports_error(scripts_dir):
    (scripts_dir / "boom.py").write_text("import sys\nsys.exit('bad plan')\n", encoding="utf-8")
    ok, out = sched._run_job_post_script({"id": "pj", "post_script": "boom.py"}, "x")
    assert not ok and "bad plan" in out


def test_merge_post_script_output():
    assert sched._merge_post_script_output("report", "") == "report"
    assert sched._merge_post_script_output("report", "wrote 2 rows") == "report\n\nwrote 2 rows"
    assert sched._merge_post_script_output("[SILENT]", "wrote 2 rows") == "wrote 2 rows"
    assert sched._merge_post_script_output("[SILENT]", "") == "[SILENT]"


def test_create_rejects_post_script_on_no_agent_job(monkeypatch, scripts_dir):
    monkeypatch.setattr(cj, "_creator_grant", lambda: None)
    (scripts_dir / "s.py").write_text("print(1)\n", encoding="utf-8")
    out = json.loads(cj.cronjob(action="create", schedule="every 5m", script="s.py",
                                no_agent=True, post_script="s.py"))
    assert out.get("success") is False and "no_agent" in out["error"]


def test_create_stores_post_script(monkeypatch, scripts_dir):
    monkeypatch.setattr(cj, "_creator_grant", lambda: None)
    (scripts_dir / "apply2.py").write_text("print(1)\n", encoding="utf-8")
    out = json.loads(cj.cronjob(action="create", schedule="every 5m", prompt="p",
                                enabled_toolsets=["file"], post_script="apply2.py"))
    assert out["success"] is True, out
    assert out["job"]["post_script"] == "apply2.py"


def test_post_script_needs_shell_granted_creator(monkeypatch):
    from agent.automation_ownership import Identity

    class _Policy:
        enabled = True
        def allowed_toolsets(self, u, requested, c=None):
            return frozenset(requested)
        def can_use_tool(self, u, toolset, c=None):
            return toolset not in {"terminal", "code_execution"}

    monkeypatch.setattr("gateway.session_context.get_session_env", lambda *a, **k: "")
    monkeypatch.setattr("agent.automation_ownership.current_identity",
                        lambda: Identity("slack", "U1", "Bob"))
    monkeypatch.setattr("gateway.tool_access.policy_for_platform", lambda name: _Policy())
    err = cj._rbac_creation_error(enabled_toolsets=["file"], has_script=True, is_no_agent=False)
    assert err and "terminal" in err


# --- ownership transfer ---------------------------------------------------------

def test_transfer_capability_loss_reported(monkeypatch):
    import tools.ownership_tool as ot
    from agent.automation_ownership import Identity

    monkeypatch.setattr("cron.jobs.get_job", lambda jid: {
        "id": jid, "enabled_toolsets": ["file", "google_sheets"], "origin": {}})
    monkeypatch.setattr("cron.capability_preflight.grant_for_identity",
                        lambda ident, chat: frozenset({"file"}))
    monkeypatch.setattr("toolsets.validate_toolset", lambda name: True)
    loss = ot._cron_capability_loss("cron:abc", Identity("slack", "U2", "Pazit"))
    assert "google_sheets" in loss
    assert ot._cron_capability_loss("skill:x", Identity("slack", "U2", "Pazit")) == ""
