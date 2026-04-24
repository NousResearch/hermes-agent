import json
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills" / "productivity" / "demo-skill").mkdir(parents=True)
    (home / "cron").mkdir(parents=True)
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", home / "cron" / "output")
    return home


def write_skill(home: Path, name="demo-skill", body="") -> Path:
    skill_dir = home / "skills" / "productivity" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_skill_doctor_reports_missing_env_var(hermes_home, monkeypatch):
    write_skill(
        hermes_home,
        body="""---
name: demo-skill
prerequisites:
  env_vars: [MISSING_DEMO_TOKEN]
---
# Demo
""",
    )
    monkeypatch.delenv("MISSING_DEMO_TOKEN", raising=False)

    from hermes_cli.reliability_doctor import doctor_skill

    result = doctor_skill("demo-skill")

    assert result["status"] == "fail"
    assert result["errors"] == [
        {"type": "env_var", "name": "MISSING_DEMO_TOKEN", "message": "missing"}
    ]


def test_skill_smoke_runs_builtin_and_command_probes(hermes_home, monkeypatch):
    script = hermes_home / "scripts" / "smoke_ok.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    write_skill(
        hermes_home,
        body=f"""---
name: demo-skill
prerequisites:
  smoke:
    safe: true
    timeout_seconds: 5
    probes:
      - type: file-exists
        path: {script}
      - type: command
        command: python {script}
---
# Demo
""",
    )

    from hermes_cli.reliability_doctor import doctor_skill

    result = doctor_skill("demo-skill", smoke=True)

    assert result["status"] == "ok"
    assert [p["status"] for p in result["smoke"]] == ["ok", "ok"]


def test_cron_doctor_reports_missing_script(hermes_home):
    from cron.jobs import create_job
    from hermes_cli.reliability_doctor import doctor_cron

    job = create_job(
        prompt="Run missing script",
        schedule="every 1h",
        script="missing_script.py",
    )

    result = doctor_cron(job["id"])

    assert result["status"] == "fail"
    assert {e["type"] for e in result["errors"]} == {"script"}


def test_cron_doctor_reports_missing_absolute_path_inside_prompt_command(hermes_home):
    from cron.jobs import create_job
    from hermes_cli.reliability_doctor import doctor_cron

    job = create_job(
        prompt='Run this.\n\nCommand:\n/bin/bash -lc "/tmp/definitely_missing_wrapper.sh run"',
        schedule="every 1h",
    )

    result = doctor_cron(job["id"])

    assert result["status"] == "fail"
    assert {e["type"] for e in result["errors"]} == {"prompt_path"}


def test_cron_smoke_runs_job_level_smoke_declaration(hermes_home):
    from cron.jobs import create_job, update_job
    from hermes_cli.reliability_doctor import doctor_cron

    script = hermes_home / "scripts" / "cron_auth_ok.py"
    script.write_text("print('auth ok')\n", encoding="utf-8")
    job = create_job(prompt="Check auth", schedule="every 1h")
    update_job(job["id"], {
        "smoke": {
            "safe": True,
            "timeout_seconds": 5,
            "probes": [{"type": "command", "command": f"python {script}"}],
        }
    })

    result = doctor_cron(job["id"], smoke=True)

    assert result["status"] == "ok"
    assert result["smoke"][0]["status"] == "ok"


def test_cron_doctor_includes_attached_skill_failures(hermes_home, monkeypatch):
    write_skill(
        hermes_home,
        body="""---
name: demo-skill
prerequisites:
  env_vars: [MISSING_DEMO_TOKEN]
---
# Demo
""",
    )
    monkeypatch.delenv("MISSING_DEMO_TOKEN", raising=False)

    from cron.jobs import create_job
    from hermes_cli.reliability_doctor import doctor_cron

    job = create_job(
        prompt="Use skill",
        schedule="every 1h",
        skills=["demo-skill"],
    )

    result = doctor_cron(job["id"])

    assert result["status"] == "fail"
    assert result["skill_results"][0]["skill"] == "demo-skill"
    assert result["skill_results"][0]["status"] == "fail"


def test_static_dependency_primitives_cover_env_command_file_directory_import_and_mcp(hermes_home, monkeypatch):
    needed_file = hermes_home / "scripts" / "exists.txt"
    needed_file.write_text("ok", encoding="utf-8")
    needed_dir = hermes_home / "present-dir"
    needed_dir.mkdir()
    write_skill(
        hermes_home,
        body=f"""---
name: demo-skill
prerequisites:
  env_vars: [DEMO_PRESENT_TOKEN]
  commands: [python]
  files: [{needed_file}]
  directories: [{needed_dir}]
  python_imports: [json]
  mcp_servers: [jbrain]
---
# Demo
""",
    )
    monkeypatch.setenv("DEMO_PRESENT_TOKEN", "set")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"mcp_servers": {"jbrain": {"command": "python"}}},
    )

    from hermes_cli.reliability_doctor import doctor_skill

    result = doctor_skill("demo-skill")

    assert result["status"] == "ok"
    assert result["errors"] == []


def test_static_mcp_configured_primitive_reports_missing_server(hermes_home, monkeypatch):
    write_skill(
        hermes_home,
        body="""---
name: demo-skill
prerequisites:
  mcp_servers: [missing-mcp]
---
# Demo
""",
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"mcp_servers": {}})

    from hermes_cli.reliability_doctor import doctor_skill

    result = doctor_skill("demo-skill")

    assert result["status"] == "fail"
    assert result["errors"] == [
        {"type": "mcp_server", "name": "missing-mcp", "message": "not_configured"}
    ]


def test_smoke_mcp_configured_primitive(hermes_home, monkeypatch):
    write_skill(
        hermes_home,
        body="""---
name: demo-skill
prerequisites:
  smoke:
    safe: true
    probes:
      - type: mcp-configured
        name: jbrain
---
# Demo
""",
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"mcp_servers": {"jbrain": {"command": "python"}}},
    )

    from hermes_cli.reliability_doctor import doctor_skill

    result = doctor_skill("demo-skill", smoke=True)

    assert result["status"] == "ok"
    assert result["smoke"][0] == {
        "type": "mcp-configured",
        "name": "jbrain",
        "status": "ok",
        "message": "configured",
    }


def test_doctor_all_skill_and_cron_paths(hermes_home):
    write_skill(hermes_home, body="""---
name: demo-skill
---
# Demo
""")
    from cron.jobs import create_job
    from hermes_cli.reliability_doctor import doctor_all_crons, doctor_all_skills

    create_job(prompt="ok", schedule="every 1h", skills=["demo-skill"])

    skill_result = doctor_all_skills()
    cron_result = doctor_all_crons()

    assert skill_result["kind"] == "skill_collection"
    assert any(item["skill"] == "demo-skill" for item in skill_result["results"])
    assert cron_result["kind"] == "cron_collection"
    assert len(cron_result["results"]) == 1


def test_doctor_command_all_skill_and_cron_paths(hermes_home, monkeypatch, capsys):
    skill_path = write_skill(hermes_home, body="""---
name: demo-skill
---
# Demo
""")
    from argparse import Namespace
    from cron.jobs import create_job
    from hermes_cli import reliability_doctor
    from hermes_cli.reliability_doctor import doctor_command

    monkeypatch.setattr(reliability_doctor, "_iter_skill_files", lambda: [skill_path])
    create_job(prompt="ok", schedule="every 1h", skills=["demo-skill"])

    skill_rc = doctor_command(Namespace(doctor_target="skill", doctor_name=None, all=True, smoke=False, json=True))
    skill_out = capsys.readouterr().out
    cron_rc = doctor_command(Namespace(doctor_target="cron", doctor_name=None, all=True, smoke=False, json=True))
    cron_out = capsys.readouterr().out

    assert skill_rc == 0
    assert json.loads(skill_out)["kind"] == "skill_collection"
    assert cron_rc == 0
    assert json.loads(cron_out)["kind"] == "cron_collection"


def test_json_output_shape_contract_is_stable(hermes_home):
    write_skill(hermes_home, body="""---
name: demo-skill
---
# Demo
""")
    from hermes_cli.reliability_doctor import doctor_skill

    result = doctor_skill("demo-skill")
    encoded = json.dumps(result, sort_keys=True)

    assert '"skill": "demo-skill"' in encoded
    assert set(result) == {"kind", "skill", "path", "status", "errors", "warnings", "smoke"}
    assert isinstance(result["errors"], list)
    assert isinstance(result["warnings"], list)
    assert isinstance(result["smoke"], list)


def test_prompt_path_extraction_catches_multiple_absolute_paths_and_ignores_relative(hermes_home):
    from cron.jobs import create_job
    from hermes_cli.reliability_doctor import doctor_cron

    existing = hermes_home / "scripts" / "exists.py"
    existing.write_text("print('ok')\n", encoding="utf-8")
    job = create_job(
        prompt=f"Run {existing} then /tmp/missing_one.sh and relative_script.py and /opt/missing_two.py",
        schedule="every 1h",
    )

    result = doctor_cron(job["id"])

    missing_prompt_paths = [e["name"] for e in result["errors"] if e["type"] == "prompt_path"]
    assert missing_prompt_paths == ["/tmp/missing_one.sh", "/opt/missing_two.py"]


def test_cron_preflight_defaults_to_smoke_warn_only(monkeypatch, capsys):
    from hermes_cli import cron as cron_cli

    seen = {}

    def fake_doctor(job_id, *, smoke=False):
        seen["job_id"] = job_id
        seen["smoke"] = smoke
        return {
            "status": "fail",
            "errors": [{"type": "smoke", "name": "auth", "message": "exit 1"}],
            "warnings": [],
        }

    monkeypatch.setattr("hermes_cli.reliability_doctor.doctor_cron", fake_doctor)

    preflight_ok, preflight_result = cron_cli._print_cron_preflight("job-1")

    assert seen == {"job_id": "job-1", "smoke": True}
    assert preflight_ok is False
    assert preflight_result["status"] == "fail"
    assert "Preflight: fail" in capsys.readouterr().out


def test_cron_create_preflight_warn_only_strict_blocks_and_skip_bypasses(monkeypatch):
    from argparse import Namespace
    from hermes_cli import cron as cron_cli

    api_calls = []
    preflight_calls = []

    def fake_api(**kwargs):
        api_calls.append(kwargs)
        return {
            "success": True,
            "job_id": "job-1",
            "name": "Demo",
            "schedule": "every 1h",
            "next_run_at": "soon",
            "job": {"job_id": "job-1", "name": "Demo", "schedule": "every 1h"},
        }

    def fake_preflight(job_id):
        preflight_calls.append(job_id)
        return {
            "status": "fail",
            "errors": [{"type": "smoke", "name": "auth", "message": "exit 1"}],
            "warnings": [],
        }

    monkeypatch.setattr(cron_cli, "_cron_api", fake_api)
    monkeypatch.setattr(cron_cli, "_run_cron_preflight", fake_preflight)

    base = dict(schedule="every 1h", prompt="demo", name="Demo", deliver=None, repeat=None, skill=None, skills=None, script=None)
    assert cron_cli.cron_create(Namespace(**base, skip_preflight=False, strict_preflight=False)) == 0
    assert preflight_calls == ["job-1"]

    assert cron_cli.cron_create(Namespace(**base, skip_preflight=False, strict_preflight=True)) == 1
    assert preflight_calls == ["job-1", "job-1"]

    assert cron_cli.cron_create(Namespace(**base, skip_preflight=True, strict_preflight=True)) == 0
    assert preflight_calls == ["job-1", "job-1"]


def test_strict_preflight_failure_warns_that_persisted_job_needs_repair(monkeypatch, capsys):
    from argparse import Namespace
    from hermes_cli import cron as cron_cli

    def fake_api(**kwargs):
        return {
            "success": True,
            "job_id": "job-1",
            "name": "Demo",
            "schedule": "every 1h",
            "next_run_at": "soon",
            "job": {"job_id": "job-1", "name": "Demo", "schedule": "every 1h"},
        }

    def fake_preflight(job_id):
        assert job_id == "job-1"
        return {
            "status": "fail",
            "errors": [{"type": "smoke", "name": "gmail-auth", "message": "exit 1"}],
            "warnings": [],
        }

    monkeypatch.setattr(cron_cli, "_cron_api", fake_api)
    monkeypatch.setattr(cron_cli, "_run_cron_preflight", fake_preflight)

    args = Namespace(
        schedule="every 1h",
        prompt="demo",
        name="Demo",
        deliver=None,
        repeat=None,
        skill=None,
        skills=None,
        script=None,
        skip_preflight=False,
        strict_preflight=True,
    )
    assert cron_cli.cron_create(args) == 1

    captured = capsys.readouterr()
    assert "Cron job job-1 was saved, but smoke preflight failed" in captured.err
    assert "ERROR smoke gmail-auth: exit 1" in captured.err
    assert "hermes doctor cron job-1 --smoke" in captured.err


def test_strict_preflight_failure_after_edit_warns_that_updated_job_persisted(monkeypatch, capsys):
    from argparse import Namespace
    from cron import jobs as cron_jobs
    from hermes_cli import cron as cron_cli

    monkeypatch.setattr(cron_jobs, "get_job", lambda job_id: {"id": job_id, "job_id": job_id, "name": "Demo", "skills": []})

    def fake_api(**kwargs):
        assert kwargs["action"] == "update"
        return {
            "success": True,
            "job": {"job_id": kwargs["job_id"], "name": "Demo", "schedule": "every 2h", "skills": []},
        }

    def fake_preflight(job_id):
        assert job_id == "job-1"
        return {
            "status": "fail",
            "errors": [{"type": "prompt_path", "name": "/missing/script.py", "message": "missing"}],
            "warnings": [],
        }

    monkeypatch.setattr(cron_cli, "_cron_api", fake_api)
    monkeypatch.setattr(cron_cli, "_run_cron_preflight", fake_preflight)

    args = Namespace(
        job_id="job-1",
        schedule=None,
        prompt=None,
        name=None,
        deliver=None,
        repeat=None,
        skill=None,
        skills=None,
        add_skills=None,
        remove_skills=None,
        clear_skills=False,
        script=None,
        skip_preflight=False,
        strict_preflight=True,
    )
    assert cron_cli.cron_edit(args) == 1
    captured = capsys.readouterr()
    assert "Cron job job-1 was saved, but smoke preflight failed" in captured.err
    assert "updated" in captured.err
    assert "ERROR prompt_path /missing/script.py: missing" in captured.err
