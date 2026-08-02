from __future__ import annotations

import json
import sys

import pytest

import hermes_cli.reliability_doctor as rd
from hermes_cli.reliability_doctor import (
    DiagnosticResult,
    SmokeValidationError,
    diagnose_all_crons,
    diagnose_all_skills,
    diagnose_cron,
    diagnose_skill,
    evaluate_static_probes,
    resolve_probe_path,
    validate_smoke_spec,
)


def test_resolve_probe_path_stays_within_selected_root(tmp_path):
    root = tmp_path / "root"
    nested = root / "scripts"
    nested.mkdir(parents=True)

    resolved = resolve_probe_path("workdir", "scripts/check.py", {"workdir": root})

    assert resolved == nested / "check.py"


def test_resolve_probe_path_rejects_symlink_escape(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SmokeValidationError, match="path_escape"):
        resolve_probe_path("workdir", "escape/secret.txt", {"workdir": root})


def test_env_present_reports_presence_without_value(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "PLANTED_SECRET")
    smoke = validate_smoke_spec({
        "version": 1,
        "probes": [{"type": "env-present", "name": "GH_TOKEN"}],
    })

    [result] = evaluate_static_probes(
        smoke,
        subject_type="skill",
        subject="github",
        roots={},
        env={"GH_TOKEN": "PLANTED_SECRET"},
    )

    assert result.status == "pass"
    assert result.reason == "env_present"
    assert "PLANTED_SECRET" not in repr(result)
    assert "PLANTED_SECRET" not in str(result.to_dict())


def test_static_env_present_defaults_to_current_process_environment(monkeypatch):
    monkeypatch.setenv("SMOKE_PRESENT_IN_PROCESS", "PLANTED_SECRET")
    smoke = validate_smoke_spec({
        "version": 1,
        "probes": [{"type": "env-present", "name": "SMOKE_PRESENT_IN_PROCESS"}],
    })

    [result] = evaluate_static_probes(
        smoke,
        subject_type="skill",
        subject="local",
        roots={},
    )

    assert result.status == "pass"
    assert result.reason == "env_present"
    assert "PLANTED_SECRET" not in str(result.to_dict())


def test_command_exists_uses_which_without_running_command(monkeypatch):
    called = {}

    def fake_which(name):
        called["name"] = name
        return "/usr/bin/gh"

    monkeypatch.setattr("hermes_cli.reliability_doctor.shutil.which", fake_which)
    smoke = validate_smoke_spec({
        "version": 1,
        "probes": [{"type": "command-exists", "name": "gh"}],
    })

    [result] = evaluate_static_probes(
        smoke, subject_type="skill", subject="github", roots={}, env={}
    )

    assert called == {"name": "gh"}
    assert result.status == "pass"
    assert result.reason == "command_found"


def test_python_import_checks_top_level_and_dotted_module_without_import_side_effect(
    tmp_path,
):
    package = tmp_path / "sideeffectpkg"
    package.mkdir()
    marker = tmp_path / "imported"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
        encoding="utf-8",
    )
    (package / "submodule.py").write_text("VALUE = 1\n", encoding="utf-8")
    smoke = validate_smoke_spec({
        "version": 1,
        "probes": [
            {"type": "python-import", "module": "sideeffectpkg"},
            {"type": "python-import", "module": "sideeffectpkg.submodule"},
        ],
    })

    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(tmp_path))
        results = evaluate_static_probes(
            smoke, subject_type="skill", subject="local", roots={}, env={}
        )
    finally:
        sys.path[:] = old_path

    assert [r.status for r in results] == ["pass", "pass"]
    assert not marker.exists()


def test_mcp_configured_reads_config_presence_without_connecting(tmp_path, monkeypatch):
    def fail_import(name, *args, **kwargs):
        if name == "tools.mcp_tool":  # pragma: no cover - should never happen
            raise AssertionError("static MCP check must not connect/import mcp tool")
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_import)
    smoke = validate_smoke_spec({
        "version": 1,
        "probes": [{"type": "mcp-configured", "server": "github"}],
    })

    [result] = evaluate_static_probes(
        smoke,
        subject_type="cron",
        subject="daily",
        roots={},
        env={},
        mcp_servers={"github": {"command": "mcp-server-github"}},
    )

    assert result == DiagnosticResult(
        subject_type="cron",
        subject="daily",
        probe_type="mcp-configured",
        target="github",
        status="pass",
        reason="mcp_configured",
    )


def test_path_probe_symlink_escape_returns_stable_failure(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    smoke = validate_smoke_spec({
        "version": 1,
        "probes": [{"type": "file-exists", "root": "workdir", "path": "escape/x"}],
    })

    [result] = evaluate_static_probes(
        smoke,
        subject_type="cron",
        subject="daily",
        roots={"workdir": root},
    )

    assert result.status == "fail"
    assert result.reason == "path_escape"


def test_diagnose_skill_reports_required_env_and_commands(monkeypatch):
    monkeypatch.setattr(rd, "_find_all_skills", lambda: [{"name": "github"}])
    monkeypatch.setattr(
        rd,
        "skill_view",
        lambda name, preprocess=False, metadata_only=False: json.dumps({
            "success": True,
            "frontmatter": {
                "name": "github",
                "required_environment_variables": [{"name": "GH_TOKEN"}],
                "prerequisites": {"commands": ["gh"]},
            },
        }),
    )
    monkeypatch.setattr(
        rd.shutil, "which", lambda name: "/usr/bin/gh" if name == "gh" else None
    )

    results = diagnose_skill("github")

    assert [(r.probe_type, r.target, r.status, r.reason) for r in results] == [
        ("env-present", "GH_TOKEN", "fail", "env_missing"),
        ("command-exists", "gh", "pass", "command_found"),
    ]


def test_diagnose_skill_defaults_to_current_process_environment(monkeypatch):
    monkeypatch.setattr(rd, "_find_all_skills", lambda: [{"name": "github"}])
    monkeypatch.setattr(
        rd,
        "_load_skill_frontmatter",
        lambda name: {"required_environment_variables": ["GH_TOKEN"]},
    )
    monkeypatch.setenv("GH_TOKEN", "PLANTED_SECRET")

    results = diagnose_skill("github")

    [env_result] = [result for result in results if result.probe_type == "env-present"]
    assert env_result.status == "pass"
    assert env_result.reason == "env_present"
    assert "PLANTED_SECRET" not in str(env_result.to_dict())


def test_diagnose_skill_does_not_fail_for_missing_optional_environment(monkeypatch):
    monkeypatch.setattr(rd, "_find_all_skills", lambda: [{"name": "optional"}])
    monkeypatch.setattr(
        rd,
        "_load_skill_frontmatter",
        lambda name: {
            "required_environment_variables": [
                {"name": "OPTIONAL_TOKEN", "optional": True}
            ]
        },
    )

    results = diagnose_skill("optional", env={})

    assert all(result.status != "fail" for result in results)
    assert all(result.target != "OPTIONAL_TOKEN" for result in results)


def test_diagnose_skill_reports_malformed_legacy_prerequisites_without_crashing(
    monkeypatch,
):
    monkeypatch.setattr(rd, "_find_all_skills", lambda: [{"name": "legacy"}])
    monkeypatch.setattr(
        rd,
        "_load_skill_frontmatter",
        lambda name: {"prerequisites": {"commands": 7}},
    )

    results = diagnose_skill("legacy", env={})

    assert results == [
        DiagnosticResult(
            "skill",
            "legacy",
            "skill-metadata",
            "legacy",
            "fail",
            "invalid_skill_metadata",
        )
    ]


def test_diagnose_skill_does_not_return_environment_value(monkeypatch):
    monkeypatch.setattr(rd, "_find_all_skills", lambda: [{"name": "github"}])
    monkeypatch.setattr(
        rd,
        "skill_view",
        lambda name, preprocess=False, metadata_only=False: json.dumps({
            "success": True,
            "frontmatter": {
                "name": "github",
                "prerequisites": {"env_vars": ["GH_TOKEN"]},
            },
        }),
    )

    results = diagnose_skill("github", env={"GH_TOKEN": "PLANTED_SECRET"})

    assert results[0].status == "pass"
    assert "PLANTED_SECRET" not in repr(results)
    assert "PLANTED_SECRET" not in str([r.to_dict() for r in results])


def test_diagnose_skill_not_found_is_stable_failure(monkeypatch):
    monkeypatch.setattr(rd, "_find_all_skills", lambda: [{"name": "other"}])

    [result] = diagnose_skill("missing")

    assert result == DiagnosticResult(
        subject_type="skill",
        subject="missing",
        probe_type="skill",
        target="missing",
        status="fail",
        reason="skill_not_found",
    )


def test_diagnose_all_skills_uses_current_skill_discovery(monkeypatch):
    monkeypatch.setattr(
        rd, "_find_all_skills", lambda: [{"name": "one"}, {"name": "two"}]
    )
    monkeypatch.setattr(
        rd,
        "diagnose_skill",
        lambda name: [
            DiagnosticResult("skill", name, "skill", name, "pass", "skill_found")
        ],
    )

    result = diagnose_all_skills()

    assert list(result) == ["one", "two"]
    assert result["one"][0].reason == "skill_found"


def test_diagnose_cron_reports_prompt_or_skill_and_schedule(monkeypatch):
    monkeypatch.setattr(
        rd,
        "resolve_job_ref",
        lambda ref: {
            "id": "abc123",
            "name": "daily",
            "prompt": "Summarize",
            "skills": ["github"],
            "schedule": {"kind": "interval"},
            "schedule_display": "every 1h",
        },
    )

    results = diagnose_cron("daily")

    assert ("cron-prompt", "prompt", "pass", "cron_prompt_present") in [
        (r.probe_type, r.target, r.status, r.reason) for r in results
    ]
    assert ("cron-skill", "github", "pass", "cron_skill_attached") in [
        (r.probe_type, r.target, r.status, r.reason) for r in results
    ]
    assert ("cron-schedule", "every 1h", "pass", "cron_schedule_present") in [
        (r.probe_type, r.target, r.status, r.reason) for r in results
    ]


def test_diagnose_cron_reports_missing_stored_script(monkeypatch, tmp_path):
    monkeypatch.setattr(rd, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        rd,
        "resolve_job_ref",
        lambda ref: {
            "id": "abc123",
            "prompt": "Run",
            "script": "reports/check.py",
            "schedule": {"kind": "interval"},
        },
    )

    results = diagnose_cron("daily")

    assert ("cron-script", "reports/check.py", "fail", "script_missing") in [
        (r.probe_type, r.target, r.status, r.reason) for r in results
    ]


def test_diagnose_no_agent_cron_does_not_require_prompt(monkeypatch, tmp_path):
    script = tmp_path / "scripts" / "watchdog.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(rd, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        rd,
        "resolve_job_ref",
        lambda ref: {
            "id": "abc123",
            "prompt": "",
            "skills": [],
            "no_agent": True,
            "script": "watchdog.py",
            "schedule": {"kind": "interval"},
        },
    )

    results = diagnose_cron("watchdog")

    assert all(result.status != "fail" for result in results)
    assert any(result.reason == "cron_prompt_not_required" for result in results)
    assert any(result.reason == "script_exists" for result in results)


def test_diagnose_cron_reports_existing_stored_script(monkeypatch, tmp_path):
    script = tmp_path / "scripts" / "reports" / "check.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(rd, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        rd,
        "resolve_job_ref",
        lambda ref: {
            "id": "abc123",
            "prompt": "Run",
            "script": "reports/check.py",
            "schedule": {"kind": "interval"},
        },
    )

    results = diagnose_cron("daily")

    assert ("cron-script", "reports/check.py", "pass", "script_exists") in [
        (r.probe_type, r.target, r.status, r.reason) for r in results
    ]


def test_diagnose_cron_rejects_hand_edited_script_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(rd, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        rd,
        "resolve_job_ref",
        lambda ref: {
            "id": "abc123",
            "prompt": "Run",
            "script": "../outside.py",
            "schedule": {"kind": "interval"},
        },
    )

    results = diagnose_cron("daily")

    assert ("cron-script", "../outside.py", "fail", "script_path_escape") in [
        (r.probe_type, r.target, r.status, r.reason) for r in results
    ]


def test_diagnose_cron_diagnoses_attached_skill_prerequisites(monkeypatch):
    monkeypatch.setattr(
        rd,
        "resolve_job_ref",
        lambda ref: {
            "id": "abc123",
            "name": "daily",
            "prompt": "",
            "skills": ["github"],
            "schedule": {"kind": "interval"},
        },
    )
    monkeypatch.setattr(
        rd,
        "diagnose_skill",
        lambda name: [
            DiagnosticResult(
                "skill", name, "env-present", "GH_TOKEN", "fail", "env_missing"
            )
        ],
    )

    results = diagnose_cron("daily")

    assert ("cron-skill", "github", "pass", "cron_skill_attached") in [
        (r.probe_type, r.target, r.status, r.reason) for r in results
    ]
    assert ("skill", "github", "env-present", "GH_TOKEN", "fail", "env_missing") in [
        (r.subject_type, r.subject, r.probe_type, r.target, r.status, r.reason)
        for r in results
    ]


def test_diagnose_cron_runs_static_smoke_only(monkeypatch, tmp_path):
    marker = tmp_path / "marker.txt"
    marker.write_text("ok", encoding="utf-8")
    monkeypatch.setattr(
        rd,
        "resolve_job_ref",
        lambda ref: {
            "id": "abc123",
            "name": "daily",
            "prompt": "Run",
            "schedule": {"kind": "interval"},
            "workdir": str(tmp_path),
            "smoke": {
                "version": 1,
                "probes": [
                    {
                        "type": "file-exists",
                        "root": "workdir",
                        "path": "marker.txt",
                    }
                ],
            },
        },
    )

    results = diagnose_cron("daily")

    assert any(
        result.probe_type == "file-exists" and result.status == "pass"
        for result in results
    )


def test_diagnose_cron_reports_invalid_hand_edited_smoke_without_crashing(monkeypatch):
    monkeypatch.setattr(
        rd,
        "resolve_job_ref",
        lambda ref: {
            "id": "abc123",
            "name": "daily",
            "prompt": "Summarize",
            "schedule": {"kind": "interval"},
            "smoke": {
                "version": 1,
                "probes": [{"type": "shell", "command": "echo nope"}],
            },
        },
    )

    results = diagnose_cron("daily")

    assert any(
        r.probe_type == "smoke-schema" and r.reason == "invalid_smoke" for r in results
    )


def test_diagnose_cron_ref_preserves_ambiguous_name_failure(monkeypatch):
    from cron.jobs import AmbiguousJobReference

    def raise_ambiguous(ref):
        raise AmbiguousJobReference(ref, [{"id": "one"}, {"id": "two"}])

    monkeypatch.setattr(rd, "resolve_job_ref", raise_ambiguous)

    [result] = diagnose_cron("daily")

    assert result.status == "fail"
    assert result.reason == "cron_ref_ambiguous"


def test_diagnose_all_crons_includes_disabled_jobs(monkeypatch):
    monkeypatch.setattr(
        rd,
        "list_jobs",
        lambda include_disabled=False: (
            [
                {"id": "enabled", "name": "Enabled", "enabled": True},
                {"id": "disabled", "name": "Disabled", "enabled": False},
            ]
            if include_disabled
            else [{"id": "enabled", "name": "Enabled", "enabled": True}]
        ),
    )
    monkeypatch.setattr(
        rd,
        "diagnose_cron",
        lambda ref: [DiagnosticResult("cron", ref, "cron", ref, "pass", "cron_found")],
    )

    result = diagnose_all_crons()

    assert list(result) == ["enabled", "disabled"]
