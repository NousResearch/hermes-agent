import json
import os
from pathlib import Path

from hermes_cli import path_authority_doctor as pad


def test_parse_paths_env_redacts_secret_values(tmp_path):
    env_file = tmp_path / "paths.env"
    target = tmp_path / "hermes-agent-kanban-v2026.5.7"
    redacted_value = "fixture_" + "redaction_value"
    env_file.write_text(
        "\n".join(
            [
                f'export HERMES_KANBAN_AGENT_DIR="{target}"',
                "GITHUB_" + "TOKEN" + "=" + redacted_value,
                "PLAIN_FLAG=enabled",
            ]
        ),
        encoding="utf-8",
    )

    parsed = pad.parse_paths_env(env_file)

    assert parsed.values["HERMES_KANBAN_AGENT_DIR"] == str(target)
    assert parsed.redacted["GITHUB_TOKEN"] == "[REDACTED]"
    assert redacted_value not in json.dumps(parsed.to_report())


def test_systemd_dropin_execstart_reset_overrides_base(tmp_path):
    target = tmp_path / "agent"
    target.mkdir()
    unit_dir = tmp_path / "systemd" / "user"
    dropin_dir = unit_dir / "hermes-gateway.service.d"
    dropin_dir.mkdir(parents=True)
    (unit_dir / "hermes-gateway.service").write_text(
        "\n".join(
            [
                "[Service]",
                "ExecStart=/old/venv/bin/python -m hermes_cli.main gateway run --replace",
                "WorkingDirectory=/old",
                'Environment="PATH=/old/bin" "HERMES_AGENT_DIR=/old"',
            ]
        ),
        encoding="utf-8",
    )
    (dropin_dir / "10-kanban.conf").write_text(
        "\n".join(
            [
                "[Service]",
                "ExecStart=",
                f"ExecStart={target}/.venv/bin/python -m hermes_cli.main gateway run",
                f"WorkingDirectory={target}",
                f'Environment="PATH={target}/.venv/bin:/usr/bin" "HERMES_AGENT_DIR={target}"',
            ]
        ),
        encoding="utf-8",
    )

    report = pad.build_path_authority_report(
        paths_env=pad.ParsedPathsEnv(
            path=tmp_path / "paths.env",
            exists=True,
            values={"HERMES_KANBAN_AGENT_DIR": str(target)},
            redacted={"HERMES_KANBAN_AGENT_DIR": str(target)},
        ),
        systemd_user_dir=unit_dir,
        include_live=False,
    )

    gateway = report.units["gateway"]
    assert gateway.effective.exec_start == [
        f"{target}/.venv/bin/python -m hermes_cli.main gateway run"
    ]
    assert gateway.effective.working_directory == str(target)
    assert gateway.effective.environment["PATH"] == f"{target}/.venv/bin:/usr/bin"
    assert gateway.effective.environment["HERMES_AGENT_DIR"] == str(target)
    assert not any(check.id == "gateway_execstart_replace" for check in report.failed_checks())


def test_gateway_execstart_replace_is_p0(tmp_path):
    target = tmp_path / "agent"
    target.mkdir()
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "hermes-gateway.service").write_text(
        "\n".join(
            [
                "[Service]",
                f"ExecStart={target}/.venv/bin/python -m hermes_cli.main gateway run --replace",
                f"WorkingDirectory={target}",
            ]
        ),
        encoding="utf-8",
    )

    report = pad.build_path_authority_report(
        paths_env=pad.ParsedPathsEnv(
            path=tmp_path / "paths.env",
            exists=True,
            values={"HERMES_KANBAN_AGENT_DIR": str(target)},
            redacted={"HERMES_KANBAN_AGENT_DIR": str(target)},
        ),
        systemd_user_dir=unit_dir,
        include_live=False,
    )

    failed = {check.id: check for check in report.failed_checks()}
    assert failed["gateway_execstart_replace"].severity == "P0"
    assert report.overall_severity == "P0"


def test_gateway_path_authority_requires_exec_and_workdir(tmp_path):
    target = tmp_path / "agent"
    stale = tmp_path / "stale-agent"
    target.mkdir()
    stale.mkdir()
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / "hermes-gateway.service").write_text(
        "\n".join(
            [
                "[Service]",
                f"ExecStart={target}/.venv/bin/python -m hermes_cli.main gateway run",
                f"WorkingDirectory={stale}",
                f'Environment="HERMES_AGENT_DIR={target}"',
            ]
        ),
        encoding="utf-8",
    )

    report = pad.build_path_authority_report(
        paths_env=pad.ParsedPathsEnv(
            path=tmp_path / "paths.env",
            exists=True,
            values={"HERMES_KANBAN_AGENT_DIR": str(target)},
            redacted={"HERMES_KANBAN_AGENT_DIR": str(target)},
        ),
        systemd_user_dir=unit_dir,
        include_live=False,
    )

    failed = {check.id: check for check in report.failed_checks()}
    assert failed["gateway_path_authority"].severity == "P0"


def test_report_redacts_secret_like_execstart_values(tmp_path):
    unit = pad.EffectiveUnit(
        exec_start=[
            "/srv/agent/.venv/bin/python -m hermes_cli.main gateway run "
            "--api-key sk-super-secret-token"
        ],
        working_directory="/srv/agent",
        environment={"OPENROUTER_API_KEY": "sk-super-secret-token"},
    )

    encoded = json.dumps(unit.to_report())

    assert "sk-super-secret-token" not in encoded
    assert "[REDACTED]" in encoded


def test_live_runtime_authority_missing_is_p0_when_live_checks_enabled(tmp_path, monkeypatch):
    target = tmp_path / "agent"
    target.mkdir()
    unit_dir = tmp_path / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    for service in pad.UNIT_NAMES.values():
        (unit_dir / service).write_text(
            "\n".join(
                [
                    "[Service]",
                    f"ExecStart={target}/.venv/bin/python -m hermes_cli.main gateway run",
                    f"WorkingDirectory={target}",
                    f'Environment="HERMES_AGENT_DIR={target}"',
                ]
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(pad, "_read_runtime_import_authority", lambda: None)

    report = pad.build_path_authority_report(
        paths_env=pad.ParsedPathsEnv(
            path=tmp_path / "paths.env",
            exists=True,
            values={"HERMES_KANBAN_AGENT_DIR": str(target)},
            redacted={"HERMES_KANBAN_AGENT_DIR": str(target)},
        ),
        systemd_user_dir=unit_dir,
        include_live=True,
    )

    failed = {check.id: check for check in report.failed_checks()}
    assert failed["runtime_import_authority_missing"].severity == "P0"


def test_cron_shadow_registry_requires_non_authoritative_marker(tmp_path):
    hermes_home = tmp_path / "hermes"
    active_dir = hermes_home / "cron"
    shadow_dir = active_dir / "cron"
    shadow_dir.mkdir(parents=True)
    (active_dir / "jobs.json").write_text('{"jobs":[]}', encoding="utf-8")
    (shadow_dir / "jobs.json").write_text(
        json.dumps({"jobs": [{"id": "old", "enabled": True, "script": "/root/stale.sh"}]}),
        encoding="utf-8",
    )
    paths_env = pad.ParsedPathsEnv(
        path=tmp_path / "paths.env",
        exists=True,
        values={"HERMES_HOME": str(hermes_home)},
        redacted={"HERMES_HOME": str(hermes_home)},
    )

    checks = pad._check_cron_shadow_authority(paths_env)

    failed = {check.id: check for check in checks if check.status != "ok"}
    assert failed["cron_shadow_registry_unclassified"].severity == "P1"


def test_cron_shadow_registry_marker_classifies_shadow_as_quarantined(tmp_path):
    hermes_home = tmp_path / "hermes"
    active_dir = hermes_home / "cron"
    shadow_dir = active_dir / "cron"
    shadow_dir.mkdir(parents=True)
    (active_dir / "jobs.json").write_text('{"jobs":[]}', encoding="utf-8")
    (shadow_dir / "jobs.json").write_text(
        json.dumps({"jobs": [{"id": "old", "enabled": True, "script": "/root/stale.sh"}]}),
        encoding="utf-8",
    )
    (shadow_dir / "NON_AUTHORITATIVE").write_text(
        "This nested cron registry is retained for audit only.\n",
        encoding="utf-8",
    )
    paths_env = pad.ParsedPathsEnv(
        path=tmp_path / "paths.env",
        exists=True,
        values={"HERMES_HOME": str(hermes_home)},
        redacted={"HERMES_HOME": str(hermes_home)},
    )

    checks = pad._check_cron_shadow_authority(paths_env)

    by_id = {check.id: check for check in checks}
    assert by_id["cron_shadow_registry_quarantined"].status == "ok"


def test_webui_graphify_authority_detects_systemd_env_drift(tmp_path):
    expected_graph = tmp_path / "artifacts" / "graph.json"
    paths_env = pad.ParsedPathsEnv(
        path=tmp_path / "paths.env",
        exists=True,
        values={
            "GRAPHIFY_OUTPUT_DIR": str(expected_graph.parent),
            "GRAPHIFY_GRAPH_JSON": str(expected_graph),
        },
        redacted={},
    )
    units = {
        "webui": pad.UnitReport(
            role="webui",
            name="hermes-webui.service",
            path=tmp_path / "hermes-webui.service",
            exists=True,
            dropins=[],
            effective=pad.EffectiveUnit(
                environment={
                    "GRAPHIFY_OUTPUT_DIR": str(tmp_path / "vault" / "Graphify"),
                    "GRAPHIFY_GRAPH_JSON": str(tmp_path / "vault" / "Graphify" / "combined-graph.json"),
                }
            ),
        )
    }

    checks = pad._check_webui_graphify_authority(paths_env, units)

    assert checks[0].id == "webui_graphify_path_authority"
    assert checks[0].severity == "P1"


def test_webui_graphify_authority_passes_when_systemd_matches_paths_env(tmp_path):
    expected_graph = tmp_path / "artifacts" / "graph.json"
    paths_env = pad.ParsedPathsEnv(
        path=tmp_path / "paths.env",
        exists=True,
        values={
            "GRAPHIFY_OUTPUT_DIR": str(expected_graph.parent),
            "GRAPHIFY_GRAPH_JSON": str(expected_graph),
        },
        redacted={},
    )
    units = {
        "webui": pad.UnitReport(
            role="webui",
            name="hermes-webui.service",
            path=tmp_path / "hermes-webui.service",
            exists=True,
            dropins=[],
            effective=pad.EffectiveUnit(
                environment={
                    "GRAPHIFY_OUTPUT_DIR": str(expected_graph.parent),
                    "GRAPHIFY_GRAPH_JSON": str(expected_graph),
                }
            ),
        )
    }

    checks = pad._check_webui_graphify_authority(paths_env, units)

    assert checks[0].status == "ok"


def test_graphify_freshness_warns_when_governance_is_newer_than_graph(tmp_path):
    vault = tmp_path / "vault"
    source = vault / "Graphify" / "graph-manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    graph = tmp_path / "artifacts" / "graph.json"
    graph.parent.mkdir(parents=True)
    graph.write_text("{}", encoding="utf-8")
    os.utime(graph, (100, 100))
    os.utime(source, (200, 200))
    paths_env = pad.ParsedPathsEnv(
        path=tmp_path / "paths.env",
        exists=True,
        values={
            "OBSIDIAN_VAULT_PATH": str(vault),
            "GRAPHIFY_GRAPH_JSON": str(graph),
        },
        redacted={},
    )

    checks = pad._check_graphify_freshness(paths_env)

    assert checks[0].id == "graphify_freshness"
    assert checks[0].severity == "P1"
    assert checks[0].status == "warn"


def test_graphify_freshness_passes_when_graph_is_current(tmp_path):
    vault = tmp_path / "vault"
    source = vault / "Graphify" / "graph-manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    graph = tmp_path / "artifacts" / "graph.json"
    graph.parent.mkdir(parents=True)
    graph.write_text("{}", encoding="utf-8")
    os.utime(source, (100, 100))
    os.utime(graph, (200, 200))
    paths_env = pad.ParsedPathsEnv(
        path=tmp_path / "paths.env",
        exists=True,
        values={
            "OBSIDIAN_VAULT_PATH": str(vault),
            "GRAPHIFY_GRAPH_JSON": str(graph),
        },
        redacted={},
    )

    checks = pad._check_graphify_freshness(paths_env)

    assert checks[0].status == "ok"


def test_dirty_active_repo_classification():
    other = pad.classify_dirty_repo([" M docs/readme.md"])
    authority = pad.classify_dirty_repo([" M gateway/status.py"])

    assert other.severity == "P1"
    assert authority.severity == "P0"
