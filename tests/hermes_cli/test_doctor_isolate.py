"""Clean-room isolation preserves source state and attributes failing slices."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sqlite3

import yaml

from hermes_cli.doctor_isolate import run_isolation_diagnostic
from hermes_cli.subcommands.doctor import build_doctor_parser


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _source_profile(tmp_path: Path) -> Path:
    source = tmp_path / "profiles" / "broken"
    (source / "plugins" / "poisoned").mkdir(parents=True)
    (source / "memories").mkdir()
    (source / "config.yaml").write_text(
        "_config_version: 1\nmodel:\n  provider: custom\n  default: model-x\n"
        "plugins:\n  enabled: [poisoned]\ndisplay:\n  skin: mono\n",
        encoding="utf-8",
    )
    (source / ".env").write_text("CUSTOM_API_KEY=never-report-me\n", encoding="utf-8")
    (source / "state.db-wal").write_bytes(b"live-wal")
    (source / "plugins" / "poisoned" / "plugin.yaml").write_text(
        "name: poisoned\nversion: 1.0.0\n", encoding="utf-8"
    )
    (source / "plugins" / "poisoned" / "__init__.py").write_text(
        "def register(ctx):\n    pass\n", encoding="utf-8"
    )
    (source / "memories" / "MEMORY.md").write_text("memory\n", encoding="utf-8")
    return source


def test_isolate_preserves_source_excludes_secrets_and_discovers_each_fresh_slice(tmp_path, monkeypatch):
    source = _source_profile(tmp_path)
    monkeypatch.setenv("ISOLATE_SECRET", "expanded-secret")
    config = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    config["providers"] = {"custom": {"api_key": "${ISOLATE_SECRET}"}}
    config["custom_providers"] = [{"name": "other", "api_key": "literal-secret"}]
    (source / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    before = _tree_hash(source)
    candidates: list[Path] = []
    candidate_configs: list[str] = []

    def probe(candidate, config, runtime):
        from hermes_cli.plugins import discover_plugins, get_plugin_manager

        candidates.append(candidate)
        candidate_configs.append((candidate / "config.yaml").read_text(encoding="utf-8"))
        discover_plugins()
        discovered = {item["name"] for item in get_plugin_manager().list_plugins()}
        if "poisoned" in discovered:
            return {
                "status": "fail",
                "failed_phase": "plugin_hook_initialization",
                "error_class": "RuntimeError",
                "timings": {"hermes_first_chunk_ms": 10.0},
            }
        return {"status": "ok", "timings": {"hermes_first_chunk_ms": 10.0}}

    report = run_isolation_diagnostic(
        source=source,
        probe=probe,
        runtime_override={"provider": "custom", "api_key": "never-report-me"},
    )
    payload = report.to_dict()

    assert _tree_hash(source) == before
    assert payload["classification"] == "profile_state_regression"
    assert payload["control"]["status"] == "pass"
    assert payload["culprit"] == {
        "slice": "plugins",
        "component": "plugin_hook_initialization",
        "evidence": ["plugins changed the acceptance probe from pass to fail"],
    }
    assert payload["cleanup_status"] == "removed"
    assert len(set(candidates)) == len(candidates)
    assert all(not candidate.exists() for candidate in candidates)
    assert ".env" not in repr(payload) and "state.db-wal" not in repr(payload)
    assert "never-report-me" not in repr(payload)
    assert all("expanded-secret" not in text and "literal-secret" not in text for text in candidate_configs)


def test_isolate_snapshots_sqlite_and_ignores_concurrent_runtime_writers(tmp_path, monkeypatch):
    source = _source_profile(tmp_path)
    (source / "state.db-wal").unlink()
    with sqlite3.connect(source / "state.db") as conn:
        conn.execute("create table evidence(value text)")
        conn.execute("insert into evidence values ('closed')")

    seen_values: list[str] = []

    def probe(candidate, config, runtime):
        if (candidate / "state.db").exists():
            with sqlite3.connect(candidate / "state.db") as conn:
                seen_values.extend(row[0] for row in conn.execute("select value from evidence"))
        logs = source / "logs"
        logs.mkdir(exist_ok=True)
        (logs / "gateway.log").write_text(str(len(seen_values)), encoding="utf-8")
        return {"status": "ok", "timings": {"hermes_first_chunk_ms": 10.0}}

    with sqlite3.connect(source / "state.db") as writer:
        writer.execute("pragma journal_mode=wal")
        writer.execute("insert into evidence values ('wal')")
        writer.commit()
        report = run_isolation_diagnostic(
            source=source, probe=probe, runtime_override={"provider": "custom", "api_key": "secret"}
        )

    assert report.classification == "healthy"
    assert {"closed", "wal"}.issubset(seen_values)
    assert next(item for item in report.slices if item.id == "session_state").status == "pass"

    monkeypatch.setattr("hermes_cli.doctor_isolate._snapshot_state_db", lambda *args, **kwargs: False)
    unresolved = run_isolation_diagnostic(
        source=source, probe=probe, runtime_override={"provider": "custom", "api_key": "secret"}
    )
    assert unresolved.classification == "needs_quiescence"
    assert next(item for item in unresolved.slices if item.id == "session_state").status == "needs_quiescence"


def test_isolate_stops_after_failed_sterile_control_and_parser_modes_are_exclusive(tmp_path):
    source = _source_profile(tmp_path)
    calls = 0

    def failed_control(candidate, config, runtime):
        nonlocal calls
        calls += 1
        return {"status": "fail", "error_class": "ProviderError", "timings": {}}

    report = run_isolation_diagnostic(
        source=source,
        probe=failed_control,
        runtime_override={"provider": "custom", "api_key": "secret"},
    )
    assert report.classification == "below_profile_layer"
    assert report.slices == []
    assert calls == 1

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_doctor_parser(subparsers, cmd_doctor=lambda args: None)
    assert parser.parse_args(["doctor", "--isolate", "--json"]).isolate is True
    try:
        parser.parse_args(["doctor", "--isolate", "--runtime"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("diagnostic modes must be mutually exclusive")
