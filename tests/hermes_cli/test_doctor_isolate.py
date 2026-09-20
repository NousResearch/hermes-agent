"""Clean-room isolation preserves source state and attributes failing slices."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

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
        "plugins:\n  enabled: true\ndisplay:\n  skin: mono\n",
        encoding="utf-8",
    )
    (source / ".env").write_text("CUSTOM_API_KEY=never-report-me\n", encoding="utf-8")
    (source / "state.db-wal").write_bytes(b"live-wal")
    (source / "plugins" / "poisoned" / "plugin.py").write_text("raise RuntimeError('bad')\n", encoding="utf-8")
    (source / "memories" / "MEMORY.md").write_text("memory\n", encoding="utf-8")
    return source


def test_isolate_preserves_source_excludes_secrets_and_finds_poisoned_plugin(tmp_path):
    source = _source_profile(tmp_path)
    before = _tree_hash(source)
    candidates: list[Path] = []

    def probe(candidate, config, runtime):
        candidates.append(candidate)
        if (candidate / "plugins" / "poisoned" / "plugin.py").exists():
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
    assert all(not candidate.exists() for candidate in candidates)
    assert ".env" not in repr(payload) and "state.db-wal" not in repr(payload)
    assert "never-report-me" not in repr(payload)


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
