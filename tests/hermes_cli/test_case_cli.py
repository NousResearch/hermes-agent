from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hermes_cli import case as case_cli


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def test_open_case_creates_file_backed_case(hermes_home):
    data = case_cli.open_case(
        case_id="CASE-2026-00001",
        title="Debug empty response",
        kind="debug",
        surface="backend",
        ladder_id="generic-debug",
    )

    assert data["case_id"] == "CASE-2026-00001"
    case_dir = hermes_home / "cases" / "CASE-2026-00001"
    assert (case_dir / "case.yaml").exists()
    assert (case_dir / "carry_forward.yaml").exists()
    assert (case_dir / "events.jsonl").exists()
    assert (case_dir / "artifacts" / "traces").is_dir()

    saved = yaml.safe_load((case_dir / "case.yaml").read_text())
    assert saved["verification"]["ladder_id"] == "generic-debug"
    assert saved["phase"] == "intake"


def test_open_case_defaults_ladder_for_ci_repair(hermes_home):
    data = case_cli.open_case(
        case_id="CASE-2026-00010",
        title="Fix CI",
        kind="ci-repair",
        surface="infra-ci",
    )

    assert data["verification"]["ladder_id"] == "generic-ci-repair"


def test_open_case_leaves_ladder_null_for_feature(hermes_home):
    data = case_cli.open_case(
        case_id="CASE-2026-00011",
        title="Add feature",
        kind="feature",
        surface="backend",
    )

    assert data["verification"]["ladder_id"] is None


def test_open_case_rejects_unknown_ladder(hermes_home):
    with pytest.raises(ValueError, match="unknown ladder id"):
        case_cli.open_case(
            case_id="CASE-2026-00012",
            title="Bad ladder",
            kind="debug",
            surface="backend",
            ladder_id="generic-feature",
        )


def test_next_case_id_is_monotonic(hermes_home):
    first = case_cli._next_case_id()
    second = case_cli._next_case_id()
    assert first != second
    assert int(first.rsplit("-", 1)[-1]) + 1 == int(second.rsplit("-", 1)[-1])


def test_next_case_id_respects_existing_directories(hermes_home):
    case_cli.open_case(
        case_id="CASE-2026-00099",
        title="Existing",
        kind="debug",
        surface="backend",
    )

    next_id = case_cli._next_case_id()
    assert int(next_id.rsplit("-", 1)[-1]) >= 100


def test_auto_open_case_allocates_unique_ids(hermes_home):
    ids: set[str] = set()

    def _open_one(index: int) -> str:
        data = case_cli.open_case(
            title=f"Concurrent {index}",
            kind="debug",
            surface="backend",
        )
        return data["case_id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_open_one, i) for i in range(8)]
        for future in as_completed(futures):
            ids.add(future.result())

    assert len(ids) == 8


def test_advance_case_validates_phase_transitions(hermes_home):
    case_path = hermes_home / "cases" / "CASE-2026-00002"
    case_cli.open_case(
        case_id="CASE-2026-00002",
        title="Fix CI",
        kind="ci-repair",
        surface="infra-ci",
    )

    updated = case_cli.advance_case("CASE-2026-00002", "repro")
    assert updated["phase"] == "repro"
    assert updated["status"] == "open"

    with pytest.raises(ValueError, match="cannot close case without verification"):
        case_cli.advance_case("CASE-2026-00002", "closed")

    verification = case_path / "artifacts" / "verification.json"
    verification.parent.mkdir(parents=True, exist_ok=True)
    verification.write_text('{"result": "verified success"}', encoding="utf-8")

    closed = case_cli.advance_case("CASE-2026-00002", "closed")
    assert closed["phase"] == "closed"
    assert closed["status"] == "closed"

    with pytest.raises(ValueError):
        case_cli.advance_case("CASE-2026-00002", "diagnose")


def test_advance_case_force_close_bypasses_verification(hermes_home):
    case_cli.open_case(
        case_id="CASE-2026-00020",
        title="Force close",
        kind="debug",
        surface="backend",
    )

    closed = case_cli.advance_case("CASE-2026-00020", "closed", force_close=True)
    assert closed["phase"] == "closed"
    assert closed["status"] == "closed"


def test_attach_artifact_records_case_event(hermes_home):
    case_cli.open_case(
        case_id="CASE-2026-00003",
        title="Review auth",
        kind="review",
        surface="backend",
    )

    entry = case_cli.attach_artifact(
        "CASE-2026-00003",
        "artifacts/review.md",
        kind="report",
        description="Review notes",
    )

    assert entry["kind"] == "report"
    data = case_cli.show_case("CASE-2026-00003")
    assert data["artifacts"][0]["path"] == "artifacts/review.md"
    events = (hermes_home / "cases" / "CASE-2026-00003" / "events.jsonl").read_text()
    assert "artifact_attached" in events


def test_case_command_open_outputs_json(hermes_home, capsys):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    case_parser = case_cli.build_parser(sub)
    case_parser.set_defaults(func=case_cli.case_command)
    args = parser.parse_args(
        [
            "case",
            "open",
            "--case-id",
            "CASE-2026-00004",
            "--title",
            "Generic debug",
            "--kind",
            "debug",
            "--surface",
            "backend",
            "--json",
        ]
    )

    rc = args.func(args)
    out = capsys.readouterr().out

    assert rc == 0
    payload = json.loads(out)
    assert payload["case_id"] == "CASE-2026-00004"
    assert payload["verification"]["ladder_id"] == "generic-debug"


def test_case_command_rejects_invalid_id(hermes_home, capsys):
    rc = case_cli.case_command(
        argparse.Namespace(
            case_action="show",
            case_id="bad-id",
            json=False,
            _case_parser=None,
        )
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "invalid case id" in captured.err


def test_run_slash_open_and_list(hermes_home):
    out = case_cli.run_slash(
        'open --case-id CASE-2026-00050 --title "Slash open" --kind debug --surface backend'
    )
    assert "Opened CASE-2026-00050" in out

    listed = case_cli.run_slash("list")
    assert "CASE-2026-00050" in listed


def test_run_slash_rejects_close_without_verification(hermes_home):
    case_cli.open_case(
        case_id="CASE-2026-00051",
        title="Needs verification",
        kind="debug",
        surface="backend",
    )

    out = case_cli.run_slash("advance CASE-2026-00051 --phase closed")
    assert "cannot close case without verification" in out


def test_cli_process_command_routes_case_slash(hermes_home):
    from cli import HermesCLI

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}

    with patch("builtins.print") as mock_print:
        cli_obj.process_command(
            '/case open --case-id CASE-2026-00060 --title "CLI route" --kind debug --surface backend'
        )

    printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "Opened CASE-2026-00060" in printed
