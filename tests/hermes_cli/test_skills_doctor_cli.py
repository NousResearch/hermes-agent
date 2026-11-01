"""Tests for `hermes skills doctor` — parser wiring and exit-code contract.

Acceptance (ORCA-06 follow-up): the command exits non-zero on a simulated
active+pinned registry entry whose skill dir is missing, and zero when the
registry reconciles clean.
"""

import argparse
import json
from pathlib import Path

from hermes_cli import skills_doctor
from hermes_cli.subcommands.skills import build_skills_parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    build_skills_parser(parser.add_subparsers(dest="command"), cmd_skills=lambda args: None)
    return parser


def _make_skill(root: Path, *parts: str) -> None:
    d = root.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {parts[-1]}\ndescription: test\n---\n\n# {parts[-1]}\n", encoding="utf-8")


def _write_registry(root: Path, entries: dict) -> None:
    (root / ".usage.json").write_text(json.dumps(entries), encoding="utf-8")


def _recent() -> dict:
    return {"state": "active", "pinned": False, "use_count": 1, "view_count": 1,
            "last_used_at": "2026-10-25T00:00:00+00:00",
            "last_viewed_at": "2026-10-25T00:00:00+00:00",
            "created_at": "2026-10-01T00:00:00+00:00", "archived_at": None}


def test_doctor_subcommand_parses():
    args = _parser().parse_args(["skills", "doctor"])
    assert args.skills_action == "doctor"
    assert args.json is False


def test_doctor_json_flag_parses():
    args = _parser().parse_args(["skills", "doctor", "--json"])
    assert args.json is True


def test_doctor_exits_nonzero_on_active_pinned_missing_dir(tmp_path, monkeypatch, capsys):
    """The ORCA-06 desync signature: active + pinned + no dir + no archive copy."""
    root = tmp_path / "skills"
    root.mkdir()
    _make_skill(root, "healthy")
    _write_registry(root, {
        "healthy": _recent(),
        "gone-pinned": {**_recent(), "pinned": True, "use_count": 0,
                        "last_used_at": "2026-09-01T00:00:00+00:00",
                        "last_viewed_at": "2026-09-01T00:00:00+00:00"},
    })
    monkeypatch.setattr(skills_doctor, "default_skills_root", lambda: root)

    rc = skills_doctor.skills_doctor_command(_parser().parse_args(["skills", "doctor"]))

    assert rc == 1
    out = capsys.readouterr().out
    assert "CRITICAL" in out
    assert "gone-pinned" in out


def test_doctor_exits_zero_when_reconciled(tmp_path, monkeypatch, capsys):
    root = tmp_path / "skills"
    root.mkdir()
    _make_skill(root, "healthy")
    _make_skill(root, "devops", "path-skill")
    _write_registry(root, {"healthy": _recent(), "devops/path-skill": _recent()})
    monkeypatch.setattr(skills_doctor, "default_skills_root", lambda: root)

    rc = skills_doctor.skills_doctor_command(_parser().parse_args(["skills", "doctor"]))

    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_doctor_json_flag_reports_and_preserves_exit_code(tmp_path, monkeypatch, capsys):
    root = tmp_path / "skills"
    root.mkdir()
    _write_registry(root, {"gone-pinned": {**_recent(), "pinned": True}})
    monkeypatch.setattr(skills_doctor, "default_skills_root", lambda: root)

    rc = skills_doctor.skills_doctor_command(
        _parser().parse_args(["skills", "doctor", "--json"]))

    assert rc == 1
    report = json.loads(capsys.readouterr().out)
    assert [c["key"] for c in report["critical"]] == ["gone-pinned"]
    assert report["registry_present"] is True


def test_doctor_missing_registry_exits_nonzero(tmp_path, monkeypatch, capsys):
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr(skills_doctor, "default_skills_root", lambda: root)

    rc = skills_doctor.skills_doctor_command(_parser().parse_args(["skills", "doctor"]))

    assert rc == 1
    assert "NO REGISTRY" in capsys.readouterr().out


def test_cmd_skills_routes_doctor_to_the_doctor_command(monkeypatch):
    """`hermes skills doctor` must not fall through to the hub router."""
    from hermes_cli.main_agent_cmds import cmd_skills

    seen = {}

    def _fake_command(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr("hermes_cli.skills_doctor.skills_doctor_command", _fake_command)
    rc = cmd_skills(argparse.Namespace(skills_action="doctor"))
    assert rc == 0
    assert seen["args"].skills_action == "doctor"
