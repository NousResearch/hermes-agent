from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from hermes_cli.engineering_memory_commands import cmd_engineering_memory
from hermes_cli.subcommands.engineering_memory import build_engineering_memory_parser


def _parser():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="root")
    build_engineering_memory_parser(subparsers, cmd_engineering_memory=cmd_engineering_memory)
    return parser


def test_cli_requires_explicit_shared_paths(capsys) -> None:
    args = _parser().parse_args(["engineering-memory", "search", "timeout", "--json"])
    args.func(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["contract"] == "hermes.engineering_memory.v1"
    assert payload["ok"] is False
    assert "explicit --vault and --index" in payload["error"]


def test_cli_json_contract_propose_rebuild_search(tmp_path: Path, capsys) -> None:
    vault = tmp_path / "vault"
    index = tmp_path / "index.sqlite3"
    vault.mkdir()
    source = vault / "candidate.md"
    source.write_text(
        """---
record_id: cli-1
schema_name: agent_engineering_record_v1
sync_owned: true
classification: diagnostic-only
authority: source-index
status: candidate
title: CLI timeout
summary: Worker timeout evidence.
agent: codex
repository: repo
memory_area: runtime
task_type: diagnosis
component: runner
verified_head: abc
observed_at: '2026-09-21T00:00:00Z'
verified_at: '2026-09-21T01:00:00Z'
source_label: receipt
source_digest: cli
evidence_refs:
  - receipt-1
---

Inspect the runner ancestor.
""",
        encoding="utf-8",
    )
    common = ["engineering-memory", "--vault", str(vault), "--index", str(index)]
    args = _parser().parse_args([*common, "propose", str(source), "--json"])
    args.func(args)
    proposed = json.loads(capsys.readouterr().out)
    assert proposed["ok"] is True
    assert proposed["status"] == "candidate"

    args = _parser().parse_args([*common, "review", "cli-1", "approve", "--reviewer", "human", "--reason", "checked", "--json"])
    args.func(args)
    assert json.loads(capsys.readouterr().out)["status"] == "approved"

    args = _parser().parse_args([*common, "rebuild", "--json"])
    args.func(args)
    assert json.loads(capsys.readouterr().out)["ok"] is True

    args = _parser().parse_args([*common, "search", "timeout", "--json"])
    args.func(args)
    result = json.loads(capsys.readouterr().out)
    assert result["records"][0]["record_id"] == "cli-1"


def test_cli_rejects_record_outside_configured_vault(tmp_path: Path, capsys) -> None:
    vault = tmp_path / "vault"
    index = tmp_path / "index.sqlite3"
    source = tmp_path / "outside.md"
    source.write_text(
        """---
record_id: outside
schema_name: agent_engineering_record_v1
sync_owned: true
classification: diagnostic-only
authority: source-index
status: candidate
title: Outside
summary: Outside source.
agent: codex
repository: repo
memory_area: runtime
task_type: diagnosis
component: runner
observed_at: '2026-09-21T00:00:00Z'
source_label: receipt
source_digest: outside
---

Evidence.
""",
        encoding="utf-8",
    )

    args = _parser().parse_args(
        ["engineering-memory", "--vault", str(vault), "--index", str(index), "propose", str(source), "--json"]
    )
    args.func(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "outside configured vault" in payload["error"]


def test_laya_doctor_is_explicit_and_read_only(capsys) -> None:
    args = _parser().parse_args(["engineering-memory", "laya", "doctor", "--json"])
    args.func(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "laya.doctor"
    assert payload["ok"] is False
    assert payload["diagnostics"] == ["laya_unavailable"]
