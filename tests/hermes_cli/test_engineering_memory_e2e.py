from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from hermes_cli.engineering_memory_commands import cmd_engineering_memory
from hermes_cli.subcommands.engineering_memory import build_engineering_memory_parser


def test_cli_e2e_keeps_result_envelope_bounded(tmp_path: Path, capsys) -> None:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="root")
    build_engineering_memory_parser(subparsers, cmd_engineering_memory=cmd_engineering_memory)
    vault = tmp_path / "vault"
    vault.mkdir()
    index = tmp_path / "index.sqlite3"

    args = parser.parse_args(["engineering-memory", "--vault", str(vault), "--index", str(index), "verify", "--json"])
    args.func(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["contract"] == "hermes.engineering_memory.v1"
    assert set(payload) >= {"contract", "operation", "ok", "health", "diagnostics"}
    assert len(capsys.readouterr().out) == 0
