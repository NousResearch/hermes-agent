from __future__ import annotations

import json
import os
from argparse import ArgumentParser


def test_normalize_receipt_caps_command_output_and_drops_raw_payloads():
    from hermes_cli.agent_work_wiki import normalize_receipt

    receipt = normalize_receipt(
        {
            "goal": "Fix auth tests",
            "changed_files": ["src/auth.py"],
            "raw_log": "drop-me",
            "commands_run": [
                {
                    "command": "pytest tests/auth -q",
                    "exit_code": 1,
                    "stdout": "A" * 500,
                    "stderr": "B" * 500,
                    "summary": "1 failed",
                }
            ],
            "next_prompt": "continue from evidence",
        },
        max_excerpt_chars=96,
    )

    assert "raw_log" not in receipt
    command = receipt["commands_run"][0]
    assert "stdout" not in command
    assert "stderr" not in command
    assert command["command"] == "pytest tests/auth -q"
    assert command["exit_code"] == 1
    assert command["summary"] == "1 failed"
    assert len(command["excerpt"]) <= 96
    assert "truncated" in command["excerpt"]


def test_write_work_wiki_bundle_creates_bounded_receipt_and_wiki_layers(tmp_path):
    from hermes_cli.agent_work_wiki import write_work_wiki_bundle

    result = write_work_wiki_bundle(
        tmp_path,
        {
            "goal": "Fix Auth Tests",
            "changed_files": ["src/auth.py", "tests/test_auth.py"],
            "commands_run": [
                {
                    "command": "pytest tests/test_auth.py -q",
                    "exit_code": 0,
                    "summary": "2 passed",
                    "stdout": "PASS\n" * 100,
                }
            ],
            "open_risks": ["integration path not checked"],
            "next_prompt": "inspect src/auth.py then run integration check",
        },
        now="2026-06-21T00:00:00Z",
        max_excerpt_chars=80,
    )

    assert result.receipt_path == tmp_path / ".hermes/work/receipts/20260621T000000Z-fix-auth-tests.json"
    assert result.handoff_path == tmp_path / ".hermes/wiki/handoffs/20260621T000000Z-fix-auth-tests.md"
    assert result.schema_path == tmp_path / ".hermes/wiki/SCHEMA.md"
    assert result.index_path == tmp_path / ".hermes/wiki/index.md"
    assert result.log_path == tmp_path / ".hermes/wiki/log.md"

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["goal"] == "Fix Auth Tests"
    assert receipt["changed_files"] == ["src/auth.py", "tests/test_auth.py"]
    assert len(receipt["commands_run"][0]["excerpt"]) <= 80

    handoff = result.handoff_path.read_text(encoding="utf-8")
    assert "```json" in handoff
    assert "Fix Auth Tests" in handoff
    assert "integration path not checked" in handoff

    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert ".hermes/work/tmp/" in gitignore
    assert ".hermes/work/receipts/" in gitignore


def test_prune_work_artifacts_deletes_old_runtime_files_but_preserves_wiki(tmp_path):
    from hermes_cli.agent_work_wiki import prune_work_artifacts

    tmp_dir = tmp_path / ".hermes/work/tmp"
    receipts_dir = tmp_path / ".hermes/work/receipts"
    wiki_dir = tmp_path / ".hermes/wiki/handoffs"
    tmp_dir.mkdir(parents=True)
    receipts_dir.mkdir(parents=True)
    wiki_dir.mkdir(parents=True)

    old_tmp = tmp_dir / "old.log"
    old_receipt = receipts_dir / "old.json"
    newest_receipt = receipts_dir / "new.json"
    wiki_page = wiki_dir / "keep.md"
    for path in (old_tmp, old_receipt, newest_receipt, wiki_page):
        path.write_text("x", encoding="utf-8")

    os.utime(old_tmp, (100, 100))
    os.utime(old_receipt, (100, 100))
    os.utime(newest_receipt, (900, 900))
    os.utime(wiki_page, (100, 100))

    result = prune_work_artifacts(
        tmp_path,
        now_timestamp=1000,
        max_age_seconds=60,
        keep_latest_receipts=1,
    )

    assert not old_tmp.exists()
    assert not old_receipt.exists()
    assert newest_receipt.exists()
    assert wiki_page.exists()
    assert result.deleted_count == 2
    assert result.bytes_deleted == 2


def test_work_wiki_cli_record_emits_machine_readable_paths(tmp_path, capsys):
    from hermes_cli.agent_work_wiki import agent_work_wiki_command, build_parser

    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(
        json.dumps(
            {
                "goal": "CLI Receipt",
                "commands_run": [{"command": "pytest -q", "exit_code": 0, "summary": "ok"}],
                "next_prompt": "continue",
            }
        ),
        encoding="utf-8",
    )

    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    work_parser = build_parser(sub)
    work_parser.set_defaults(func=agent_work_wiki_command)
    args = parser.parse_args(
        [
            "work-wiki",
            "record",
            "--root",
            str(tmp_path),
            "--receipt-file",
            str(receipt_file),
            "--now",
            "2026-06-21T00:00:00Z",
            "--json",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["receipt_path"].endswith(".hermes/work/receipts/20260621T000000Z-cli-receipt.json")
    assert payload["handoff_path"].endswith(".hermes/wiki/handoffs/20260621T000000Z-cli-receipt.md")


def test_work_wiki_cli_gc_emits_deleted_count(tmp_path, capsys):
    from hermes_cli.agent_work_wiki import agent_work_wiki_command, build_parser

    old_dir = tmp_path / ".hermes/work/tmp"
    old_dir.mkdir(parents=True)
    old_file = old_dir / "old.log"
    old_file.write_text("xx", encoding="utf-8")
    os.utime(old_file, (100, 100))

    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    work_parser = build_parser(sub)
    work_parser.set_defaults(func=agent_work_wiki_command)
    args = parser.parse_args(
        [
            "work-wiki",
            "gc",
            "--root",
            str(tmp_path),
            "--now-timestamp",
            "1000",
            "--max-age-seconds",
            "60",
            "--json",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["deleted_count"] == 1
    assert payload["bytes_deleted"] == 2


def test_work_wiki_is_registered_as_builtin_cli_subcommand():
    import hermes_cli.main as main_mod

    assert "work-wiki" in main_mod._BUILTIN_SUBCOMMANDS
    assert hasattr(main_mod, "cmd_work_wiki")
