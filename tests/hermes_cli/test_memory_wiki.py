"""Tests for the `hermes memory wiki-index` CLI helper."""

from __future__ import annotations

import argparse
import json
import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_memory_wiki_index_cli_writes_json(tmp_path, capsys):
    home = tmp_path / ".hermes"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("Hermes uses pytest for tests", encoding="utf-8")

    token = set_hermes_home_override(home)
    try:
        from hermes_cli.memory_wiki import run_memory_wiki_index

        out = tmp_path / "index.json"
        rc = run_memory_wiki_index(argparse.Namespace(out=str(out), query=None, max_chars=1200))
    finally:
        reset_hermes_home_override(token)

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["stats"]["entries"] == 1
    assert payload["entries"][0]["text"] == "Hermes uses pytest for tests"
    assert "Wrote memory wiki index" in capsys.readouterr().out


def test_memory_wiki_index_cli_query_prints_selection(tmp_path, capsys):
    home = tmp_path / ".hermes"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text(
        "Hermes uses pytest for tests\n§\nCloudflare uses proxied DNS records",
        encoding="utf-8",
    )

    token = set_hermes_home_override(home)
    try:
        from hermes_cli.memory_wiki import run_memory_wiki_index

        rc = run_memory_wiki_index(argparse.Namespace(out=None, query="pytest", max_chars=200))
    finally:
        reset_hermes_home_override(token)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selection"]["entries"]
    context = payload["selection"]["context"]
    assert context.startswith("<memory-wiki-context>\n")
    assert context.endswith("\n</memory-wiki-context>")
    assert "pytest" in context.lower()
    assert "Cloudflare" not in context


def test_memory_wiki_index_parser_routes_command_and_rejects_bad_budget():
    import argparse

    from hermes_cli.subcommands.memory import build_memory_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    sentinel = object()
    build_memory_parser(subparsers, cmd_memory=sentinel)

    args = parser.parse_args(["memory", "wiki-index", "--query", "pytest", "--max-chars", "200"])
    assert args.func is sentinel
    assert args.memory_command == "wiki-index"
    assert args.query == "pytest"
    assert args.max_chars == 200

    with pytest.raises(SystemExit):
        parser.parse_args(["memory", "wiki-index", "--max-chars", "0"])
