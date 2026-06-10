"""Tests for the Unforgit memory provider plugin."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

from plugins.memory.unforgit import UnforgitMemoryProvider, _resolve_cli


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "memory-repo"
    (repo / ".unforgit").mkdir(parents=True)
    (repo / ".unforgit" / "unforgit.yaml").write_text("repoId: test\n", encoding="utf-8")
    return repo


def _make_cli(tmp_path: Path) -> Path:
    cli = tmp_path / "unforgit"
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o755)
    return cli


def test_resolve_cli_prefers_path_executable(tmp_path):
    cli = _make_cli(tmp_path)

    assert _resolve_cli(str(cli), tmp_path) == str(cli)


def test_resolve_cli_falls_back_to_hermes_scripts(tmp_path, monkeypatch):
    scripts_cli = tmp_path / "scripts" / "unforgit"
    scripts_cli.parent.mkdir()
    scripts_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    scripts_cli.chmod(0o755)
    monkeypatch.setattr("plugins.memory.unforgit.shutil.which", lambda _: None)

    assert _resolve_cli("unforgit", tmp_path) == str(scripts_cli)


def test_is_available_requires_cli_and_initialized_unforgit_repo(tmp_path):
    repo = _make_repo(tmp_path)
    cli = _make_cli(tmp_path)

    with patch(
        "plugins.memory.unforgit._load_config",
        return_value={"repo_path": str(repo), "cli_path": str(cli)},
    ), patch("plugins.memory.unforgit.get_hermes_home", return_value=tmp_path, create=True):
        provider = UnforgitMemoryProvider()
        assert provider.is_available()


def test_search_tool_invokes_unforgit_cli_with_local_only(tmp_path):
    repo = _make_repo(tmp_path)
    cli = _make_cli(tmp_path)

    with patch(
        "plugins.memory.unforgit._load_config",
        return_value={
            "repo_path": str(repo),
            "cli_path": str(cli),
            "recall_top_k": 3,
            "local_only": True,
            "mirror_builtin_writes": True,
            "sync_turns": False,
        },
    ), patch("plugins.memory.unforgit.get_hermes_home", return_value=tmp_path, create=True):
        provider = UnforgitMemoryProvider()
        provider.initialize("session-1")

        completed = Mock(
            returncode=0,
            stdout=json.dumps({"results": [{"text": "Hermes supports Unforgit", "type": "semantic"}]}),
            stderr="",
        )
        with patch("plugins.memory.unforgit.subprocess.run", return_value=completed) as run:
            result = json.loads(provider.handle_tool_call("unforgit_search", {"query": "Hermes", "top_k": 2}))

        assert result["results"][0]["text"] == "Hermes supports Unforgit"
        cmd = run.call_args.args[0]
        assert cmd == [str(cli), "--json", "recall", "Hermes", "-k", "2", "--local-only"]
        assert run.call_args.kwargs["cwd"] == str(repo)


def test_builtin_memory_writes_are_mirrored_when_enabled(tmp_path):
    repo = _make_repo(tmp_path)
    cli = _make_cli(tmp_path)

    with patch(
        "plugins.memory.unforgit._load_config",
        return_value={
            "repo_path": str(repo),
            "cli_path": str(cli),
            "recall_top_k": 5,
            "local_only": True,
            "mirror_builtin_writes": True,
            "sync_turns": False,
        },
    ), patch("plugins.memory.unforgit.get_hermes_home", return_value=tmp_path, create=True):
        provider = UnforgitMemoryProvider()
        provider.initialize("session-1")

        completed = Mock(returncode=0, stdout=json.dumps({"id": "mem-1"}), stderr="")
        with patch("plugins.memory.unforgit.subprocess.run", return_value=completed) as run:
            provider.on_memory_write("add", "memory", "Project uses Unforgit", {"platform": "telegram"})

        cmd = run.call_args.args[0]
        assert cmd == [
            str(cli),
            "--json",
            "add",
            "Project uses Unforgit",
            "--type",
            "semantic",
            "--tags",
            "hermes-memory,memory,telegram",
        ]
