from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from hermes_cli import hki_cmd
from hki.inventory import build_inventory, write_inventory
from hki.manifest import build_manifest, source_id_for, write_manifest
from hki.paths import find_git_worktree_root, resolve_scope
from hki.report import write_sources_report


def _by_path(inventory):
    return {item.relative_path: item for item in inventory.files}


def _run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    hki_parser = hki_cmd.build_parser(sub)
    hki_parser.set_defaults(func=hki_cmd.hki_command)
    args = parser.parse_args(["hki", *argv])
    return hki_cmd.hki_command(args)


def test_basic_inventory_creation_records_text_binary_and_hash(tmp_path):
    (tmp_path / "README.md").write_text("# Hello\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"abc\x00def")

    scope = resolve_scope(tmp_path)
    inventory = build_inventory(scope)
    path = write_inventory(inventory, scope)

    assert path == tmp_path / ".hermes" / "hki" / "inventory.json"
    assert path.exists()

    by_path = _by_path(inventory)
    assert by_path["README.md"].is_text is True
    assert by_path["README.md"].classification == "text"
    assert by_path["README.md"].suffix == ".md"
    assert by_path["README.md"].sha256

    assert by_path["blob.bin"].is_text is False
    assert by_path["blob.bin"].classification == "binary"
    assert by_path["blob.bin"].sha256


def test_resolve_scope_uses_git_root_when_cwd_is_repo_root(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    scope = resolve_scope(tmp_path)

    assert scope.cwd == tmp_path.resolve()
    assert scope.root == tmp_path.resolve()
    assert scope.inventory_path == tmp_path / ".hermes" / "hki" / "inventory.json"


def test_cli_from_nested_git_directory_writes_to_repo_root(tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    (nested / "module.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Repo\n", encoding="utf-8")

    assert _run_cli(["inventory", "--cwd", str(nested)]) == 0
    assert "Wrote HKI inventory" in capsys.readouterr().out

    root_inventory_path = tmp_path / ".hermes" / "hki" / "inventory.json"
    nested_inventory_path = nested / ".hermes" / "hki" / "inventory.json"
    assert root_inventory_path.exists()
    assert not nested_inventory_path.exists()

    inventory = json.loads(root_inventory_path.read_text(encoding="utf-8"))
    assert inventory["root"] == str(tmp_path.resolve())
    assert [item["relative_path"] for item in inventory["files"]] == [
        "README.md",
        "src/pkg/module.py",
    ]


def test_resolve_scope_outside_repo_falls_back_to_cwd(tmp_path):
    nested = tmp_path / "notes"
    nested.mkdir()

    scope = resolve_scope(nested)

    assert scope.cwd == nested.resolve()
    assert scope.root == nested.resolve()
    assert scope.inventory_path == nested / ".hermes" / "hki" / "inventory.json"


def test_find_git_worktree_root_accepts_git_file_marker(tmp_path):
    (tmp_path / ".git").write_text("gitdir: /tmp/example/worktrees/repo\n", encoding="utf-8")
    nested = tmp_path / "src"
    nested.mkdir()

    assert find_git_worktree_root(nested) == tmp_path.resolve()


def test_inventory_excludes_common_generated_and_secret_paths(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("TOKEN=secret\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("module.exports = 1\n", encoding="utf-8")
    (tmp_path / ".hermes" / "hki" / "cache").mkdir(parents=True)
    (tmp_path / ".hermes" / "hki" / "cache" / "x").write_text("cached\n", encoding="utf-8")

    inventory = build_inventory(resolve_scope(tmp_path))
    paths = {item.relative_path for item in inventory.files}

    assert paths == {"src/app.py"}
    assert inventory.skipped_by_reason["secret_env_file"] == 2
    assert inventory.skipped_by_reason["excluded_directory"] >= 2
    assert inventory.skipped_by_reason["hki_output"] == 1
    assert inventory.skipped_count >= 5


def test_manifest_ordering_and_source_ids_are_deterministic(tmp_path):
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")

    inventory = build_inventory(resolve_scope(tmp_path))
    manifest_one = build_manifest(inventory)
    manifest_two = build_manifest(inventory)

    paths = [source.relative_path for source in manifest_one.sources]
    assert paths == ["a.txt", "b.txt"]
    assert [source.source_id for source in manifest_one.sources] == [
        source.source_id for source in manifest_two.sources
    ]
    assert manifest_one.sources[0].source_id == source_id_for("a.txt")
    assert len(manifest_one.sources[0].source_id) == len("src_") + 20
    assert len({source.source_id for source in manifest_one.sources}) == 2


def test_sources_report_file_creation(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "docs" / "README.md").write_text("# Docs\n", encoding="utf-8")

    scope = resolve_scope(tmp_path)
    inventory = build_inventory(scope)
    write_inventory(inventory, scope)
    manifest = build_manifest(inventory, inventory_path=scope.inventory_path)
    write_manifest(manifest, scope)
    report_path = write_sources_report(manifest, scope)

    content = report_path.read_text(encoding="utf-8")
    assert report_path == tmp_path / ".hermes" / "hki" / "reports" / "sources.md"
    assert "# HKI Source Inventory Report" in content
    assert "not a semantic dossier yet" in content
    assert "Manifest: `.hermes/hki/manifest.json`" in content
    assert "Total skipped/excluded entries: 0" in content
    assert "`python`: 1" in content
    assert "`markdown`: 1" in content


def test_symlink_outside_root_is_skipped(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("outside\n", encoding="utf-8")

    link = root / "outside-link.txt"
    try:
        os.symlink(outside / "secret.txt", link)
    except (AttributeError, NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    inventory = build_inventory(resolve_scope(root))

    assert "outside-link.txt" not in {item.relative_path for item in inventory.files}
    assert inventory.skipped_by_reason["outside_root"] == 1


def test_cli_commands_write_expected_files(tmp_path, capsys):
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")

    assert _run_cli(["inventory", "--cwd", str(tmp_path)]) == 0
    assert "Wrote HKI inventory" in capsys.readouterr().out
    assert (tmp_path / ".hermes" / "hki" / "inventory.json").exists()

    assert _run_cli(["manifest", "--cwd", str(tmp_path)]) == 0
    manifest_path = tmp_path / ".hermes" / "hki" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["inventory_path"] == ".hermes/hki/inventory.json"
    assert [source["relative_path"] for source in manifest["sources"]] == ["note.txt"]

    assert _run_cli(["report", "sources", "--cwd", str(tmp_path)]) == 0
    assert (tmp_path / ".hermes" / "hki" / "reports" / "sources.md").exists()


def test_resolve_scope_rejects_file_cwd(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cwd is not a directory"):
        resolve_scope(Path(file_path))


def test_resolve_scope_rejects_missing_cwd(tmp_path):
    with pytest.raises(ValueError, match="cwd does not exist"):
        resolve_scope(tmp_path / "missing")
