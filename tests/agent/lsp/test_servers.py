"""Tests for language-server definitions and project-root resolution."""
from __future__ import annotations

from pathlib import Path

from agent.lsp.servers import _root_rust


def test_root_rust_uses_enclosing_cargo_workspace(tmp_path: Path):
    repo = tmp_path / "repo"
    workspace = repo / "native"
    workspace.mkdir(parents=True)
    (workspace / "Cargo.toml").write_text(
        '[workspace]\nmembers = ["crates/cli", "crates/engine"]\n',
        encoding="utf-8",
    )

    roots = set()
    for name in ("cli", "engine"):
        member = workspace / "crates" / name
        source = member / "src" / "main.rs"
        source.parent.mkdir(parents=True)
        source.write_text("fn main() {}\n", encoding="utf-8")
        (member / "Cargo.toml").write_text(
            f'[package]\nname = "{name}"\nversion = "0.1.0"\n',
            encoding="utf-8",
        )
        roots.add(_root_rust(str(source), str(repo)))

    assert roots == {str(workspace)}


def test_root_rust_uses_nearest_enclosing_workspace(tmp_path: Path):
    repo = tmp_path / "repo"
    nested = repo / "independent"
    member = nested / "crates" / "app"
    source = member / "src" / "main.rs"
    source.parent.mkdir(parents=True)
    source.write_text("fn main() {}\n", encoding="utf-8")
    (repo / "Cargo.toml").write_text(
        '[workspace]\nexclude = ["independent"]\n',
        encoding="utf-8",
    )
    (nested / "Cargo.toml").write_text(
        '[workspace]\nmembers = ["crates/app"]\n',
        encoding="utf-8",
    )
    (member / "Cargo.toml").write_text(
        '[package]\nname = "app"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    assert _root_rust(str(source), str(repo)) == str(nested)


def test_root_rust_falls_back_within_supplied_workspace(tmp_path: Path):
    outer = tmp_path / "outer"
    repo = outer / "repo"
    member = repo / "crate"
    source = member / "src" / "lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    (outer / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    (member / "Cargo.toml").write_text(
        '[package]\nname = "standalone"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    assert _root_rust(str(source), str(repo)) == str(member)
    assert _root_rust(str(repo / "untracked.rs"), str(repo)) == str(repo)
