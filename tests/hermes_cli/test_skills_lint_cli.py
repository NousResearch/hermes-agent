"""`hermes skills lint` — scope resolution, output shapes, and exit-status propagation.

The linter rules themselves are covered in tests/tools/test_skill_linter.py; this file covers the
CLI surface: which skills a call lints, that collection rules run only over a collection scope,
the --json payload, and that the int exit status survives the whole dispatch chain
(do_lint -> skills_command -> cmd_skills -> args.func) so CI can gate on it.
"""

import argparse
import json
from io import StringIO

import pytest
from rich.console import Console

from hermes_cli.skills_hub import do_lint, handle_skills_slash, skills_command

CLEAN = """---
name: {name}
description: {description}
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [x]
    related_skills: [{related}]
---

# {name}

## When to Use
- When the user wants X.

## Procedure
1. Use `read_file` to load it.
"""


def _skill(root, name, description="Search arXiv papers by keyword, author, or ID.", related="",
           frontmatter_name=None):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        CLEAN.format(name=frontmatter_name or name, description=description, related=related),
        encoding="utf-8")
    return skill_dir


def _console():
    sink = StringIO()
    return Console(file=sink, force_terminal=False, color_system=None, width=200), sink


@pytest.fixture()
def profile(tmp_path, monkeypatch):
    """A fake active profile: two clean skills, one with a dangling related_skills name."""
    import agent.skill_utils as skill_utils
    root = tmp_path / "profile"
    _skill(root, "clean-a")
    _skill(root, "clean-b", description="Extract text and tables from PDF files.")
    _skill(root, "dangling", description="Render Mermaid diagrams to SVG.", related="nonesuch")
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [root])
    monkeypatch.setattr(skill_utils, "get_project_skills_dirs", lambda: [])
    return root


def test_default_scope_is_the_whole_profile_with_collection_rules(profile):
    c, sink = _console()
    assert do_lint(console=c) == 0  # related-skill-missing is a warning
    out = sink.getvalue()
    assert "3 skill(s) linted" in out
    assert "related-skill-missing" in out and "'nonesuch'" in out


def test_explicit_target_by_name_skips_collection_rules(profile):
    c, sink = _console()
    assert do_lint(targets=["dangling"], console=c) == 0
    out = sink.getvalue()
    assert "1 skill(s) linted" in out
    assert "related-skill-missing" not in out


def test_explicit_target_by_path_and_skill_md(profile):
    c, sink = _console()
    assert do_lint(targets=[str(profile / "clean-a"), str(profile / "clean-b" / "SKILL.md")], console=c) == 0
    assert "2 skill(s) linted" in sink.getvalue()


def test_unknown_target_is_a_usage_error(profile):
    c, sink = _console()
    assert do_lint(targets=["not-installed"], console=c) == 2
    assert "not-installed" in sink.getvalue()


def test_all_with_targets_is_a_usage_error(profile):
    c, _ = _console()
    assert do_lint(targets=["clean-a"], all_skills=True, console=c) == 2


def test_dir_scope_walks_a_tree_and_runs_collection_rules(tmp_path):
    tree = tmp_path / "repo-skills"
    _skill(tree / "research", "arxiv")
    _skill(tree / "research", "paper-finder", description="Search arXiv papers by author, keyword, or ID.")
    c, sink = _console()
    assert do_lint(lint_dir=str(tree), console=c) == 0
    out = sink.getvalue()
    assert "2 skill(s) linted" in out and "alias-overlap" in out


def test_dir_scope_missing_directory_is_a_usage_error(tmp_path):
    c, _ = _console()
    assert do_lint(lint_dir=str(tmp_path / "nope"), console=c) == 2


def test_error_severity_finding_exits_1_and_strict_promotes_warnings(tmp_path):
    tree = tmp_path / "t"
    _skill(tree, "mismatch", frontmatter_name="other-name")  # name-dir-mismatch is an ERROR
    c, _ = _console()
    assert do_lint(lint_dir=str(tree), console=c) == 1
    warn_tree = tmp_path / "w"
    _skill(warn_tree, "warned", description="x" * 80)  # description-length is a WARNING
    assert do_lint(lint_dir=str(warn_tree), console=c) == 0
    assert do_lint(lint_dir=str(warn_tree), strict=True, console=c) == 1


def test_json_payload_shape(profile, capsys):
    c, sink = _console()
    assert do_lint(as_json=True, console=c) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"skills", "collection", "summary"}
    assert payload["summary"] == {"skills": 3, "errors": 0, "warnings": 1}
    assert [s["name"] for s in payload["skills"]] == ["clean-a", "clean-b", "dangling"]
    assert all(set(s) == {"name", "path", "findings"} for s in payload["skills"])
    assert payload["collection"][0]["rule"] == "related-skill-missing"
    assert payload["collection"][0]["skill"] == "dangling"
    assert sink.getvalue() == ""  # JSON mode prints nothing to the Rich console


def test_empty_collection_reports_zero_and_exits_0(tmp_path):
    (tmp_path / "empty").mkdir()
    c, sink = _console()
    assert do_lint(lint_dir=str(tmp_path / "empty"), console=c) == 0
    assert "0 skill(s) linted" in sink.getvalue()


# --- exit status propagation through the real dispatch chain ---

def _parse(argv):
    from hermes_cli.main_agent_cmds import cmd_skills
    from hermes_cli.subcommands.skills import build_skills_parser
    parser = argparse.ArgumentParser(prog="hermes")
    build_skills_parser(parser.add_subparsers(dest="command"), cmd_skills=cmd_skills)
    return parser.parse_args(argv)


def test_lint_parser_defaults():
    args = _parse(["skills", "lint"])
    assert args.skills_action == "lint"
    assert args.targets == [] and args.all_skills is False and args.lint_dir is None
    assert args.strict is False and args.json is False
    args = _parse(["skills", "lint", "a", "b", "--strict", "--json", "--dir", "x"])
    assert args.targets == ["a", "b"] and args.strict and args.json and args.lint_dir == "x"


def test_exit_status_reaches_args_func(tmp_path, monkeypatch):
    """`hermes skills lint` must return the lint status through cmd_skills so main() can
    sys.exit on it — a validate command whose status is discarded exits 0 with findings."""
    tree = tmp_path / "t"
    _skill(tree, "mismatch", frontmatter_name="other-name")
    monkeypatch.setattr("hermes_cli.skills_hub._console", _console()[0])
    args = _parse(["skills", "lint", "--dir", str(tree)])
    assert args.func(args) == 1
    assert skills_command(args) == 1
    ok_tree = tmp_path / "ok"
    _skill(ok_tree, "fine")
    args = _parse(["skills", "lint", "--dir", str(ok_tree)])
    assert args.func(args) == 0


def test_other_actions_still_return_none(monkeypatch):
    monkeypatch.setattr("hermes_cli.skills_hub.do_check", lambda **_: None)
    assert skills_command(argparse.Namespace(skills_action="check", name=None)) is None


def test_slash_lint_runs_over_dir(tmp_path):
    tree = tmp_path / "t"
    _skill(tree, "fine")
    c, sink = _console()
    handle_skills_slash(f"/skills lint --dir {tree}", console=c)
    assert "1 skill(s) linted: 0 error(s), 0 warning(s)" in sink.getvalue()
