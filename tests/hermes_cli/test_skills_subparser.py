"""Test that skills subparser doesn't conflict (regression test for #898)."""

import argparse


def test_no_duplicate_skills_subparser():
    """Ensure 'skills' subparser is only registered once to avoid Python 3.11+ crash.

    Python 3.11 changed argparse to raise an exception on duplicate subparser
    names instead of silently overwriting (see CPython #94331).

    This test will fail with:
        argparse.ArgumentError: argument command: conflicting subparser: skills

    if the duplicate 'skills' registration is reintroduced.
    """
    # Force fresh import of the module where parser is constructed
    # If there are duplicate 'skills' subparsers, this import will raise
    # argparse.ArgumentError at module load time
    import sys

    # Remove cached module if present
    if 'hermes_cli.main' in sys.modules:
        del sys.modules['hermes_cli.main']

    try:
        import hermes_cli.main  # noqa: F401
    except argparse.ArgumentError as e:
        if "conflicting subparser" in str(e):
            raise AssertionError(
                f"Duplicate subparser detected: {e}. "
                "See issue #898 for details."
            ) from e
        raise


def test_skills_lint_parses_with_defaults():
    """`hermes skills lint` argparse wiring: defaults and flags round-trip."""
    import argparse as _argparse

    from hermes_cli.subcommands.skills import build_skills_parser

    parser = _argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_skills_parser(subparsers, cmd_skills=lambda args: None)

    args = parser.parse_args(["skills", "lint"])
    assert args.skills_action == "lint"
    assert args.targets == []
    assert args.all_skills is False
    assert args.json is False
    assert args.fail_on == "error"

    args = parser.parse_args(
        ["skills", "lint", "my-skill", "--json", "--fail-on", "warning"]
    )
    assert args.targets == ["my-skill"]
    assert args.json is True
    assert args.fail_on == "warning"
