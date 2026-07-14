"""Test that skills subparser doesn't conflict (regression test for #898)."""

import argparse


def test_skills_parser_accepts_inventory_and_all_active_audit():
    from hermes_cli.subcommands.skills import build_skills_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_skills_parser(subparsers, cmd_skills=lambda _args: None)

    audit_args = parser.parse_args(["skills", "audit", "--all-active", "--deep"])
    assert audit_args.skills_action == "audit"
    assert audit_args.all_active is True
    assert audit_args.deep is True

    inventory_args = parser.parse_args(["skills", "inventory", "--json"])
    assert inventory_args.skills_action == "inventory"
    assert inventory_args.json is True


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
