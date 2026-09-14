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


def test_skills_tap_refresh_subparser():
    """Verify 'hermes skills tap refresh [repo]' parses both specific and omitted repo."""
    from hermes_cli.main import build_skills_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_skills_parser(subparsers, cmd_skills=lambda a: None)

    # Specific tap repo
    args = parser.parse_args(["skills", "tap", "refresh", "callacat/hermes-capabilities"])
    assert args.command == "skills"
    assert args.skills_action == "tap"
    assert args.tap_action == "refresh"
    assert args.repo == "callacat/hermes-capabilities"

    # All taps (repo omitted)
    args_all = parser.parse_args(["skills", "tap", "refresh"])
    assert args_all.command == "skills"
    assert args_all.skills_action == "tap"
    assert args_all.tap_action == "refresh"
    assert args_all.repo == ""
