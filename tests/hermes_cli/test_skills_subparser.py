"""Test that skills subparser doesn't conflict (regression test for #898)."""

import argparse


def test_no_duplicate_skills_subparser(purged_hermes_modules):
    """Ensure 'skills' subparser is only registered once to avoid Python 3.11+ crash.

    Python 3.11 changed argparse to raise an exception on duplicate subparser
    names instead of silently overwriting (see CPython #94331).

    This test will fail with:
        argparse.ArgumentError: argument command: conflicting subparser: skills

    if the duplicate 'skills' registration is reintroduced.
    """
    # Force fresh import of the module where parser is constructed.
    # If there are duplicate 'skills' subparsers, this import will raise
    # argparse.ArgumentError at module load time.
    #
    # purged_hermes_modules evicts hermes_cli* before the test and restores
    # the original module objects on teardown, so the re-import below neither
    # leaks a discarded copy nor splits module identity for later tests
    # in the same pytest process (t_63128384 / t_19f4c777).
    try:
        import hermes_cli.main  # noqa: F401
    except argparse.ArgumentError as e:
        if "conflicting subparser" in str(e):
            raise AssertionError(
                f"Duplicate subparser detected: {e}. "
                "See issue #898 for details."
            ) from e
        raise
