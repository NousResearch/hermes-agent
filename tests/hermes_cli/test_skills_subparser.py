"""Test that skills subparser doesn't conflict (regression test for #898)."""

import argparse


def test_no_duplicate_skills_subparser(monkeypatch):
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

    # Drop the cached module so the import below really re-executes -- but
    # put BOTH bindings back afterwards. A bare `del sys.modules[...]` leaves
    # a second hermes_cli.main object installed for the rest of the session,
    # while every module that already did `from hermes_cli import main as
    # cli_main` still holds the first one. Their patch.object(cli_main, ...)
    # then patches an orphan and never reaches the module under test: that is
    # how test_update_orphan_backend_reap.py lost its PROJECT_ROOT patch and
    # let _cmd_update_impl run git checkout/merge/stash against the real
    # checkout, stashing the developer's uncommitted work mid-run.
    #
    # `from hermes_cli import main` resolves through the PACKAGE ATTRIBUTE,
    # which the re-import rebinds too -- so restoring sys.modules alone is not
    # enough. monkeypatch records the current value of each and restores both
    # at teardown.
    import hermes_cli

    cached_main = sys.modules.get('hermes_cli.main')
    if cached_main is not None:
        monkeypatch.setattr(hermes_cli, 'main', cached_main, raising=False)
    monkeypatch.delitem(sys.modules, 'hermes_cli.main', raising=False)

    try:
        import hermes_cli.main  # noqa: F401
    except argparse.ArgumentError as e:
        if "conflicting subparser" in str(e):
            raise AssertionError(
                f"Duplicate subparser detected: {e}. "
                "See issue #898 for details."
            ) from e
        raise
