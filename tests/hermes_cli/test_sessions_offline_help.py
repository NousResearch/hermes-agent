"""The gated store-rewrite commands must advertise the offline requirement in their help.

`hermes sessions optimize-storage` refuses while any other process holds state.db
(`hermes_state_holders.held_store_refusal`, #110054), but its help text used to claim it
"throttles so a running gateway stays responsive" — which reads as permission to run it
live. An operator (or an agent) reading only `--help` would start the rewrite underneath a
running gateway and get the refusal as a surprise. The same gap existed on `optimize` and
`prune`. `set-journal-mode` already states the requirement correctly; these now match it.

Runs everywhere: it inspects the argparse metadata, no holder scan needed (unlike
`test_sessions_held_store_gate.py`, which is posix-only).
"""

import argparse

from hermes_cli import sessions_cmd
from hermes_cli.subcommands.sessions import build_sessions_parser


def _sessions_action() -> argparse._SubParsersAction:
    parser = argparse.ArgumentParser()
    build_sessions_parser(parser.add_subparsers(dest="command"), cmd_sessions=sessions_cmd.cmd_sessions)
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    sessions = subparsers.choices["sessions"]
    return next(a for a in sessions._actions if isinstance(a, argparse._SubParsersAction))


def _advertised_help(action: str) -> str:
    """The text `hermes sessions --help` and `hermes sessions <action> --help` show.

    ``help=`` is stored on the parent's ``_choices_actions`` (the subcommand listing);
    ``description=`` is stored on the child parser itself.
    """
    nested = _sessions_action()
    sub = nested.choices[action]
    choice = next(a for a in nested._choices_actions if a.dest == action)
    return f"{choice.help or ''} {sub.description or ''}".lower()


def test_gated_actions_advertise_the_offline_requirement():
    for action in sorted(sessions_cmd._HELD_STORE_ACTIONS):
        text = _advertised_help(action)
        # `set-journal-mode`'s phrasing: "offline (every holder stopped)".
        assert "offline" in text or "holder stopped" in text, (
            f"`hermes sessions {action}` is gated on the store being quiet but its help "
            "does not say so — a live operator would hit the refusal as a surprise"
        )


def test_optimize_storage_does_not_claim_it_runs_live():
    description = _sessions_action().choices["optimize-storage"].description or ""
    assert "throttles so a running gateway stays responsive" not in description
    assert "refuses while any other process holds state.db" in description
