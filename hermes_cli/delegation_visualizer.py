"""CLI subcommand: ``hermes delegation <subcommand>``.

A thin wrapper around the standalone ``hermes_delegation`` package (the M1+M2
delegation-visualizer). It exposes the three subcommands that make sense inside
a Hermes session — ``status``, ``verify``, and ``report`` — and delegates the
actual work to the M1+M2 public API:

* ``status``  → check whether the verifier daemon is listening on its Unix
                socket (uses ``hermes_delegation.cli._daemon_is_listening``).
* ``verify``  → re-derive the authoritative snapshot from the ledger with
                ``hermes_delegation.compute_snapshot`` and print it as JSON.
* ``report``  → same snapshot, rendered to Markdown via
                ``hermes_delegation.report.render_report``.

``watch`` and ``dashboard`` stay standalone-only (the live TUI is awkward in an
interactive session); point users at ``hermes-delegation dashboard <task>`` in a
separate terminal for the live view.

This module mirrors ``hermes_cli/curator.py``: no side effects at import time,
``register_cli(parent)`` wires the argparse subparsers, and ``cli_main(argv)`` is
a standalone entry point. If the ``hermes_delegation`` package is not installed,
every subcommand fails gracefully with an install hint rather than crashing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

try:
    import hermes_delegation as hd  # noqa: F401

    HD_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via _require_hd in tests
    HD_AVAILABLE = False


def _hermes_agent_root() -> Path | None:
    """Best-effort locate the hermes-agent checkout root at call time.

    The ``hermes_cli`` package lives directly under the hermes-agent root, so
    the root is the parent of the ``hermes_cli`` package directory. Returns
    ``None`` when the package has no on-disk origin (e.g. a namespace package,
    a zipimport, or a PyPI wheel without an editable checkout) — in that case
    the caller falls back to the PyPI install command.
    """
    try:
        spec = importlib.util.find_spec("hermes_cli")
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return None
    if spec is None or not spec.origin:
        return None
    # spec.origin -> .../hermes-agent/hermes_cli/__init__.py
    # parent       -> .../hermes-agent/hermes_cli
    # parent.parent-> .../hermes-agent  (the checkout root)
    return Path(spec.origin).resolve().parent.parent


def _install_hint() -> str:
    """Build the install hint for the missing ``hermes_delegation`` package.

    Computed at call time (not frozen as a module-level constant) so it adapts
    to wherever hermes-agent actually lives. When a hermes-agent checkout is
    found on disk, point at the editable optional-extra install from that root;
    otherwise (PyPI / wheel installs) point at the published distribution.
    The package name and the Python 3.11+ requirement are always mentioned.
    """
    root = _hermes_agent_root()
    if root is not None:
        install_cmd = (
            f"from the hermes-agent root ({root}) run "
            '`pip install -e ".[delegation]"`'
        )
    else:  # pragma: no cover - PyPI/wheel path, no editable checkout
        install_cmd = (
            'install the optional extra with `pip install "hermes-agent[delegation]"` '
            '(or, from a checkout, `pip install -e ".[delegation]"`)'
        )
    return (
        "delegation: the `hermes_delegation` package is not installed or the "
        "wrong Python version. To enable the delegation visualizer "
        f"(hermes-delegation), {install_cmd}. Requires Python 3.11+."
    )

# Defaults mirror the M1+M2 package (see hermes_delegation.cli).
_DEFAULT_BASE_DIR = Path.home() / ".hermes" / "delegation" / "ledgers"
_DEFAULT_SOCKET_PATH = Path.home() / ".hermes" / "delegation" / "verifier.sock"


def _require_hd() -> bool:
    """Print the install hint and return False when the package is missing."""
    if not HD_AVAILABLE:
        print(_install_hint(), file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_status(args) -> int:
    """Report whether the verifier daemon is running.

    Returns 0 when the daemon is listening, 1 otherwise (so the exit code is
    usable as a health check in scripts / `&&` chains).
    """
    if not _require_hd():
        return 1

    # NOTE: ``_daemon_is_listening`` is a private (leading-underscore) symbol in
    # the M1+M2 ``hermes_delegation.cli`` module, so it carries the usual risk of
    # a private API: a future refactor could rename or remove it without a
    # deprecation cycle. We rely on it because M1+M2 exposes no public liveness
    # check for the verifier daemon — there is no public alternative today. The
    # try/except below turns a vanished symbol into an actionable "re-install"
    # message instead of a raw, unexplained ImportError traceback.
    try:
        from hermes_delegation.cli import _daemon_is_listening
    except ImportError:
        print(
            "delegation: the M1+M2 `hermes_delegation` API has changed "
            "(`hermes_delegation.cli._daemon_is_listening` is gone); please "
            "re-install the delegation-visualizer package. " + _install_hint(),
            file=sys.stderr,
        )
        return 1

    socket_path = Path(args.socket_path).expanduser()
    if _daemon_is_listening(socket_path):
        print("daemon: running")
        print(f"  socket: {socket_path}")
        return 0

    print("daemon: not running")
    print(f"  socket: {socket_path}")
    print("  hint:   start it with `hermes-delegation watch &`")
    return 1


def _cmd_verify(args) -> int:
    """Re-derive the snapshot for a task from its ledger and print it as JSON.

    Pure: reads ``<base-dir>/<task_id>.jsonl`` and runs ``compute_snapshot`` —
    no daemon required. Returns 1 if the ledger is missing or malformed.
    """
    if not _require_hd():
        return 1

    from hermes_delegation import compute_snapshot

    base_dir = Path(args.base_dir).expanduser()
    ledger_path = base_dir / f"{args.task_id}.jsonl"

    if not ledger_path.exists():
        print(
            f"delegation: no ledger found for task {args.task_id!r} at {ledger_path}",
            file=sys.stderr,
        )
        return 1

    try:
        snapshot = compute_snapshot(ledger_path)
    except Exception as exc:  # empty/malformed ledger, validation error, ...
        print(
            f"delegation: could not compute snapshot for {args.task_id!r}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def _cmd_report(args) -> int:
    """Render the Markdown report for a task to a file or stdout.

    Computes the snapshot the same way ``verify`` does, then renders the §10
    Markdown report. Writes to ``--output`` when given, otherwise stdout.
    """
    if not _require_hd():
        return 1

    from hermes_delegation import compute_snapshot
    from hermes_delegation.report import render_report

    base_dir = Path(args.base_dir).expanduser()
    ledger_path = base_dir / f"{args.task_id}.jsonl"

    if not ledger_path.exists():
        print(
            f"delegation: no ledger found for task {args.task_id!r} at {ledger_path}",
            file=sys.stderr,
        )
        return 1

    try:
        snapshot = compute_snapshot(ledger_path)
    except Exception as exc:
        print(
            f"delegation: could not compute snapshot for {args.task_id!r}: {exc}",
            file=sys.stderr,
        )
        return 1

    markdown = render_report(snapshot)

    output = getattr(args, "output", None)
    if output:
        out_path = Path(output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"delegation: wrote report to {out_path}")
    else:
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring (called from hermes_cli.main / cli.py)
# ---------------------------------------------------------------------------

def _add_base_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-dir",
        default=str(_DEFAULT_BASE_DIR),
        help="directory holding <task_id>.jsonl ledgers "
        "(default: ~/.hermes/delegation/ledgers)",
    )


def _add_socket_path(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--socket-path",
        default=str(_DEFAULT_SOCKET_PATH),
        help="path to the verifier daemon's Unix socket "
        "(default: ~/.hermes/delegation/verifier.sock)",
    )


def register_cli(parent: argparse.ArgumentParser) -> None:
    """Attach ``delegation`` subcommands to *parent*.

    Mirrors ``hermes_cli.curator.register_cli``: the caller passes the
    ArgumentParser returned by ``subparsers.add_parser("delegation", ...)``.
    """
    parent.set_defaults(func=lambda a: (parent.print_help(), 0)[1])
    subs = parent.add_subparsers(dest="delegation_command")

    p_status = subs.add_parser(
        "status", help="Show whether the verifier daemon is running"
    )
    _add_socket_path(p_status)
    p_status.set_defaults(func=_cmd_status)

    p_verify = subs.add_parser(
        "verify",
        help="Re-derive a task's snapshot from its ledger and print JSON",
    )
    p_verify.add_argument("task_id", help="task id to verify")
    _add_base_dir(p_verify)
    p_verify.set_defaults(func=_cmd_verify)

    p_report = subs.add_parser(
        "report",
        help="Render the Markdown delegation report for a task",
    )
    p_report.add_argument("task_id", help="task id to report on")
    _add_base_dir(p_report)
    p_report.add_argument(
        "--output",
        default=None,
        help="path to write the Markdown report (default: stdout)",
    )
    p_report.set_defaults(func=_cmd_report)


def cli_main(argv=None) -> int:
    """Standalone entry (also usable by cli.py's slash-command dispatch).

    Builds the parser, dispatches to the matching handler, and returns the
    handler's exit code. With no subcommand, prints help and returns 0.
    """
    parser = argparse.ArgumentParser(prog="hermes delegation")
    register_cli(parser)
    args = parser.parse_args(argv)
    fn = getattr(args, "func", None)
    if fn is None:
        parser.print_help()
        return 0
    return int(fn(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli_main())
