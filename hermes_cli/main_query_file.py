"""``--query-file``: the single-query transport that never passes through a shell.

Split out of ``hermes_cli/main.py``.
"""

import sys


def _read_query_file(args) -> None:
    """--query-file: read the single query from a file (or stdin via '-').

    Callers never have to shell-quote message bodies; this is the transport
    the Bot Mode DM protocol uses; interpolating arbitrary text into a
    double-quoted shell argument truncates on quotes and executes $(...)
    (see tools/bot_mode_probe.py).
    """
    _qfile = getattr(args, "query_file", None)
    if not _qfile:
        return
    if args.query:
        # argparse's mutually-exclusive group catches the normal CLI path;
        # this guards programmatic callers that fill the namespace directly.
        print("Error: -q/--query and --query-file are mutually exclusive", file=sys.stderr)
        sys.exit(2)
    try:
        if _qfile == "-":
            args.query = sys.stdin.read()
        else:
            with open(_qfile, "r", encoding="utf-8-sig", errors="replace") as _fh:
                args.query = _fh.read()
    except OSError as _e:
        print(f"Error: cannot read --query-file {_qfile}: {_e}", file=sys.stderr)
        sys.exit(2)
    if not (args.query or "").strip():
        print(f"Error: --query-file {_qfile} is empty", file=sys.stderr)
        sys.exit(2)


def resolve_oneshot_prompt(args) -> bool:
    """Resolve top-level ``-Q/--query-file/--prompt-file`` or ``-z -`` into ``args.oneshot``.

    Returns True when ``args`` triggers top-level one-shot execution, False otherwise.
    Raises ValueError on mutually-exclusive, unreadable, or empty prompt inputs.
    """
    if getattr(args, "_oneshot_resolved", False):
        return bool(getattr(args, "oneshot", None))

    qfile = getattr(args, "oneshot_query_file", None)
    oneshot = getattr(args, "oneshot", None)
    if qfile and oneshot:
        raise ValueError("-z/--oneshot and -Q/--query-file are mutually exclusive")

    if qfile:
        try:
            if qfile == "-":
                text = sys.stdin.read()
            else:
                with open(qfile, "r", encoding="utf-8-sig", errors="replace") as fh:
                    text = fh.read()
        except OSError as exc:
            raise ValueError(f"cannot read --query-file {qfile}: {exc}") from exc
        if not (text or "").strip():
            raise ValueError(f"--query-file {qfile} is empty")
        args.oneshot = text
        args.oneshot_query_file = None
        args._oneshot_resolved = True
        return True

    if isinstance(oneshot, str) and oneshot == "-":
        text = sys.stdin.read()
        if not (text or "").strip():
            raise ValueError("stdin prompt for -z/--oneshot is empty")
        args.oneshot = text
        args._oneshot_resolved = True
        return True

    if oneshot:
        args._oneshot_resolved = True
        return True

    return False


def resolve_oneshot_args_or_exit(args) -> bool:
    """CLI wrapper around :func:`resolve_oneshot_prompt` that prints ``Error:`` and exits 2."""
    try:
        return resolve_oneshot_prompt(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
