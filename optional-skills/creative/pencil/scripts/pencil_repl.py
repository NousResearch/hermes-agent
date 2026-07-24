#!/usr/bin/env python3
"""Drive Pencil's interactive MCP-tool shell non-interactively.

Pencil (pencil.dev) ships ``pencil interactive`` — a REPL that calls its MCP
design tools directly on a ``.pen`` file (``batch_design``, ``batch_get``,
``get_screenshot``, ``snapshot_layout``, ``get_editor_state``,
``get_variables``, ``export_nodes`` …). This wrapper feeds a fixed sequence of
tool-call lines to that REPL over stdin, appends ``save()``/``exit()`` in
headless mode, and returns the captured output. It lets an agent perform
deterministic design operations WITHOUT an LLM/agent API key and WITHOUT an MCP
bridge — the agent supplies the intent, Pencil executes the tool calls.

The ``pencil`` binary is a *lazy* runtime dependency: install it only if you
use this skill, with ``npm install -g @pencil.dev/cli``. Auth is required even
headless — run ``pencil_doctor.py`` first.

Examples
--------
Headless edit of a new/existing file::

    python pencil_repl.py --out design.pen \\
        --cmd 'get_editor_state({ include_schema: true })' \\
        --cmd 'batch_design({ operations: "hero=I(document,{type:\\"frame\\",width:1440,height:900,fill:\\"#0A0A0A\\"})" })'

Drive a running desktop app (changes apply live, no save() needed)::

    python pencil_repl.py --app desktop --in design.pen \\
        --cmd 'batch_get({ patterns: [{ reusable: true }] })'

Read the command list from a file (one REPL call per line)::

    python pencil_repl.py --out design.pen --cmds-file ops.txt
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from typing import List, Optional, Sequence


PENCIL_INSTALL_HINT = (
    "`pencil` not found on PATH. Install the Pencil CLI:\n"
    "  npm install -g @pencil.dev/cli\n"
    "then authenticate with `pencil login` (or set PENCIL_CLI_KEY)."
)


def build_pencil_argv(
    *,
    pencil_bin: str = "pencil",
    out: Optional[str] = None,
    in_path: Optional[str] = None,
    app: Optional[str] = None,
) -> List[str]:
    """Assemble the ``pencil interactive`` argv.

    App mode connects to a running Pencil instance; headless mode requires an
    output path. Raises ``ValueError`` when neither ``app`` nor ``out`` is set,
    matching the CLI's own requirement.
    """
    if not app and not out:
        raise ValueError("headless mode requires --out (or use --app <name>)")
    argv: List[str] = [pencil_bin, "interactive"]
    if app:
        argv += ["--app", app]
    if in_path:
        argv += ["--in", in_path]
    if out:
        argv += ["--out", out]
    return argv


def build_stdin(commands: Sequence[str], *, save: bool = True) -> str:
    """Build the REPL stdin payload: the tool-call lines, an optional
    ``save()`` (headless only), then ``exit()`` so the shell terminates."""
    lines: List[str] = [c.strip() for c in commands if c.strip()]
    if save:
        lines.append("save()")
    lines.append("exit()")
    return "\n".join(lines) + "\n"


def run_repl(
    commands: Sequence[str],
    *,
    pencil_bin: str = "pencil",
    out: Optional[str] = None,
    in_path: Optional[str] = None,
    app: Optional[str] = None,
    save: bool = True,
    timeout: int = 180,
    _run=subprocess.run,
    _which=shutil.which,
) -> subprocess.CompletedProcess:
    """Run the interactive shell with the given REPL command lines piped in.

    ``_run`` / ``_which`` are injectable for testing. In app mode ``save`` is
    forced off because changes apply live.
    """
    if _which(pencil_bin) is None:
        raise FileNotFoundError(PENCIL_INSTALL_HINT)
    argv = build_pencil_argv(
        pencil_bin=pencil_bin, out=out, in_path=in_path, app=app
    )
    stdin_payload = build_stdin(commands, save=save and not app)
    return _run(
        argv,
        input=stdin_payload,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Drive `pencil interactive` non-interactively over stdin.",
    )
    p.add_argument("--out", "-o", help="Output .pen file (required in headless mode)")
    p.add_argument("--in", "-i", dest="in_path", help="Input .pen file (optional)")
    p.add_argument("--app", "-a", help="Connect to a running Pencil app, e.g. 'desktop'")
    p.add_argument(
        "--cmd",
        action="append",
        default=[],
        help="A REPL tool-call line, e.g. 'batch_get()'. Repeatable.",
    )
    p.add_argument("--cmds-file", help="File with one REPL call per line")
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not append save() (headless mode only)",
    )
    p.add_argument("--pencil-bin", default="pencil", help="Path to the pencil binary")
    p.add_argument("--timeout", type=int, default=180, help="Seconds before abort")
    args = p.parse_args(argv)

    commands: List[str] = list(args.cmd)
    if args.cmds_file:
        with open(args.cmds_file, "r", encoding="utf-8") as f:
            commands += f.read().splitlines()
    if not commands:
        print("No REPL commands given (use --cmd or --cmds-file).", file=sys.stderr)
        return 2

    try:
        proc = run_repl(
            commands,
            pencil_bin=args.pencil_bin,
            out=args.out,
            in_path=args.in_path,
            app=args.app,
            save=not args.no_save,
            timeout=args.timeout,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 127
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print(f"pencil interactive timed out after {args.timeout}s", file=sys.stderr)
        return 124

    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
