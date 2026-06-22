#!/usr/bin/env python3
"""identity_ledger.py — mechanical evidence ledger for Friday's identity loop.

The *court record* half of self-awareness: a dated, append-only log of what the
agent verifiably did, derived mechanically from kanban completion receipts. No
model, no prose generation — it cannot flatter, only transcribe receipts.

Subcommands
-----------
rollup
    Read newly-``completed`` kanban work (``task_events`` joined to its closing
    ``task_runs`` row) since a stored watermark and append one dated entry per
    completion to ``~/.hermes/identity/LEDGER.md``. Idempotent: a re-run with no
    new completions is a no-op, and entries are never duplicated.

append
    Gated manual entry for direct (un-carded) conversational work. Requires a
    receipt — at least one of ``--file`` (must exist on disk), ``--command``
    (executed; exit code + captured output stored), or ``--diff``. A prose-only
    entry exits non-zero and writes nothing. Same evidence bar the intake
    quality gate enforces, applied to direct work.

Deployed to ``~/.hermes/scripts/`` and run by a daily ``no_agent`` cron job.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

def _candidate_import_roots(script_path: Path, hermes_home: Path) -> list[Path]:
    """Source roots to try before importing Hermes modules.

    In a checkout, this script lives under ``repo/scripts``. After
    ``setup_identity_loop.py`` deploys it, it lives under
    ``~/.hermes/scripts`` while the source checkout commonly remains at
    ``~/.hermes/hermes-agent``. Add both shapes so direct host smoke commands
    and no-agent cron can import ``hermes_cli`` consistently.
    """
    roots = [
        script_path.resolve().parent.parent,
        hermes_home / "hermes-agent",
    ]
    env_root = (os.environ.get("HERMES_AGENT_ROOT") or "").strip()
    if env_root:
        roots.insert(0, Path(env_root).expanduser())
    return roots


def _add_import_roots() -> None:
    raw_home = (os.environ.get("HERMES_HOME") or "").strip()
    hermes_home = Path(raw_home).expanduser() if raw_home else Path.home() / ".hermes"
    for root in reversed(_candidate_import_roots(Path(__file__), hermes_home)):
        root_s = str(root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)


_add_import_roots()

try:
    from hermes_constants import get_hermes_home
except ImportError:  # pragma: no cover - fallback for detached deployments
    def get_hermes_home() -> Path:  # type: ignore[misc]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val) if val else Path.home() / ".hermes"


def _identity_dir() -> Path:
    d = get_hermes_home() / "identity"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ledger_path() -> Path:
    return _identity_dir() / "LEDGER.md"


def _watermark_path() -> Path:
    return _identity_dir() / ".ledger_watermark"


def _read_watermark() -> int:
    p = _watermark_path()
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return 0


def _write_watermark(value: int) -> None:
    _watermark_path().write_text(str(int(value)), encoding="utf-8")


def _utc_date(epoch: Optional[int]) -> str:
    ts = epoch if epoch is not None else time.time()
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def _ensure_ledger_header() -> None:
    p = _ledger_path()
    if not p.exists() or not p.read_text(encoding="utf-8").strip():
        p.write_text(
            "# LEDGER.md — Friday's evidence record\n\n"
            "Append-only. Mechanical. Each entry is a receipt of verifiable "
            "work — never a self-narration.\n",
            encoding="utf-8",
        )


def _append_entry(text: str) -> None:
    _ensure_ledger_header()
    with _ledger_path().open("a", encoding="utf-8") as fh:
        fh.write("\n" + text.rstrip() + "\n")


def _fmt_receipt_list(label: str, value) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, (list, tuple)):
        items = ", ".join(str(v) for v in value if str(v).strip())
        if not items:
            return None
        return f"- {label}: {items}"
    return f"- {label}: {value}"


# ---------------------------------------------------------------------------
# rollup
# ---------------------------------------------------------------------------

def cmd_rollup(_args: argparse.Namespace) -> int:
    try:
        from hermes_cli import kanban_db as kb
    except ImportError as e:  # pragma: no cover - installed deployments have it
        print(f"identity_ledger: cannot import kanban_db ({e})", file=sys.stderr)
        return 1

    conn = kb.connect()
    try:
        watermark = _read_watermark()
        rows = conn.execute(
            "SELECT id, task_id, run_id, created_at FROM task_events "
            "WHERE kind = 'completed' AND id > ? ORDER BY id ASC",
            (watermark,),
        ).fetchall()

        if not rows:
            return 0

        max_id = watermark
        appended = 0
        for row in rows:
            ev_id = int(row["id"])
            task_id = row["task_id"]
            run_id = row["run_id"]
            created_at = row["created_at"]
            max_id = max(max_id, ev_id)

            run = kb.get_run(conn, int(run_id)) if run_id is not None else None

            header = f"## {_utc_date(created_at)} — task {task_id}"
            if run is not None:
                header += f" (run {run.id})"

            lines = [header]
            if run is not None:
                if run.profile:
                    lines.append(f"- profile: {run.profile}")
                if run.outcome:
                    lines.append(f"- outcome: {run.outcome}")
                if run.summary:
                    lines.append(f"- summary: {run.summary.strip()}")
                meta = run.metadata or {}
                for label, key in (
                    ("changed_files", "changed_files"),
                    ("tests_run", "tests_run"),
                    ("artifacts", "artifacts"),
                ):
                    line = _fmt_receipt_list(label, meta.get(key))
                    if line:
                        lines.append(line)
            else:
                lines.append("- (no closing run recorded)")

            _append_entry("\n".join(lines))
            appended += 1

        _write_watermark(max_id)
        print(f"identity_ledger: appended {appended} entr"
              f"{'y' if appended == 1 else 'ies'} (watermark -> {max_id})")
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# append (gated)
# ---------------------------------------------------------------------------

def cmd_append(args: argparse.Namespace) -> int:
    receipts: list[str] = []

    # --file: each must exist on disk.
    for f in args.file or []:
        path = Path(f).expanduser()
        if not path.exists():
            print(
                f"identity_ledger: refusing append — file does not exist: {f}",
                file=sys.stderr,
            )
            return 2
        receipts.append(f"- file: {path}")

    # --command: execute, capture exit code + output.
    for cmd in args.command or []:
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120,
            )
        except Exception as e:
            print(f"identity_ledger: command failed to run: {cmd} ({e})",
                  file=sys.stderr)
            return 2
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        captured = out or err or "(no output)"
        # Keep the ledger readable: cap captured output.
        if len(captured) > 800:
            captured = captured[:800] + " …[truncated]"
        receipts.append(
            f"- command: `{cmd}` (exit {proc.returncode})\n"
            f"  output: {captured}"
        )

    # --diff: literal text, or @path to read from a file.
    if args.diff:
        diff_val = args.diff
        if diff_val.startswith("@"):
            dp = Path(diff_val[1:]).expanduser()
            if not dp.exists():
                print(f"identity_ledger: diff file does not exist: {dp}",
                      file=sys.stderr)
                return 2
            diff_val = dp.read_text(encoding="utf-8")
        if diff_val.strip():
            receipts.append("- diff:\n```\n" + diff_val.rstrip() + "\n```")

    if not receipts:
        print(
            "identity_ledger: refusing prose-only append — a receipt is "
            "required (--file / --command / --diff). Nothing written.",
            file=sys.stderr,
        )
        return 2

    header = f"## {_utc_date(None)} — direct work (manual)"
    if args.task:
        header += f" — task {args.task}"
    lines = [header, f"- summary: {args.summary.strip()}", *receipts]
    _append_entry("\n".join(lines))
    print("identity_ledger: appended manual entry.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # Default action is rollup, so a bare invocation (how the cron scheduler
    # runs the script — `python identity_ledger.py` with no args) does a rollup.
    p.set_defaults(func=cmd_rollup)
    sub = p.add_subparsers(dest="command_name")

    sp_rollup = sub.add_parser(
        "rollup", help="Append ledger entries for newly-completed kanban work.")
    sp_rollup.set_defaults(func=cmd_rollup)

    sp_append = sub.add_parser(
        "append", help="Gated manual entry for direct work (requires a receipt).")
    sp_append.add_argument("--summary", required=True, help="One-line description.")
    sp_append.add_argument("--task", help="Optional related task id.")
    sp_append.add_argument(
        "--file", action="append",
        help="Receipt: a file that must exist on disk (repeatable).")
    sp_append.add_argument(
        "--command", action="append",
        help="Receipt: a command to execute; its exit code + output are stored "
             "(repeatable).")
    sp_append.add_argument(
        "--diff", help="Receipt: a unified diff (literal text, or @path).")
    sp_append.set_defaults(func=cmd_append)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
