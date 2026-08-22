"""``hermes sync-fork`` — CLI surface and interactive screen.

Kept separate from :mod:`hermes_cli.sync_fork` so the merge logic stays free of
printing and prompts, and can be driven from a cron script, the dashboard, or a
test without a terminal.
"""

from __future__ import annotations

import json
from pathlib import Path

from hermes_cli import sync_fork


def _repo_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def _render_state(state: sync_fork.ForkState, plan: sync_fork.SyncPlan) -> list[str]:
    """Human-readable status block, shared by the CLI and the curses screen."""
    # First, because "up to date" is what an unreadable state used to render as,
    # and the curses screen reaches this function without passing the CLI guard.
    if state.error:
        return [state.error]
    if state.behind == 0 and not state.diverged:
        headline = "Up to date with upstream."
    elif state.diverged:
        headline = (
            f"Diverged: {state.behind} commit(s) behind, "
            f"{state.ahead} local commit(s) upstream does not have."
        )
    else:
        headline = f"{state.behind} commit(s) behind upstream (no local commits)."

    lines = [headline]
    if state.dirty:
        lines.append("Worktree has uncommitted changes — sync will refuse until it is clean.")
    if state.behind:
        lines.append(f"Merge plan: {len(plan.steps)} step(s).")
        if plan.conflict_prone:
            lines.append(
                f"{len(plan.conflict_prone)} upstream commit(s) touch files this fork "
                "also changed — those are the steps that can conflict."
            )
        else:
            lines.append("No upstream commit touches a file this fork changed.")
    if state.diverged and state.behind:
        lines.append(
            "`hermes update` skips this fork by design (its sync is fast-forward "
            "only). That is what sync-fork is for."
        )
    return lines


def run(args) -> int:
    """Entry point for ``hermes sync-fork``."""
    root = _repo_root(getattr(args, "repo", None))
    upstream_ref = getattr(args, "upstream_ref", None) or sync_fork.DEFAULT_UPSTREAM_REF

    if getattr(args, "ui", False):
        return _run_ui(root, upstream_ref)

    state = sync_fork.inspect(root, upstream_ref)

    # Before the plan, and before every action path below. A state that could not
    # be read must not reach code that branches on `behind`, because `behind` is
    # 0 in that case and 0 means "nothing to do" everywhere downstream.
    #
    # Exit 2, distinct from 1: 1 already means "a sync is available" for --check,
    # so reusing it would tell a cron the opposite of the truth. Consumers that
    # branch on the exit code (hermes-webui's fork_keeper_bridge does) get a
    # value that cannot be confused with either success or work-pending.
    if state.error:
        if getattr(args, "json", False):
            print(json.dumps({"error": state.error}))
        else:
            print(f"  {state.error}")
        return 2

    plan = sync_fork.plan(root, upstream_ref)

    if getattr(args, "json", False):
        print(json.dumps({
            "behind": state.behind,
            "ahead": state.ahead,
            "diverged": state.diverged,
            "dirty": state.dirty,
            "steps": len(plan.steps),
            "conflict_prone": len(plan.conflict_prone),
        }))
        return 0

    for line in _render_state(state, plan):
        print(f"  {line}")

    if getattr(args, "check", False):
        # Exit 1 signals "action available" so a cron or CI step can branch on it.
        return 1 if state.behind else 0

    if state.behind == 0:
        return 0

    result = sync_fork.sync(root, upstream_ref, dry_run=getattr(args, "dry_run", False))
    print(f"  {result.reason}")
    if not result.ok and result.conflicted_paths:
        print("  conflicted:")
        for path in result.conflicted_paths:
            print(f"    {path}")
    return 0 if result.ok else 1


def _run_ui(root: Path, upstream_ref: str) -> int:
    """Minimal curses screen: status, plan, and a guarded sync action.

    Deliberately plain — it is a status-and-confirm surface, not a git client.
    Falls back to the plain CLI output when there is no usable terminal, so
    ``--ui`` over a pipe degrades instead of crashing.
    """
    import curses

    def draw(stdscr) -> int:
        curses.use_default_colors()
        message = ""
        while True:
            state = sync_fork.inspect(root, upstream_ref)
            plan = sync_fork.plan(root, upstream_ref)
            stdscr.erase()
            stdscr.addstr(0, 0, "hermes sync-fork", curses.A_BOLD)
            stdscr.addstr(1, 0, f"repo: {root}")
            stdscr.addstr(2, 0, f"upstream: {upstream_ref}")
            row = 4
            for line in _render_state(state, plan):
                for chunk in _wrap(line, max(20, stdscr.getmaxyx()[1] - 4)):
                    stdscr.addstr(row, 2, chunk)
                    row += 1
            if message:
                row += 1
                for chunk in _wrap(message, max(20, stdscr.getmaxyx()[1] - 4)):
                    stdscr.addstr(row, 2, chunk, curses.A_BOLD)
                    row += 1
            row += 1
            actions = "[s] sync   [d] dry-run   [r] refresh   [q] quit"
            stdscr.addstr(row, 2, actions, curses.A_DIM)
            stdscr.refresh()

            key = stdscr.getch()
            if key in (ord("q"), 27):
                return 0
            if key == ord("r"):
                message = ""
                continue
            if key in (ord("s"), ord("d")):
                dry = key == ord("d")
                stdscr.addstr(row + 2, 2, "working…")
                stdscr.refresh()
                result = sync_fork.sync(root, upstream_ref, dry_run=dry)
                message = result.reason
                if result.conflicted_paths:
                    message += " | conflicted: " + ", ".join(result.conflicted_paths[:4])

    def _wrap(text: str, width: int) -> list[str]:
        words, lines, cur = text.split(), [], ""
        for word in words:
            candidate = f"{cur} {word}".strip()
            if len(candidate) <= width:
                cur = candidate
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines or [""]

    try:
        return curses.wrapper(draw)
    except Exception:
        state = sync_fork.inspect(root, upstream_ref)
        plan = sync_fork.plan(root, upstream_ref)
        print("  (no usable terminal for --ui; showing status instead)")
        for line in _render_state(state, plan):
            print(f"  {line}")
        return 0


def register(subparsers) -> None:
    """Attach ``sync-fork`` to the main argument parser."""
    parser = subparsers.add_parser(
        "sync-fork",
        help="Merge upstream into a fork that has diverged (update skips this case)",
    )
    parser.add_argument("--repo", help="Repository path (default: this install)")
    parser.add_argument(
        "--upstream-ref",
        dest="upstream_ref",
        default=sync_fork.DEFAULT_UPSTREAM_REF,
        help=f"Upstream ref to merge (default: {sync_fork.DEFAULT_UPSTREAM_REF})",
    )
    parser.add_argument("--check", action="store_true",
                        help="Report status only; exit 1 when a sync is available")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Show the merge plan without merging")
    parser.add_argument("--json", action="store_true", help="Machine-readable status")
    parser.add_argument("--ui", action="store_true", help="Interactive screen")
    parser.set_defaults(func=run)
