"""Argparse command wiring for hermes_parity."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from . import __version__, bisect as bisect_mod, buckets, forkdelta, gates, gitops, state


DEFAULT_WORKTREE_ROOT = Path.home() / ".hermes" / "worktrees"


def _repo(args: argparse.Namespace) -> Path:
    repo = gitops.repo_root(Path.cwd())
    gitops.require_remotes(repo)
    return repo


def _print_status(repo: Path, *, fail_behind: int | None) -> int:
    gitops.fetch(repo, "origin")
    gitops.fetch(repo, "fork")
    branch = gitops.current_branch(repo)
    ahead, behind = gitops.ahead_behind(repo, "fork/main", "origin/main")
    base = gitops.merge_base(repo, "fork/main", "origin/main")
    conflicts = gitops.conflict_entries(repo)
    marker_count = len(gitops.conflict_marker_lines(repo))
    print(f"repo: {repo}")
    print(f"branch: {branch}")
    print(f"fork/main ahead origin/main: {ahead}")
    print(f"fork/main behind origin/main: {behind}")
    print(f"merge-base: {base}")
    print(f"unmerged entries: {len(conflicts)}")
    print(f"conflict marker lines: {marker_count}")
    for bucket in buckets.bucket_conflicts(repo):
        print(f"bucket {bucket.name}: {len(bucket.files)}")
    if fail_behind is not None and behind >= fail_behind:
        return 2
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    return _print_status(_repo(args), fail_behind=args.fail_behind)


def cmd_start(args: argparse.Namespace) -> int:
    repo = _repo(args)
    if gitops.is_dirty(repo):
        print("error: live checkout is dirty; refusing to start parity worktree", file=sys.stderr)
        return 2
    gitops.fetch(repo, "origin")
    gitops.fetch(repo, "fork")
    root = gitops.ensure_worktree_root(args.worktree_root)
    stamp = datetime.now().strftime("%Y-%m-%d")
    name = args.name or f"parity-{stamp}"
    worktree = root / name
    branch = args.branch or f"sync/upstream-{stamp}"
    target_ref = args.target or args.upstream
    target_sha = gitops.rev_parse(repo, target_ref)
    fork_main_at_start = gitops.rev_parse(repo, args.fork)
    merge_base = gitops.merge_base(repo, args.fork, target_sha)
    if worktree.exists():
        print(f"error: worktree already exists: {worktree}", file=sys.stderr)
        return 2
    gitops.add_worktree(repo, worktree, branch, args.fork)
    gitops.run_git(worktree, ["config", "merge.conflictStyle", "diff3"])
    merge = gitops.merge_no_commit(worktree, target_sha)
    bucket_stats = buckets.write_conflict_report(worktree)
    created = state.ParityState(
        tree_sha=gitops.tree_sha(worktree),
        data={
            "created": datetime.now(timezone.utc).isoformat(),
            "target_sha": target_sha,
            "merge_base": merge_base,
            "fork_main_at_start": fork_main_at_start,
            "branch": branch,
            "remote": args.remote,
            "upstream": args.upstream,
            "fork": args.fork,
            "rollback": {
                "fork_main": fork_main_at_start,
                "local_main": gitops.rev_parse(repo, "HEAD"),
            },
            "buckets": bucket_stats,
            "gates": {},
        },
    )
    state.save_state(worktree, created)
    print(f"created worktree: {worktree}")
    print(f"merge exit: {merge.returncode}")
    print(f"conflict report: {worktree / 'docs/sync/review/conflict-buckets.md'}")
    print("next: resolve conflicts, then python3.11 -m hermes_parity gates")
    return 0


def cmd_gates(args: argparse.Namespace) -> int:
    repo = Path(args.worktree).resolve() if args.worktree else _repo(args)
    gitops.require_remotes(repo)
    results = gates.run_gates(
        repo,
        fast=args.fast,
        stage=args.stage,
        resume=args.resume,
        strict=args.strict,
        base=args.base,
        fork_ref=args.fork_ref,
        tests=args.tests,
    )
    print(gates.format_table(results))
    print()
    for reminder in gates.ci_reminders(repo):
        print(f"reminder: {reminder}")
    return 0 if all(result.passed for result in results) else 1


def _tests_from_args(args: argparse.Namespace) -> list[str]:
    tests = list(args.tests or [])
    for path in args.from_file or []:
        tests.extend(bisect_mod.parse_from_file(path))
    return tests


def _load_required_state(repo: Path) -> state.ParityState:
    loaded = state.load_state(repo, invalidate_on_tree_change=False)
    if not loaded:
        raise gitops.GitError(f"missing {state.STATE_FILE}; run from a parity worktree or pass overrides")
    return loaded


def _ensure_baseline_worktree(repo: Path, root: Path, baseline_arg: Path | None) -> Path:
    if baseline_arg:
        return baseline_arg.resolve()
    loaded = _load_required_state(repo)
    baseline = root.expanduser().resolve() / "parity-baseline"
    fork_sha = str(loaded.data.get("fork_main_at_start") or "")
    if not fork_sha:
        raise gitops.GitError("state missing fork_main_at_start; cannot auto-manage baseline")
    if baseline.exists():
        current = gitops.rev_parse(baseline, "HEAD")
        if current != fork_sha:
            gitops.run_git(repo, ["worktree", "remove", "--force", str(baseline)], check=False)
            print(f"rollback: git -C {repo} worktree remove --force {baseline}")
            gitops.run_git(repo, ["worktree", "add", "--detach", str(baseline), fork_sha])
    else:
        print(f"rollback: git -C {repo} worktree remove --force {baseline}")
        gitops.run_git(repo, ["worktree", "add", "--detach", str(baseline), fork_sha])
    return baseline


def cmd_bisect(args: argparse.Namespace) -> int:
    merge = Path(args.merge).resolve() if args.merge else _repo(args)
    baseline = _ensure_baseline_worktree(merge, args.worktree_root, args.baseline)
    tests = _tests_from_args(args)
    if not tests:
        print("error: no pytest node ids supplied", file=sys.stderr)
        return 2
    jobs = bisect_mod.bounded_jobs(args.jobs)
    results = bisect_mod.classify_many(
        baseline_repo=baseline,
        merge_repo=merge,
        tests=tests,
        runner=bisect_mod.pytest_runner,
        jobs=jobs,
    )
    print(bisect_mod.format_table(results))
    return 1 if any(result.classification == bisect_mod.Classification.REGRESSION for result in results) else 0


def _fork_main_unchanged(repo: Path, loaded: state.ParityState) -> bool:
    recorded = str(loaded.data.get("fork_main_at_start") or "")
    if not recorded:
        return False
    gitops.fetch(repo, "fork")
    return gitops.rev_parse(repo, "fork/main") == recorded


def _all_gates_green(repo: Path) -> bool:
    loaded = state.load_state(repo, invalidate_on_tree_change=False)
    if not loaded:
        return False
    valid = state.valid_gate_names_for_current_tree(repo, gates.ORDERED_STAGE_NAMES)
    return set(gates.ORDERED_STAGE_NAMES).issubset(valid)


def _write_finish_files(repo: Path, loaded: state.ParityState, branch: str, force_reason: str | None) -> tuple[Path, Path]:
    body = repo / "docs" / "sync" / "review" / "parity-pr-body.md"
    commit_msg = repo / "docs" / "sync" / "review" / "parity-merge-commit-message.txt"
    body.parent.mkdir(parents=True, exist_ok=True)
    buckets_data = loaded.data.get("buckets") or {}
    behavior = "## Behavior changes\n\n"
    body_text = "\n".join(
        [
            "## Parity sync",
            "",
            f"- target SHA: {loaded.data.get('target_sha', '')}",
            f"- merge-base: {loaded.data.get('merge_base', '')}",
            f"- bucket stats: `{json.dumps(buckets_data, sort_keys=True)}`",
            f"- force reason: {force_reason}" if force_reason else "",
            "",
            behavior,
            "## Validation",
            "",
            "- `python3.11 -m hermes_parity gates --resume`",
            "",
        ]
    )
    body.write_text(body_text, encoding="utf-8")
    commit_msg.write_text(
        "\n".join(
            [
                f"Merge upstream parity into {branch}",
                "",
                f"Target SHA: {loaded.data.get('target_sha', '')}",
                f"Merge-base: {loaded.data.get('merge_base', '')}",
                f"Bucket stats: {json.dumps(buckets_data, sort_keys=True)}",
                "",
                behavior,
            ]
        ),
        encoding="utf-8",
    )
    return body, commit_msg


def cmd_finish(args: argparse.Namespace) -> int:
    repo = Path(args.worktree).resolve() if args.worktree else _repo(args)
    gitops.require_remotes(repo)
    loaded = _load_required_state(repo)
    if not _fork_main_unchanged(repo, loaded):
        print("finish blocked: fork/main moved since start. Recovery: git merge fork/main in this branch; never rebase.", file=sys.stderr)
        return 2
    results = gates.run_gates(
        repo,
        fast=args.fast,
        resume=args.resume,
        strict=args.strict,
        base=args.base,
        fork_ref=args.fork_ref,
        tests=args.tests,
    )
    print(gates.format_table(results))
    failures = [result for result in results if not result.passed]
    if failures and not args.force:
        print("finish blocked: gates failed. Re-run with --force --force-reason <reason> to print PR command anyway.")
        return 1
    if args.force and not args.force_reason:
        print("error: --force requires --force-reason", file=sys.stderr)
        return 2
    if args.force_reason:
        print(f"force reason: {args.force_reason}")
    if failures and args.force:
        print("finish continuing under --force; red gates are shown above.")
    if not failures and not _all_gates_green(repo):
        print("finish blocked: not every gate is green at current tree SHA. Run python3.11 -m hermes_parity gates.", file=sys.stderr)
        return 1
    if not gitops.merge_head_exists(repo) and not gitops.has_staged_changes(repo):
        print("finish blocked: no in-progress merge or staged merge changes found", file=sys.stderr)
        return 2
    branch = gitops.current_branch(repo)
    body_file, commit_msg = _write_finish_files(repo, loaded, branch, args.force_reason)
    gitops.commit(repo, commit_msg)
    remote = args.remote or str(loaded.data.get("remote") or "origin")
    gitops.push(repo, remote, branch)
    title = args.title or f"Merge upstream parity into {branch}"
    print("landing rules: merge with --merge; never squash. If BEHIND, merge fork/main; never rebase.")
    print("PR command:")
    print(f"gh pr create --base main --head {branch} --title {title!r} --body-file {str(body_file)!r}")
    return 0 if not failures or args.force else 1


def cmd_ack(args: argparse.Namespace) -> int:
    repo = Path(args.worktree).resolve() if args.worktree else _repo(args)
    _load_required_state(repo)
    for path in args.paths:
        state.record_ack(repo, path, args.reason)
        print(f"acknowledged: {path} — {args.reason}")
    print("note: re-run gates; the manifest+forkdelta stage treats acknowledged paths as reviewed-and-intentional.")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    repo = _repo(args)
    worktree = args.worktree.resolve()
    if not str(worktree).startswith(str(args.worktree_root.expanduser().resolve())):
        print(f"error: refusing to remove worktree outside {args.worktree_root}", file=sys.stderr)
        return 2
    gitops.remove_worktree(repo, worktree, force=args.force)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes_parity")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--worktree-root", type=Path, default=DEFAULT_WORKTREE_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="show live parity status")
    status.add_argument("--fail-behind", type=int, metavar="N", help="exit 2 when fork/main is behind origin/main by at least N commits")
    status.set_defaults(func=cmd_status)

    start = sub.add_parser("start", help="create a parity merge worktree")
    start.add_argument("--name", default=None)
    start.add_argument("--branch", default=None)
    start.add_argument("--target")
    start.add_argument("--upstream", default="origin/main")
    start.add_argument("--fork", default="fork/main")
    start.add_argument("--remote", default="origin")
    start.set_defaults(func=cmd_start)

    gate_cmd = sub.add_parser("gates", help="run the six-stage parity gate ladder")
    gate_cmd.add_argument("--worktree", type=Path)
    gate_cmd.add_argument("--fast", action="store_true", help="skip scripts/run_tests.sh")
    gate_cmd.add_argument("--stage", choices=gates.ORDERED_STAGE_NAMES)
    gate_cmd.add_argument("--resume", action="store_true")
    gate_cmd.add_argument("--strict", action="store_true")
    gate_cmd.add_argument("--base")
    gate_cmd.add_argument("--fork-ref", default="fork/main")
    gate_cmd.add_argument("tests", nargs="*")
    gate_cmd.set_defaults(func=cmd_gates)

    bisect = sub.add_parser("bisect", help="classify baseline vs merge test results")
    bisect.add_argument("--baseline", type=Path)
    bisect.add_argument("--merge", type=Path)
    bisect.add_argument("--from-file", action="append", default=[])
    bisect.add_argument("--jobs", type=int)
    bisect.add_argument("tests", nargs="*")
    bisect.set_defaults(func=cmd_bisect)

    finish = sub.add_parser("finish", help="run gates and print PR creation command")
    finish.add_argument("--worktree", type=Path)
    finish.add_argument("--fast", action="store_true")
    finish.add_argument("--resume", action="store_true")
    finish.add_argument("--strict", action="store_true")
    finish.add_argument("--force", action="store_true")
    finish.add_argument("--force-reason")
    finish.add_argument("--remote")
    finish.add_argument("--base")
    finish.add_argument("--fork-ref", default="fork/main")
    finish.add_argument("--title")
    finish.add_argument("--body")
    finish.add_argument("tests", nargs="*")
    finish.set_defaults(func=cmd_finish)

    ack = sub.add_parser("ack", help="acknowledge an intentionally dropped/renamed fork file (fork-delta gate clearance)")
    ack.add_argument("--worktree", type=Path)
    ack.add_argument("--reason", required=True, help="why this fork-only path is intentionally not surviving the merge")
    ack.add_argument("paths", nargs="+")
    ack.set_defaults(func=cmd_ack)

    clean = sub.add_parser("clean", help="remove a parity worktree")
    clean.add_argument("worktree", type=Path)
    clean.add_argument("--force", action="store_true")
    clean.set_defaults(func=cmd_clean)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        gitops.require_git_version()
        return int(args.func(args))
    except gitops.GitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
