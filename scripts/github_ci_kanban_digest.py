#!/usr/bin/env python3
"""Post concise GitHub CI failure digests to active Kanban PR-gate cards.

Intended usage from the Hermes repo (or any checkout with the right GitHub
remote/auth):

    python scripts/github_ci_kanban_digest.py --repo NousResearch/hermes-agent

The command is silent when no active PR-gate card has a new red-check digest.
It stores one digest key per task under the shared Kanban home so cron can run
it repeatedly without comment spam.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Keep script runnable directly from a source checkout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import kanban_db as kb  # noqa: E402
from agent.redact import redact_sensitive_text  # noqa: E402

ACTIVE_STATUSES = ("triage", "todo", "scheduled", "ready", "running", "blocked", "review")
RED_CONCLUSIONS = {"failure", "timed_out", "startup_failure", "cancelled", "action_required"}
GREEN_CONCLUSIONS = {"success", "neutral", "skipped"}
PR_RE = re.compile(r"(?:PR\s*#|pull/(?:request/)?|github\.com/[^/]+/[^/]+/pull/)(\d+)", re.I)
PR_GATE_RE = re.compile(r"\bpr[-_ ]?gate\b|\bmergeable\b|\bci\b", re.I)
RUN_ID_RE = re.compile(r"/actions/runs/(\d+)")
PATH_RE = re.compile(r"(?P<path>(?:[\w./-]+/)?[\w.-]+\.(?:tsx|ts|jsx|js|mjs|cjs|py|sh|yml|yaml|json|md|css|scss|html))(?::(?P<line>\d+)(?::\d+)?)?")
PKG_RE = re.compile(r"(?:packages|apps|plugins|tools|gateway|hermes_cli|agent|tests)/[\w./-]+")
ERROR_RE = re.compile(
    r"(error:|failed|failure|traceback|assertionerror|exception|\bE\s+assert|panic|npm ERR!|ERR_PNPM|biome|vitest|pytest|ruff|mypy)",
    re.I,
)

Runner = Callable[[list[str]], str]


@dataclass
class GateCard:
    task_id: str
    title: str
    body: str
    pr_number: int


@dataclass
class Check:
    name: str
    state: str
    details_url: str = ""
    run_id: str = ""

    @property
    def is_red(self) -> bool:
        return self.state.lower() in RED_CONCLUSIONS

    @property
    def is_green(self) -> bool:
        return self.state.lower() in GREEN_CONCLUSIONS


def run_gh(args: list[str]) -> str:
    cmd = ["gh", *args]
    return subprocess.run(cmd, text=True, capture_output=True, check=True).stdout


def gh_json(runner: Runner, args: list[str]) -> Any:
    out = runner(args)
    return json.loads(out or "null")


def infer_repo_from_git() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"], text=True, capture_output=True, check=True
        ).stdout.strip()
    except Exception:
        return None
    m = re.search(r"github\.com[:/]([^/]+/[^/.]+)(?:\.git)?$", out)
    return m.group(1) if m else None


def find_pr_number(text: str) -> Optional[int]:
    match = PR_RE.search(text or "")
    return int(match.group(1)) if match else None


def looks_like_pr_gate(text: str) -> bool:
    return bool(PR_GATE_RE.search(text or ""))


def active_pr_gate_cards(conn) -> list[GateCard]:
    cards: list[GateCard] = []
    for status in ACTIVE_STATUSES:
        for task in kb.list_tasks(conn, status=status, include_archived=False, order_by="created-desc"):
            text = f"{task.title}\n{task.body or ''}"
            pr = find_pr_number(text)
            if pr is None or not looks_like_pr_gate(text):
                continue
            cards.append(GateCard(task.id, task.title, task.body or "", pr))
    return cards


def normalize_check(node: dict[str, Any]) -> Optional[Check]:
    name = str(node.get("name") or node.get("context") or "").strip()
    if not name:
        return None
    conclusion = str(node.get("conclusion") or node.get("state") or node.get("status") or "").lower()
    if conclusion in {"completed", "queued", "in_progress", "pending", "expected"}:
        conclusion = str(node.get("state") or node.get("status") or conclusion).lower()
    details = str(node.get("detailsUrl") or node.get("targetUrl") or "")
    run_match = RUN_ID_RE.search(details)
    return Check(name=name, state=conclusion, details_url=details, run_id=run_match.group(1) if run_match else "")


def pr_status(repo: str, pr: int, runner: Runner = run_gh) -> dict[str, Any]:
    data = gh_json(
        runner,
        [
            "pr",
            "view",
            str(pr),
            "--repo",
            repo,
            "--json",
            "number,url,headRefName,headRefOid,baseRefName,statusCheckRollup",
        ],
    )
    checks = [c for c in (normalize_check(n) for n in data.get("statusCheckRollup") or []) if c]
    return {
        "number": int(data["number"]),
        "url": data.get("url", ""),
        "branch": data.get("headRefName", ""),
        "head": data.get("headRefOid", ""),
        "base": data.get("baseRefName", ""),
        "checks": checks,
    }


def state_path(board: Optional[str] = None) -> Path:
    suffix = f"-{board}" if board else ""
    return kb.kanban_home() / "kanban" / f"github-ci-digests{suffix}.json"


def load_state(path: Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def save_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest_key(task_id: str, pr: int, head: str, red_checks: Iterable[Check]) -> str:
    raw = json.dumps(
        {
            "task": task_id,
            "pr": pr,
            "head": head,
            "red": sorted((c.name, c.state, c.run_id) for c in red_checks),
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def collect_run_logs(repo: str, run_ids: Iterable[str], runner: Runner = run_gh) -> str:
    chunks: list[str] = []
    for rid in dict.fromkeys(r for r in run_ids if r):
        try:
            chunks.append(runner(["run", "view", rid, "--repo", repo, "--log-failed"]))
        except subprocess.CalledProcessError as exc:
            # gh includes useful auth/not-found context on stderr; keep it short and non-secret.
            chunks.append(f"[could not read failed log for run {rid}: exit {exc.returncode}]")
    return "\n".join(chunks)


def concise_failures(log_text: str, max_items: int = 6) -> list[str]:
    """Extract the first actionable package/file errors without dumping logs."""
    seen: set[str] = set()
    items: list[str] = []
    for raw in (log_text or "").splitlines():
        line = raw.strip()
        if not line or not ERROR_RE.search(line):
            continue
        # GitHub log-failed rows often start with job/step/timestamp columns.
        parts = line.split("\t")
        message = parts[-1].strip() if parts else line
        path_match = PATH_RE.search(message)
        pkg_match = PKG_RE.search(message)
        if path_match:
            subject = path_match.group("path")
            if path_match.group("line"):
                subject += f":{path_match.group('line')}"
        elif pkg_match:
            subject = pkg_match.group(0).rstrip("/:")
        else:
            subject = message[:120]
        summary = redact_sensitive_text(re.sub(r"\s+", " ", message), force=True)
        if len(summary) > 180:
            summary = summary[:177].rstrip() + "…"
        item = f"{subject} — {summary}"
        if item in seen:
            continue
        seen.add(item)
        items.append(item)
        if len(items) >= max_items:
            break
    return items


def _failure_paths(failures: list[str]) -> list[str]:
    paths: list[str] = []
    for failure in failures:
        subject = failure.split(" — ", 1)[0]
        match = PATH_RE.search(subject) or PATH_RE.search(failure)
        if not match:
            continue
        path = match.group("path")
        if path not in paths:
            paths.append(path)
    return paths


def _workspace_filter(path: str) -> str:
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] in {"apps", "packages", "plugins"}:
        return f"./{parts[0]}/{parts[1]}"
    return ""


def _workspace_relative_path(path: str) -> str:
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] in {"apps", "packages", "plugins"}:
        return "/".join(parts[2:])
    return path


def infer_repro_commands(red_checks: list[Check], failures: list[str]) -> list[str]:
    text = "\n".join([c.name for c in red_checks] + failures).lower()
    paths = _failure_paths(failures)
    py_paths = [p for p in paths if p.endswith(".py")]
    js_paths = [p for p in paths if re.search(r"\.(tsx?|jsx?|mjs|cjs)$", p)]
    commands: list[str] = []
    if "biome" in text or "lint" in text:
        commands.append(f"pnpm biome check {shlex.quote(js_paths[0])}" if js_paths else "pnpm biome check .")
    if "vitest" in text or "unit" in text or re.search(r"\.(tsx?|jsx?)", text):
        if js_paths:
            workspace = _workspace_filter(js_paths[0])
            if workspace:
                commands.append(
                    f"pnpm --filter {shlex.quote(workspace)} test {shlex.quote(_workspace_relative_path(js_paths[0]))}"
                )
            else:
                commands.append(f"pnpm test {shlex.quote(js_paths[0])}")
        else:
            commands.append("pnpm test")
    if "pytest" in text or ".py" in text:
        commands.append(
            f"python -m pytest -o 'addopts=' -q {shlex.quote(py_paths[0])}"
            if py_paths
            else "python -m pytest -o 'addopts=' -q"
        )
    if "typecheck" in text or "tsc" in text:
        workspace = _workspace_filter(js_paths[0]) if js_paths else ""
        commands.append(f"pnpm --filter {shlex.quote(workspace)} typecheck" if workspace else "pnpm typecheck")
    if "build" in text:
        workspace = _workspace_filter(js_paths[0]) if js_paths else ""
        commands.append(f"pnpm --filter {shlex.quote(workspace)} build" if workspace else "pnpm build")
    if not commands:
        commands.append("gh run view <run-id> --log-failed")
    return list(dict.fromkeys(commands))[:4]


def latest_base_red_check_names(repo: str, base: str, runner: Runner = run_gh) -> Optional[set[str]]:
    if not base:
        return None
    try:
        runs = gh_json(
            runner,
            [
                "run",
                "list",
                "--repo",
                repo,
                "--branch",
                base,
                "--limit",
                "20",
                "--json",
                "name,conclusion,status",
            ],
        )
    except Exception:
        return None
    names = set()
    for run in runs or []:
        conclusion = str(run.get("conclusion") or run.get("status") or "").lower()
        if conclusion in RED_CONCLUSIONS:
            names.add(str(run.get("name") or ""))
    return names


def _normalize_ci_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def classify_failure(red_checks: list[Check], base_red_names: Optional[set[str]]) -> str:
    if base_red_names is None:
        return "uncertain: could not compare against the base branch checks"
    red_names = {c.name for c in red_checks}
    overlap = red_names & base_red_names
    if not overlap:
        normalized_base = {_normalize_ci_name(n): n for n in base_red_names}
        overlap = {name for name in red_names if _normalize_ci_name(name) in normalized_base}
    if overlap:
        listed = ", ".join(sorted(overlap)[:3])
        return f"baseline/stack-order debt likely: base branch recently red on {listed}"
    if base_red_names:
        listed = ", ".join(sorted(base_red_names)[:3])
        return (
            "uncertain: base branch has red workflow run(s) "
            f"({listed}), but PR red check/job names do not match those workflow names"
        )
    return "branch-specific likely: red checks are not currently red on the base branch"


def format_digest(pr_info: dict[str, Any], red_checks: list[Check], green_checks: list[Check], failures: list[str], repro: list[str], classification: str) -> str:
    run_ids = sorted({c.run_id for c in red_checks if c.run_id})
    head = str(pr_info.get("head") or "")
    short_head = head[:12] if head else "unknown"
    red_names = ", ".join(c.name for c in red_checks) or "none"
    green_names = ", ".join(c.name for c in green_checks[:10]) or "none"
    more_green = f" (+{len(green_checks) - 10} more)" if len(green_checks) > 10 else ""
    run_text = ", ".join(run_ids) if run_ids else "unknown"
    failure_lines = "\n".join(f"- {f}" for f in failures) if failures else "- No concise file/package failure found in failed logs; inspect the run UI."
    repro_lines = "\n".join(f"- `{cmd}`" for cmd in repro)
    return textwrap.dedent(
        f"""
        CI digest for PR #{pr_info['number']} ({pr_info.get('branch') or 'unknown'} @ {short_head})
        Run id(s): {run_text}
        Red checks: {red_names}
        Green checks: {green_names}{more_green}

        First actionable failures:
        {failure_lines}

        Focused local repro:
        {repro_lines}

        Assessment: {classification}
        """
    ).strip()


def maybe_post_digest(conn, card: GateCard, repo: str, state: dict[str, str], runner: Runner = run_gh, *, dry_run: bool = False) -> Optional[str]:
    info = pr_status(repo, card.pr_number, runner)
    checks: list[Check] = info["checks"]
    red = [c for c in checks if c.is_red]
    if not red:
        return None
    green = [c for c in checks if c.is_green]
    key = digest_key(card.task_id, card.pr_number, info.get("head", ""), red)
    if state.get(card.task_id) == key:
        return None
    logs = collect_run_logs(repo, [c.run_id for c in red], runner)
    failures = concise_failures(logs)
    repro = infer_repro_commands(red, failures)
    classification = classify_failure(red, latest_base_red_check_names(repo, info.get("base", ""), runner))
    body = redact_sensitive_text(format_digest(info, red, green, failures, repro, classification), force=True)
    if not dry_run:
        kb.add_comment(conn, card.task_id, "ci-digest", body)
    state[card.task_id] = key
    return body


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=None, help="GitHub owner/repo (defaults to git origin)")
    p.add_argument("--board", default=None, help="Kanban board slug")
    p.add_argument("--task-id", action="append", help="Limit to specific Kanban task id(s)")
    p.add_argument("--pr", type=int, help="Probe one PR even if no PR-gate task exists (requires --task-id for posting)")
    p.add_argument("--dry-run", action="store_true", help="Print digest instead of commenting/state write")
    p.add_argument("--state-file", type=Path, default=None, help="Override dedupe state path")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo or infer_repo_from_git()
    if not repo:
        print("github-ci-digest: --repo is required outside a GitHub checkout", file=sys.stderr)
        return 2
    path = args.state_file or state_path(args.board)
    state = load_state(path)
    posted: list[str] = []
    with kb.connect_closing(board=args.board) as conn:
        cards = active_pr_gate_cards(conn)
        if args.task_id:
            wanted = set(args.task_id)
            cards = [c for c in cards if c.task_id in wanted]
        if args.pr is not None:
            if not args.task_id:
                cards = [GateCard("dry-run", f"PR-gate PR #{args.pr}", "", args.pr)]
            elif not cards:
                cards = [GateCard(args.task_id[0], f"PR-gate PR #{args.pr}", "", args.pr)]
        for card in cards:
            try:
                body = maybe_post_digest(conn, card, repo, state, dry_run=args.dry_run)
            except subprocess.CalledProcessError as exc:
                print(
                    f"github-ci-digest: skipped task {card.task_id} PR #{card.pr_number}: gh exited {exc.returncode}",
                    file=sys.stderr,
                )
                continue
            if body:
                posted.append(f"{card.task_id}: PR #{card.pr_number}\n{body}")
    if posted and not args.dry_run:
        save_state(path, state)
    elif posted and args.dry_run:
        print("\n\n---\n\n".join(posted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
