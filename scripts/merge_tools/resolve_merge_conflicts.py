#!/usr/bin/env python3
"""Resolve or classify upstream merge conflicts using the shared policy map."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from upstream_merge_policy import Classification, Strategy, classify_paths, load_strategy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY_FILE = REPO_ROOT / "scripts" / "merge_tools" / "hermes-merge-conflict-strategies.json"


OVERLAY_ACTIONS = frozenset({"official_with_overlay", "manual_api_followup"})


def merge_base_sha(upstream_ref: str, old_head: str) -> str:
    result = run_git(["merge-base", old_head, upstream_ref], check=False)
    return result.stdout.strip()


def merge_file_overlay(
    target_path: str,
    upstream_ref: str,
    base_sha: str,
    old_head: str,
    *,
    dry_run: bool,
    sanitizers: dict[str, dict[str, object]] | None = None,
) -> bool:
    """Take upstream file, then replay custom delta from merge-base..old_head."""
    if dry_run:
        return True

    if sanitizers and target_path in sanitizers:
        from apply_three_way_overlay import three_way_merge

        code, merged = three_way_merge(
            target_path,
            base_sha,
            upstream_ref,
            old_head,
            sanitizers=sanitizers,
        )
        if code == 2 or "<<<<<<<" in merged:
            return False
        target = REPO_ROOT / target_path
        target.write_text(merged, encoding="utf-8", newline="\n")
        run_git(["add", "--", target_path], check=False)
        return True

    run_git(["checkout", upstream_ref, "--", target_path], check=False)
    patch_cmd = ["git", "diff", f"{base_sha}..{old_head}", "--", target_path]
    patch_payload = subprocess.run(
        patch_cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout
    if not patch_payload.strip():
        run_git(["add", "--", target_path], check=False)
        return True

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".diff",
        delete=False,
    ) as handle:
        handle.write(patch_payload)
        patch_file = handle.name

    apply_res = subprocess.run(
        ["git", "apply", "--3way", "--whitespace=nowarn", patch_file],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    Path(patch_file).unlink(missing_ok=True)
    if apply_res.returncode != 0:
        return False
    run_git(["add", "--", target_path], check=False)
    return True


class Resolver:
    def __init__(
        self,
        upstream_ref: str,
        dry_run: bool,
        *,
        old_head: str = "",
        merge_base: str = "",
        overlay_sanitizers: dict[str, dict[str, object]] | None = None,
    ):
        self.upstream_ref = upstream_ref
        self.dry_run = dry_run
        self.old_head = old_head or run_git(["rev-parse", "HEAD"], check=False).stdout.strip()
        self.merge_base = merge_base or merge_base_sha(upstream_ref, self.old_head)
        self.overlay_sanitizers = overlay_sanitizers or {}
        self.actions: list[dict[str, str]] = []

    def run(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        print(f"  $ {' '.join(cmd)}")
        if self.dry_run:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def record(self, classification: Classification, result: str) -> None:
        self.actions.append(
            {
                "path": classification.path,
                "action": classification.action,
                "result": result,
                "note": classification.note,
                "pattern": classification.pattern,
            },
        )

    def git_checkout_upstream(self, path: str) -> subprocess.CompletedProcess[str]:
        return self.run(["git", "checkout", self.upstream_ref, "--", path], check=False)

    def git_checkout_head(self, path: str) -> subprocess.CompletedProcess[str]:
        return self.run(["git", "checkout", "HEAD", "--", path], check=False)

    def git_add(self, path: str) -> None:
        self.run(["git", "add", "--", path], check=False)

    def git_rm(self, path: str) -> None:
        self.run(["git", "rm", "-f", "--", path], check=False)

    def resolve_upstream(self, path: str) -> None:
        result = self.git_checkout_upstream(path)
        if result.returncode != 0:
            self.git_rm(path)
            return
        self.git_add(path)

    def resolve_preserve_custom(self, path: str) -> None:
        self.git_checkout_head(path)
        self.git_add(path)

    def resolve_drop_generated(self, path: str) -> None:
        result = self.run(["git", "checkout", self.upstream_ref, "--", path], check=False)
        if result.returncode == 0:
            self.git_add(path)
            return
        file_path = REPO_ROOT / path
        if not self.dry_run and file_path.exists():
            file_path.unlink()
        self.git_rm(path)

    def apply_action(self, path: str, action: str) -> str:
        if action == "upstream":
            self.resolve_upstream(path)
            return "resolved"
        if action == "preserve_custom":
            self.resolve_preserve_custom(path)
            return "resolved"
        if action == "drop_generated":
            self.resolve_drop_generated(path)
            return "resolved"
        if action in OVERLAY_ACTIONS:
            if self.dry_run:
                return "overlay_planned"
            if merge_file_overlay(
                path,
                self.upstream_ref,
                self.merge_base,
                self.old_head,
                dry_run=False,
                sanitizers=self.overlay_sanitizers,
            ):
                return "overlay_applied"
            return "overlay_failed"
        raise ValueError(f"Unknown action: {action}")


def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def unresolved_files() -> list[str]:
    result = run_git(["diff", "--name-only", "--diff-filter=U"], check=False)
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def load_preclassified_paths(path: Path) -> list[Classification] | None:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped.startswith("{") and not stripped.startswith("["):
        return None

    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = payload.get("classifications")
    if not isinstance(payload, list):
        return None

    classifications: list[Classification] = []
    for item in payload:
        if not isinstance(item, dict) or "path" not in item or "action" not in item:
            return None
        classifications.append(
            Classification(
                path=str(item["path"]),
                action=str(item["action"]),
                note=str(item.get("note", "")),
                pattern=str(item.get("pattern", "*")),
                touched_upstream=bool(item.get("touched_upstream", False)),
                touched_custom=bool(item.get("touched_custom", False)),
            ),
        )
    return classifications


def read_paths_file(path: Path) -> list[str]:
    return [
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_classifications_for_unresolved(
    classifications: list[Classification],
    unresolved: list[str],
    strategy: Strategy,
) -> list[Classification]:
    by_path = {classification.path: classification for classification in classifications}
    missing_paths = [path for path in unresolved if path not in by_path]
    fallback_classifications = {
        classification.path: classification
        for classification in classify_paths(missing_paths, strategy)
    }
    return [
        by_path.get(path) or fallback_classifications[path]
        for path in unresolved
    ]


def summarize_results(
    classifications: list[Classification],
    resolver: Resolver,
    blocker_actions: frozenset[str],
) -> tuple[list[str], list[str]]:
    blocked_paths = [
        classification.path
        for classification in classifications
        if classification.action in blocker_actions
        and not any(
            action["path"] == classification.path
            and action["result"] in {"overlay_planned", "overlay_applied"}
            for action in resolver.actions
        )
    ]
    overlay_failed = [
        action["path"]
        for action in resolver.actions
        if action["result"] == "overlay_failed"
    ]
    blocked_paths = sorted(set(blocked_paths) | set(overlay_failed))
    unresolved = unresolved_files()
    return blocked_paths, unresolved


def write_markdown_log(
    log_path: Path,
    resolver: Resolver,
    classifications: list[Classification],
    blocked_paths: list[str],
    unresolved: list[str],
) -> None:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Merge Conflict Resolution Log",
        "",
        f"- Timestamp (UTC): `{timestamp}`",
        f"- Upstream ref: `{resolver.upstream_ref}`",
        f"- Dry run: `{resolver.dry_run}`",
        f"- Classified paths: `{len(classifications)}`",
        f"- Blocked paths: `{len(blocked_paths)}`",
        f"- Remaining unresolved conflicts: `{len(unresolved)}`",
        "",
        "## Actions",
        "",
    ]
    if resolver.actions:
        for action in resolver.actions:
            note_text = f" - {action['note']}" if action["note"] else ""
            lines.append(
                f"- `{action['path']}`: `{action['action']}` -> `{action['result']}`{note_text}",
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Blocked Paths", ""])
    if blocked_paths:
        lines.extend([f"- `{path}`" for path in blocked_paths])
    else:
        lines.append("- none")
    lines.extend(["", "## Remaining Conflicts", ""])
    if unresolved:
        lines.extend([f"- `{path}`" for path in unresolved])
    else:
        lines.append("- none")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json_report(
    report_path: Path,
    strategy: Strategy,
    classifications: list[Classification],
    blocked_paths: list[str],
    unresolved: list[str],
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "default_action": strategy.default_action,
        "blocker_actions": sorted(strategy.blocker_actions),
        "classifications": [
            {
                "path": classification.path,
                "action": classification.action,
                "note": classification.note,
                "pattern": classification.pattern,
                "touched_upstream": classification.touched_upstream,
                "touched_custom": classification.touched_custom,
            }
            for classification in classifications
        ],
        "blocked_paths": blocked_paths,
        "unresolved_conflicts": unresolved,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve merge conflicts using a strategy map.")
    parser.add_argument("--upstream-ref", default="upstream/main")
    parser.add_argument("--strategy-file", default=str(DEFAULT_STRATEGY_FILE))
    parser.add_argument("--paths-file", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-unresolved", action="store_true")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--old-head",
        default="",
        help="Pre-merge HEAD for overlay replay (defaults to current HEAD).",
    )
    parser.add_argument(
        "--merge-base",
        default="",
        help="Explicit merge-base SHA for overlay replay.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategy_file = Path(args.strategy_file)
    if not strategy_file.is_absolute():
        strategy_file = (REPO_ROOT / strategy_file).resolve()
    if not strategy_file.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_file}")

    strategy = load_strategy(strategy_file)
    strategy_payload = json.loads(strategy_file.read_text(encoding="utf-8"))
    from overlay_sanitize import load_overlay_sanitizers

    overlay_sanitizers = load_overlay_sanitizers(strategy_payload)
    if args.paths_file:
        paths_file = Path(args.paths_file)
        if not paths_file.is_absolute():
            paths_file = (REPO_ROOT / paths_file).resolve()
        classifications = load_preclassified_paths(paths_file)
        if classifications is None:
            paths = read_paths_file(paths_file)
            classifications = classify_paths(paths, strategy)
    else:
        classifications = classify_paths(unresolved_files(), strategy)
    if args.only_unresolved:
        classifications = select_classifications_for_unresolved(
            classifications,
            unresolved_files(),
            strategy,
        )
    resolver = Resolver(
        upstream_ref=args.upstream_ref,
        dry_run=args.dry_run,
        old_head=args.old_head,
        merge_base=args.merge_base,
        overlay_sanitizers=overlay_sanitizers,
    )
    print(f"Detected paths: {len(classifications)}")
    if resolver.merge_base:
        print(f"Overlay merge-base: {resolver.merge_base}")
        print(f"Overlay old-head: {resolver.old_head}")

    for classification in classifications:
        result = resolver.apply_action(classification.path, classification.action)
        resolver.record(classification, result)

    blocked_paths, unresolved = summarize_results(
        classifications=classifications,
        resolver=resolver,
        blocker_actions=strategy.blocker_actions,
    )

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_path = REPO_ROOT / "_docs" / f"merge-conflict-resolution-{stamp}.md"
    if not log_path.is_absolute():
        log_path = (REPO_ROOT / log_path).resolve()
    write_markdown_log(log_path, resolver, classifications, blocked_paths, unresolved)
    print(f"Wrote log: {log_path}")

    if args.report_json:
        report_path = Path(args.report_json)
        if not report_path.is_absolute():
            report_path = (REPO_ROOT / report_path).resolve()
        write_json_report(report_path, strategy, classifications, blocked_paths, unresolved)
        print(f"Wrote report: {report_path}")

    if blocked_paths:
        print("Blocked paths:")
        for path in blocked_paths:
            print(f" - {path}")
    if unresolved:
        print("Remaining unresolved files:")
        for path in unresolved:
            print(f" - {path}")

    if args.strict and (blocked_paths or unresolved):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
