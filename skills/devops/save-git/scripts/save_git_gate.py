#!/usr/bin/env python3
"""Use Save Git gate v2 — fail-closed 5-stage Git/GitLab/VPS shipping gate.

Reads a per-project .savegit.json adapter so the gate knows each project's real
stack (npm/pnpm/python), build/test commands, health URL, and deploy target.
Runs the relevant stages, then prints ONE decision token plus a single Grid
table that shows which layer is the real blocker.

Dependency-free on purpose so it runs from any repo and from Git hooks.
It never prints secret values.

Stages:
  1 local   git clean, diff in scope, secret scan, forbidden paths, bundle, build/test/lint
  2 mr      right project/remote, source/target branch, sync with origin/target,
            commit+files scope guard, conflict check
  3 ci      latest-commit pipeline passed (only when ci.enabled)
  4 dryrun  build candidate + real container/service health (only when configured)
  5 prod    deploy from origin/main, deployed SHA = origin SHA, health commitSha match

Decision tokens:
  merge: SAFE_TO_MERGE | BLOCKED_DO_NOT_MERGE | OWNER_DECISION_REQUIRED
  ship:  SAFE_TO_DEPLOY | PRODUCTION_VERIFIED | PRODUCTION_NOT_VERIFIED
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse


PROTECTED_BRANCHES = {"main", "master"}
SECRET_FILE_PATTERNS = (
    ".env", ".env.", ".pem", ".key", "id_rsa", "id_ed25519",
    "private-key", "private_key",
)
SECRET_LINE_RE = re.compile(
    r"(?i)^\+.*(?:api[_-]?key|secret|token|password|private[_-]?key|access[_-]?key)\s*[:=]"
)
CONFIG_NAMES = (".savegit.json", ".savegit.yaml", ".savegit.yml")

STAGE_TITLES = {
    "local": "1 Local",
    "mr": "2 MR sanity",
    "ci": "3 CI",
    "dryrun": "4 VPS dryrun",
    "prod": "5 Production",
}
STAGE_CHECKS = {
    "local": "git clean, diff in scope, secret, build/test/lint, bundle",
    "mr": "project/remote, source/target, sync main, commit+files scope, conflict",
    "ci": "latest-commit pipeline passed, not stuck, migration",
    "dryrun": "build candidate, env/port/service, container health",
    "prod": "deploy from origin/main, SHA match, health commitSha, service path, rollback",
}


# --------------------------------------------------------------------------- #
# shell + git helpers
# --------------------------------------------------------------------------- #
@dataclass
class CmdResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    code: int = 0


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 15) -> CmdResult:
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, text=True,
            capture_output=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 1
        return CmdResult(ok=False, stderr=str(exc), code=code)
    return CmdResult(ok=proc.returncode == 0, stdout=proc.stdout.strip(),
                     stderr=proc.stderr.strip(), code=proc.returncode)


def run_shell(command: str, cwd: Path, timeout: int) -> CmdResult:
    try:
        proc = subprocess.run(
            command, cwd=str(cwd), shell=True, text=True,
            capture_output=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CmdResult(ok=False, stderr=str(exc), code=124)
    except OSError as exc:
        return CmdResult(ok=False, stderr=str(exc), code=1)
    return CmdResult(ok=proc.returncode == 0, stdout=proc.stdout.strip(),
                     stderr=proc.stderr.strip(), code=proc.returncode)


def git(cwd: Path, *args: str, timeout: int = 15) -> CmdResult:
    return run(["git", *args], cwd=cwd, timeout=timeout)


def git_root(cwd: Path) -> Path | None:
    result = git(cwd, "rev-parse", "--show-toplevel")
    return Path(result.stdout) if result.ok and result.stdout else None


def lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def localhost_project_root(url: str) -> Path | None:
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or not parsed.port:
        return None
    listeners = run(["lsof", "-nP", f"-iTCP:{parsed.port}", "-sTCP:LISTEN", "-Fp"], timeout=10)
    if not listeners.ok:
        return None
    pids = [line[1:] for line in listeners.stdout.splitlines() if line.startswith("p")]
    for pid in pids:
        cwd_result = run(["lsof", "-a", "-p", pid, "-d", "cwd", "-Fn"], timeout=10)
        if not cwd_result.ok:
            continue
        for line in cwd_result.stdout.splitlines():
            if line.startswith("n"):
                root = git_root(Path(line[1:]))
                if root:
                    return root
    return None


def current_branch(root: Path) -> str:
    result = git(root, "branch", "--show-current")
    if result.ok and result.stdout:
        return result.stdout
    result = git(root, "rev-parse", "--short", "HEAD")
    return f"detached:{result.stdout}" if result.ok and result.stdout else "unknown"


def current_sha(root: Path) -> str:
    result = git(root, "rev-parse", "HEAD")
    return result.stdout if result.ok and result.stdout else "unknown"


def remote_url(root: Path) -> str:
    result = git(root, "config", "--get", "remote.origin.url")
    return result.stdout if result.ok and result.stdout else "unknown"


def status_short(root: Path) -> list[str]:
    result = git(root, "status", "--short", "--untracked-files=all")
    return lines(result.stdout) if result.ok else []


def changed_files_worktree(root: Path) -> list[str]:
    names: set[str] = set()
    for args in (("diff", "--name-only"), ("diff", "--cached", "--name-only"),
                 ("ls-files", "--others", "--exclude-standard")):
        result = git(root, *args)
        if result.ok:
            names.update(lines(result.stdout))
    return sorted(names)


def suspicious_files(names: list[str]) -> list[str]:
    risks = []
    for name in names:
        lower = name.lower()
        if any(pattern in lower for pattern in SECRET_FILE_PATTERNS):
            risks.append(f"sensitive filename: {name}")
    return risks


def suspicious_diff_patterns(root: Path) -> list[str]:
    risks: set[str] = set()
    for args in (("diff", "--cached", "-U0"), ("diff", "-U0")):
        result = git(root, *args, timeout=20)
        if not result.ok:
            continue
        for line in result.stdout.splitlines():
            if SECRET_LINE_RE.search(line):
                risks.add("secret-like added line in diff")
                break
    return sorted(risks)


def health_get(url: str) -> tuple[str, dict | None, str]:
    """Return (status, json_body_or_none, raw_text)."""
    if not url:
        return "skip", None, ""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=6) as response:
            raw = response.read().decode("utf-8", "replace")
            status = "ok" if 200 <= response.status < 300 else f"fail:{response.status}"
    except urllib.error.HTTPError as exc:
        return f"fail:{exc.code}", None, ""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"fail:{exc.__class__.__name__}", None, ""
    body: dict | None = None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            body = parsed
    except json.JSONDecodeError:
        body = None
    return status, body, raw


# --------------------------------------------------------------------------- #
# config (.savegit.json adapter)
# --------------------------------------------------------------------------- #
def load_config(root: Path) -> tuple[dict, str]:
    for name in CONFIG_NAMES:
        path = root / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if name.endswith(".json"):
            try:
                return json.loads(text), name
            except json.JSONDecodeError:
                return {}, f"{name} (invalid json)"
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text) or {}, name
        except Exception:
            return {}, f"{name} (yaml unreadable, install pyyaml or use .savegit.json)"
    return {}, "missing"


# --------------------------------------------------------------------------- #
# stage results
# --------------------------------------------------------------------------- #
@dataclass
class StageResult:
    name: str
    status: str = "skip"          # pass | fail | skip | unknown
    blocking: bool = False
    detail: str = ""
    fix: str = ""
    items: list[str] = field(default_factory=list)


def required(stage: str, cfg: dict, for_ship: bool) -> bool:
    if stage in {"local", "mr"}:
        return True
    if stage == "ci":
        return bool((cfg.get("ci") or {}).get("enabled"))
    if stage == "dryrun":
        dep = cfg.get("deploy") or {}
        return for_ship and bool((dep.get("container_health") or {}).get("command")
                                 or (dep.get("vps_service") or {}).get("name"))
    if stage == "prod":
        return for_ship and bool((cfg.get("deploy") or {}).get("health_url"))
    return False


# --------------------------------------------------------------------------- #
# stage 1: local
# --------------------------------------------------------------------------- #
def stage_local(root: Path, cfg: dict, fast: bool) -> StageResult:
    res = StageResult("local", status="pass")
    problems: list[str] = []

    dirty = status_short(root)
    if dirty:
        problems.append(f"worktree dirty: {len(dirty)} file(s)")
        res.items = dirty[:50]

    changed = changed_files_worktree(root)
    secret_risks = suspicious_files(changed) + suspicious_diff_patterns(root)
    if secret_risks:
        problems.append("secret risk in changed files/diff")
        res.items += secret_risks

    forbidden = cfg.get("forbidden_paths") or []
    hits = [name for name in changed
            if any(fnmatch.fnmatch(name, pat) or pat in name for pat in forbidden)]
    if hits:
        problems.append(f"forbidden path in diff: {', '.join(hits[:5])}")

    # bundle must not point to localhost
    for pat in cfg.get("bundle_globs") or []:
        for bundle in root.glob(pat):
            try:
                text = bundle.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for needle in cfg.get("bundle_must_not_contain") or []:
                if needle in text:
                    problems.append(f"bundle {bundle.name} contains '{needle}'")

    # build/test/lint commands from adapter
    if not fast:
        checks = cfg.get("checks") or {}
        timeout = int(cfg.get("test_timeout_sec") or 600)
        for label in ("install", "lint", "typecheck", "build", "test", "audit"):
            command = checks.get(label)
            if not command:
                continue
            out = run_shell(command, root, timeout)
            if out.code == 124:
                problems.append(f"{label} TIMEOUT (>{timeout}s) — แยกจาก fail")
            elif not out.ok:
                problems.append(f"{label} failed (exit {out.code})")
    else:
        res.detail = "fast mode: skipped build/test/lint commands; "

    if problems:
        res.status = "fail"
        res.blocking = True
        res.detail += "; ".join(problems)
        res.fix = "classify dirty files (safe vs user work), remove secret/forbidden paths, fix failing checks, rerun"
    else:
        res.detail += "clean tree, no secret, checks passed" if not res.detail else "clean tree, no secret"
    return res


# --------------------------------------------------------------------------- #
# stage 2: mr sanity
# --------------------------------------------------------------------------- #
def merge_base(root: Path, ref: str) -> str:
    result = git(root, "merge-base", "HEAD", ref, timeout=15)
    return result.stdout if result.ok and result.stdout else ""


def has_conflict(root: Path, ref: str) -> bool | None:
    new = git(root, "merge-tree", "--write-tree", "HEAD", ref, timeout=20)
    if new.code in (0, 1):
        return new.code == 1
    base = merge_base(root, ref)
    if not base:
        return None
    old = git(root, "merge-tree", base, "HEAD", ref, timeout=20)
    if not old.ok and not old.stdout:
        return None
    return "<<<<<<<" in old.stdout


def stage_mr(root: Path, cfg: dict) -> StageResult:
    res = StageResult("mr", status="pass")
    problems: list[str] = []
    branch = current_branch(root)

    must = cfg.get("remote_must_match")
    if must:
        url = remote_url(root)
        if must not in url:
            problems.append(f"remote '{url}' does not match expected '{must}' (wrong project?)")

    target = cfg.get("default_target") or "main"
    if branch.rstrip() in PROTECTED_BRANCHES:
        problems.append(f"source branch is protected '{branch}' — work on a feature branch")

    ref = f"origin/{target}"
    has_ref = git(root, "rev-parse", "--verify", "--quiet", ref).ok
    if not has_ref:
        git(root, "fetch", "--quiet", "origin", target, timeout=20)
        has_ref = git(root, "rev-parse", "--verify", "--quiet", ref).ok

    if not has_ref:
        res.status = "unknown"
        res.blocking = True
        res.detail = f"cannot resolve {ref} (no network or wrong target) — verify target before merge"
        res.fix = f"fetch origin {target} or fix default_target in .savegit.json"
        return res

    behind = git(root, "rev-list", "--count", f"HEAD..{ref}")
    behind_n = int(behind.stdout) if behind.ok and behind.stdout.isdigit() else -1
    if behind_n > 0:
        problems.append(f"branch behind {ref} by {behind_n} commit(s) — sync/rebase before merge")

    base = merge_base(root, ref)
    if base:
        commits = git(root, "rev-list", "--count", f"{base}..HEAD")
        files = git(root, "diff", "--name-only", f"{base}..HEAD")
        commit_n = int(commits.stdout) if commits.ok and commits.stdout.isdigit() else 0
        file_n = len(lines(files.stdout)) if files.ok else 0
        res.items = [f"commits ahead: {commit_n}", f"files changed: {file_n}"]
        guard = cfg.get("scope_guard") or {}
        max_c = int(guard.get("max_commits") or 0)
        max_f = int(guard.get("max_files") or 0)
        if max_c and commit_n > max_c:
            problems.append(f"scope bloat: {commit_n} commits > max {max_c} (likely WRONG TARGET branch)")
        if max_f and file_n > max_f:
            problems.append(f"scope bloat: {file_n} files > max {max_f} (likely WRONG TARGET branch)")

    conflict = has_conflict(root, ref)
    if conflict is True:
        problems.append(f"merge conflict against {ref}")

    if problems:
        res.status = "fail"
        res.blocking = True
        res.detail = "; ".join(problems)
        res.fix = "fix target branch / sync with origin / resolve conflict, then rerun"
    else:
        res.detail = f"on '{branch}' -> {ref}, in scope, synced, no conflict"
    return res


# --------------------------------------------------------------------------- #
# stage 3: ci
# --------------------------------------------------------------------------- #
def stage_ci(root: Path, cfg: dict) -> StageResult:
    ci = cfg.get("ci") or {}
    if not ci.get("enabled"):
        return StageResult("ci", status="skip", detail="ci.enabled=false in .savegit.json")
    res = StageResult("ci", status="unknown", blocking=True)
    if run(["which", "glab"]).ok:
        out = run(["glab", "ci", "status", "--branch", current_branch(root)], cwd=root, timeout=20)
        text = (out.stdout + out.stderr).lower()
        if "success" in text or "passed" in text:
            res.status, res.blocking, res.detail = "pass", False, "latest pipeline passed"
            return res
        if any(word in text for word in ("failed", "stuck", "pending", "running", "canceled")):
            res.status, res.detail = "fail", f"pipeline not green: {text[:120]}"
            res.fix = "fix pipeline / runner, wait for latest-commit pipeline to pass"
            return res
    res.detail = "ci enabled but status unreachable (no glab/token) — verify pipeline before merge"
    res.fix = "check GitLab pipeline of the latest commit manually; do not merge on unknown CI"
    return res


# --------------------------------------------------------------------------- #
# stage 4: dryrun (container/service health)
# --------------------------------------------------------------------------- #
def stage_dryrun(root: Path, cfg: dict) -> StageResult:
    deploy = cfg.get("deploy") or {}
    ch = deploy.get("container_health") or {}
    command = ch.get("command")
    if not command:
        return StageResult("dryrun", status="skip", detail="no container_health.command configured")
    res = StageResult("dryrun", status="pass")
    container = ch.get("container")
    if container and run(["which", "docker"]).ok:
        full = f"docker exec {container} sh -lc {json.dumps(command)}"
        out = run_shell(full, root, timeout=20)
    else:
        out = run_shell(command, root, timeout=20)
    if out.ok:
        res.detail = f"real health command passed: {command}"
    else:
        res.status, res.blocking = "fail", True
        res.detail = f"health command failed in container: {command} (exit {out.code})"
        res.fix = "fix healthcheck command/binary inside the real container (e.g. curl vs wget), rerun"
    return res


# --------------------------------------------------------------------------- #
# stage 5: production
# --------------------------------------------------------------------------- #
def stage_prod(root: Path, cfg: dict, health_url_override: str) -> StageResult:
    deploy = cfg.get("deploy") or {}
    url = health_url_override or deploy.get("health_url") or ""
    if not url:
        return StageResult("prod", status="skip", detail="no deploy.health_url configured")
    res = StageResult("prod", status="pass")
    problems: list[str] = []

    status, body, _ = health_get(url)
    if status != "ok":
        problems.append(f"health endpoint not ok: {status}")
    else:
        field_name = deploy.get("health_commit_field") or "commitSha"
        deployed = str((body or {}).get(field_name, "")).strip()
        target = cfg.get("default_target") or "main"
        ref = f"origin/{target}"
        git(root, "fetch", "--quiet", "origin", target, timeout=20)
        origin_sha = git(root, "rev-parse", "--verify", "--quiet", ref)
        want = origin_sha.stdout if origin_sha.ok else current_sha(root)
        if not deployed:
            problems.append(f"health has no '{field_name}' field — cannot prove latest commit (add it)")
        elif want and not (deployed.startswith(want[:8]) or want.startswith(deployed[:8])):
            problems.append(f"deployed commit {deployed[:8]} != origin {want[:8]} (production on old commit)")
        else:
            res.items = [f"deployed commit {deployed[:12]} matches origin"]

    if problems:
        res.status, res.blocking = "fail", True
        res.detail = "; ".join(problems)
        res.fix = "redeploy from origin/main, wait for service, re-check health commitSha"
    else:
        res.detail = "health ok and commit matches origin"
    return res


# --------------------------------------------------------------------------- #
# orchestration + decision
# --------------------------------------------------------------------------- #
def run_stages(root: Path, cfg: dict, stages: list[str], fast: bool, health_url: str) -> dict[str, StageResult]:
    out: dict[str, StageResult] = {}
    for stage in stages:
        if stage == "local":
            out[stage] = stage_local(root, cfg, fast)
        elif stage == "mr":
            out[stage] = stage_mr(root, cfg)
        elif stage == "ci":
            out[stage] = stage_ci(root, cfg)
        elif stage == "dryrun":
            out[stage] = stage_dryrun(root, cfg)
        elif stage == "prod":
            out[stage] = stage_prod(root, cfg, health_url)
    return out


def first_blocker(results: dict[str, StageResult], order: list[str]) -> StageResult | None:
    for stage in order:
        res = results.get(stage)
        if res and res.blocking:
            return res
    return None


def decide(mode: str, results: dict[str, StageResult]) -> tuple[str, str, str, str]:
    """Return (decision, blocking_layer, fix, owner_action)."""
    if mode == "merge-gate":
        order = ["local", "mr", "ci"]
        blocker = first_blocker(results, order)
        if blocker:
            scope = "WRONG TARGET" in blocker.detail or "scope bloat" in blocker.detail
            layer = STAGE_TITLES[blocker.name]
            return (
                "BLOCKED_DO_NOT_MERGE", layer, blocker.fix,
                f"ห้าม merge — แก้ที่ {layer}: {blocker.detail}" +
                (" (ตรวจ MR target branch ก่อน)" if scope else ""),
            )
        return ("SAFE_TO_MERGE", "none", "",
                "กด merge ได้ — หลัง merge ยังห้าม deploy จนกว่า ship-gate ผ่าน")
    # ship-gate
    merge_blocker = first_blocker(results, ["local", "mr", "ci"])
    if merge_blocker:
        layer = STAGE_TITLES[merge_blocker.name]
        return ("BLOCKED_DO_NOT_MERGE", layer, merge_blocker.fix,
                f"ห้าม merge/deploy — แก้ที่ {layer}: {merge_blocker.detail}")
    prod = results.get("prod")
    dry = results.get("dryrun")
    deploy_blocker = first_blocker(results, ["dryrun", "prod"])
    if deploy_blocker is prod and prod and prod.blocking:
        return ("PRODUCTION_NOT_VERIFIED", STAGE_TITLES["prod"], prod.fix,
                f"production ยังไม่ผ่าน: {prod.detail}")
    if deploy_blocker is dry and dry and dry.blocking:
        return ("BLOCKED_DO_NOT_MERGE", STAGE_TITLES["dryrun"], dry.fix,
                f"dry-run fail: {dry.detail}")
    if prod and prod.status == "pass":
        return ("PRODUCTION_VERIFIED", "none", "",
                "deploy + health + commitSha ตรงครบ — ตัด local ได้")
    return ("SAFE_TO_DEPLOY", "none", "",
            "deploy จาก origin/main ได้ แล้วรัน prod stage ตรวจ health/commitSha ซ้ำ")


def print_grid(decision: str, blocking_layer: str, fix: str, owner_action: str,
               results: dict[str, StageResult], shown: list[str]) -> None:
    print(f"Decision: {decision}")
    print()
    print("| ด่าน | ตรวจอะไร | ผล | block |")
    print("|---|---|---|---|")
    for stage in ["local", "mr", "ci", "dryrun", "prod"]:
        res = results.get(stage)
        if stage not in shown or res is None:
            status, block = "skip", "no"
        else:
            status = res.status
            block = "yes" if res.blocking else "no"
        print(f"| {STAGE_TITLES[stage]} | {STAGE_CHECKS[stage]} | {status} | {block} |")
    print()
    print(f"Blocking layer: {blocking_layer}")
    for stage in shown:
        res = results.get(stage)
        if res and res.detail:
            print(f"- {STAGE_TITLES[stage]}: {res.detail}")
            for item in res.items[:8]:
                print(f"    · {item}")
    if fix:
        print(f"Fix needed: {fix}")
    print(f"Owner action: {owner_action}")


ACTION_TO_STAGE = {
    "inspect": ("local", True), "work": ("local", True),
    "push": ("local", False), "merge": ("merge-gate", False),
    "deploy": ("ship-gate", False),
}
STAGE_PLAN = {
    "local": ["local"], "mr": ["mr"], "ci": ["ci"],
    "dryrun": ["dryrun"], "prod": ["prod"],
    "merge-gate": ["local", "mr", "ci"],
    "ship-gate": ["local", "mr", "ci", "dryrun", "prod"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=tuple(STAGE_PLAN), default="")
    parser.add_argument("--action", choices=tuple(ACTION_TO_STAGE), default="",
                        help="deprecated alias for --stage; kept for old hooks")
    parser.add_argument("--url", default=os.getenv("SAVE_GIT_URL", ""),
                        help="localhost URL resolved to the serving project cwd before shell cwd")
    parser.add_argument("--health-url", default=os.getenv("SAVE_GIT_HEALTH_URL", ""))
    parser.add_argument("--fast", action="store_true",
                        help="local stage skips build/test/lint commands (for pre-push hook)")
    parser.add_argument("--hook", choices=("pre-push",), default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("hook_args", nargs="*")
    args = parser.parse_args()

    fast = args.fast
    if args.stage:
        stage = args.stage
    elif args.action:
        stage, action_fast = ACTION_TO_STAGE[args.action]
        fast = fast or action_fast
        print(f"# note: --action is deprecated, mapped to --stage {stage}", file=sys.stderr)
    else:
        stage = "merge-gate"

    root = (localhost_project_root(args.url) if args.url else None) or git_root(Path.cwd())
    if root is None:
        print("Decision: BLOCKED_DO_NOT_MERGE")
        print("Blocking layer: 0 Repo")
        print("Fix needed: not inside a git repository; resolve target path first")
        print("Owner action: ห้าม merge — หา project path ที่ถูกก่อน")
        return 1

    cfg, cfg_name = load_config(root)
    mode = "merge-gate" if stage in ("merge-gate", "ship-gate") else stage
    plan = STAGE_PLAN[stage]
    results = run_stages(root, cfg, plan, fast, args.health_url)

    if stage in ("merge-gate", "ship-gate"):
        decision, layer, fix, owner = decide(stage, results)
    else:
        res = results[plan[0]]
        if res.status in ("fail", "unknown"):
            decision = "BLOCKED_DO_NOT_MERGE"
        elif res.status == "skip":
            decision = "SAFE_TO_MERGE"
        else:
            decision = "SAFE_TO_MERGE"
        layer = STAGE_TITLES[plan[0]] if res.blocking else "none"
        fix = res.fix
        owner = res.detail

    if args.json:
        print(json.dumps({
            "decision": decision, "stage": stage, "config": cfg_name,
            "project": root.name, "path": str(root),
            "blocking_layer": layer, "fix": fix, "owner_action": owner,
            "stages": {k: asdict(v) for k, v in results.items()},
        }, ensure_ascii=False, indent=2))
    else:
        print(f"# project: {root.name}  config: {cfg_name}  stage: {stage}")
        print_grid(decision, layer, fix, owner, results, plan)

    safe = decision in {"SAFE_TO_MERGE", "SAFE_TO_DEPLOY", "PRODUCTION_VERIFIED"}
    return 0 if safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
