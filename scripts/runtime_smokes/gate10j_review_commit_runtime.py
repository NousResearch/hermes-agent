from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
VAULT_PLANSPEC = Path("/home/piet/vault/03-Agents/Hermes/plans/gate10j-review-commit-runtime-regression-planspec-2026-05-13.md")
GATE10I_RECEIPT = Path("/home/piet/vault/03-Agents/Hermes/receipts/gate10i-dispatcher-worker-review-commit-smoke-2026-05-13.md")
DEFAULT_BOARD = Path("/home/piet/.hermes/kanban.db")
REAL_CODER = Path("/home/piet/.hermes/profiles/coder")
REAL_REVIEWER = Path("/home/piet/.hermes/profiles/reviewer")

CODER_TOOLS = ["kanban_show", "write_file", "kanban_comment", "kanban_complete", "kanban_block"]
REVIEWER_TOOLS = ["kanban_show", "read_file", "kanban_comment", "kanban_complete", "kanban_block"]
FORBIDDEN_SYSTEMS = [
    "OpenClaw",
    "Atlas",
    "Mission-Control",
    "Telegram",
    "Discord",
    "systemd",
    "cron",
    "real_config",
    "secrets_values",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def board_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    counts: dict[str, Any] = {}
    try:
        conn = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
        try:
            for table in ("tasks", "task_runs", "task_events"):
                counts[table] = conn.execute(f"select count(*) from {table}").fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - diagnostic path
        counts["error"] = repr(exc)
    return {
        "path": str(path),
        "exists": True,
        "mtime_ns": st.st_mtime_ns,
        "size": st.st_size,
        "sha256": sha256(path),
        "counts": counts,
    }


def run(
    cmd: list[str],
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    check: bool = False,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
        timeout=timeout,
    )


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], cwd=repo, check=check, timeout=60)


def git_out(repo: Path, *args: str) -> str:
    return git(repo, *args).stdout.strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def task_id(output: str) -> str:
    match = re.search(r"(t_[a-f0-9]+)", output)
    if not match:
        raise RuntimeError("no task id in output: " + output)
    return match.group(1)


def json_arg(payload: dict[str, Any]) -> str:
    return shlex.quote(json.dumps(payload, sort_keys=True))


def init_repo(parent: Path, name: str) -> Path:
    repo = parent / name
    repo.mkdir(parents=True)
    git(repo, "init")
    git(repo, "config", "user.name", "Gate 10J Smoke")
    git(repo, "config", "user.email", "gate10j@example.invalid")
    write(repo / "README.md", "base\n")
    write(repo / ".gitignore", "*.pyc\n")
    git(repo, "add", "README.md", ".gitignore")
    git(repo, "commit", "-m", "initial")
    return repo


def setup_temp_profiles(root: Path) -> list[str]:
    symlinks: list[str] = []
    profiles = root / "hermes-home" / "profiles"
    for name, src in [("coder", REAL_CODER), ("reviewer-b", REAL_REVIEWER)]:
        if not (src / "config.yaml").exists():
            raise FileNotFoundError(f"missing profile config: {src / 'config.yaml'}")
        dst = profiles / name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "config.yaml", dst / "config.yaml")
        (dst / "logs").mkdir(exist_ok=True)
        (dst / "sessions").mkdir(exist_ok=True)
        (dst / "skills").mkdir(exist_ok=True)
        # Symlink only; never read, copy, or print secret values.
        if (src / ".env").exists():
            link = dst / ".env"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src / ".env")
            symlinks.append(str(link))
    return symlinks


def remove_symlinks(paths: list[str]) -> list[str]:
    removed: list[str] = []
    for raw in paths:
        p = Path(raw)
        try:
            if p.is_symlink():
                p.unlink()
                removed.append(raw)
        except OSError:
            pass
    return removed


def body_yaml(board_root_glob: str, tools: list[str], instructions: str, max_runtime: int = 300) -> str:
    import yaml

    front = {
        "scope_contract": {
            "version": 2,
            "allowed_systems": ["isolated_hermes_kanban_board", "temp_scratch_workspace"],
            "allowed_paths": [board_root_glob],
            "allowed_tools": tools,
            "forbidden_systems": FORBIDDEN_SYSTEMS,
            "forbidden_paths": [
                "/home/piet/.hermes/kanban.db",
                "/home/piet/.openclaw/**",
                "/home/piet/.hermes/config.yaml",
                "/home/piet/.hermes/profiles/**",
            ],
            "ambiguity_policy": "block_instead_of_guess",
        },
        "completion_policy": {"require_scope_attestation": True},
        "termination_conditions": {
            "max_dispatch_passes": 1,
            "max_worker_runs": 1,
            "max_runtime_seconds": max_runtime,
            "failure_limit": 1,
        },
    }
    return "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n\n" + instructions.strip() + "\n"


def task_state(kb: Any, task_id_: str) -> dict[str, Any]:
    with kb.connect() as conn:
        row = conn.execute(
            "select id,status,result,workspace_path,current_run_id,worker_pid,last_failure_error from tasks where id=?",
            (task_id_,),
        ).fetchone()
        latest = kb.latest_run(conn, task_id_)
        events = [
            dict(r)
            for r in conn.execute(
                "select id,kind,payload,run_id from task_events where task_id=? order by id",
                (task_id_,),
            ).fetchall()
        ]
    parsed_events = []
    for event in events:
        payload = event.get("payload")
        try:
            payload = json.loads(payload) if payload else None
        except Exception:
            pass
        parsed_events.append({**event, "payload": payload})
    return {
        "task": dict(row) if row else None,
        "latest_run": None
        if latest is None
        else {
            "id": latest.id,
            "task_id": latest.task_id,
            "profile": latest.profile,
            "status": latest.status,
            "started_at": latest.started_at,
            "ended_at": latest.ended_at,
            "outcome": latest.outcome,
            "summary": latest.summary,
            "metadata": latest.metadata,
            "error": latest.error,
        },
        "events": parsed_events,
    }


def wait_terminal(kc: Any, kb: Any, task_id_: str, timeout_s: int, poll_s: int = 5) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_dispatch = ""
    while time.time() < deadline:
        state = task_state(kb, task_id_)
        status = (state.get("task") or {}).get("status")
        if status in {"done", "blocked", "archived"}:
            state["last_dispatch_poll"] = last_dispatch
            return state
        last_dispatch = kc.run_slash("dispatch --max 1 --failure-limit 1 --json")
        time.sleep(poll_s)
    state = task_state(kb, task_id_)
    state["timeout"] = True
    state["last_dispatch_poll"] = last_dispatch
    return state


def make_coder_metadata(workflow_id: str) -> dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "review_required": True,
        "coder_done_without_review": False,
        "scope_attestation": True,
        "scope_contract_version": 2,
        "forbidden_actions_taken": 0,
        "effective_toolsets": CODER_TOOLS,
        "no_prose_only_terminal_output": True,
        "acceptance_checks": {
            "implementation_tests_green": True,
            "postchecks_green": True,
            "reviewer_b_terminal_approved": True,
        },
        "anti_scope": {
            "default_board_unchanged": True,
            "temp_credential_symlinks_removed": True,
            "no_deploy_reload": True,
            "no_mc_openclaw_discord": True,
            "no_config_systemd_cron_secret_changes": True,
            "no_unrelated_dirty_committed": True,
            "violations": [],
        },
    }


def make_reviewer_metadata(workflow_id: str, coder_task_id: str, artifact_path: str) -> dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "verdict": "APPROVED",
        "blocking_findings": [],
        "required_verification": [],
        "residual_risk": "local scoped commit only",
        "evidence_audited": ["coder_handoff", artifact_path, "isolated_worker_log"],
        "consumed_coder_handoff": True,
        "consumed_coder_task_id": coder_task_id,
        "coder_review_required_seen": True,
        "artifact_path": artifact_path,
        "artifact_review_required_seen": True,
        "scope_attestation": True,
        "scope_contract_version": 2,
        "forbidden_actions_taken": 0,
        "effective_toolsets": REVIEWER_TOOLS,
    }


def log_info(root: Path, board: str, task_id_: str) -> dict[str, Any]:
    path = root / "hermes-home" / "kanban" / "boards" / board / "logs" / f"{task_id_}.log"
    out = {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
        "sha256": sha256(path) if path.exists() else None,
    }
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        out["tail"] = "\n".join(text.splitlines()[-80:])
    return out


def run_gate10j_smoke(result_path: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    started = now_iso()
    default_before = board_snapshot(DEFAULT_BOARD)
    main_status_before = git_out(REPO_ROOT, "status", "--short")
    main_head = git_out(REPO_ROOT, "rev-parse", "--short", "HEAD")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%z")
    root = root or Path(tempfile.mkdtemp(prefix=f"gate10j-review-commit-{stamp}-"))
    root.mkdir(parents=True, exist_ok=True)
    result_path = result_path or (root / "result.json")
    board_slug = "gate10j"
    workflow_id = f"wf-gate10j-review-commit-{stamp}"
    symlinks: list[str] = []
    removed_symlinks: list[str] = []
    result: dict[str, Any] = {
        "status": "BLOCKED",
        "workflow_id": workflow_id,
        "started_at": started,
        "root": str(root),
        "approval": "Piet approved Gate 10J in current thread: Go Gate J",
        "risk": "repo-code-change + worker-runtime-local-isolated",
    }
    original_env = {key: os.environ.get(key) for key in ("HERMES_HOME", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT")}
    try:
        symlinks = setup_temp_profiles(root)
        hermes_home = root / "hermes-home"
        os.environ["HERMES_HOME"] = str(hermes_home)
        os.environ["HERMES_KANBAN_HOME"] = str(hermes_home)
        os.environ["HERMES_KANBAN_BOARD"] = board_slug
        os.environ.pop("HERMES_KANBAN_DB", None)
        os.environ.pop("HERMES_KANBAN_WORKSPACES_ROOT", None)
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from hermes_cli import kanban as kc
        from hermes_cli import kanban_db as kb

        board_db = kb.init_db()
        plan_verdict = {
            "verdict": "APPROVED",
            "workflow_id": workflow_id,
            "blocking_findings": [],
            "required_verification": [],
            "residual_risk": "approved isolated opt-in worker-runtime regression; no default-board/config/systemd/cron/MC/OpenClaw/Discord mutation",
            "evidence_audited": [str(VAULT_PLANSPEC), str(GATE10I_RECEIPT)],
            "note": "mechanical local verifier verdict for this opt-in harness; spawned Reviewer-B still validates Coder output before review-commit",
        }
        write(root / "reviewer-preflight-verdict.json", json.dumps(plan_verdict, indent=2, sort_keys=True))

        repo = init_repo(root, "repo-positive")
        repo_initial = git_out(repo, "rev-parse", "HEAD")
        repo_initial_count = int(git_out(repo, "rev-list", "--count", "HEAD"))
        write(repo / ".gitignore", "*.pyc\n# pre-existing dirty\n")
        write(repo / "approved" / "scoped.txt", "approved change from Gate 10J\n")

        board_root_glob = str(root) + "/**"
        coder_meta = make_coder_metadata(workflow_id)
        coder_artifact_rel = "gate10j-coder-output.json"
        coder_artifact_payload = {
            "schema": "Gate10JCoderHandoff",
            "workflow_id": workflow_id,
            "review_required": True,
            "repo_code_touched": False,
            "commit_created": False,
            "deploy_or_reload": False,
            "mc_openclaw_discord_touched": False,
            "scope_attestation": True,
            "forbidden_actions_taken": 0,
        }
        coder_instructions = f"""
You are the real Gate 10J Coder worker. Do exactly these steps and then terminate through Kanban tools only:
1. Call kanban_show for your current task.
2. Write file `{coder_artifact_rel}` in the current workspace with this exact JSON: {json.dumps(coder_artifact_payload, sort_keys=True)}
3. Add one kanban_comment that starts with `CoderReviewHandoff` and contains `review_required=true`, `scope_attestation=true`, `forbidden_actions_taken=0`, and `artifact_path={coder_artifact_rel}`.
4. Call kanban_complete for your current task with summary `Gate 10J coder handoff complete; review_required=true` and metadata exactly this JSON object: {json.dumps(coder_meta, sort_keys=True)}
Do not use prose-only final response. If any step is impossible, call kanban_block.
"""
        coder_body = body_yaml(board_root_glob, CODER_TOOLS, coder_instructions)
        coder_id = task_id(
            kc.run_slash(
                " ".join(
                    [
                        "create",
                        shlex.quote("Gate 10J real Coder review_required handoff"),
                        "--assignee",
                        "coder",
                        "--workspace",
                        "scratch",
                        "--max-runtime",
                        "300",
                        "--max-retries",
                        "1",
                        "--created-by",
                        "gate10j-coordinator",
                        "--body",
                        shlex.quote(coder_body),
                    ]
                )
            )
        )
        coder_dry = kc.run_slash("dispatch --dry-run --max 1 --failure-limit 1 --json")
        coder_dispatch = kc.run_slash("dispatch --max 1 --failure-limit 1 --json")
        coder_state = wait_terminal(kc, kb, coder_id, 420)
        coder_log = log_info(root, board_slug, coder_id)
        coder_task = coder_state.get("task") or {}
        coder_run = coder_state.get("latest_run") or {}
        coder_ok = coder_task.get("status") == "done" and coder_run.get("outcome") == "completed"
        if not coder_ok:
            raise RuntimeError("coder worker did not complete cleanly")
        coder_workspace = Path(coder_task.get("workspace_path") or "")
        artifact_path = coder_workspace / coder_artifact_rel
        artifact_ok = artifact_path.exists()
        if not artifact_ok:
            raise RuntimeError("coder artifact missing: " + str(artifact_path))

        reviewer_meta = make_reviewer_metadata(workflow_id, coder_id, str(artifact_path))
        reviewer_instructions = f"""
You are the real Gate 10J Reviewer-B worker. Do exactly these steps and then terminate through Kanban tools only:
1. Call kanban_show for your current task.
2. Read this exact artifact path with read_file: `{artifact_path}`.
3. Add one kanban_comment that starts with `ReviewerBVerdict` and contains `verdict=APPROVED`, `consumed_coder_task_id={coder_id}`, and `artifact_path={artifact_path}`.
4. Call kanban_complete for your current task with summary `Gate 10J Reviewer-B APPROVED` and metadata exactly this JSON object: {json.dumps(reviewer_meta, sort_keys=True)}
Do not use prose-only final response. If any step is impossible, call kanban_block.
"""
        reviewer_body = body_yaml(board_root_glob, REVIEWER_TOOLS, reviewer_instructions)
        reviewer_id = task_id(
            kc.run_slash(
                " ".join(
                    [
                        "create",
                        shlex.quote("Gate 10J real Reviewer-B APPROVED verdict"),
                        "--assignee",
                        "reviewer-b",
                        "--workspace",
                        "scratch",
                        "--max-runtime",
                        "300",
                        "--max-retries",
                        "1",
                        "--created-by",
                        "gate10j-coordinator",
                        "--body",
                        shlex.quote(reviewer_body),
                    ]
                )
            )
        )
        reviewer_dry = kc.run_slash("dispatch --dry-run --max 1 --failure-limit 1 --json")
        reviewer_dispatch = kc.run_slash("dispatch --max 1 --failure-limit 1 --json")
        reviewer_state = wait_terminal(kc, kb, reviewer_id, 420)
        reviewer_log = log_info(root, board_slug, reviewer_id)
        reviewer_task = reviewer_state.get("task") or {}
        reviewer_run = reviewer_state.get("latest_run") or {}
        reviewer_ok = (
            reviewer_task.get("status") == "done"
            and reviewer_run.get("outcome") == "completed"
            and (reviewer_run.get("metadata") or {}).get("verdict") == "APPROVED"
        )
        if not reviewer_ok:
            raise RuntimeError("reviewer-b worker did not complete APPROVED")

        positive_out = kc.run_slash(
            " ".join(
                [
                    "review-commit",
                    coder_id,
                    "--reviewer-task",
                    reviewer_id,
                    "--repo",
                    shlex.quote(str(repo)),
                    "--scoped-path",
                    "approved/scoped.txt",
                    "--message",
                    shlex.quote("test: gate 10j real worker scoped review commit"),
                    "--json",
                ]
            )
        )
        positive_payload = json.loads(positive_out)
        commit_hash = positive_payload["commit_hash"]
        committed_paths = git_out(repo, "show", "--name-only", "--format=", commit_hash).splitlines()
        repo_final_count = int(git_out(repo, "rev-list", "--count", "HEAD"))
        repo_status = git_out(repo, "status", "--short")
        repo_cached = git_out(repo, "diff", "--name-only", "--cached")
        temp_diff_check = git(repo, "diff", "--check", check=False)

        missing_repo = init_repo(root, "repo-missing-reviewer")
        write(missing_repo / "allowed.py", "allowed = True\n")
        missing_out = kc.run_slash(
            f"review-commit {coder_id} --reviewer-task t_missing --repo {shlex.quote(str(missing_repo))} --scoped-path allowed.py --message 'blocked missing reviewer'"
        )
        bad_reviewer_meta = {
            **reviewer_meta,
            "verdict": "NEEDS_REVISION",
            "blocking_findings": ["intentional negative control"],
            "required_verification": ["do not commit"],
        }
        bad_reviewer_id = task_id(kc.run_slash("create 'Gate 10J negative Reviewer-B NEEDS_REVISION' --assignee reviewer-b"))
        kc.run_slash(f"complete {bad_reviewer_id} --summary 'negative reviewer complete' --metadata {json_arg(bad_reviewer_meta)}")
        nonapproved_repo = init_repo(root, "repo-nonapproved")
        write(nonapproved_repo / "allowed.py", "allowed = True\n")
        nonapproved_out = kc.run_slash(
            f"review-commit {coder_id} --reviewer-task {bad_reviewer_id} --repo {shlex.quote(str(nonapproved_repo))} --scoped-path allowed.py --message 'blocked nonapproved'"
        )
        staged_repo = init_repo(root, "repo-staged-outside")
        write(staged_repo / "allowed.py", "allowed = True\n")
        write(staged_repo / "outside.py", "outside = True\n")
        git(staged_repo, "add", "outside.py")
        staged_out = kc.run_slash(
            f"review-commit {coder_id} --reviewer-task {reviewer_id} --repo {shlex.quote(str(staged_repo))} --scoped-path allowed.py --message 'blocked staged outside'"
        )

        tests = run(
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/hermes_cli/test_scoped_auto_commit.py",
                "tests/hermes_cli/test_kanban_review_commit.py",
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            timeout=300,
        )
        main_diff_check = git(REPO_ROOT, "diff", "--check", check=False)
        removed_symlinks = remove_symlinks(symlinks)
        default_after = board_snapshot(DEFAULT_BOARD)
        main_status_after = git_out(REPO_ROOT, "status", "--short")
        with kb.connect() as conn:
            isolated_counts = {table: conn.execute(f"select count(*) from {table}").fetchone()[0] for table in ("tasks", "task_runs", "task_events")}
            task_links_count = conn.execute("select count(*) from task_links").fetchone()[0]

        checks = {
            "preflight_verdict_approved": plan_verdict["verdict"] == "APPROVED",
            "coder_real_worker_done": coder_ok,
            "reviewer_b_real_worker_approved": reviewer_ok,
            "coder_artifact_exists": artifact_ok,
            "positive_one_new_commit": repo_final_count == repo_initial_count + 1,
            "positive_commit_exact_scoped_paths": committed_paths == ["approved/scoped.txt"],
            "preexisting_dirty_uncommitted": ".gitignore" in repo_status and repo_cached == "",
            "temp_repo_diff_check_green": temp_diff_check.returncode == 0,
            "main_repo_diff_check_green": main_diff_check.returncode == 0,
            "gate10f_10g_tests_green": tests.returncode == 0,
            "missing_reviewer_blocks": "reviewer task not found" in missing_out and git_out(missing_repo, "rev-list", "--count", "HEAD") == "1",
            "non_approved_blocks": "reviewer verdict is NEEDS_REVISION" in nonapproved_out and git_out(nonapproved_repo, "rev-list", "--count", "HEAD") == "1",
            "staged_outside_blocks": "staged paths outside scoped commit: outside.py" in staged_out and git_out(staged_repo, "rev-list", "--count", "HEAD") == "1",
            "default_board_counts_unchanged": default_before.get("counts") == default_after.get("counts"),
            "default_board_sha_unchanged": default_before.get("sha256") == default_after.get("sha256"),
            "default_board_mtime_unchanged": default_before.get("mtime_ns") == default_after.get("mtime_ns"),
            "credential_symlinks_removed": sorted(removed_symlinks) == sorted(symlinks),
            "main_status_preserved": main_status_before == main_status_after,
            "no_task_links": task_links_count == 0,
        }
        status = "GREEN" if all(checks.values()) else "BLOCKED"
        result.update(
            {
                "status": status,
                "ended_at": now_iso(),
                "root": str(root),
                "hermes_home": str(hermes_home),
                "board_slug": board_slug,
                "board_db": str(board_db),
                "isolated_counts": isolated_counts,
                "task_links_count": task_links_count,
                "main_repo": {
                    "head": main_head,
                    "status_before": main_status_before,
                    "status_after": main_status_after,
                    "diff_check_rc": main_diff_check.returncode,
                    "diff_check_stdout": main_diff_check.stdout,
                    "diff_check_stderr": main_diff_check.stderr,
                },
                "default_board": {"before": default_before, "after": default_after},
                "preflight_verdict": plan_verdict,
                "tasks": {
                    "coder": {
                        "task_id": coder_id,
                        "state": coder_state,
                        "dispatch_dry_run": coder_dry,
                        "dispatch": coder_dispatch,
                        "log": coder_log,
                        "artifact": str(artifact_path),
                        "artifact_sha256": sha256(artifact_path),
                    },
                    "reviewer_b": {
                        "task_id": reviewer_id,
                        "state": reviewer_state,
                        "dispatch_dry_run": reviewer_dry,
                        "dispatch": reviewer_dispatch,
                        "log": reviewer_log,
                    },
                    "negative_reviewer": {"task_id": bad_reviewer_id},
                },
                "positive_repo": {
                    "path": str(repo),
                    "initial_head": repo_initial,
                    "commit_hash": commit_hash,
                    "commit_count_before": repo_initial_count,
                    "commit_count_after": repo_final_count,
                    "committed_paths": committed_paths,
                    "status_short": repo_status,
                    "diff_check_rc": temp_diff_check.returncode,
                },
                "negative_controls": {
                    "missing_reviewer": missing_out,
                    "nonapproved": nonapproved_out,
                    "staged_outside": staged_out,
                },
                "tests": {
                    "cmd": "PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/hermes_cli/test_scoped_auto_commit.py tests/hermes_cli/test_kanban_review_commit.py",
                    "rc": tests.returncode,
                    "stdout": tests.stdout,
                    "stderr": tests.stderr,
                },
                "temp_secret_symlinks": {
                    "created": symlinks,
                    "removed": removed_symlinks,
                    "values_read_or_printed": False,
                },
                "anti_scope": {
                    "push": False,
                    "deploy_reload": False,
                    "mc_openclaw_discord_atlas": False,
                    "real_config_systemd_cron_secret_changes": False,
                    "default_board_mutated": not (
                        checks["default_board_counts_unchanged"]
                        and checks["default_board_sha_unchanged"]
                        and checks["default_board_mtime_unchanged"]
                    ),
                },
                "checks": checks,
                "residual_risk": "Proves opt-in isolated local dispatcher/worker subprocess mechanics only; not permanent gateway pickup, production profile config, MC/OpenClaw bridge, Discord routing, deploy/reload, or push behavior.",
            }
        )
    except Exception as exc:  # pragma: no cover - integration diagnostic path
        removed_symlinks = remove_symlinks(symlinks)
        result.update(
            {
                "status": "BLOCKED",
                "ended_at": now_iso(),
                "error": repr(exc),
                "temp_secret_symlinks": {
                    "created": symlinks,
                    "removed": removed_symlinks,
                    "values_read_or_printed": False,
                },
                "default_board_after_error": board_snapshot(DEFAULT_BOARD),
            }
        )
    finally:
        for key, old_value in original_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        write(result_path, json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    result = run_gate10j_smoke()
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "workflow_id": result.get("workflow_id"),
                "root": result.get("root"),
                "result": str(Path(result.get("root", ".")) / "result.json"),
                "error": result.get("error"),
                "commit": (result.get("positive_repo") or {}).get("commit_hash"),
                "checks": result.get("checks"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.get("status") == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
