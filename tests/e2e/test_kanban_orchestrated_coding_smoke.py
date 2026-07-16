"""Live E2E smoke test for the Kanban Orchestrated Coding MVP.

Command:
    python -m pytest tests/e2e/test_kanban_orchestrated_coding_smoke.py -q -m integration

This test intentionally uses real local Git/worktree/process operations and a
real DeepSeek-family LLM call for the executor lane. Planner/auditor lanes are
deterministic so the expensive models are not used for routine worker work.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

from hermes_cli import kanban_db as kb


DEFAULT_DEEPSEEK_DIRECT_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_OPENROUTER_MODEL = "deepseek/deepseek-chat"
DEFAULT_DEEPSEEK_OPENCODE_GO_MODEL = "deepseek-v4-flash"
VERIFY_COMMAND = [sys.executable, "-m", "pytest", "-q"]
VERIFY_COMMAND_LABEL = f"{sys.executable} -m pytest -q"


def _load_user_env() -> None:
    env_file = Path.home() / ".hermes" / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_user_env()


@dataclass(frozen=True)
class LiveDeepSeekRoute:
    provider: str
    model: str
    api_key: str = field(default="", repr=False)
    base_url: str = ""


def _resolve_live_deepseek_route() -> LiveDeepSeekRoute | None:
    from agent.auxiliary_client import resolve_provider_client
    from hermes_cli.auth import resolve_api_key_provider_credentials

    candidates = (
        (
            "opencode-go",
            os.getenv("HERMES_E2E_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_OPENCODE_GO_MODEL),
        ),
        (
            "deepseek",
            os.getenv("HERMES_E2E_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_DIRECT_MODEL),
        ),
        (
            "openrouter",
            os.getenv("HERMES_E2E_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_OPENROUTER_MODEL),
        ),
    )
    for provider, model in candidates:
        if provider == "openrouter":
            creds = {
                "api_key": os.getenv("OPENROUTER_API_KEY", ""),
                "base_url": "",
            }
        elif provider == "opencode-go" and os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENCODE_GO_API_KEY"):
            creds = {
                "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                "base_url": os.getenv("OPENCODE_GO_BASE_URL", ""),
            }
        else:
            try:
                creds = resolve_api_key_provider_credentials(provider)
            except Exception:
                creds = {}
        api_key = str(creds.get("api_key") or "").strip()
        base_url = str(creds.get("base_url") or "").strip()
        if not api_key:
            continue
        client, resolved_model = resolve_provider_client(
            provider=provider,
            model=model,
            explicit_api_key=api_key,
            explicit_base_url=base_url or None,
        )
        if client is not None:
            return LiveDeepSeekRoute(
                provider=provider,
                model=resolved_model or model,
                api_key=api_key,
                base_url=base_url,
            )
    return None


def _run(cmd: list[str], cwd: Path, *, check: bool = True, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(cmd)}\n"
            f"cwd={cwd}\n"
            f"exit={result.returncode}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    return result


def _init_tiny_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "kanban-e2e@example.com"], repo)
    _run(["git", "config", "user.name", "Kanban E2E"], repo)
    (repo / "math_bug.py").write_text(
        "def add(a, b):\n"
        "    return a - b\n",
        encoding="utf-8",
    )
    (repo / "test_math_bug.py").write_text(
        "from math_bug import add\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "initial failing math fixture"], repo)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def _call_deepseek_executor(
    route: LiveDeepSeekRoute,
    failure_output: str,
    call_llm: Callable[..., Any],
) -> tuple[str, dict[str, Any]]:
    prompt = (
        "You are the low-cost executor lane in a Kanban coding workflow. "
        "Return ONLY a JSON object with keys explanation and files. "
        "files must map relative file paths to complete replacement file content. "
        "Fix the failing repository. Do not include markdown.\n\n"
        "Repository files:\n"
        "math_bug.py:\n"
        "def add(a, b):\n"
        "    return a - b\n\n"
        "test_math_bug.py:\n"
        "from math_bug import add\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        f"Failing command output:\n{failure_output[-2000:]}\n"
    )
    started = time.monotonic()
    try:
        response = call_llm(
            task="kanban_e2e_executor",
            provider=route.provider,
            model=route.model,
            base_url=route.base_url or None,
            api_key=route.api_key or None,
            messages=[
                {"role": "system", "content": "Return compact JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=350,
            timeout=45,
        )
    except Exception as exc:
        message = str(exc)
        if route.api_key:
            message = message.replace(route.api_key, "<REDACTED>")
        pytest.fail(
            "DeepSeek-family live executor call failed\n"
            f"provider={route.provider}\n"
            f"model={route.model}\n"
            f"base_url={route.base_url or '<provider default>'}\n"
            f"error_type={type(exc).__name__}\n"
            f"error={message}"
        )
    elapsed = time.monotonic() - started
    content = response.choices[0].message.content or ""
    payload = _extract_json_object(content)
    usage = getattr(response, "usage", None)
    usage_dict = {
        "provider": route.provider,
        "model": route.model,
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    return content, usage_dict | {"payload": payload}


def _real_call_llm(**kwargs: Any) -> Any:
    from agent.auxiliary_client import call_llm

    return call_llm(**kwargs)


def _deterministic_deepseek_call_llm(**kwargs: Any) -> Any:
    assert kwargs["provider"] in {"deepseek", "openrouter", "opencode-go"}
    assert "deepseek" in kwargs["model"].lower()
    assert kwargs["max_tokens"] <= 350
    assert kwargs["timeout"] <= 45
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "explanation": "Replace subtraction with addition.",
                            "files": {
                                "math_bug.py": "def add(a, b):\n    return a + b\n",
                            },
                        }
                    )
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_live_deepseek_route_resolves_opencode_key_exported_as_deepseek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-opencode-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)

    route = _resolve_live_deepseek_route()

    assert route is not None
    assert route.provider == "opencode-go"
    assert route.model == DEFAULT_DEEPSEEK_OPENCODE_GO_MODEL


def test_live_deepseek_route_resolves_opencode_go_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-opencode-test")

    route = _resolve_live_deepseek_route()

    assert route is not None
    assert route.provider == "opencode-go"
    assert route.model == DEFAULT_DEEPSEEK_OPENCODE_GO_MODEL


def test_live_deepseek_route_resolves_openrouter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-test")

    route = _resolve_live_deepseek_route()

    assert route is not None
    assert route.provider == "openrouter"
    assert route.model == DEFAULT_DEEPSEEK_OPENROUTER_MODEL


def _apply_executor_patch(worktree: Path, patch_payload: dict[str, Any]) -> list[str]:
    files = patch_payload.get("files")
    assert isinstance(files, dict) and files, patch_payload
    changed: list[str] = []
    for rel, content in files.items():
        rel_path = Path(str(rel))
        assert not rel_path.is_absolute()
        assert ".." not in rel_path.parts
        target = worktree / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        changed.append(rel_path.as_posix())
    return changed


def _event_kinds(conn, task_id: str) -> list[str]:
    return [event.kind for event in kb.list_events(conn, task_id)]


@pytest.mark.integration
def test_live_kanban_orchestrated_coding_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    route = _resolve_live_deepseek_route()
    if route is None:
        pytest.fail(
            "DeepSeek-family executor credential missing: configure "
            "OPENCODE_GO_API_KEY, DEEPSEEK_API_KEY, or OPENROUTER_API_KEY "
            "in env/~/.hermes/.env/Hermes auth store"
        )
    _run_kanban_orchestrated_coding_smoke(
        tmp_path,
        monkeypatch,
        route=route,
        call_llm=_real_call_llm,
    )


def test_kanban_orchestrated_coding_git_process_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real Git/Kanban/process path without a network credential.

    This is not the live DoD because the DeepSeek-family/OpenCode call is deterministic here.
    The live test above is the only one that proves external model invocation.
    """

    _run_kanban_orchestrated_coding_smoke(
        tmp_path,
        monkeypatch,
        route=LiveDeepSeekRoute(provider="deepseek", model=DEFAULT_DEEPSEEK_DIRECT_MODEL),
        call_llm=_deterministic_deepseek_call_llm,
    )


def _run_kanban_orchestrated_coding_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    route: LiveDeepSeekRoute,
    call_llm: Callable[..., Any],
) -> None:

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "30000")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    repo = tmp_path / "repo"
    _init_tiny_repo(repo)

    board = "e2e-koc"
    kb.create_board(board, name="E2E KOC", default_workdir=str(repo))
    kb.init_db(board=board)

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    with kb.connect(board=board) as conn:
        root_id = kb.create_task(
            conn,
            title="Fix add() bug with orchestrated coding",
            body="User goal: make pytest pass, commit locally, merge to integration.",
            assignee="orchestrator",
            use_default_budget=False,
            board=board,
        )
        planner_id = kb.create_task(
            conn,
            title="Plan fix",
            body="Plan: run pytest, ask DeepSeek-family/OpenCode executor for minimal patch, verify, commit.",
            assignee="planner",
            idempotency_key=f"{root_id}:planner:r1",
            use_default_budget=False,
            board=board,
        )
        auditor_id = kb.create_task(
            conn,
            title="Audit plan",
            body="Read planner output and open executor gate if concrete.",
            assignee="plan-auditor",
            parents=(planner_id,),
            use_default_budget=False,
            board=board,
        )
        executor_id = kb.create_task(
            conn,
            title="Execute approved plan",
            body="Use DeepSeek-family/OpenCode worker lane. Work in isolated git worktree.",
            assignee="deepseek-worker",
            parents=(auditor_id,),
            idempotency_key=f"{root_id}:executor:r1",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="koc/e2e-worker",
            plan_audit_required=True,
            plan_audit_max_rounds=2,
            budget_usd=0.25,
            use_default_budget=False,
            board=board,
        )

        assert kb.complete_task(
            conn,
            planner_id,
            summary="Plan names files, test command, worktree, commit, merge, and audit.",
            metadata={"planned_files": ["math_bug.py"], "test_command": VERIFY_COMMAND_LABEL},
        )
        kb.record_plan_audit_verdict(
            conn,
            executor_id,
            approved=True,
            reviewer="deterministic-plan-auditor",
            reason="Plan is concrete enough for the executor.",
            metadata={"round": 1},
        )
        assert kb.complete_task(
            conn,
            auditor_id,
            summary=f"Approved plan for executor {executor_id}.",
            metadata={"executor_task_id": executor_id, "round": 1, "approved": True},
        )
        kb.recompute_ready(conn)
        assert kb.claim_task(conn, executor_id, claimer="deepseek-worker") is not None
        executor_task = kb.get_task(conn, executor_id)
        assert executor_task is not None
        worktree = kb.resolve_workspace(executor_task, board=board)
        kb.set_workspace_path(conn, executor_id, str(worktree))

    assert worktree.exists()
    assert worktree != repo
    assert (worktree / ".git").exists()
    assert _run(["git", "branch", "--show-current"], worktree).stdout.strip() == "koc/e2e-worker"

    first_verify = _run(VERIFY_COMMAND, worktree, check=False, timeout=60)
    first_evidence = {
        "command": VERIFY_COMMAND_LABEL,
        "cwd": str(worktree),
        "exit_code": first_verify.returncode,
        "stdout": first_verify.stdout,
        "stderr": first_verify.stderr,
    }
    (artifact_dir / "first_verify.json").write_text(
        json.dumps(first_evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert first_verify.returncode != 0

    raw_deepseek, deepseek_result = _call_deepseek_executor(
        route,
        first_verify.stdout + "\n" + first_verify.stderr,
        call_llm,
    )
    (artifact_dir / "deepseek_raw.txt").write_text(raw_deepseek, encoding="utf-8")
    (artifact_dir / "deepseek_usage.json").write_text(
        json.dumps(deepseek_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    changed_files = _apply_executor_patch(worktree, deepseek_result["payload"])
    assert "math_bug.py" in changed_files

    second_verify = _run(VERIFY_COMMAND, worktree, check=False, timeout=60)
    second_evidence = {
        "command": VERIFY_COMMAND_LABEL,
        "cwd": str(worktree),
        "exit_code": second_verify.returncode,
        "stdout": second_verify.stdout,
        "stderr": second_verify.stderr,
    }
    (artifact_dir / "second_verify.json").write_text(
        json.dumps(second_evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert second_verify.returncode == 0

    diff = _run(["git", "diff"], worktree).stdout
    assert "return a + b" in diff
    _run(["git", "add", "."], worktree)
    commit_message = (
        "fix: complete kanban e2e worker task\n\n"
        f"Hermes-Task-ID: {executor_id}\n"
        f"Hermes-Root-Task-ID: {root_id}\n"
        f"Hermes-Agent: deepseek-worker\n"
        f"Hermes-Provider: {route.provider}\n"
        f"Hermes-Model: {route.model}\n"
        f"Hermes-Tests: {VERIFY_COMMAND_LABEL}\n"
    )
    _run(["git", "commit", "-m", commit_message], worktree)
    commit_sha = _run(["git", "rev-parse", "HEAD"], worktree).stdout.strip()

    with kb.connect(board=board) as conn:
        assert kb.complete_task(
            conn,
            executor_id,
            summary="DeepSeek-family/OpenCode worker fixed math_bug.py and pytest passed.",
            metadata={
                "model": route.model,
                "provider": route.provider,
                "changed_files": changed_files,
                "verify": second_evidence,
                "process_evidence": [str(artifact_dir / "first_verify.json"), str(artifact_dir / "second_verify.json")],
                "commit": commit_sha,
                "worktree": str(worktree),
                "branch": "koc/e2e-worker",
            },
        )
        reviewer_id = kb.create_task(
            conn,
            title="Final read-only audit and local merge",
            body="Read diff/test evidence, merge worker branch to local integration branch.",
            assignee="final-auditor",
            parents=(executor_id,),
            use_default_budget=False,
            board=board,
        )
        kb.recompute_ready(conn)
        assert kb.claim_task(conn, reviewer_id, claimer="final-auditor") is not None

    _run(["git", "checkout", "-b", "integration/koc-e2e"], repo)
    merge = _run(["git", "merge", "--no-ff", "koc/e2e-worker", "-m", "merge: kanban e2e worker"], repo, check=False)
    merge_evidence = {
        "command": "git merge --no-ff koc/e2e-worker",
        "cwd": str(repo),
        "exit_code": merge.returncode,
        "stdout": merge.stdout,
        "stderr": merge.stderr,
        "integration_branch": "integration/koc-e2e",
    }
    (artifact_dir / "merge.json").write_text(
        json.dumps(merge_evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assert merge.returncode == 0
    final_verify = _run(VERIFY_COMMAND, repo, check=False, timeout=60)
    final_evidence = {
        "command": VERIFY_COMMAND_LABEL,
        "cwd": str(repo),
        "exit_code": final_verify.returncode,
        "stdout": final_verify.stdout,
        "stderr": final_verify.stderr,
    }
    (artifact_dir / "final_verify.json").write_text(
        json.dumps(final_evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_verdict = "COMPLETED" if final_verify.returncode == 0 else "BLOCKED"
    report = {
        "job": {"root_task_id": root_id, "executor_task_id": executor_id},
        "model": {"provider": route.provider, "model": route.model},
        "process_evidence": {
            "first_verify": first_evidence,
            "second_verify": second_evidence,
            "final_verify": final_evidence,
        },
        "git": {
            "worktree": str(worktree),
            "worker_branch": "koc/e2e-worker",
            "commit": commit_sha,
            "merge": merge_evidence,
        },
        "final_audit": {
            "mode": "read-only",
            "verdict": final_verdict,
            "reason": "Final pytest passed after local merge." if final_verdict == "COMPLETED" else "Final pytest failed after merge.",
        },
    }
    report_path = artifact_dir / "kanban_orchestrated_coding_e2e_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with kb.connect(board=board) as conn:
        if final_verdict == "COMPLETED":
            assert kb.complete_task(
                conn,
                reviewer_id,
                summary="Read-only final audit passed after local merge.",
                metadata={"report": str(report_path), "verdict": final_verdict},
            )
            assert kb.complete_task(
                conn,
                root_id,
                summary="Kanban Orchestrated Coding E2E completed.",
                metadata={"report": str(report_path), "verdict": final_verdict},
            )
        else:
            fix_id = kb.create_task(
                conn,
                title="Fix post-merge verification failure",
                body=final_verify.stdout + "\n" + final_verify.stderr,
                assignee="deepseek-worker",
                parents=(reviewer_id,),
                use_default_budget=False,
                board=board,
            )
            assert kb.block_task(conn, reviewer_id, kind="dependency", reason=f"Created fix task {fix_id}")
            assert kb.block_task(conn, root_id, kind="dependency", reason=f"Waiting on fix task {fix_id}")

        transitions = {
            root_id: _event_kinds(conn, root_id),
            planner_id: _event_kinds(conn, planner_id),
            auditor_id: _event_kinds(conn, auditor_id),
            executor_id: _event_kinds(conn, executor_id),
            reviewer_id: _event_kinds(conn, reviewer_id),
        }
        (artifact_dir / "task_transitions.json").write_text(
            json.dumps(transitions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        final_root = kb.get_task(conn, root_id)
        final_executor = kb.get_task(conn, executor_id)
        final_reviewer = kb.get_task(conn, reviewer_id)

    assert "plan_audit_approved" in transitions[executor_id]
    assert "completed" in transitions[executor_id]
    assert final_executor is not None and final_executor.status == "done"
    assert final_reviewer is not None and final_reviewer.status == "done"
    assert final_root is not None and final_root.status == "done"
    assert report["final_audit"]["verdict"] == "COMPLETED"
    assert report_path.exists()
