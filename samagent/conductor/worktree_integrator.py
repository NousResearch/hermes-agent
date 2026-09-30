"""Git Worktree Swarm Executor & Single Integrator (05-final-plan.md §5).

Builds directly on Hermes's ``tools.subagent_worktree``:
1. Spawns an isolated git worktree per module worker in a wave (`create_subagent_worktree`).
2. Executes module workers in parallel (via ThreadPoolExecutor) or sequentially inside their worktrees.
3. Enforces the authoritative Layer-2 post-hoc diff guard (`check_git_diff_ownership`) against
   `contract/ownership.yaml` before any commit/merge.
4. Rejects and discards any worktree that modified unowned files or `.samagent/contract/**`.
5. Single Integrator merges verified module branches into the integration branch in deterministic order.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from samagent.contract.freeze import (
    OwnershipVerdict,
    check_git_diff_ownership,
    load_ownership_map,
)
from tools.subagent_worktree import (
    _run_git,
    build_worktree_context_note,
    create_subagent_worktree,
)


@dataclass
class WorktreeModuleRun:
    module_name: str
    worktree_path: str
    branch: str
    base_commit: str
    ownership_verdict: Dict[str, Any]
    committed_sha: Optional[str] = None
    merged: bool = False
    rejected_reason: Optional[str] = None
    context_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrationReport:
    success: bool
    parallel_executed: bool
    module_runs: List[WorktreeModuleRun] = field(default_factory=list)
    merged_branches: List[str] = field(default_factory=list)
    rejected_modules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "parallel_executed": self.parallel_executed,
            "module_runs": [m.to_dict() for m in self.module_runs],
            "merged_branches": list(self.merged_branches),
            "rejected_modules": list(self.rejected_modules),
        }


def ensure_git_repo_initialized(project_dir: Path) -> str:
    """Ensure *project_dir* is a git repo with at least one commit so worktrees can branch from HEAD."""
    root = Path(project_dir)
    if not (root / ".git").exists():
        _run_git(["init", "-b", "main"], cwd=str(root))
    _run_git(["add", "-A"], cwd=str(root))
    status = _run_git(["status", "--porcelain"], cwd=str(root))
    head = _run_git(["rev-parse", "--verify", "HEAD"], cwd=str(root))
    if head.returncode != 0 or status.stdout.strip():
        _run_git(
            [
                "-c",
                "user.name=SamAgent Integrator",
                "-c",
                "user.email=integrator@samagent.local",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "-m",
                "chore(samagent): baseline before worktree swarm wave",
            ],
            cwd=str(root),
        )
    rev = _run_git(["rev-parse", "HEAD"], cwd=str(root))
    return rev.stdout.strip()


class WorktreeSwarmIntegrator:
    """Runs module workers in isolated git worktrees and merges verified branches via a single integrator."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)

    def execute_wave(
        self,
        module_names: List[str],
        worker_fn: Callable[[str, Path], None],
        *,
        run_id: str = "wave1",
        parallel: bool = True,
    ) -> IntegrationReport:
        ensure_git_repo_initialized(self.project_dir)
        ownership_map = load_ownership_map(self.project_dir)

        # 1. Create worktrees on main thread (git worktree add locks .git/worktrees)
        wt_infos: Dict[str, Dict[str, str]] = {}
        for mod_name in module_names:
            info = create_subagent_worktree(str(self.project_dir), subagent_id=f"{run_id}-{mod_name}")
            if info is None:
                raise RuntimeError(f"Failed to create git worktree for module {mod_name!r}")
            wt_infos[mod_name] = info

        # 2. Run module workers inside their isolated worktrees (parallel or sequential)
        def _run_one(m_name: str) -> None:
            wt_dir = Path(wt_infos[m_name]["path"])
            worker_fn(m_name, wt_dir)

        if parallel and len(module_names) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(module_names))) as pool:
                futures = [pool.submit(_run_one, m) for m in module_names]
                for fut in futures:
                    fut.result()
        else:
            for m in module_names:
                _run_one(m)

        # 3. Layer-2 post-hoc git diff ownership check + commit + single-integrator merge
        module_runs: List[WorktreeModuleRun] = []
        merged_branches: List[str] = []
        rejected_modules: List[str] = []

        for mod_name in module_names:
            info = wt_infos[mod_name]
            wt_path = Path(info["path"])
            branch = info["branch"]
            base_commit = info["base_commit"]
            verdict: OwnershipVerdict = check_git_diff_ownership(
                wt_path,
                module_name=mod_name,
                ownership_map=ownership_map,
                base_ref=base_commit,
            )
            run_record = WorktreeModuleRun(
                module_name=mod_name,
                worktree_path=str(wt_path),
                branch=branch,
                base_commit=base_commit,
                ownership_verdict=verdict.to_dict(),
                context_note=build_worktree_context_note(info),
            )

            if not verdict.allowed:
                run_record.rejected_reason = verdict.reason
                rejected_modules.append(mod_name)
            else:
                # Stage and commit owned changes inside the worker worktree
                _run_git(["add", "-A"], cwd=str(wt_path))
                st = _run_git(["status", "--porcelain"], cwd=str(wt_path))
                if st.stdout.strip():
                    _run_git(
                        [
                            "-c",
                            "user.name=SamAgent Worker",
                            "-c",
                            "user.email=worker@samagent.local",
                            "-c",
                            "commit.gpgsign=false",
                            "commit",
                            "-m",
                            f"feat({mod_name}): implement owned module in isolated worktree",
                        ],
                        cwd=str(wt_path),
                    )
                head_sha = _run_git(["rev-parse", "HEAD"], cwd=str(wt_path)).stdout.strip()
                run_record.committed_sha = head_sha

                # Single integrator merges the verified branch into the main checkout
                merge_res = _run_git(
                    [
                        "-c",
                        "user.name=SamAgent Integrator",
                        "-c",
                        "user.email=integrator@samagent.local",
                        "-c",
                        "commit.gpgsign=false",
                        "merge",
                        "--no-ff",
                        "-m",
                        f"merge({mod_name}): integrate verified worktree {branch}",
                        branch,
                    ],
                    cwd=str(self.project_dir),
                )
                if merge_res.returncode == 0:
                    run_record.merged = True
                    merged_branches.append(branch)
                else:
                    _run_git(["merge", "--abort"], cwd=str(self.project_dir))
                    run_record.rejected_reason = f"Merge conflict: {merge_res.stderr.strip()}"
                    rejected_modules.append(mod_name)

            # Clean up worktree directory and temporary branch
            _run_git(["worktree", "remove", "--force", str(wt_path)], cwd=str(self.project_dir))
            _run_git(["branch", "-D", branch], cwd=str(self.project_dir))
            module_runs.append(run_record)

        return IntegrationReport(
            success=(len(rejected_modules) == 0),
            parallel_executed=(parallel and len(module_names) > 1),
            module_runs=module_runs,
            merged_branches=merged_branches,
            rejected_modules=rejected_modules,
        )
