"""Autonomous mode - Swarm execution for Hermes.

This module provides the core logic for spawning a swarm of Codex agents
with GSD skills to execute tasks autonomously.
"""

import json
import os
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Store running processes for status tracking
RUNNING_PROCESSES: Dict[str, dict] = {}


# ----------------------------------------------------------------------
# Run ID Generation
# ----------------------------------------------------------------------

def generate_run_id(task: str = "run") -> str:
    """Generate a unique run ID.
    
    Format: YYYYMMDD-<6char>
    """
    date_str = datetime.now().strftime("%Y%m%d")
    short_id = uuid.uuid4().hex[:6]
    # Create a simple headline from task
    headline = task.lower().replace(" ", "-")[:20]
    return f"{date_str}-{short_id}", headline


# ----------------------------------------------------------------------
# Run Directory Management
# ----------------------------------------------------------------------

def get_runs_dir() -> Path:
    """Get the runs directory."""
    return Path.home() / ".hermes" / "autonomous" / "runs"


def get_vault_dir() -> Path:
    """Get the vault backup directory."""
    return Path.home() / "Documents" / "great-vault" / "Automations" / "autonomous" / "runs"


def create_run_dirs(run_id: str, headline: str) -> tuple[Path, Path]:
    """Create run directories in both local and vault.
    
    Returns: (local_dir, vault_dir)
    """
    runs_dir = get_runs_dir()
    vault_dir = get_vault_dir()
    
    local_dir = runs_dir / run_id
    vault_dir = vault_dir / f"{run_id}_{headline}"
    
    # Create local dirs
    (local_dir / "input").mkdir(parents=True, exist_ok=True)
    (local_dir / "output" / "logs").mkdir(parents=True, exist_ok=True)
    
    # Create vault dirs
    (vault_dir / "input").mkdir(parents=True, exist_ok=True)
    (vault_dir / "output").mkdir(parents=True, exist_ok=True)
    
    return local_dir, vault_dir


def save_manifest(run_id: str, task: str, repo: str, status: str = "started") -> Path:
    """Save run manifest."""
    runs_dir = get_runs_dir()
    manifest = {
        "run_id": run_id,
        "task": task,
        "status": status,
        "started_at": datetime.now().isoformat() + "Z",
        "repo": repo,
    }
    manifest_path = runs_dir / run_id / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def update_manifest(run_id: str, **updates) -> None:
    """Update run manifest with new values."""
    runs_dir = get_runs_dir()
    manifest_path = runs_dir / run_id / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest.update(updates)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


# ----------------------------------------------------------------------
# Vault Sync
# ----------------------------------------------------------------------

def sync_to_vault(run_id: str, headline: str) -> None:
    """Sync run data to vault."""
    runs_dir = get_runs_dir()
    vault_dir = get_vault_dir()
    
    local_dir = runs_dir / run_id
    vault_subdir = vault_dir / f"{run_id}_{headline}"
    
    if not local_dir.exists():
        return
    
    # Copy input files
    input_dir = local_dir / "input"
    if input_dir.exists():
        for f in input_dir.glob("*"):
            if f.is_file():
                import shutil
                shutil.copy(f, vault_subdir / "input" / f.name)
    
    # Copy output report
    output_dir = local_dir / "output"
    report_path = output_dir / "report.md"
    if report_path.exists():
        import shutil
        shutil.copy(report_path, vault_subdir / "output" / "report.md")


# ----------------------------------------------------------------------
# Codex Swarm Execution
# ----------------------------------------------------------------------

def spawn_codex_agent(
    repo_path: str,
    task: str,
    gsd_skill: str = "gsd-executor",
    model: str = "o3",
) -> str:
    """Spawn a Codex agent with GSD skill.
    
    Returns the command that would be run.
    """
    # Build the Codex command
    cmd = f"""codex exec \\
  --dangerously-bypass-approvals-and-sandbox \\
  -C {repo_path} \\
  --skill {gsd_skill} \\
  "{task}" """
    return cmd


def build_swarm_prompt(
    problem: str,
    scope: str,
    tests: str,
    constraints: str,
    done_criteria: str,
    repo_path: str,
) -> str:
    """Build the full prompt for the swarm to execute.
    
    This is what gets sent to Codex with GSD skills.
    """
    return f"""You are an autonomous coding agent. Your mission is to solve the following problem:

## PROBLEM
{problem}

## SCOPE
{scope}

## TESTS / VALIDATION
{tests}

## CONSTRAINTS
{constraints}

## DONE CRITERIA
{done_criteria}

## REPO PATH
{repo_path}

## YOUR TASK
1. Understand the problem thoroughly
2. Map the relevant parts of the codebase
3. Plan your approach
4. Execute the solution
5. Test your solution
6. If tests fail, debug and retry (up to 3 times)
7. When done, provide a summary of what you changed

## OUTPUT FORMAT
When complete, provide:
- Files changed
- Tests run and their results
- Any issues encountered and how you resolved them
- Whether you're ready for PR or need manual review

Go forth and build!"""


# ----------------------------------------------------------------------
# PR Submission
# ----------------------------------------------------------------------

def get_git_remote() -> Optional[str]:
    """Get the current git remote URL."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def add_fork_remote(fork_url: str) -> None:
    """Add user's fork as a remote."""
    import subprocess
    try:
        subprocess.run(
            ["git", "remote", "add", "fork", fork_url],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
    except Exception:
        pass  # Might already exist


def create_pr_branch(branch_name: str, commit_message: str, files: list[str]) -> bool:
    """Create branch, commit and push.
    
    Returns True on success.
    """
    import subprocess
    
    try:
        # Check if branch exists
        result = subprocess.run(
            ["git", "rev-parse", "--verify", branch_name],
            capture_output=True,
            cwd=os.getcwd(),
        )
        
        if result.returncode == 0:
            # Branch exists, checkout
            subprocess.run(["git", "checkout", branch_name], cwd=os.getcwd())
        else:
            # Create new branch from main
            subprocess.run(["git", "checkout", "-b", branch_name, "main"], cwd=os.getcwd())
        
        # Add files
        subprocess.run(["git", "add"] + files, cwd=os.getcwd())
        
        # Commit
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=os.getcwd(),
            env={**os.environ, "GIT_AUTHOR_NAME": "Hermes Dralha", 
                 "GIT_AUTHOR_EMAIL": "tenzindjampa23@gmail.com",
                 "GIT_COMMITTER_NAME": "Hermes Dralha",
                 "GIT_COMMITTER_EMAIL": "tenzindjampa23@gmail.com"},
        )
        
        # Push to fork
        subprocess.run(
            ["git", "push", "-u", "fork", branch_name],
            cwd=os.getcwd(),
            env={**os.environ, "GIT_SSH_COMMAND": "ssh -o StrictHostKeyChecking=no"},
        )
        
        return True
    except Exception as e:
        print(f"Error creating PR: {e}")
        return False


# ----------------------------------------------------------------------
# Report Generation
# ----------------------------------------------------------------------

def generate_report(
    run_id: str,
    task: str,
    status: str,
    files_changed: list[str],
    test_results: str,
    notes: str = "",
) -> str:
    """Generate the final report."""
    report = f"""# Autonomous Run Report: {task}

## Summary
- **Run ID:** {run_id}
- **Task:** {task}
- **Status:** {status}
- **Files Changed:** {len(files_changed)}

## Files Changed
"""
    for f in files_changed:
        report += f"- {f}\n"
    
    report += f"""
## Test Results
{test_results}

## Notes
{notes}

---
Generated by Hermes Autonomous Mode
"""
    return report


def save_report(run_id: str, report: str) -> Path:
    """Save report to run directory."""
    runs_dir = get_runs_dir()
    report_path = runs_dir / run_id / "output" / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    return report_path


# ----------------------------------------------------------------------
# Swarm Execution (Full Implementation)
# ----------------------------------------------------------------------

def spawn_swarm(
    repo_path: str,
    task: str,
    gsd_skill: str = "gsd-executor",
    run_id: str = None,
) -> Dict[str, Any]:
    """Spawn a Codex agent in background to execute the task.
    
    Returns dict with run_id, command, and initial status.
    """
    global RUNNING_PROCESSES
    
    if run_id is None:
        run_id, _ = generate_run_id(task[:30])
    
    # Build the full prompt with GSD context
    full_prompt = build_full_autonomous_prompt(task, repo_path)
    
    # Build the Codex command
    cmd = [
        "codex", "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C", repo_path,
        "--skill", gsd_skill,
        full_prompt
    ]
    
    # Start the process in background
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=repo_path,
        )
        
        # Track the process
        RUNNING_PROCESSES[run_id] = {
            "process": proc,
            "task": task,
            "repo": repo_path,
            "started_at": datetime.now().isoformat(),
            "status": "running",
        }
        
        # Update manifest
        update_manifest(run_id, status="running", process_id=proc.pid)
        
        return {
            "run_id": run_id,
            "status": "started",
            "process_id": proc.pid,
            "command": " ".join(cmd),
        }
    except Exception as e:
        return {
            "run_id": run_id,
            "status": "failed",
            "error": str(e),
        }


def build_full_autonomous_prompt(task: str, repo_path: str) -> str:
    """Build the full prompt for autonomous execution.
    
    This includes context about what the agent should do.
    """
    return f"""You are running in AUTONOMOUS MODE for Hermes.

Your mission: Execute the following task autonomously using GSD methodology.

## TASK
{task}

## REPO
{repo_path}

## INSTRUCTIONS
1. First, understand the task fully
2. Map the relevant codebase (use GSD skills if needed)
3. Plan your approach
4. Execute the solution
5. Test thoroughly
6. If tests fail, debug and retry (up to 3 times)
7. When done, summarize what you changed

## OUTPUT
When complete, provide:
- Files changed (list)
- Tests run and results
- Any issues and resolutions
- Ready for PR or needs manual review

Go!"""


def check_run_status(run_id: str) -> Dict[str, Any]:
    """Check the status of a running autonomous task."""
    global RUNNING_PROCESSES
    
    if run_id not in RUNNING_PROCESSES:
        # Check if it's a completed run
        runs_dir = get_runs_dir()
        manifest_path = runs_dir / run_id / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            return manifest
        return {"status": "not_found", "run_id": run_id}
    
    proc_info = RUNNING_PROCESSES[run_id]
    proc = proc_info["process"]
    
    if proc.poll() is None:
        return {
            "run_id": run_id,
            "status": "running",
            "started_at": proc_info["started_at"],
            "task": proc_info["task"],
        }
    else:
        # Process completed
        stdout, stderr = proc.communicate()
        RUNNING_PROCESSES[run_id]["status"] = "completed"
        RUNNING_PROCESSES[run_id]["returncode"] = proc.returncode
        RUNNING_PROCESSES[run_id]["stdout"] = stdout[:5000]  # Limit size
        RUNNING_PROCESSES[run_id]["stderr"] = stderr[:2000]
        
        # Update manifest
        update_manifest(run_id, status="completed", returncode=proc.returncode)
        
        return {
            "run_id": run_id,
            "status": "completed",
            "returncode": proc.returncode,
            "output": stdout[:2000],
            "error": stderr[:1000] if stderr else None,
        }


def run_iteration(
    repo_path: str,
    task: str,
    iteration: int = 1,
    max_iterations: int = 3,
    failure_context: str = None,
) -> Dict[str, Any]:
    """Run one iteration of the autonomous task.
    
    If iteration > 1, includes failure_context from previous attempt.
    """
    if iteration > max_iterations:
        return {
            "status": "failed",
            "reason": f"Max iterations ({max_iterations}) reached",
        }
    
    # Build prompt with iteration context
    if failure_context:
        prompt = f"""{task}

## PREVIOUS ITERATION FAILED
{failure_context}

## THIS ITERATION
Fix the above issues and try again. You have {max_iterations - iteration} retries left after this."""
    else:
        prompt = task
    
    # Run the swarm
    result = spawn_swarm(repo_path, prompt, run_id=f"{repo_path.split('/')[-1]}-{iteration}")
    
    return result


def complete_vault_sync(run_id: str, headline: str) -> None:
    """Complete vault sync - copy all run data to vault."""
    runs_dir = get_runs_dir()
    vault_dir = get_vault_dir()
    
    local_dir = runs_dir / run_id
    vault_subdir = vault_dir / f"{run_id}_{headline}"
    
    if not local_dir.exists():
        return
    
    # Create vault dirs
    (vault_subdir / "input").mkdir(parents=True, exist_ok=True)
    (vault_subdir / "output").mkdir(parents=True, exist_ok=True)
    
    # Copy all input files
    input_dir = local_dir / "input"
    if input_dir.exists():
        for f in input_dir.glob("*"):
            if f.is_file():
                import shutil
                shutil.copy(f, vault_subdir / "input" / f.name)
    
    # Copy all output files
    output_dir = local_dir / "output"
    if output_dir.exists():
        for f in output_dir.glob("*"):
            if f.is_file():
                import shutil
                if f.is_dir():
                    shutil.copytree(f, vault_subdir / "output" / f.name, dirs_exist_ok=True)
                else:
                    shutil.copy(f, vault_subdir / "output" / f.name)


def execute_autonomous_task(
    task: str,
    problem: str = None,
    scope: str = None,
    tests: str = None,
    constraints: str = None,
    done_criteria: str = None,
    repo_path: str = None,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """Main entry point for executing an autonomous task.
    
    This orchestrates the full workflow:
    1. Create run directories
    2. Save input spec
    3. Spawn swarm
    4. Handle iterations
    5. Sync to vault
    """
    # Generate run ID
    run_id, headline = generate_run_id(task[:30])
    
    # Default repo path
    if repo_path is None:
        repo_path = os.getcwd()
    
    # Create directories
    local_dir, vault_dir = create_run_dirs(run_id, headline)
    
    # Save input
    if problem:
        with open(local_dir / "input" / "problem.md", "w") as f:
            f.write(problem)
    
    # Save plan if provided
    plan_content = f"""# Plan for: {task}

## Problem
{problem or 'Not specified'}

## Scope
{scope or 'Not specified'}

## Tests
{tests or 'Not specified'}

## Constraints
{constraints or 'Not specified'}

## Done Criteria
{done_criteria or 'Not specified'}
"""
    with open(local_dir / "input" / "plan.md", "w") as f:
        f.write(plan_content)
    
    # Save manifest
    save_manifest(run_id, task, repo_path, "running")
    
    # Build the task prompt
    full_task = f"""{task}

## Context:
- Problem: {problem or 'Not specified'}
- Scope: {scope or 'Not specified'}
- Tests: {tests or 'Not specified'}
- Constraints: {constraints or 'Not specified'}
- Done: {done_criteria or 'Not specified'}
"""
    
    # Execute first iteration
    result = spawn_swarm(repo_path, full_task, run_id=run_id)
    
    return {
        "run_id": run_id,
        "headline": headline,
        "status": "started",
        "message": f"Autonomous task started. Run ID: {run_id}",
    }


# ----------------------------------------------------------------------
# Main Entry Point (for imports)
# ----------------------------------------------------------------------

__all__ = [
    "generate_run_id",
    "get_runs_dir",
    "get_vault_dir",
    "create_run_dirs",
    "save_manifest",
    "update_manifest",
    "sync_to_vault",
    "spawn_codex_agent",
    "build_swarm_prompt",
    "get_git_remote",
    "add_fork_remote",
    "create_pr_branch",
    "generate_report",
    "save_report",
    # New swarm functions
    "spawn_swarm",
    "check_run_status",
    "run_iteration",
    "complete_vault_sync",
    "execute_autonomous_task",
    "RUNNING_PROCESSES",
]