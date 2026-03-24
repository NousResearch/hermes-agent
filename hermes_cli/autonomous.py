"""Autonomous mode - Swarm execution for Hermes.

This module provides the core logic for spawning a swarm of Codex agents
with GSD skills to execute tasks autonomously.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


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
]