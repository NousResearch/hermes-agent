#!/usr/bin/env python3
"""
Hermes Swarm Orchestration Helper

Usage:
    python3 swarm.py init <project_name> [--desc <description>]
    python3 swarm.py status <swarm_id>
    python3 swarm.py spawn <swarm_id> <agent_name> <task> [--model <model>]
    python3 swarm.py progress <swarm_id>
    python3 swarm.py complete <swarm_id>
    python3 swarm.py cleanup <swarm_id> [--archive]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Config
HERMES_HOME = Path.home() / ".hermes"
SWARM_DIR = HERMES_HOME / "swarm"
ARCHIVE_DIR = SWARM_DIR / "archive"


def ensure_dirs():
    """Create necessary directories."""
    SWARM_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def init_swarm(project_name: str, description: str = "", tech_stack: list = None) -> str:
    """
    Initialize a new swarm with directory structure and base config.
    Returns swarm_id.
    """
    ensure_dirs()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    swarm_id = f"{project_name.lower().replace(' ', '-')}-{timestamp}"

    swarm_path = SWARM_DIR / swarm_id
    memory_path = swarm_path / "memory"
    context_path = memory_path / "context"
    artifacts_path = memory_path / "artifacts"
    task_queue_path = memory_path / "task_queue"
    progress_path = memory_path / "progress"
    logs_path = swarm_path / "logs"

    # Create structure
    for p in [memory_path, context_path, artifacts_path, task_queue_path, progress_path, logs_path]:
        p.mkdir(parents=True, exist_ok=True)

    # Create project overview
    project_overview = {
        "swarm_id": swarm_id,
        "project": project_name,
        "description": description,
        "tech_stack": tech_stack or [],
        "created_at": datetime.now().isoformat(),
        "status": "initialized",
        "agents": {},
        "tasks": {}
    }

    with open(context_path / "project_overview.json", "w") as f:
        json.dump(project_overview, f, indent=2)

    # Create task queue
    task_queue = {
        "swarm_id": swarm_id,
        "created_at": datetime.now().isoformat(),
        "tasks": {}
    }

    with open(memory_path / "task_queue.json", "w") as f:
        json.dump(task_queue, f, indent=2)

    # Create progress tracker
    progress = {
        "swarm_id": swarm_id,
        "last_updated": datetime.now().isoformat(),
        "agents": {},
        "tasks": {}
    }

    with open(memory_path / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)

    print(f"✅ Swarm initialized: {swarm_id}")
    print(f"   Path: {swarm_path}")
    print(f"   Context: {context_path / 'project_overview.json'}")

    return swarm_id


def add_task(swarm_id: str, task_id: str, description: str, agent: str = None,
             dependencies: list = None, instructions: str = "") -> dict:
    """
    Add a task to the swarm's task queue.
    """
    swarm_path = SWARM_DIR / swarm_id
    if not swarm_path.exists():
        print(f"❌ Swarm not found: {swarm_id}")
        sys.exit(1)

    task_queue_path = swarm_path / "memory" / "task_queue.json"
    with open(task_queue_path) as f:
        task_queue = json.load(f)

    task = {
        "id": task_id,
        "description": description,
        "instructions": instructions,
        "assigned_to": agent,
        "status": "pending",
        "dependencies": dependencies or [],
        "artifacts": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    task_queue["tasks"][task_id] = task

    with open(task_queue_path, "w") as f:
        json.dump(task_queue, f, indent=2)

    # Also update progress
    progress_path = swarm_path / "memory" / "progress.json"
    with open(progress_path) as f:
        progress = json.load(f)

    progress["tasks"][task_id] = {
        "id": task_id,
        "description": description,
        "assigned_to": agent,
        "status": "pending",
        "dependencies": dependencies or [],
        "step": "queued",
        "artifacts_created": [],
        "blocked_by": None,
        "ready_for_handoff": False,
        "handoff_notes": None,
        "updated_at": datetime.now().isoformat()
    }

    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"✅ Task added: {task_id}")
    return task


def spawn_agent(swarm_id: str, agent_name: str, task: str = None,
                model: str = None, profile: str = None, toolsets: str = None,
                context_mode: str = "minimal") -> str:
    """
    Spawn a tmux agent for the swarm.

    context_mode: "minimal" (small models, <1500 tokens) or "full" (large models)
    """
    swarm_path = SWARM_DIR / swarm_id
    if not swarm_path.exists():
        print(f"❌ Swarm not found: {swarm_id}")
        sys.exit(1)

    session_name = f"swarm-{swarm_id}-{agent_name}"
    agent_dir = swarm_path / "agents" / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Build context based on mode
    context_path = swarm_path / "memory" / "context"
    artifacts_path = swarm_path / "memory" / "artifacts"
    progress_path = swarm_path / "memory" / "progress.json"

    if context_mode == "minimal":
        prompt = f"""You are {agent_name} in swarm {swarm_id}.

READ YOUR CONTEXT:
- Project: {context_path / 'project_overview.json'}
- Your tasks: {context_path / f'{agent_name}_task.json'} (if exists)

YOUR ROLE: {task or "Specialist agent"}

COORDINATION:
- Progress: {progress_path}
- Write artifacts: {artifacts_path}/
- Task queue: {swarm_path / 'memory' / 'task_queue.json'}

PROTOCOL:
1. Read project_overview.json to understand the project
2. Read your task spec if it exists
3. Update progress.json when you complete steps
4. Write final deliverables to artifacts/
5. Signal completion in progress.json

START WORK NOW. Be concise. Write progress updates to {progress_path}.
"""
    else:  # full context
        # Read all context files and dump them
        context_files = list(context_path.glob("*.json"))
        full_context = {}
        for cf in context_files:
            with open(cf) as f:
                full_context[cf.name] = json.load(f)

        prompt = f"""You are {agent_name} in swarm {swarm_id}.

PROJECT OVERVIEW:
{json.dumps(full_context.get("project_overview.json", {}), indent=2)}

YOUR TASK: {task}

FULL PROJECT CONTEXT:
{json.dumps(full_context, indent=2)}

COORDINATION:
- Progress: {progress_path}
- Artifacts: {artifacts_path}/

PROTOCOL:
1. Read all context files for full project understanding
2. Execute your task
3. Write deliverables to artifacts/
4. Update progress.json with status updates
5. Signal completion when done

START NOW.
"""

    # Save agent prompt
    with open(agent_dir / "prompt.txt", "w") as f:
        f.write(prompt)

    # Build tmux command
    cmd = f'tmux new-session -d -s {session_name} -x 120 -y 40 "hermes'

    if profile:
        cmd += f' -p {profile}'
    if model:
        cmd += f' -m {model}'
    if toolsets:
        cmd += f' -t {toolsets}'

    cmd += f' -s {agent_dir / "prompt.txt"}"\n'

    # Also update progress with agent info
    with open(progress_path) as f:
        progress = json.load(f)

    progress["agents"][agent_name] = {
        "session": session_name,
        "spawned_at": datetime.now().isoformat(),
        "status": "running",
        "context_mode": context_mode,
        "task": task
    }

    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    # Run tmux
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Agent spawned: {agent_name}")
        print(f"   Session: {session_name}")
        print(f"   Task: {task}")
        print(f"   Context mode: {context_mode}")
        print(f"   Monitor: tmux attach -t {session_name}")
        return session_name
    else:
        print(f"❌ Failed to spawn agent: {result.stderr}")
        sys.exit(1)


def send_to_agent(session_name: str, message: str):
    """Send a message to a running tmux agent."""
    # Escape the message for tmux
    escaped = message.replace("'", "'\\''")
    cmd = f"tmux send-keys -t {session_name} '{escaped}' Enter"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Message sent to {session_name}")
    else:
        print(f"❌ Failed: {result.stderr}")


def read_agent_output(session_name: str, lines: int = 50) -> str:
    """Read output from a tmux agent."""
    cmd = f"tmux capture-pane -t {session_name} -p | tail -{lines}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout


def get_swarm_status(swarm_id: str) -> dict:
    """Get full swarm status."""
    swarm_path = SWARM_DIR / swarm_id
    if not swarm_path.exists():
        return {"error": f"Swarm not found: {swarm_id}"}

    status = {
        "swarm_id": swarm_id,
        "path": str(swarm_path),
        "exists": True
    }

    # Read progress
    progress_path = swarm_path / "memory" / "progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
            status["agents"] = progress.get("agents", {})
            status["tasks"] = progress.get("tasks", {})
            status["last_updated"] = progress.get("last_updated", "unknown")

    # Check tmux sessions
    result = subprocess.run("tmux list-sessions 2>/dev/null | grep 'swarm-'", shell=True,
                            capture_output=True, text=True)
    status["tmux_sessions"] = result.stdout.strip().split("\n") if result.stdout else []

    return status


def print_swarm_status(swarm_id: str):
    """Pretty print swarm status."""
    status = get_swarm_status(swarm_id)

    if "error" in status:
        print(f"❌ {status['error']}")
        return

    print(f"\n{'='*60}")
    print(f"SWARM: {swarm_id}")
    print(f"PATH: {status['path']}")
    print(f"{'='*60}\n")

    # Agents
    print("🤖 AGENTS:")
    if status.get("agents"):
        for name, info in status["agents"].items():
            print(f"   {name}: {info['status']} | session: {info['session']}")
    else:
        print("   No agents spawned yet")

    # Tasks
    print("\n📋 TASKS:")
    if status.get("tasks"):
        for task_id, info in status["tasks"].items():
            status_icon = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "blocked": "🚫"}
            icon = status_icon.get(info["status"], "❓")
            print(f"   {icon} {task_id}: {info['status']} - {info['description'][:50]}...")
            if info.get("step"):
                print(f"      → {info['step']}")
    else:
        print("   No tasks defined yet")

    # Tmux sessions
    print("\n🖥️  TMUX SESSIONS:")
    if status.get("tmux_sessions"):
        for session in status["tmux_sessions"]:
            if session:
                print(f"   {session}")
    else:
        print("   No active sessions")

    print()


def update_progress(swarm_id: str, agent_name: str = None, task_id: str = None,
                     status: str = None, step: str = None, blocked_by: str = None):
    """Update progress for an agent or task."""
    swarm_path = SWARM_DIR / swarm_id
    progress_path = swarm_path / "memory" / "progress.json"

    with open(progress_path) as f:
        progress = json.load(f)

    if agent_name and agent_name in progress["agents"]:
        progress["agents"][agent_name]["status"] = status or progress["agents"][agent_name].get("status")

    if task_id and task_id in progress["tasks"]:
        if status:
            progress["tasks"][task_id]["status"] = status
        if step:
            progress["tasks"][task_id]["step"] = step
        if blocked_by:
            progress["tasks"][task_id]["blocked_by"] = blocked_by
            progress["tasks"][task_id]["status"] = "blocked"

    progress["last_updated"] = datetime.now().isoformat()

    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"✅ Progress updated for {agent_name or task_id}")


def monitor_swarm(swarm_id: str, timeout_minutes: int = 60, poll_interval: int = 30):
    """
    Monitor swarm until all tasks complete or timeout.
    Returns "success", "timeout", or "partial"
    """
    print(f"📡 Monitoring swarm {swarm_id}...")
    print(f"   Timeout: {timeout_minutes} minutes")
    print(f"   Poll interval: {poll_interval} seconds\n")

    start = time.time()
    timeout_seconds = timeout_minutes * 60

    while time.time() - start < timeout_seconds:
        status = get_swarm_status(swarm_id)

        # Count statuses
        tasks = status.get("tasks", {})
        if not tasks:
            print("⏳ No tasks defined yet...")
            time.sleep(poll_interval)
            continue

        task_statuses = [t["status"] for t in tasks.values()]
        completed = task_statuses.count("completed")
        total = len(task_statuses)
        blocked = task_statuses.count("blocked")
        in_progress = task_statuses.count("in_progress")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {completed}/{total} completed, "
              f"{in_progress} in progress, {blocked} blocked")

        if completed == total:
            print("\n✅ All tasks completed!")
            return "success"

        if blocked > 0:
            blocked_tasks = [t for t in tasks.values() if t["status"] == "blocked"]
            print(f"\n🚫 Blocked tasks detected:")
            for bt in blocked_tasks:
                print(f"   {bt['id']}: blocked by {bt.get('blocked_by', 'unknown')}")
            return "blocked"

        time.sleep(poll_interval)

    print(f"\n⏰ Timeout reached after {timeout_minutes} minutes")
    return "timeout"


def collect_results(swarm_id: str) -> dict:
    """Collect all results from swarm artifacts."""
    swarm_path = SWARM_DIR / swarm_id
    artifacts_path = swarm_path / "memory" / "artifacts"
    progress_path = swarm_path / "memory" / "progress.json"

    results = {
        "swarm_id": swarm_id,
        "collected_at": datetime.now().isoformat(),
        "completed_tasks": [],
        "artifacts": [],
        "summaries": []
    }

    # Read progress
    with open(progress_path) as f:
        progress = json.load(f)

    for task_id, info in progress.get("tasks", {}).items():
        if info["status"] == "completed":
            results["completed_tasks"].append({
                "id": task_id,
                "description": info.get("description"),
                "step": info.get("step"),
                "artifacts": info.get("artifacts_created", [])
            })

    # Read artifacts
    if artifacts_path.exists():
        for artifact in artifacts_path.glob("*.json"):
            with open(artifact) as f:
                try:
                    data = json.load(f)
                    results["artifacts"].append({
                        "file": artifact.name,
                        "data": data
                    })
                except:
                    pass

    return results


def complete_swarm(swarm_id: str, kill_agents: bool = True):
    """Mark swarm as complete and optionally kill agents."""
    swarm_path = SWARM_DIR / swarm_id
    progress_path = swarm_path / "memory" / "progress.json"

    # Update status
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)

        progress["status"] = "completed"
        progress["completed_at"] = datetime.now().isoformat()

        # Kill tmux sessions
        if kill_agents:
            for agent_name, info in progress.get("agents", {}).items():
                session = info.get("session")
                if session:
                    subprocess.run(f"tmux kill-session -t {session} 2>/dev/null", shell=True)
                    print(f"🪦 Killed session: {session}")

        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

    # Collect results
    results = collect_results(swarm_id)
    results_path = swarm_path / "memory" / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Swarm {swarm_id} marked complete")
    print(f"   Results: {results_path}")
    print(f"   Completed {len(results['completed_tasks'])} tasks")


def cleanup_swarm(swarm_id: str, archive: bool = False):
    """Clean up swarm directory."""
    swarm_path = SWARM_DIR / swarm_id

    if not swarm_path.exists():
        print(f"❌ Swarm not found: {swarm_id}")
        return

    if archive:
        # Kill agents first
        status = get_swarm_status(swarm_id)
        for agent_name, info in status.get("agents", {}).items():
            session = info.get("session")
            if session:
                subprocess.run(f"tmux kill-session -t {session} 2>/dev/null", shell=True)

        # Archive
        archive_name = f"{swarm_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar.gz"
        archive_path = ARCHIVE_DIR / archive_name
        subprocess.run(f"tar -czf {archive_path} {swarm_path}", shell=True)
        shutil.rmtree(swarm_path)
        print(f"📦 Archived to: {archive_path}")
    else:
        shutil.rmtree(swarm_path)
        print(f"🗑️  Deleted: {swarm_path}")


def list_swarms():
    """List all swarms."""
    if not SWARM_DIR.exists():
        print("No swarms found")
        return

    print("\n📦 SWARMS:\n")
    for swarm_dir in sorted(SWARM_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if swarm_dir.is_dir():
            progress_file = swarm_dir / "memory" / "progress.json"
            status = "unknown"
            if progress_file.exists():
                with open(progress_file) as f:
                    p = json.load(f)
                    status = p.get("status", "unknown")

            created = swarm_dir.stat().st_ctime
            created_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")

            print(f"  {swarm_dir.name} | {status} | created {created_str}")


# CLI
def main():
    parser = argparse.ArgumentParser(description="Hermes Swarm Orchestration Helper")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init
    p_init = subparsers.add_parser("init", help="Initialize a new swarm")
    p_init.add_argument("project_name", help="Project name")
    p_init.add_argument("--desc", default="", help="Project description")
    p_init.add_argument("--stack", nargs="+", help="Tech stack (e.g., node react postgres)")

    # add-task
    p_task = subparsers.add_parser("add-task", help="Add a task to swarm")
    p_task.add_argument("swarm_id", help="Swarm ID")
    p_task.add_argument("task_id", help="Task ID")
    p_task.add_argument("description", help="Task description")
    p_task.add_argument("--agent", help="Assigned agent")
    p_task.add_argument("--deps", nargs="+", help="Dependency task IDs")
    p_task.add_argument("--instructions", default="", help="Detailed instructions")

    # spawn
    p_spawn = subparsers.add_parser("spawn", help="Spawn an agent")
    p_spawn.add_argument("swarm_id", help="Swarm ID")
    p_spawn.add_argument("agent_name", help="Agent name")
    p_spawn.add_argument("--task", help="Task description")
    p_spawn.add_argument("--model", help="Model to use")
    p_spawn.add_argument("--profile", help="Hermes profile")
    p_spawn.add_argument("--toolsets", help="Toolsets (comma-separated)")
    p_spawn.add_argument("--context", choices=["minimal", "full"], default="minimal",
                         help="Context mode (minimal=small models, full=large models)")

    # send
    p_send = subparsers.add_parser("send", help="Send message to agent")
    p_send.add_argument("session", help="Session name")
    p_send.add_argument("message", help="Message to send")

    # read
    p_read = subparsers.add_parser("read", help="Read agent output")
    p_read.add_argument("session", help="Session name")
    p_read.add_argument("--lines", type=int, default=50, help="Lines to read")

    # status
    p_status = subparsers.add_parser("status", help="Get swarm status")
    p_status.add_argument("swarm_id", nargs="?", help="Swarm ID (omit to list all)")

    # update
    p_update = subparsers.add_parser("update", help="Update progress")
    p_update.add_argument("swarm_id", help="Swarm ID")
    p_update.add_argument("--agent", help="Agent name")
    p_update.add_argument("--task", help="Task ID")
    p_update.add_argument("--status", help="Status")
    p_update.add_argument("--step", help="Current step")

    # monitor
    p_monitor = subparsers.add_parser("monitor", help="Monitor swarm progress")
    p_monitor.add_argument("swarm_id", help="Swarm ID")
    p_monitor.add_argument("--timeout", type=int, default=60, help="Timeout in minutes")
    p_monitor.add_argument("--poll", type=int, default=30, help="Poll interval in seconds")

    # complete
    p_complete = subparsers.add_parser("complete", help="Mark swarm complete")
    p_complete.add_argument("swarm_id", help="Swarm ID")
    p_complete.add_argument("--keep-agents", action="store_true", help="Don't kill tmux agents")

    # cleanup
    p_cleanup = subparsers.add_parser("cleanup", help="Clean up swarm")
    p_cleanup.add_argument("swarm_id", help="Swarm ID")
    p_cleanup.add_argument("--archive", action="store_true", help="Archive before deleting")

    args = parser.parse_args()

    if args.command == "init":
        swarm_id = init_swarm(args.project_name, args.desc, args.stack)
        print(f"\nNext steps:")
        print(f"  swarm.py add-task {swarm_id} <task_id> <description>")
        print(f"  swarm.py spawn {swarm_id} <agent_name> --task <task>")

    elif args.command == "add-task":
        add_task(args.swarm_id, args.task_id, args.description, args.agent, args.deps, args.instructions)

    elif args.command == "spawn":
        spawn_agent(args.swarm_id, args.agent_name, args.task, args.model, args.profile,
                   args.toolsets, args.context)

    elif args.command == "send":
        send_to_agent(args.session, args.message)

    elif args.command == "read":
        output = read_agent_output(args.session, args.lines)
        print(output)

    elif args.command == "status":
        if args.swarm_id:
            print_swarm_status(args.swarm_id)
        else:
            list_swarms()

    elif args.command == "update":
        update_progress(args.swarm_id, args.agent, args.task, args.status, args.step)

    elif args.command == "monitor":
        result = monitor_swarm(args.swarm_id, args.timeout, args.poll)
        print(f"\nResult: {result}")

    elif args.command == "complete":
        complete_swarm(args.swarm_id, not args.keep_agents)

    elif args.command == "cleanup":
        cleanup_swarm(args.swarm_id, args.archive)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
