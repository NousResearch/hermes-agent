---
name: orca-control
version: 1.0.0
description: "Manage, orchestrate, inspect, and automate the Orca IDE and Multi-Agent Runtime Server — projects, worktrees, supervised workers, interactive terminals, decision gates, accounts, and automations."
author: "Rafa Martins <rafacpti@gmail.com>"
credits: "Rafa Martins (rafacpti@gmail.com)"
homepage: "https://github.com/rafacpti23/orca-control-skill"
license: "MIT"
platforms: [linux]
metadata:
  hermes:
    category: devops
    tags: [orca, devops, agents, orchestration, worktrees, terminals, automations, multi-agent]
    author: "Rafa Martins"
    email: "rafacpti@gmail.com"
---

# Orca Control & Multi-Agent Orchestrator

**Author**: Rafa Martins (`rafacpti@gmail.com`)  
**Repository**: [github.com/rafacpti23/orca-control-skill](https://github.com/rafacpti23/orca-control-skill)  
**License**: MIT  

Comprehensive management, operation, diagnostics, and orchestration toolkit for the **Orca IDE / Multi-Agent Runtime Server**.

---

## 🏗️ Architecture & Server Layout

* **System User:** `orca` (`uid: 993`, `gid: 984`)
* **Home Directory:** `/home/orca`
* **Configuration & SQLite DB:** `/home/orca/.config/orca/` (`orchestration.db`, sockets, runtime config)
* **Projects & Workspaces:** `/home/orca/orca/projects/` & `/home/orca/orca/workspaces/`
* **Service Manager:** `orca-serve.service` (systemd unit running port `6768`, WebSocket `ws://127.0.0.1:6768` or `wss://<your-domain>`)
* **Global CLI Entrypoint:** `/usr/local/bin/orca` (wraps Node execution of unpacked Orca CLI engine with `HOME=/home/orca`)

---

## 🚦 Diagnostics & Health Checks

### 1. Runtime & Graph State
```bash
# Check runtime readiness, PID, window status and graph state
orca status

# Check systemd service status
systemctl status orca-serve --no-pager

# Restart service cleanly if required
systemctl restart orca-serve
```

### 2. Resource & Memory Diagnostics
```bash
orca diagnostics memory
```

---

## 📁 Project & Worktree Management

### 1. Projects and Repositories
```bash
# List all registered repositories
orca repo list

# List durable projects known to Orca
orca project list

# Register an existing filesystem directory as an Orca repo
orca repo add /home/orca/orca/projects/<project_name>

# Make a project available on host by cloning a repository
orca project setup-clone <git_url>
```

### 2. Worktrees (Isolated Task Environments)
```bash
# List all active Orca-managed worktrees
orca worktree list

# Compact orchestration summary across worktrees
orca worktree ps

# Create a new isolated worktree based on a git branch
orca worktree create <branch_name>

# Inspect a specific worktree
orca worktree show <worktree_id>

# Remove a worktree safely
orca worktree rm <worktree_id>
```

---

## 🤖 Multi-Agent Orchestration & Task Lifecycle

### 1. Runs and Tasks
```bash
# List active and past orchestration Runs
orca orchestration run-list

# Create and bind a new orchestration Run
orca orchestration run-create --title "Backend Refactor Sprint"

# List orchestration tasks
orca orchestration task-list

# Filter tasks by status (pending | ready | dispatched | completed | failed | blocked)
orca orchestration task-list --status pending

# Create a new task with explicit specifications
orca orchestration task-create --title "Implement Auth Middleware" --spec "Use JWT with HMAC-SHA256..."

# Update task status
orca orchestration task-update --task <task_id> --status completed
```

### 2. Supervised Workers (Agents in Action)
```bash
# List active supervised worker accounting
orca orchestration worker-list

# Start a worker attached to a task
orca orchestration worker-start --task <task_id>

# Read output / logs from a supervised worker
orca orchestration worker-read --worker <worker_id>

# Release worker terminal when task completes
orca orchestration worker-release --worker <worker_id>

# Retain worker terminal for live debugging
orca orchestration worker-retain --worker <worker_id>
```

### 3. Decision Gates & Inter-Agent Inbox
```bash
# List decision gates awaiting human/coordinator resolution
orca orchestration gate-list

# Resolve a decision gate (approve or reject agent proposed change)
orca orchestration gate-resolve --gate <gate_id> --decision approve

# Inspect inbox for inter-agent communication
orca orchestration inbox

# Send direct message to a terminal / worker thread
orca orchestration reply --thread <thread_id> --content "Approved. Proceed with database migration."
```

---

## 💻 Live Terminals & Session Control

```bash
# List all live Orca-managed terminals with output previews
orca terminal list

# Create a new terminal session in the current worktree
orca terminal create

# Send commands or input to a running terminal
orca terminal send --terminal <terminal_id> "npm test\n"

# Read bounded terminal output
orca terminal read --terminal <terminal_id>

# Close a specific terminal pane
orca terminal close --terminal <terminal_id>
```

---

## 🔑 Agent Accounts & Credentials (Claude & Codex)

```bash
# List registered agent accounts
orca account list

# Add/Authenticate OpenAI Codex CLI account
orca account add --agent codex

# Add/Authenticate Anthropic Claude CLI account
orca account add --agent claude
```

---

## ⏰ Automations & Scheduled Workflows

```bash
# List registered automations
orca automations list

# Trigger an automation run manually
orca automations run <automation_id>

# View execution history for an automation
orca automations runs <automation_id>
```

---

## 🌐 Browser Automation & Mobile Emulators

Orca provides native computer use and viewport control tools:

```bash
# Capture DOM / accessibility tree snapshot
orca snapshot

# Navigate browser tab to URL
orca goto https://example.com

# Capture page screenshot
orca screenshot

# List connected mobile emulators
orca emulator list

# Send tap gesture to emulator
orca emulator tap --x 120 --y 340
```

---

## ⚠️ Pitfalls & Pro Tips

1. **User Permissions:** Ensure files created in `/home/orca/` belong to `orca:orca` (`chown -R orca:orca /home/orca/orca/projects/`).
2. **Sender Terminal Resolution:** When invoking `orca orchestration` outside of an interactive terminal session, pass `--from <terminal_handle>` (discoverable via `orca terminal list`).
3. **Daemon Safety:** Use `systemctl restart orca-serve` instead of killing Electron PIDs directly to prevent stale lockfiles.
