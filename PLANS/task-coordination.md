# Task Coordination Framework

## Project Overview

**Autolycus Agent** - FreeBSD-compatible fork of Hermes Agent  
Repository: `waym0reom3ga/autolycus-agent`  
Status: Prototype / FreeBSD Only (not production-ready)

---

## Team Roles & Responsibilities

### waym0re
- **Role**: Project owner/collaborator
- **Responsibilities**: 
  - Overall project direction
  - Periodic review of task progress
  - FreeBSD testing environment access
  - Final approval on major changes

### om3ga (me)
- **Role**: Repository manager/supervisor
- **Access Level**: Full read/write via fine-grained token
- **Responsibilities**:
  1. Manage and track development tasks
  2. Assign work to programming assistant
  3. Review code changes for FreeBSD compatibility focus
  4. Maintain documentation and task backlog
  5. Coordinate between team members

### Programming Assistant (TBD)
- **Role**: Implementation and debugging
- **Responsibilities**:
  - Execute assigned tasks from this document
  - Write code, fix bugs, implement features
  - Report progress and blockers
  - Test changes where possible (Linux environment)

---

## Communication Protocol

### Primary Channel: This Document
Since GitHub Issues are disabled in this repository, we use **PLANS/task-coordination.md** as our task tracking system.

### Workflow
1. **om3ga** creates/updates tasks in this document
2. **Programming Assistant** picks up tasks and marks progress
3. **waym0re** periodically reviews and provides feedback
4. All changes committed to `main` branch

### Status Markers
- `[ ]` - Not started / Available for pickup
- `[~]` - In progress
- `[x]` - Completed
- `[!]` - Blocked / Needs attention

---

## Repository Statistics (as of 2026-04-08)

| Metric | Value |
|--------|-------|
| **Total Commits** | 3,505 |
| **Python Files** | 744 |
| **Repository Size** | 66 MB |
| **Primary Language** | Python |
| **License** | LGPL v2.1 |

### Key Components
- `run_agent.py` (477KB) - Core AIAgent class with conversation loop
- `cli.py` (390KB) - HermesCLI interactive orchestrator  
- `model_tools.py` - Tool orchestration and discovery
- `toolsets.py` - Toolset definitions
- `hermes_state.py` - SQLite session store with FTS5 search
- `tools/` - 744 Python files implementing various tools
- `gateway/` - Multi-platform messaging (Telegram, Discord, Slack, etc.)
- `agent/` - Agent internals (prompt building, compression, caching)

---

## Active Tasks

### Task #1: FreeBSD Compatibility Audit
**Status**: [ ] Not started  
**Priority**: High  
**Assigned to**: Programming Assistant  

**Description**: 
Perform comprehensive audit of codebase to identify Linux-specific dependencies and system calls that need FreeBSD equivalents.

**Deliverables**:
- List of all `os.uname()` checks for "Linux"
- Identify PTY-related code (won't work on FreeBSD)
- Find Docker/singularity backend compatibility issues
- Document missing FreeBSD ports/packages

---

### Task #2: Setup Documentation Review  
**Status**: [ ] Not started  
**Priority**: Medium  
**Assigned to**: Programming Assistant  

**Description**:
Review `README.md` installation instructions and verify all FreeBSD package names are correct. Cross-reference with official FreeBSD ports collection.

**Checklist**:
- [ ] Verify `pkg install rust` is correct
- [ ] Verify `pkg install python311` availability  
- [ ] Check if `uv` has a FreeBSD port or needs cargo install
- [ ] Validate all optional dependencies have FreeBSD equivalents

---

### Task #3: Create FreeBSD-Specific Error Handling
**Status**: [ ] Not started  
**Priority**: High  
**Assigned to**: Programming Assistant  

**Description**:
Add graceful degradation for features that don't work on FreeBSD (voice, PTY) with clear user messaging.

**Files to Modify**:
- `hermes_cli/config.py` - Add FreeBSD detection
- `tools/terminal_tool.py` - Handle missing PTY gracefully
- `cli.py` - Show platform-specific warnings

---

## Completed Tasks

None yet - project coordination framework just established.

---

## Blockers & Known Issues

1. **No FreeBSD Test Environment**: om3ga only has Arch Linux locally. Programming assistant may also lack FreeBSD access.
2. **Issues Disabled**: GitHub issues feature is disabled in this repo, forcing us to use file-based task tracking.
3. **Limited Token Scope**: Fine-grained token only has access to `autolycus-agent` repo, not all of waym0re's repos.

---

## Meeting Notes / Updates

### 2026-04-08: Initial Setup
- om3ga cloned repository and verified full read/write access
- Discovered GitHub issues are disabled - switched to file-based task tracking
- Created this coordination framework document
- Identified key team roles and communication protocol

---

## How to Use This Document

1. **Adding Tasks**: om3ga adds new tasks under "Active Tasks" section with clear description and deliverables
2. **Claiming Tasks**: Programming assistant changes `[ ]` to `[~]` when starting work
3. **Progress Updates**: Add sub-bullets or notes under task as work progresses
4. **Completion**: Change `[~]` to `[x]` and move to "Completed Tasks" section
5. **Blockers**: Add to "Blockers & Known Issues" if stuck

---

*Last updated: 2026-04-08 by om3ga (repo manager)*  
*Document maintained in PLANS/task-coordination.md for version control and team visibility*
