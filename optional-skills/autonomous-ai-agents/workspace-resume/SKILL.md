---
name: workspace-resume
description: "Land in the right repo when Hermes starts, not home."
version: 0.1.0
author: Dustinn Stroud (strouddustinn-bot), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Workspace, Session, Resume, Git, Productivity]
    related_skills: [hermes-agent]
---

# Workspace Resume Skill

Resolves the workspace a session should start in when Hermes launches from the
home directory: recall the last active git-repo workspace from this profile's
session store, or rank candidate repos and pick the highest-leverage one. It
does not manage project workspaces (the `project` toolset does that) or start
new sessions — it only decides where the current one should be anchored.

## When to Use

- The user types `hermes` from `$HOME` (or any non-repo directory) and starts
  working on "the code", "the project", "the repo" without cd-ing first.
- A new session starts and the agent's working directory is the home directory
  but a previous session in this profile was clearly working in a repo.
- The user asks "where should we pick up?" / "what was I working on" at session
  start.
- Don't use for: the user explicitly cd-ed or named a directory this session
  (respect that); remote terminal backends (docker/ssh/modal) — the recall
  script only ranks directories on the local host.

## Prerequisites

- Python 3.10+ (stdlib only — `sqlite3`, `pathlib`, `subprocess` for git).
- `git` on PATH for the ranking pass (recall works without it).
- A profile-local session store: the script resolves `$HERMES_HOME` (or
  `~/.hermes`) and `$HERMES_PROFILE` exactly like the agent does, so profiles
  and the default home both work with no arguments.

## How to Run

Run through `terminal` with the skill-relative script path:

```bash
python3 scripts/workspace_recall.py            # resolve + print decision (JSON)
python3 scripts/workspace_recall.py --sync     # also update terminal.cwd in config.yaml
python3 scripts/workspace_recall.py --json     # machine-readable single object
```

Then `cd` to the returned `target` (see Procedure) when it differs from the
current directory.

## Quick Reference

| Flag | Effect |
|---|---|
| (none) | Resolve + print the decision; no side effects |
| `--sync` | Additionally write `terminal.cwd` in `config.yaml` via `hermes config set` |
| `--json` | Emit one JSON object (for scripted consumption) |
| `--candidates` | Print the ranked candidate list instead of acting |

`--sync` updates the *config* for future launches; it never moves the session
mid-conversation (cache-safe: nothing about the live session changes).

## Procedure

1. **Detect the need.** At session start, when the agent's cwd is `$HOME` (or
   any directory that is not inside a git worktree), and the user's first
   message implies repo work, run the script. Completion: the script exits 0
   with a decision.
2. **Recall pass.** The script queries this profile's session store for the
   most recent non-archived session whose `cwd` (or `git_repo_root`) is an
   existing directory. Completion: a hit yields `mode: "recall"` with `target`,
   `age_hours`, and the last session's `git_branch`.
3. **Rank pass.** On no recall hit, the script scans candidate roots
   (`~`, `~/Projects`, `~/projects`, `~/code`, `~/dev`, `~/repos`,
   `~/src`, `~/work`, plus `$CWD_ROOTS` if set) for git worktrees, scores each
   by last-commit recency, dirty-state, and session count in the store, and
   returns the top one. Completion: `mode: "rank"` with `target` and
   `score` breakdown.
4. **Hop.** `cd` to `target` via `terminal`. Announce the mode, target, and
   why (last session 2h ago on `main`, or top-ranked repo: latest commit
   yesterday, 3 recent sessions). Completion: cwd is the target; state the
   branch/repo to the user.
5. **Sync (opt-in).** Run `scripts/workspace_recall.py --sync` to persist the
   choice for future launches via `hermes config set terminal.cwd <target>`.
   Completion: `hermes config get terminal.cwd` echoes the target.

## Pitfalls

- **Respect an explicit cwd.** Never trigger when the user launched Hermes
  from inside a repo or named a directory — that's a deliberate choice. The
  script is for the "bare `hermes` from `$HOME`" case only.
- **Never cd mid-conversation for sync.** The `--sync` flag only writes
  config; the in-session hop (step 4) happens once at session start, before
  any user turns — this keeps prompt caching intact.
- **Deleted workspaces.** The recall hit is validated with `os.path.isdir`
  before it wins; a stale `cwd` row falls through to the rank pass instead of
  hopping to a missing directory.
- **Remote backends.** With a remote terminal backend, directory probes run
  inside the container — don't run this skill there; the session store's
  paths won't match the container's filesystem.
- **Profile isolation.** The script keys on `$HERMES_PROFILE` — running it in
  the default profile recalls the default profile's sessions, not another
  profile's. Cross-profile recall is out of scope by design.
- **Windows.** `git` is still required for the rank pass; recall works without
  it. Paths with spaces are quoted in all emitted commands.

Windows note: recall works without git; the rank pass requires git on PATH
(`where git` — not POSIX `which`).

## Verification

- `python3 scripts/workspace_recall.py --json` from a machine with prior
  sessions returns a decision object with non-empty `target` and a `mode` of
  `recall` or `rank` (or `none` on a fresh machine).
- After `--sync`, `hermes config get terminal.cwd` echoes the resolved target.
- On a profile with no sessions and no candidate repos, the script exits 0
  with `mode: "none"` — never a traceback, never a hang.
- The test suite covers recall-hit, rank-ordering, and empty-store cases with
  a temp `HERMES_HOME` (see `tests/skills/test_workspace_resume_skill.py`).