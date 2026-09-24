# File tools reached other agents' data, and the agent's own policy

**Found by:** a real model (Gemini 3.5 Flash) on a live worker run, 2026-09-24.

## What happened

The `operations` worker, asked to extract last quarter's refunds, searched the host for data.
With the runtime's `file` toolset it listed every `state.db` on the host (other agents'
conversation history), read the tenant bundle's objective file, and could have read the
shared board and the audit log. Nothing stopped it: the runtime's file guard
(`agent/file_safety.py`) describes itself as "defense-in-depth, NOT a security boundary". It
blocks credential files (`.env`, `auth.json`, token and vault directories) and nothing else.

Worse, found while fixing it: nothing stopped a `write_file` to the agent's **own**
`nova-policy.json` or to the policy plugin itself. An agent could have rewritten its rules
for its next run.

A second, unrelated defect surfaced in the same test: the runtime image sets
`HERMES_WRITE_SAFE_ROOT=/opt/data`, its default home. The AWS deployment moves the home to the
state volume and never overrode the root, so **every** file write by a worker on AWS was
refused by the runtime — its own task workspace included.

## What changed

- The policy plugin holds the four file tools (`read_file`, `search_files`, `write_file`,
  `patch`, including every file a V4A patch names) to an allow-list, decided in
  `nova/policy/decide.py::decide_paths`: the task's workspace read and write; the agent's own
  profile and its task's attachments read only; nothing else. Paths are resolved with
  `realpath`, so `..` and symlinks cannot step out. Outside a task nothing is writable, and
  the profiles directory never is.
- The worker unit sets `HERMES_WRITE_SAFE_ROOT` to the task workspaces, so the runtime's own
  guard and NOVA's agree.

## What remains

`terminal` and `execute_code` run as the same OS user and are not path-checked. An agent that
must not reach other agents' data must not be granted them; the example bundle denies both.

## The lesson

A guard's own description of its scope is the first thing to read. "Not a security boundary"
was written in the file, and the platform had been treating it as one.
