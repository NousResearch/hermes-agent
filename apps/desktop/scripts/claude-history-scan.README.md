# Claude Code History Scanner

A read-only Python scanner for Claude Code session transcripts. Emits
metadata (titles, timestamps, message counts) — **never** message content
or any raw data (tool_result, attachment, system-reminder).

## What it does

Walks `~/.claude/projects/<workspace>/<session>.jsonl`, parses each file in
a single pass, and writes a JSON array of session metadata records. Used
by the Hermes Desktop sidebar feature (see
`~/.hermes/plans/2026-07-01-claude-code-history-sidebar.md`) to populate
the "Claude Code 历史" sidebar section without exposing raw conversation
content.

## What it does NOT do

- It does not store `message.content` (text, tool_result, attachment, etc.).
- It does not read sub-agent jsonl files (only direct children of each
  `projects/<workspace>/` directory).
- It does not write anything under `~/.claude/` (strictly read-only there).
- It does not require a network connection.

## Schema

```json
[{
  "session_id":      "b9dd9fe9-79f3-49ce-881f-92c69517def3",
  "cwd":             "C:\\Claude\\给女朋友的桌宠",
  "first_user":      "读handoff然后接上继续",
  "first_timestamp": "2026-06-12T11:39:00.000Z",
  "last_timestamp":  "2026-06-29T11:10:00.000Z",
  "message_count":   593,
  "workspace_group": "C--Claude",
  "file_size_bytes": 8388608,
  "file_path":       "C:\\Users\\hf\\.claude\\projects\\C--Claude\\<session>.jsonl"
}]
```

- `first_user` is the first line of the first `user` message's first text
  block, truncated to 100 chars (or `null` if no such message exists).
- `message_count` counts `user` + `assistant` rows that contain at least one
  non-empty text block (raw data like `tool_result` is intentionally
  excluded from the count — we report the number of human-readable turns).
- `workspace_group` is the immediate subdirectory name under `projects/`
  (e.g. `C--Claude`, `C--Users-hf`).
- If a file fails to read, an `error` key is added to that entry; the scan
  continues. Use `--strict` to exit non-zero on any read failure.

## Usage

```bash
# Default: print to stdout
python claude-history-scan.py

# Pretty-print to a file
python claude-history-scan.py --output metadata.json --pretty

# Custom Claude home (also honors $CLAUDE_HOME)
python claude-history-scan.py --claude-home D:/Users/me/.claude
python CLAUDE_HOME=/c/Users/me/.claude python claude-history-scan.py

# CI mode: fail if any file had a read error
python claude-history-scan.py --strict
```

Exit codes: `0` = success, `2` = bad paths, `3` = `--strict` and at least
one read failure.

## Performance

On a Windows host with ~240 jsonl files totaling ~600 MB: ~14 s with the
default thread pool (size = `min(16, cpu_count)`). Bump `--workers` for
faster I/O on hosts with NVMe or many small files.

## Tests

```bash
python -m pytest tests/claude-history-scan.test.py
```

(Tests will be added in M5 alongside the end-to-end guard script. For now,
manual verification is documented in the plan.)
