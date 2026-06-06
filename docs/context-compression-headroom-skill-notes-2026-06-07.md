# Headroom context-compression skill notes — 2026-06-07

This note records a local user-skill update made during a Tomonori Hermes session.
It is intentionally a documentation artifact only; it does not enable any runtime integration.

## Scope decision

Adopt option A only:

- Use `headroom-ai==0.9.7` only in an isolated Windows venv.
- Use it only as a manual API/CLI compression helper.
- Keep telemetry disabled with `HEADROOM_TELEMETRY=off`.
- Continue STOP for automatic integration, MCP registration, proxy/wrap, Hermes/Codex config changes, Obsidian saves, commit/push of the user skill directory, and any secret-bearing logs.

## User skill files updated outside this repository

The actual skill files live in the active Hermes profile, not in this git repository:

```text
C:\Users\nitta\AppData\Local\hermes\skills\autonomous-ai-agents\codex\SKILL.md
C:\Users\nitta\AppData\Local\hermes\skills\autonomous-ai-agents\codex\references\context-compression-smoke-test.md
```

That `skills/` directory was checked and is not a git repository, so the local skill edits could not be committed in place.

## Diff-equivalent: context-compression smoke test reference

Added this Headroom-specific section to `references/context-compression-smoke-test.md`:

```diff
+## Headroom notes from Tomonori Windows/WSL trials
+
+- On Tomonori's Windows host, `headroom-ai==0.23.0` may fall back to a Rust/maturin source build and fail around MSVC `link.exe`; do not install Visual Studio Build Tools just for the first trial unless explicitly approved.
+- `headroom-ai==0.9.7` installs in an isolated Windows venv and is suitable for limited manual API/CLI trials with `HEADROOM_TELEMETRY=off`.
+- For tool-output compression in 0.9.7, pass long content as a `role: "tool"` message. Default `role: "user"` messages are protected and may produce 0% compression unless `CompressConfig(compress_user_messages=True, protect_recent=0, ...)` is intentionally used.
+- 0.9.7 performed well on structured JSON, CI/build-style logs, lint/search JSON, and tool-output lines while preserving high-signal IDs; plain pytest transcript logs, stacktrace-only text, and NDJSON may preserve evidence but produce little/no compression. Treat those as WARN/no-benefit, not success.
+- WSL can install `headroom-ai==0.23.0` in a temporary uv-created venv even when `python3-venv` is missing. Use `/home/nitta/.local/bin/uv venv ...` and `uv pip install --python ...` in that environment. Keep this as verification-only until more samples pass.
```

## Diff-equivalent: Codex App Server Windows note

The `codex` skill already had a Windows app-server stdio note. It was normalized to this non-duplicated wording:

```diff
+On Tomonori's Windows environment, do **not** use `codex remote-control start` as the primary path for Hermes → Codex App session access. The daemon lifecycle reports Unix-only support on Windows. Use the Codex CLI app-server stdio transport instead:
+
+```
+codex app-server --stdio
+```
+
+This is still Codex CLI based, but it exposes the Codex App Server JSON-RPC protocol over stdio. Practical methods verified from Hermes include:
+
+- `initialize`
+- `thread/read` with `threadId` and `includeTurns`
+- `thread/resume` with `threadId`
+- `turn/start` with `threadId` and text input
+
+This path can read/resume an existing Codex App/VS Code session ID and send a message, then observe status changes such as `active` → `idle` and read the final assistant reply.
```

## Verification evidence

### Windows option A: `headroom-ai==0.9.7`

Environment:

```text
C:\Users\nitta\AppData\Local\Temp\headroom-hermes-a-eval\venv
headroom-ai==0.9.7
HEADROOM_TELEMETRY=off
```

Result:

```text
10 samples total
7 samples passed
3 samples warned/no-benefit
```

Passing sample categories:

| Sample category | Approx. savings | Critical evidence retained |
|---|---:|---|
| structured JSON records | 98.11% | FATAL / trace_id |
| CI log | 97.89% | ERROR / FATAL |
| API JSON | 96.95% | request_id / error_code / trace_id |
| tool output | 97.37% | source_to_sink / security.py |
| build log | 97.89% | linker error / trace_id |
| search JSON | 97.51% | security.py / source_to_sink |
| lint JSON | 97.05% | unsafe call / trace_id |

WARN/no-benefit sample categories:

- pytest transcript: high-signal evidence retained, but no material compression.
- stacktrace-only text: high-signal evidence retained, but no material compression.
- NDJSON events: high-signal evidence retained, but no material compression.

Important implementation note:

```python
messages = [
    {"role": "user", "content": "Preserve ERROR/FATAL/trace_id evidence."},
    {"role": "tool", "tool_call_id": "call_eval", "content": long_output},
]
```

Passing the long output as `role: "user"` was not useful in the 0.9.7 trial because recent/user messages were protected and often produced 0% compression.

### WSL verification-only: `headroom-ai==0.23.0`

Environment:

```text
WSL Ubuntu-24.04
/tmp/headroom-wsl-023/venv
headroom-ai==0.23.0
HEADROOM_TELEMETRY=off
```

Setup note:

- `python3 -m venv` failed because `python3-venv` was missing.
- `/home/nitta/.local/bin/uv venv ...` plus `uv pip install --python ...` succeeded.

Smoke result:

| Sample | Result |
|---|---|
| JSON | 38.16% savings, FATAL/trace_id retained, PASS |
| log | high-signal evidence retained, but no material compression, WARN |

## Not done

- No Hermes config change.
- No Codex config change.
- No MCP registration.
- No proxy or shell wrapper.
- No Obsidian write/save.
- No automatic prompt/tool-output integration.
- No push.

## Current recommendation

Keep option A as a limited manual helper only. Use it for long structured JSON, CI/build logs, lint/search JSON, and tool output. Do not use it for secrets, raw user prompts, pytest transcript logs, stacktrace-only text, or NDJSON unless a new smoke test proves material compression without dropping critical evidence.
