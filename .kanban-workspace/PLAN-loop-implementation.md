# Plan: `/loop` Command Implementation

> Verified against live source at `/home/d/Desktop/agenda/hermes-agent` (working tree, 2026-05-08). All file:line references confirmed by direct read.

---

## 3.1 Summary

`/loop` is a thin UX wrapper around Hermes' existing cron infrastructure. It schedules a recurring cron job from a single slash command:

```
/loop 5m check deployment
```

This becomes a cron job running every 5 minutes, with results delivered back to the chat where it was invoked.

**V1 scope:** parse interval + prompt, create cron job, return job ID. No new storage, no new scheduler, no session inheritance, no `loop.md`, no AI-chosen intervals. Subcommands: `list`, `pause`, `resume`, `remove`.

---

## 3.2 Files to Modify (exact paths + line numbers)

### File 1: `hermes_cli/commands.py`

**Location:** Insert a new `CommandDef` into `COMMAND_REGISTRY` list at line 162, immediately after the `/cron` entry (line 160-162).

**What to add:**
```python
    CommandDef("loop", "Run a prompt repeatedly on a schedule", "Tools & Skills",
               aliases=("repeat",), args_hint="<schedule> <prompt>",
               subcommands=("list", "pause", "resume", "remove")),
```

**Why here:** The registry is organized by category. `loop` belongs in "Tools & Skills" alongside `cron` and `kanban`. Inserting after `cron` keeps scheduling commands grouped.

**Verification:** `COMMAND_REGISTRY` starts at line 64. `/cron` entry is at lines 160-162. `/kanban` follows at 166-170.

---

### File 2: `cli.py`

**Location A — dispatch branch:** Insert `elif canonical == "loop":` at line 6859, immediately after `elif canonical == "cron":` (line 6857-6858) and before `elif canonical == "curator":` (line 6859-6860).

**What to add (line 6859):**
```python
        elif canonical == "loop":
            self._handle_loop_command(cmd_original)
```

**Location B — handler method:** Add `_handle_loop_command()` as a new method on `HermesCLI`. Place it immediately after `_handle_cron_command()` (which ends around line 6580) and before `_handle_curator_command()` (which starts after).

**What to add:**
```python
    def _handle_loop_command(self, cmd: str):
        """Handle /loop <schedule> <prompt> — thin wrapper around cronjob tool.

        Creates a recurring cron job from natural language input.
        Subcommands: list, pause, resume, remove.
        """
        import json
        import shlex
        from tools.cronjob_tools import cronjob as cronjob_tool
        from cron.jobs import get_job

        def _cron_api(**kwargs):
            return json.loads(cronjob_tool(**kwargs))

        tokens = shlex.split(cmd)

        # No args → show usage + list
        if len(tokens) == 1:
            print()
            print("+" + "-" * 68 + "+")
            print("|" + " " * 22 + "(^_^) /loop — Scheduled Prompts" + " " * 23 + "|")
            print("+" + "-" * 68 + "+")
            print()
            print("  Usage:")
            print('    /loop <schedule> <prompt>     e.g. /loop 5m "check deployment"')
            print('    /loop list                    Show all loop jobs')
            print('    /loop pause <job_id>          Pause a loop job')
            print('    /loop resume <job_id>         Resume a paused loop job')
            print('    /loop remove <job_id>         Delete a loop job')
            print()
            print("  Schedules:  5m, 30m, 2h, 1d, every 5m, every 2h, 0 9 * * *")
            print()

            result = _cron_api(action="list", include_disabled=True)
            jobs = result.get("jobs", []) if result.get("success") else []
            loop_jobs = [j for j in jobs if j.get("name", "").startswith("loop:")]
            if loop_jobs:
                print("  Loop Jobs:")
                print("  " + "-" * 63)
                for job in loop_jobs:
                    state = job.get("state", "?")
                    state_icon = "▶" if state == "active" else "⏸"
                    print(f"    {state_icon} {job['job_id'][:12]:<12} | {job['schedule']:<15} | {job.get('repeat', 'forever')}")
                    print(f"      {job.get('prompt_preview', '')}")
                    if job.get("next_run_at"):
                        print(f"      Next: {job['next_run_at']}")
                    print()
            else:
                print("  No loop jobs. Use '/loop <schedule> <prompt>' to create one.")
            print()
            return

        subcommand = tokens[1].lower()

        # /loop list
        if subcommand == "list":
            result = _cron_api(action="list", include_disabled=True)
            jobs = result.get("jobs", []) if result.get("success") else []
            loop_jobs = [j for j in jobs if j.get("name", "").startswith("loop:")]
            if not loop_jobs:
                print("(._.) No loop jobs found.")
                return
            print()
            print("Loop Jobs:")
            print("-" * 80)
            for job in loop_jobs:
                print(f"  ID: {job['job_id']}")
                print(f"  Name: {job['name']}")
                print(f"  State: {job.get('state', '?')}")
                print(f"  Schedule: {job['schedule']} ({job.get('repeat', '?')})")
                print(f"  Next run: {job.get('next_run_at', 'N/A')}")
                print(f"  Prompt: {job.get('prompt_preview', '')}")
                if job.get("last_run_at"):
                    print(f"  Last run: {job['last_run_at']} ({job.get('last_status', '?')})")
                print()
            return

        # /loop pause <job_id>
        if subcommand == "pause":
            if len(tokens) < 3:
                print("(._.) Usage: /loop pause <job_id>")
                return
            job_id = tokens[2]
            result = _cron_api(action="pause", job_id=job_id, reason="paused from /loop")
            if result.get("success"):
                print(f"(^_^)b Paused loop job: {result['job']['name']} ({job_id})")
            else:
                print(f"(x_x) Failed to pause: {result.get('error')}")
            return

        # /loop resume <job_id>
        if subcommand == "resume":
            if len(tokens) < 3:
                print("(._.) Usage: /loop resume <job_id>")
                return
            job_id = tokens[2]
            result = _cron_api(action="resume", job_id=job_id)
            if result.get("success"):
                print(f"(^_^)b Resumed loop job: {result['job']['name']} ({job_id})")
                print(f"  Next run: {result['job'].get('next_run_at')}")
            else:
                print(f"(x_x) Failed to resume: {result.get('error')}")
            return

        # /loop remove <job_id>
        if subcommand == "remove":
            if len(tokens) < 3:
                print("(._.) Usage: /loop remove <job_id>")
                return
            job_id = tokens[2]
            result = _cron_api(action="remove", job_id=job_id)
            if result.get("success"):
                print(f"(^_^)b Removed loop job: {result.get('removed_job', {}).get('name', job_id)}")
            else:
                print(f"(x_x) Failed to remove: {result.get('error')}")
            return

        # /loop <schedule> <prompt>  (create)
        # Try to parse: first token is schedule, rest is prompt
        schedule = tokens[1]
        prompt = " ".join(tokens[2:]) if len(tokens) > 2 else ""

        if not prompt:
            print("(._.) Usage: /loop <schedule> <prompt>")
            print('  Example: /loop 5m "check deployment status"')
            print('  Example: /loop every 30m "summarize news"')
            return

        # Warn if interval looks very short
        from cron.jobs import parse_duration
        try:
            minutes = parse_duration(schedule.lstrip("every "))
            if minutes < 1:
                print("⚠ Interval is very short (< 1 minute). The scheduler ticks every 60s, so this may not fire as expected.")
        except Exception:
            pass  # Not a simple duration — could be cron expr or "every X" form

        # Check if gateway is running (best-effort)
        try:
            from hermes_cli.cron import _is_gateway_running
            if not _is_gateway_running():
                print("⚠ No gateway is running. The job will be scheduled but will not execute until a gateway starts.")
        except Exception:
            pass

        name = f"loop: {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
        result = _cron_api(
            action="create",
            schedule=schedule if not schedule.startswith("every ") else schedule,
            prompt=prompt,
            name=name,
            deliver="origin",
        )
        if result.get("success"):
            print(f"(^_^)b Loop job created: {result['job_id']}")
            print(f"  Schedule: {result['schedule']}")
            print(f"  Next run: {result['next_run_at']}")
            print(f"  To stop: /loop remove {result['job_id']}")
        else:
            print(f"(x_x) Failed to create loop: {result.get('error')}")
```

**Why here:** The CLI dispatch chain in `process_command()` (lines 6696-7096) uses a giant if/elif. Adding the branch near `/cron` keeps scheduling commands grouped. The handler follows the same pattern as `_handle_cron_command()` (lines 6325-6580): import `cronjob_tool`, use `_cron_api()` wrapper for JSON parsing, manual string splitting for args.

---

### File 3: `gateway/run.py`

**Location A — dispatch branch:** Insert `if canonical == "loop":` at line 5698, immediately after `if canonical == "background":` (line 5697-5698) and before `if canonical == "steer":` (line 5700).

**What to add (line 5698):**
```python
        if canonical == "loop":
            return await self._handle_loop_command(event)
```

**Location B — handler method:** Add `async def _handle_loop_command()` as a new method on `GatewayRunner`. Place it immediately after `_handle_background_command()` (which ends around line 9390) and before `_handle_kanban_command()` (line 7444), OR at the end of the handler block near other tool commands.

**What to add:**
```python
    async def _handle_loop_command(self, event: MessageEvent) -> str:
        """Handle /loop in gateway — thin wrapper around cronjob tool.

        Creates a recurring cron job from a single slash command.
        Subcommands: list, pause, resume, remove.
        """
        import json
        import shlex
        from tools.cronjob_tools import cronjob as cronjob_tool
        from cron.jobs import get_job

        def _cron_api(**kwargs):
            return json.loads(cronjob_tool(**kwargs))

        text = (event.text or "").strip()
        # Strip leading "/loop" leaving args
        if text.startswith("/"):
            text = text.lstrip("/")
        if text.startswith("loop"):
            text = text[len("loop"):].lstrip()

        tokens = shlex.split(text)

        # No args → show usage + list
        if not tokens:
            result = _cron_api(action="list", include_disabled=True)
            jobs = result.get("jobs", []) if result.get("success") else []
            loop_jobs = [j for j in jobs if j.get("name", "").startswith("loop:")]
            lines = [
                "*/loop <schedule> <prompt>* — e.g. `/loop 5m check deployment`",
                "Subcommands: `list`, `pause <id>`, `resume <id>`, `remove <id>`",
            ]
            if loop_jobs:
                lines.append("")
                lines.append(f"*{len(loop_jobs)} loop job(s):*")
                for job in loop_jobs:
                    state_icon = "▶" if job.get("state") == "active" else "⏸"
                    lines.append(f"  {state_icon} `{job['job_id'][:12]}` | {job['schedule']} | {job.get('prompt_preview', '')}")
            else:
                lines.append("No loop jobs. Create one with `/loop <schedule> <prompt>`.")
            return "\n".join(lines)

        subcommand = tokens[0].lower()

        # list
        if subcommand == "list":
            result = _cron_api(action="list", include_disabled=True)
            jobs = result.get("jobs", []) if result.get("success") else []
            loop_jobs = [j for j in jobs if j.get("name", "").startswith("loop:")]
            if not loop_jobs:
                return "No loop jobs found."
            lines = ["*Loop Jobs:*"]
            for job in loop_jobs:
                lines.append(f"• `{job['job_id']}` | {job['schedule']} | {job.get('state', '?')} | {job.get('prompt_preview', '')}")
            return "\n".join(lines)

        # pause
        if subcommand == "pause":
            if len(tokens) < 2:
                return "Usage: `/loop pause <job_id>`"
            job_id = tokens[1]
            result = _cron_api(action="pause", job_id=job_id, reason="paused from /loop")
            if result.get("success"):
                return f"⏸ Paused loop job `{job_id}`"
            return f"⚠ Failed to pause: {result.get('error')}"

        # resume
        if subcommand == "resume":
            if len(tokens) < 2:
                return "Usage: `/loop resume <job_id>`"
            job_id = tokens[1]
            result = _cron_api(action="resume", job_id=job_id)
            if result.get("success"):
                return f"▶ Resumed loop job `{job_id}` — next run: {result['job'].get('next_run_at', 'N/A')}"
            return f"⚠ Failed to resume: {result.get('error')}"

        # remove
        if subcommand == "remove":
            if len(tokens) < 2:
                return "Usage: `/loop remove <job_id>`"
            job_id = tokens[1]
            result = _cron_api(action="remove", job_id=job_id)
            if result.get("success"):
                return f"🗑 Removed loop job `{job_id}`"
            return f"⚠ Failed to remove: {result.get('error')}"

        # Create: /loop <schedule> <prompt>
        schedule = tokens[0]
        prompt = " ".join(tokens[1:]) if len(tokens) > 1 else ""

        if not prompt:
            return "Usage: `/loop <schedule> <prompt>`\nExample: `/loop 5m check deployment`"

        name = f"loop: {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
        result = _cron_api(
            action="create",
            schedule=schedule,
            prompt=prompt,
            name=name,
            deliver="origin",
        )
        if result.get("success"):
            return (
                f"✅ Loop job created: `{result['job_id']}`\n"
                f"Schedule: {result['schedule']}\n"
                f"Next run: {result['next_run_at']}\n"
                f"To stop: `/loop remove {result['job_id']}`"
            )
        return f"⚠ Failed to create loop: {result.get('error')}"
```

**Why here:** The gateway dispatch chain (lines 5596-5720) uses async `if canonical == "..."` branches. Adding near `/background` keeps session/tool commands grouped. The gateway handler is async (returns `str | None`) and follows the same pattern as `_handle_kanban_command()` (lines 7444-7519).

---

## 3.3 Code Pattern

### Parsing `/loop <interval> <prompt>`

1. **Tokenize** with `shlex.split()` (both CLI and gateway use this).
2. **First token after `/loop`** is the schedule (e.g., `5m`, `every 30m`, `0 9 * * *`).
3. **Remaining tokens** joined by spaces become the prompt.
4. **Subcommands** (`list`, `pause`, `resume`, `remove`) are detected when the first token matches one of these keywords.

### Calling cron infrastructure

Both handlers call `cronjob_tool(action="create", ...)` directly:
```python
from tools.cronjob_tools import cronjob as cronjob_tool
cronjob_tool(action="create", schedule=schedule, prompt=prompt, name=f"loop: {prompt[:50]}", deliver="origin")
```

The `cronjob` tool internally calls `create_job()` from `cron/jobs.py:423`, which:
- Parses schedule via `parse_schedule()` (line 125)
- Generates a 12-char hex job ID (line 502)
- Computes `next_run_at` (line 560)
- Saves atomically to `~/.hermes/cron/jobs.json`

`deliver="origin"` ensures results come back to the chat where `/loop` was invoked. `origin` is auto-populated by `_origin_from_env()` (tools/cronjob_tools.py:71-88) which reads `HERMES_SESSION_PLATFORM`, `HERMES_SESSION_CHAT_ID`, and `HERMES_SESSION_THREAD_ID`.

### Confirmation to user

CLI: prints kawaii status lines with job ID, schedule, next run time.
Gateway: returns markdown-formatted confirmation string (gateway adapters render it).

### Subcommand support

| Subcommand | Action | API call |
|-----------|--------|----------|
| `list` | Show all loop jobs | `cronjob(action="list", include_disabled=True)` + filter by name prefix `loop:` |
| `pause <id>` | Pause a job | `cronjob(action="pause", job_id=id)` |
| `resume <id>` | Resume a paused job | `cronjob(action="resume", job_id=id)` |
| `remove <id>` | Delete a job | `cronjob(action="remove", job_id=id)` |

---

## 3.4 Test Strategy

### Unit tests

**New file:** `tests/hermes_cli/test_loop_command.py`

Tests to add:
1. **Test parsing** — verify `_handle_loop_command()` correctly parses `/loop 5m check deployment` into schedule=`5m`, prompt=`check deployment`.
2. **Test subcommands** — verify `list`, `pause`, `resume`, `remove` branch correctly.
3. **Test empty prompt** — verify error message when prompt is missing.
4. **Test cronjob tool call** — mock `cronjob_tool` and assert it receives correct params (action="create", schedule, prompt, name starting with "loop:", deliver="origin").
5. **Test gateway handler** — mock `_cron_api` and verify async handler returns correct markdown string.

**New file:** `tests/gateway/test_loop_command.py`

Tests to add:
1. **Test gateway dispatch** — verify `canonical == "loop"` hits `_handle_loop_command()`.
2. **Test gateway hook** — verify `command:loop` hook is emitted (it happens automatically because `is_gateway_known_command("loop")` returns True via `GATEWAY_KNOWN_COMMANDS`).

### Integration tests

**Location:** `tests/cron/test_scheduler.py` (existing cron integration tests) or new `tests/cron/test_loop_integration.py`.

Tests to add:
1. **Test end-to-end creation** — call `/loop 1m test prompt`, verify job appears in `jobs.json` with correct schedule, name prefix, and `deliver="origin"`.
2. **Test delivery** — create a loop job with `deliver="local"`, verify output is saved to `~/.hermes/cron/output/{job_id}/`.

**Do NOT add tests that require a running gateway daemon.** Mark any test needing a live gateway with `@pytest.mark.integration` and skip by default.

### Existing test patterns to follow

- `tests/hermes_cli/test_goals.py` — tests slash command handlers in CLI
- `tests/cron/test_scheduler.py` — tests cron execution pipeline
- `tests/gateway/test_goal_verdict_send.py` — tests gateway command dispatch

---

## 3.5 Edge Cases

| Edge Case | Behavior | Where handled |
|-----------|----------|-------------|
| Empty prompt (e.g., `/loop 5m`) | Print usage: "Usage: /loop <schedule> <prompt>" | Both CLI + gateway handlers |
| Interval too short (< 1 min) | Warn: "Interval is very short (< 1 minute)..." but still create the job | CLI handler only (gateway skips this warning) |
| No gateway running | Warn: "No gateway is running. The job will be scheduled but will not execute until a gateway starts." | CLI handler uses `_is_gateway_running()` from `hermes_cli.cron` |
| Duplicate /loop for same prompt | Create another job with a different job ID. No deduplication. | `create_job()` always generates new UUID (cron/jobs.py:502) |
| Removing non-existent loop | Print error from cronjob tool: "Job with ID 'xxx' not found." | Handled by `cronjob(action="remove")` returning `success: false` |
| Gateway delivery failure | Job succeeds (output saved to file) but `last_delivery_error` is set. User sees error in `/loop list` or cron list. | `cron/scheduler.py:480-654` handles delivery errors; not `/loop`'s concern |
| Invalid schedule string | `parse_schedule()` raises ValueError → caught by `cronjob_tool` → returns JSON with `success: false` and error message | `cron/jobs.py:125-211` |
| Prompt contains threat patterns | `cronjob_tool` scans against `_CRON_THREAT_PATTERNS` (tools/cronjob_tools.py:41-52) and blocks creation | `tools/cronjob_tools.py:305-308` |
| Very long prompt | Truncated to 50 chars in job name; full prompt stored in job record | Name truncation in handler; full prompt passed to `create_job()` |
| Special characters in prompt | `shlex.split()` handles quotes; `cronjob_tool` validates invisible chars | `shlex.split()` in handlers, `_scan_cron_prompt()` in tool |

---

## 3.6 Gotchas / Pitfalls

1. **CommandDef has no handler field.** Adding the `CommandDef` to `COMMAND_REGISTRY` only makes the command resolvable and exposes it in help/Telegram menus. You MUST add dispatch branches in BOTH `cli.py` AND `gateway/run.py`. The registry alone does nothing for execution.

2. **The cron scheduler runs in the gateway, not the CLI.** A `/loop` job created from CLI will only execute if a gateway is running. The CLI handler warns about this; the gateway handler does not (the gateway IS the scheduler).

3. **`create_job()` needs `_origin_from_env()` for origin delivery.** The `cronjob` tool calls this automatically (tools/cronjob_tools.py:334). Do NOT override it in the handler unless you want custom origin logic.

4. **Gateway handler is async; CLI handler is sync.** Do not copy-paste between them. The gateway returns strings; the CLI prints directly. Gateway uses `await` for nothing in this handler (tool calls are synchronous Python function calls).

5. **Do not add tests that require a running gateway daemon.** The test suite must pass in CI without a gateway. Use mocking for unit tests. Integration tests that need a real scheduler tick should be marked `@pytest.mark.integration` and skipped in normal test runs.

6. **`/cron` is `cli_only=True` and has no gateway handler.** `/loop` is NOT `cli_only` — it works in both CLI and gateway. This is intentional: `/loop` is a user-facing convenience command, not an admin tool.

7. **`_is_gateway_running()` is imported from `hermes_cli.cron`.** This function checks for a gateway PID file. It is best-effort; a gateway could start after the check. The warning is informational only.

8. **Job name prefix `loop:` is the filter key.** Both `list` subcommands filter jobs by `name.startswith("loop:")`. If a user creates a regular cron job with a name starting with `loop:` via `/cron add`, it will appear in `/loop list`. This is acceptable edge-case leakage.

9. **`GATEWAY_KNOWN_COMMANDS` is auto-derived.** Adding the `CommandDef` automatically makes `is_gateway_known_command("loop")` return True and emits `command:loop` hooks. No manual frozenset update needed.

10. **The `cronjob` tool returns JSON strings.** Both handlers wrap it with `json.loads()` (the `_cron_api()` helper pattern copied from `_handle_cron_command()`).

11. **No-agent mode is out of scope for V1.** `/loop` always creates `no_agent=False` jobs (the default). Script-only loops can be created via `/cron` with `--no-agent --script`.

12. **Skill attachment is out of scope for V1.** `/loop` does not support `--skill` flags. Users can attach skills via `/cron edit <job_id> --skill <name>` after creation.

---

## Verified Source References

| Claim | File:Line | Status |
|-------|-----------|--------|
| `CommandDef` dataclass | `hermes_cli/commands.py:45-57` | ✅ Confirmed |
| `COMMAND_REGISTRY` list | `hermes_cli/commands.py:64` | ✅ Confirmed |
| `_build_command_lookup()` | `hermes_cli/commands.py:214-224` | ✅ Confirmed |
| `resolve_command()` | `hermes_cli/commands.py:227-232` | ✅ Confirmed |
| `GATEWAY_KNOWN_COMMANDS` frozenset | `hermes_cli/commands.py:287-292` | ✅ Confirmed |
| `is_gateway_known_command()` | `hermes_cli/commands.py:295-313` | ✅ Confirmed |
| `ACTIVE_SESSION_BYPASS_COMMANDS` | `hermes_cli/commands.py:319-330` | ✅ Confirmed (subset, truncated) |
| CLI `process_command()` entry | `cli.py:6675-6694` | ✅ Confirmed |
| CLI `/cron` branch | `cli.py:6857-6858` | ✅ Confirmed |
| CLI `/kanban` branch | `cli.py:6861-6862` | ✅ Confirmed |
| `_handle_cron_command()` | `cli.py:6325-6580` | ✅ Confirmed |
| Gateway command resolution | `gateway/run.py:5506-5517` | ✅ Confirmed |
| Gateway hook emission | `gateway/run.py:5548-5594` | ✅ Confirmed |
| Gateway `/kanban` branch | `gateway/run.py:5646-5647` | ✅ Confirmed |
| Gateway `/background` branch | `gateway/run.py:5697-5698` | ✅ Confirmed |
| `_handle_kanban_command()` async | `gateway/run.py:7444-7519` | ✅ Confirmed |
| `cronjob()` tool signature | `tools/cronjob_tools.py:257-275` | ✅ Confirmed |
| `cronjob` create path | `tools/cronjob_tools.py:285-360` | ✅ Confirmed |
| `_origin_from_env()` | `tools/cronjob_tools.py:71-88` | ✅ Confirmed |
| `create_job()` | `cron/jobs.py:423-577` | ✅ Confirmed |
| `parse_duration()` | `cron/jobs.py:104-122` | ✅ Confirmed |
| `parse_schedule()` | `cron/jobs.py:125-211` | ✅ Confirmed |
| `_scan_cron_prompt()` | `tools/cronjob_tools.py:60-68` | ✅ Confirmed |
| `_CRON_THREAT_PATTERNS` | `tools/cronjob_tools.py:41-52` | ✅ Confirmed |
