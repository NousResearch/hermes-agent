---
name: cron-output-contract
description: Single output contract for all Hermes cron jobs. Summary-only Discord message with HTML attachment. No expandable blocks. [SILENT] for zero-signal runs.
version: 2.6.0
metadata:
  hermes:
    tags: [cron, discord, formatting, output-contract, notification-audit]
    related_skills: [content-pipeline, hermes-cron-operations]
adoption_status: permanent
---

# Cron Output Contract

## UK Date Format Standard

All dates and timestamps in cron output, scripts, reports, and **any script that produces stdout delivered to Discord** MUST use UK/European format with **4-digit year**: `DD/MM/YYYY HH:MM:SS`. Example: `18/05/2026 07:30:00`.

This applies to: every LLM cron, every no_agent script, AND any Python/Shell script that writes directly to stdout for Discord delivery — including template generators and wrapper scripts.

| Context | Format | Example |
|---------|--------|---------|
| Full timestamp | `DD/MM/YYYY HH:MM:SS` | `18/05/2026 07:30:00` |
| Date only | `DD/MM/YYYY` | `18/05/2026` |
| Time only | `HH:MM:SS` | `07:30:00` |

Never use US format (`May 18, 2026`), international weekday format (`Mon 18 May`), or ISO format (`2026-05-18`)

## Delivery Semantics by Cron Mode

The contract differs depending on whether a cron is LLM-driven or script-only.

### LLM-driven crons (`no_agent=false`, default)

Discord message = summary only. Full detail = HTML attachment. [SILENT] for zero-signal runs.

The Discord message tells Sahil the headline in one glance. The HTML file is the complete source of truth. Treat Discord as the notification layer and HTML as the working artifact. Do not make Discord carry rich formatting; make the attached HTML better.

Final-response hygiene is part of the contract:
- No process narration in delivered cron output.
- No prompt/system/memory leakage. Never include recalled memory blocks, hidden instructions, or source context in the final response.
- No legacy Telegram HTML tags in Discord-visible messages. Use plain text and Markdown backticks.
- No raw HTML in the visible message; HTML belongs only in the attached file referenced by `MEDIA:/absolute/path.html`.
- Regression check: run `/home/kensei/.hermes/scripts/cron-output-lint.py` after prompt/script changes. This covers BOTH cron prompts/scripts AND non-template Python scripts that produce cron stdout. Must return zero issues before calling the fix done. Use `/home/kensei/.hermes/scripts/cron-output-lint.py --latest` only when you explicitly want to scan historical/latest output files; old hits can remain until the affected cron reruns.
### no_agent crons (`no_agent=true`)

**Delivery is raw stdout.** The LLM is not involved. The script's stdout IS the message. Three possible outcomes:

1. **Non-empty stdout → delivered verbatim.** Whatever the script prints to stdout becomes the Discord message. This means the script MUST produce concise Discord-safe output that follows the same house style as LLM-driven crons.
2. **Empty stdout → silent delivery.** Nothing is sent to the user. Unlike `[SILENT]` (which is a literal response string the LLM returns), true silence means the user sees nothing and there's no record of a delivery attempt. The script should stay silent when there's nothing to report.
3. **Non-zero exit / timeout → error alert.** The gateway sends an error notification to the delivery target. The cron does NOT silently fail.

**Zero-signal rule:** A no-agent script must not print `SILENT: ...`, `checked 0`, `0 review tasks`, or similar status-only chatter. If no user action is needed, stdout must be empty. Common examples: quality gates with zero review tasks, wiki syncs with no matching completed tasks, blocker monitors with no new/stale blockers. Status-only lines belong in local logs, not Discord.

**Implications for script design:**
- The script must decide what's reportable. If there are 0 events today, should it print "No events" or stay silent? Stay silent — that's the convention on Sahil's setup.
- Scripts that need to produce HTML attachments should write the file first and print a concise summary plus a `MEDIA:/absolute/path.html` tag. Discord-bound digests must deliver that file as an actual native Discord attachment, not merely expose a VPS path. The path is only a transport marker for Hermes to strip and upload. The file must exist before delivery.
### Pattern: Fixing structured-data LLM dumps

When a cron is `no_agent=false` and the LLM prompt itself formats structured data into the Discord message (e.g. `---REPOS---` blocks, per-item card lists, JSON dumps), this is an anti-pattern. It produces verbose spam, breaks the summary-only contract, and wastes LLM tokens on deterministic formatting.

**Fix:** Convert the job to `no_agent=true` and move all formatting into the script:

1. **Script produces stdout** → Short Discord-safe summary: headline, stats, top picks, `MEDIA:/path/to/report.html`.
2. **Script writes a dark HTML file** → Full detail, date-stamped, to an allowlisted directory.
3. **Script writes a structured text/JSON file** → Machine-parseable data (e.g. `---REPOS---` block) in the same directory for downstream crons to read instead of receiving raw context injection.
4. **Downstream crons updated** → Remove dependency on raw context dumps. Add a prompt step to read the structured file via the `terminal` or `file` toolset.
5. **Zero tasks created** → Data producers (radar, digest collectors) should not create kanban tasks. Output-only.

Real example: `github-radar-merged` on 2026-05-27. The cron prompt was dumping `---REPOS---` + full JSON into Discord. Converted to `no_agent=true`, script now emits a 5-line summary + MEDIA tag, writes HTML + `github-radar-repos.txt`, and the librarian/review crons read the text file.

**Full step-by-step conversion recipe:** `references/no-agent-conversion-recipe.md`

### When to convert to no_agent

- Script already produces complete stdout → candidate for no_agent after rewriting output to Discord-safe summary/attachment format
- Script produces JSON → needs a format wrapper (~50-100 lines Python)
- Script needs MCP tools → CANNOT convert (MCP is LLM-loop only)

**Pattern for conversion / migration review:**
1. Inventory active jobs first: job name, schedule, `deliver`, `no_agent`, script path, and prompt/output contract. Never assume old Telegram topics have been fully migrated just because one gateway message reaches Discord.
2. For “all notifications” or “all daily cron outputs” requests, include both root and profile cron files. A root-only `cronjob list` is incomplete: profile-scoped jobs may still emit scheduled output or sit stale with old routing.
3. If the script already does all the work (for example `system_report_daily.sh`), check its stdout. If it emits Telegram HTML/tags or topic-specific wording, rewrite it to concise Discord-safe plain text plus optional `MEDIA:/absolute/path.html` attachment.
4. If the script only fetches data (for example `calendar_brief_combined.py`), add a format wrapper that reads its output and prints the Discord summary plus HTML attachment path.
5. Create a shell wrapper that runs fetch → format sequentially, and make the formatter create the HTML file before printing the `MEDIA:` tag.
6. Update the cron: `--no-agent true --script path/to/wrapper.sh` when the job is fully script-rendered. The model field is ignored for no_agent jobs — set it to null.
7. Test with `bash wrapper.sh`, then run the cron manually and verify the actual Discord channel receives both the short summary and the file attachment.


## Weekly Build Ideas Digest

For scheduled "build ideas collection" runs, follow `references/weekly-build-ideas-digest.md`.

Minimum source sweep before returning `[SILENT]`:
- Recent session history for idea/build/feature/pilot/product signals
- GitHub issues/PRs/commits updated in the last 7 days
- Kanban keyword matching to determine whether each idea already has a task

Output should be a compact structured digest with: idea, context, feasibility, and kanban status, followed by a short priority call. Do not create tasks unless the cron prompt explicitly asks for creation.

## Research Digest — updated 24/05/26: summary + HTML attachment

The old `research_digest.py` (1,630-line Python collector) is **retired**. Replaced by a self-contained LLM cron prompt that generates a short Discord summary + full HTML attachment.

**Key specs:**
- **Target: 10-15 items total.** Per-section split: News 4-6, Tools 4-6, Signal 2-3
- Sources: TLDR AI (tldr.tech/ai), The Rundown AI, official blogs (Google AI, OpenAI, Anthropic, Nous, Ollama, HF), X/Twitter via xurl (31 accounts), web_search, GitHub trending
- **48h hard cutoff** on freshness — items older than 2 days excluded
- **Freshness labels** required on every item: `today`, `yesterday`, `2d ago`
- [SILENT] if fewer than 6 quality items pass freshness gate
- Delivers to `discord:#research-digest` at 09:00 UK daily on kimi-k2.6
- Discord message = 2-3 line summary (headline + count + top picks) + `MEDIA:` tag pointing to HTML file
- HTML file = full detail in dark-mode HTML at `~/.hermes/runbooks/research-digest/research-brief-YYYYMMDD-HHMM.html`

**Output format:**
```
📡 Daily Research Brief — DD/MM/YYYY
11 items · 4 news · 5 tools · 2 signal

MEDIA:/home/kensei/.hermes/runbooks/research-digest/research-brief-YYYYMMDD-HHMM.html
```

**HTML file structure:**
- Dark mode (`#11100f` background, `#fbbf24` accent, `#f5f5f4` text)
- Sections: News (red header), Tools (blue header), Signal (grey header)
- Each item: title, 1-line summary, link, source credit, age label
- Source credit footer: `Sources: TLDR AI, The Rundown AI, web_search, xurl, official blogs`

See the `research-digest` skill (devops category) for the full reference, source list, and cron implementation template.
See the `research-digest` skill (devops category) for the full reference, source list, and cron implementation template.

**The old 3-lane architecture (AI News / Tool News / Research), RSS sources (TechCrunch, Reddit, HN), Tavily keyword scoring, and additive heuristic scoring are all dead. Do not reference them.**

## Content Digest — summary + HTML, NOT per-card

Content engine (`content-engine-v2-daily` cron) generates 30 drafts (6 per 5 brands). Each draft has content_type (text, text+image, voice, video).

**Delivery: one summary message + one HTML file attached as document attachment. User replies with approval codes: `A:all`, `A:1,2,3,6-28`, `R:4,5,11,29`, or write a note like "all approved except 1,4,5,11,29. Number 4 needs different angle."

**Approval workflow:** Summary message shows brand counts + content_type counts + HTML attached. User approves/rejects by reply.

**Fallback templates** when LLM generation fails.

## Critical pitfall: MEDIA delivery can fail two ways — not just missing directory

MEDIA delivery to Discord (or any platform) can fail silently in two distinct ways. Both produce the same symptom for Sahil: the file path text appears literally in the chat message instead of a native attachment.

### Failure A: File never written (directory doesn't exist)

On 2026-05-13, the quality gate cron wrote `MEDIA:/home/kensei/.hermes/reports/quality-gate.html` in its output, but the directory `/home/kensei/.hermes/reports/` did not exist. The model cannot create directories — it can only write files if the parent path exists. Result: the HTML file was never created, and the gateway's `send_message_tool.py` silently skipped the MEDIA: tag with the log message "Media file not found, skipping: /path/..." (line 851). The user saw a text path reference with no attached file.

**Rule for any cron that produces HTML attachments:**
1. **The directory MUST exist before the LLM writes to it.** If the cron prompt references `/path/to/reports/report.html`, that directory must be created in advance (by setup, by the skill, or by a pre-flight script).
2. **The LLM must write the file FIRST, then output the MEDIA: tag.** Never output `MEDIA:/path/file.html` before confirming the file exists. Include in the prompt: "Write the HTML file to <path> BEFORE producing your summary. After writing, verify the file exists."
3. **Use a date-stamped filename** to avoid collisions across runs, e.g. `quality-gate-2026-05-13.html`.
4. **Verify after writing:** include a step like `test -f /path/to/file.html && echo "HTML written OK" || echo "FAIL: HTML not written"` in the output generation sequence.

### Failure B: Path exists but not on the `HERMES_MEDIA_ALLOW_DIRS` allowlist

On 2026-05-23, cron jobs (github-radar, mailbox-cleaner, MrHermagi) wrote HTML report files to `~/.hermes/runbooks/` and emitted `MEDIA:/home/kensei/.hermes/runbooks/...` in their output. The file existed on disk. But `validate_media_delivery_path()` in `gateway/platforms/base.py` checked the path against the `HERMES_MEDIA_ALLOW_DIRS` allowlist — and rejected it. The default allowlist only includes Hermes-managed cache directories (`image_cache`, `audio_cache`, `video_cache`, `document_cache`, `browser_screenshots`). `runbooks/` is not on the list. Result: the gateway silently drops the MEDIA tag, and the path text appears literally in Discord.

**Diagnostic:** the file EXISTS at the MEDIA: path on disk (`ls -la /path/from/MEDIA-line` returns a real file), but Discord shows the path text literally. The gateway.log will NOT log an error — the path just fails validation and is dropped.

**Fix: add the report output directories to `HERMES_MEDIA_ALLOW_DIRS` in `~/.hermes/.env`:**

```bash
# Add to ~/.hermes/.env (no quotes, no 'export' keyword, literal absolute paths)
HERMES_MEDIA_ALLOW_DIRS=/home/kensei/.hermes/runbooks:/home/kensei/.hermes/reports
```

Then restart the gateway:

```bash
sudo systemctl kill hermes-gateway
sleep 3
sudo systemctl start hermes-gateway
```

**Verify:** The env var loads inside Python via `load_hermes_dotenv()`, NOT into the process `/proc/<pid>/environ` block. So `strings /proc/<pid>/environ | grep MEDIA` returns nothing even when the var is working. Instead, send a test MEDIA message:

```bash
echo '<html><body><h1>Test</h1></body></html>' > /home/kensei/.hermes/media/test-file.html
# Then send a MEDIA message via the agent or cron
```

**Gateway restart pitfall:** `systemctl restart` may not kill all old gateway PIDs. Old processes from previous boot sessions survive when systemd's kill semantics don't reach them (happened 2026-05-23 — old PIDs from 19:02 survived through multiple `restart` and `stop/start` cycles). If the fix seems not to work, run `ps aux | grep gateway`, note start times. Manually `kill <pid>` any that started before the env edit, then `systemctl start hermes-gateway` for a clean slate.

### Failure B deep dive: why the env var fix was not enough

On 23/05/26, `HERMES_MEDIA_ALLOW_DIRS` was added to `~/.hermes/.env`. Cron deliveries started working (the gateway process loads `.env` at startup via `load_hermes_dotenv()`). But `send_message` tool calls from CLI sessions **still showed file paths as literal text** — the MEDIA tag was silently rejected even though the file existed and the env var was set.

**Root cause:** The `send_message` tool and `execute_code` tool run in **isolated subprocess contexts** that do NOT load `.env`. They start with a clean environment. My shell's `export HERMES_MEDIA_ALLOW_DIRS=...` is inherited by child processes I launch via `terminal()`, but NOT by:
- The `send_message` tool (which spawns its own Python subprocess)
- The `execute_code` tool (same pattern)

These tool processes only see env vars from the parent Hermes agent process — which may or may not have loaded `.env` depending on how it was launched.

**The actual fix:** Adding `runbooks` and `reports` to the **hardcoded** `MEDIA_DELIVERY_SAFE_ROOTS` tuple in `gateway/platforms/base.py`:

```python
MEDIA_DELIVERY_SAFE_ROOTS = (
    # ... existing cache roots ...
    _HERMES_HOME / "runbooks",
    _HERMES_HOME / "reports",
)
```

This is the durable fix because `validate_media_delivery_path()` checks `MEDIA_DELIVERY_SAFE_ROOTS` first and the env var second. Hardcoded roots survive any process context.

**Lesson:** The env var is a supplement for custom paths, but any path that cron jobs consistently output to should be added to `MEDIA_DELIVERY_SAFE_ROOTS` in `base.py`. Never rely on an env-var-only approach for paths used in cron outputs — a `send_message` tool call or `execute_code` validation will not see it.

**Verification in subprocess context:**
```python
import sys
sys.path.insert(0, '/home/kensei/.hermes/hermes-agent')
from gateway.platforms.base import validate_media_delivery_path
# Should return the resolved path (not None) even without HERMES_MEDIA_ALLOW_DIRS set
result = validate_media_delivery_path('/home/kensei/.hermes/runbooks/test.html')
```

### Failure C: File extension not in extract_media() regex

On 2026-05-27, all crons producing HTML attachments (`kanban_daily_digest_noagent`, `calendar_brief_format`, `system_report`, `content_engine_daily`, `research_digest`, `mailbox-cleaner`, `MrHermagi`) were emitting `MEDIA:/path/to/file.html`, but the `MEDIA:` tag appeared as literal text on Discord. The file existed, the directory was allowlisted, the gateway was running — but the tag was never parsed and stripped.

**Root cause:** `extract_media()` in `gateway/platforms/base.py` line 2418 uses a regex to identify valid `MEDIA:` tag file extensions. The regex only matched: `png|jpe?g|gif|webp|mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|txt|csv|apk|ipa`. HTML extensions (`.html`, `.htm`) were not in the list, so `extract_media()` never recognized the tag, and it was left as raw text in the message.

Note: `_LOCAL_MEDIA_EXTS` (used by `extract_local_files()`) already included `.html` and `.htm` — but that's a different code path for bare path detection, not for `MEDIA:` tag parsing. The two must both include the extension for it to work end-to-end.

**Fix:** Add `html?` to the regex alternation group:

```python
# Before (line 2418):
(?:png|jpe?g|gif|webp|mp4|...|txt|csv|apk|ipa)

# After:
(?:png|jpe?g|gif|webp|mp4|...|txt|csv|html?|apk|ipa)
```

Patch both copies (the git checkout and the `.hermes/hermes-agent/` deployment copy), then restart all gateway services.

**Verification:** Send a test message with `MEDIA:/path/to/test.html` via `send_message` and confirm the file appears as a native Discord attachment (not as text in the message body).

**Diagnostic shortcut:** If `MEDIA:/path.html` appears as raw text but `MEDIA:/path.png` works for the same path prefix, the extension is missing from the regex. Check `base.py` line 2418 directly — don't waste time on allowlists or directory permissions.

### All three failures checklist

When debugging "MEDIA path shown as text instead of attached file":

| Check | Command | Failure mode |
|---|---|---|
| Does the file exist? | `ls -la /path/from/MEDIA-line` | Failure A |
| Is the parent directory writable? | Check parent dir permissions | Failure A |
| Is the path in HERMES_MEDIA_ALLOW_DIRS? | `grep HERMES_MEDIA_ALLOW_DIRS ~/.hermes/.env` | Failure B |
| Did the gateway restart with the new env? | Check `ps aux` start times vs edit time | Failure B |
| File readable by gateway user? | `stat -c '%a %U %G' /path/to/file.html` | Both — must be readable by `kensei` |
| Is the file extension in the extract_media() regex? | `grep 'html?' gateway/platforms/base.py` | Failure C |
| Does MEDIA work for .png but not .html on the same path prefix? | Quick test with two file types | Failure C |

## TOON Optimisation for LLM Crons

Starting 2026-05-14, every LLM cron that consumes `kanban --json` output should pipe it through TOON encoding before reasoning over it. This saves 27-69% on structured-data tokens.

**The pattern — pipe kanban JSON through `toon_utils.toon_encode`:**

```bash
hermes kanban list --status triage --json | python3 -c "
import json,sys
sys.path.insert(0,'/home/kensei/repos/KenseiAgent/scripts')
from toon_utils import toon_encode
print(toon_encode(json.load(sys.stdin)))
"
```

Add `sys.path.insert(0,'/home/kensei/repos/KenseiAgent/scripts')` before importing because the scripts directory is not on the default Python path for cron-runner shells.

**Target crons (wired 2026-05-14, TOON removed from triage-processor 2026-05-15):**
- `kanban-daily-digest` — pipes done/running/blocked/triage lists through TOON
- `kensei-heartbeat-audit` — pipes kanban list/show output through TOON
- ~~`kensei-triage-processor` — pipes triage list through TOON~~ **REMOVED:** terminal security scanner blocks pipe-to-interpreter on cron jobs. Triage-processor now fetches `hermes kanban list --json` directly without TOON pipeline.

**What NOT to TOON:**
- Script-mode crons that collect and render data (research digest, content engine, mailbox brief) don't consume JSON in LLM context — they produce output files and Discord-safe summary text. TOON doesn't apply.
- Data written to disk for persistence (JSON files in runbooks) — TOON is a prompt encoding, not a storage format.
- Small kanban responses (< 200 chars) — the `maybe_toon()` threshold skips these.

**The utility module:**

Located at `/home/kensei/repos/KenseiAgent/scripts/toon_utils.py`. Four functions:

| Function | Purpose |
|----------|---------|
| `toon_encode(data)` | Always encode as TOON, fallback to compact JSON |
| `maybe_toon(data, threshold=200)` | Only TOON if data is big enough |
| `toon_blob(data, label="toon")` | Wrap in codeblock for LLM prompts |
| `size_report(data)` | Debug output showing TOON vs JSON savings |

**Implementation cost:** ~10 lines of Python in the utility module, ~3 lines per cron prompt for the pipe command. Near-zero risk (TOON is lossless round-trip, MIT license). Already installed at `/home/kensei/.local/lib/python3.12/site-packages/toon_format` (v0.9.0b1, official library).

## Pitfalls

- Scheduler advance-before-run trap: `cron.scheduler.tick()` advances `next_run_at` before execution. If a tick crashes or times out after that, skipped reports can look healthy because `next_run_at` is already tomorrow and `last_status` is still the old `ok`. Audits must compare `last_run_at` against the expected schedule.
- Runtime path trap on Sahil's VPS: gateway may run from `/home/kensei/.hermes/hermes-agent/` while CLI/cron tooling may use the pipx site-packages copy. For scheduler/gateway patches, inspect `ps -ef`, patch the active tree and pipx copy when needed, then restart the gateway.
- Discord MEDIA UX: the scheduler combines text with the first non-audio MEDIA file as a single Discord message (caption + attachment). Audio files send as separate follow-up messages. See `references/discord-multiple-attachments.md` for the fix and behavior table.
- Multiple MEDIA files in one cron delivery: text combines with the first non-audio file, remaining files send separately. If you want text + all files in one message, redesign your cron output to use a single MEDIA tag (e.g. zip or self-contained HTML with audio embedded as a link).
- LLM generation via Hermes gateway doesn't have a single-turn endpoint. Fallback templates are the working path.
- Telegram Bot API requires TELEGRAM_BOT_TOKEN in the Python process env. If cron runner spawns in isolated env, `os.getenv("TELEGRAM_BOT_TOKEN")` returns None and `send_document()` silently fails. Always test with a `send_message()` probe before trusting document delivery.
- **Status-only delta crons create false signal.** A cron that tracks "historical drift" (e.g., WFA delta reporting "0 live, 269 historical") and fires every 30m produces 48 identical messages/day. The growing historical count is NOT new activity — it's dead tasks accumulating keys. The cron output contract says zero-signal runs must be [SILENT] or empty stdout. A delta cron that only tracks historical drift is permanently in violation. Fix: separate `live` (actionable) from `historical` (audit-only) in state schema. Fire only when `live` changes.
- **Daily digest vs continuous alerting.** When Sahil says "daily output instead" or "I don't need this every 30 minutes", the cron is either (a) tracking the wrong thing (historical instead of live), or (b) running on the wrong cadence. Rule: live findings → alert immediately when they appear. Historical summaries → weekly audit. Nothing → daily digest. If the cron's purpose is "surveillance", reframe it as "detection" (only on change) or absorb it into a daily consolidated digest.
- Old RSS watcher state backed up to `~/.hermes/runbooks/rss-watcher-backup-20260512/`.
- Stale `__pycache__` bytecode trap: after patching `base.py` (or any gateway Python module), Python may load stale `.pyc` files from `__pycache__/` instead of the patched source. Always delete `__pycache__/` directories in both repo and `.hermes/hermes-agent/` copies before restarting gateways. Symptom: patch looks correct in source, but runtime behaviour doesn't change. No errors logged — stale bytecode silently wins.

## Discord Connection Diagnostics

When cron output is NOT reaching Discord despite `deliver` being set and the gateway showing `active (running)`:

### Step 1 — Check gateway log for Discord transport errors

```bash
grep "Failed to send Discord\|Server disconnected\|Broken pipe" \
  /home/kensei/.hermes/logs/gateway.log | tail -10
```

Common failure signatures:
- `Server disconnected` — Discord WebSocket/HTTP connection severed mid-delivery. The gateway may reconnect moments later but the cron message is lost because cron scheduler already handed it to the send call and moved on.
- `[Errno 32] Broken pipe` — TCP connection reset. Often caused by rate-limiting or keepalive expiry on long-running gateway processes.
- Silent skip (no error logged, message never arrives) — cron message fell into a gap between disconnect and reconnect.

### Step 2 — Check connection stability

```bash
grep "Disconnected\|Connected as" /home/kensei/.hermes/logs/gateway.log | tail -20
```

Look for rapid connect/disconnect cycles (flapping). If the bot is disconnecting and reconnecting every few minutes, messages in flight during a disconnect window are silently dropped. The cron scheduler does NOT retry delivery — it writes the output to disk and moves on.

### Step 3 — Verify MEDIA file delivery

For no-agent or LLM-driven crons that output `MEDIA:/absolute/path.html`, three things must be true:
1. The file must physically exist at that path at delivery time (confirmed by `ls -la`).
2. The path must use the correct date-stamped directory structure — the MEDIA: tag is literal text piped to the gateway's send handler, which strips it and uploads the file. If the file path is wrong, the handler silently skips it and the MEDIA: tag text reaches Discord as visible text.
3. The path must be in an allowlisted directory. Two levels of allowlist exist:
   - **Hardcoded roots** in `MEDIA_DELIVERY_SAFE_ROOTS` (gateway/platforms/base.py) — these work in ALL contexts: gateway cron delivery, CLI `send_message` tool, and `execute_code` validation tool. If you want to add a path that works everywhere, add it here.
   - **Env var roots** via `HERMES_MEDIA_ALLOW_DIRS` in `~/.hermes/.env` — these work in the gateway process (which loads `.env` at startup) but NOT in isolated tool subprocesses like `send_message` or `execute_code`. Use this for experimental/custom paths, but if the path is used regularly, add it to the hardcoded roots instead.

**Priority:** When diagnosing "path exists but rejected", check the hardcoded roots first. If the path is under `runbooks/` or `reports/`, it should be in `MEDIA_DELIVERY_SAFE_ROOTS` — not just the env var. Without both, a `send_message` tool test (which doesn't load .env) will reject the path silently and show the RAW path text in Discord, even though the cron delivery itself works.

Check a known-working MEDIA delivery by inspecting a recent cron output file for the MEDIA line, then verifying the referenced file AND the allowlist:

```bash
cat /home/kensei/.hermes/cron/output/<job-id>/<recent-run>.md | grep MEDIA
ls -la /path/from/MEDIA-line
grep HERMES_MEDIA_ALLOW_DIRS /home/kensei/.hermes/.env
```

### Step 4 — Confirm channel routing

The cron's `deliver` field in jobs.json shows the target. For `discord:#channel-name`, the hash-name must map to an actual Discord channel where the bot has `Send Messages` permission. If the channel doesn't exist or the bot lacks permissions, the message is silently rejected. Cross-reference with `DISCORD_HOME_CHANNEL` in `.env`:

```bash
grep DISCORD_HOME_CHANNEL /home/kensei/.hermes/.env
```

### Step 5 — Check cron output was actually produced

```bash
ls -lt /home/kensei/.hermes/cron/output/<job-id>/ | head -3
```

If the cron is LLM-driven (`no_agent=false`) and returned `[SILENT]`, no message is sent by design. If it's no_agent and the script produced empty stdout, same result. This is correct behaviour — not a delivery failure.

### When to escalate

- Discord connection flapping >5 times in 24h → check Discord bot token health and rate-limit status (Discord API allows ~30 messages/minute for bots).
- MEDIA file exists but delivery fails → first check `HERMES_MEDIA_ALLOW_DIRS` in `.env` (Failure B scenario — path not allowlisted), then check the file's parent directory permissions (Hermes gateway runs as the `kensei` user; the file must be world-readable or owned by `kensei`).
- All checks pass yet output never appears → restart the gateway (`sudo systemctl restart hermes-gateway`) to force a clean Discord session. If old PIDs survive, kill them manually first (see Failure B restart pitfall).

### Profile cron visibility trap

After a Discord migration or notification audit, root cron health is not the whole truth. Profile-scoped cron files under `/home/kensei/.hermes/profiles/*/cron/jobs.json` can contain enabled jobs with Discord delivery targets but `last_run_at: null` and no output files. Treat these as scheduler-scope problems, not Discord delivery problems: the active root gateway may not be running those profile schedulers at all.

Audit pattern:
1. Inventory root jobs and every profile `cron/jobs.json`.
2. For each enabled profile job, compare `last_run_at` and the profile-local `cron/output/<job_id>/` directory.
3. If the job has a Discord target but has never produced output, do not chase Discord channel permissions first. Decide whether to migrate the job into root cron, run the relevant profile scheduler, or disable stale profile jobs.

### HTML attachment prompt hardening

For Discord-bound digest jobs, wording like “if you generate an HTML report” is too weak. It lets LLM crons emit only a visible message, attach inconsistent files, or fall back to light-mode HTML. Use mandatory wording:

- “ALWAYS generate a dark-mode HTML report file and attach it with `MEDIA:/absolute/path`.”
- “Create the parent directory before writing.”
- “Use `color-scheme: dark`; body `#11100f`; cards `#1c1a18`/`#2c2a28`; text `#f5f5f4`; muted `#a8a29e`; accent `#fbbf24`; border `#34302c`.”
- “Do not use white/light backgrounds (`#fff`, `#fafafa`, `#f8f9fa`) or black text (`#000`, `#111`).”
- “Visible Discord message = short summary plus `MEDIA:` tag, not the full report.”

Run `cron-output-lint.py` after changes. If a script still emits Telegram HTML tags (`<b>`, `<code>`, `<blockquote expandable>`) for a Discord-bound job, rewrite its stdout to Discord-safe Markdown/plain text unless the platform renderer explicitly expects Telegram HTML.

## Output Consolidation Workflow (revive-first, join-second)

When Sahil asks you to fix gaps in output delivery, follow this sequence. Do NOT skip to "join X into Y" — the user explicitly rejected that approach.

### Step 1 — Inventory (what is this session about?)
Run the Full Output Inventory procedure below. You need a complete map of active crons, dead scripts, orphan output dirs, and platform connectivity BEFORE you make any decisions.

### Step 2 — Evaluate each dead/unwired script individually
For every script that has no active cron, ask:
- **What does this script do?** (1 line — read the code, don't guess from the filename)
- **What domain does it belong to?** (system health, kanban monitoring, content pipeline, calendar, credential management, quality improvement, drift detection)
- **What data does it need and produce?** (stdout text, JSON, HTML files, MEDIA: tags)
- **Does it overlap with an existing active cron?** If yes — same domain AND similar cadence — it's a merge candidate. If not — same domain but different cadence — it's a separate cron.

### Step 3 — Identify genuine overlaps only
Two scripts should be merged ONLY if:
- **Same domain** (system health + memory watchdog = same domain. Kanban digest + blocker push = different domain — keep separate)
- **Similar cadence** (every 2h + every 4h = merge. Every 2h + daily morning = keep separate)
- **Same output format** (both produce Discord text + HTML, or both alert via kanban tasks)

If a dead script shares a domain with an existing active cron, absorb it as a probe into the existing cron's script rather than creating a new cron (see heartbeat fusion pattern below).

### Step 4 — Join only directly overlapping items
- If two crons check the SAME system metric at DIFFERENT frequencies → absorb the less frequent one into the more frequent one
- If two crons check the SAME metric at SIMILAR frequency → merge into one cron
- If two crons are DIFFERENT domains but share a time slot → keep separate. They deliver to different channels.
- A kanban digest and a blocker scanner are DIFFERENT concerns. One summarises what happened. One pushes for action. Do not merge them.

### Step 5 — Register revived crons with correct no_agent flag
- Scripts that produce complete stdout output (Discord-safe text + optional MEDIA: tags) should use `no_agent=true` — no LLM cost, the script IS the output.
- Scripts that need LLM reasoning (summarising, classifying, prioritising) should use `no_agent=false` (LLM mode) with appropriate skills and toolsets.

### Script path convention (relative only)

When creating a no_agent cron, the `script` field MUST be just the filename — not an absolute path, not a home-relative path. The cron scheduler resolves scripts under `~/.hermes/scripts/` automatically.

```bash
# ✅ Correct — just the filename
script: system_report_daily.sh

# ❌ Wrong — absolute or home-relative
script: /home/kensei/.hermes/scripts/system_report_daily.sh
script: ~/.hermes/scripts/system_report_daily.sh
```

If you pass an absolute path, the `cronjob` tool rejects it with `Script path must be relative to ~/.hermes/scripts/.` Place all scripts there and use bare filenames.

## Discord-Safe Text Requirements

All scripts that produce user-facing output for Discord MUST strip Telegram HTML tags. Discord renders `<b>`, `<code>`, `<blockquote expandable>` and similar HTML as raw text characters.

### Tags to strip

| Telegram HTML | Discord-safe replacement |
|---|---|
| `<b>text</b>` | `**text**` or just remove |
| `<code>text</code>` | `` `text` `` (backtick code fences) |
| `<blockquote expandable>` | Delete entirely |
| `<i>text</i>` | `*text*` or just remove |
| `<a href="url">text</a>` | `text — url` |

### Pattern: check every wrapper script

When reviving a dead script that was originally authored for Telegram:
1. Read the script's stdout output logic (the `echo`, `print`, or `cat` statements)
2. Grep for `<b>`, `<code>`, `<blockquote`, `<i>`, or `<a ` patterns
3. Replace each with Discord-safe equivalents
4. Run the script manually and verify output renders without raw HTML tags

**Real example (2026-05-23):** `token_health_wrapper.sh` output contained `✅ <b>Token health</b> · all OK` — Discord rendered this as "✅ `<b>Token health</b>` · all OK". Fixed by removing `<b>` tags.

### Test every revived cron before declaring done
Run each script manually. Verify:
- stdout is Discord-safe (no `<b>`, `<code>`, `<blockquote>` Telegram HTML tags)
- MEDIA: path exists at delivery time
- HTML attachment directories exist (create with `mkdir -p` if needed)
- Exit code matches expected (0 = clean, 1+ = error)

### Step 7 — Verify format consistency across all output
Run `cron-output-lint.py` after changes. Fix any scripts that still emit Telegram HTML tags for Discord-bound delivery.

## Heartbeat Fusion Pattern (absorbs, not creates)

When a standalone watchdog script checks a system metric that the heartbeat audit already covers partially, **absorb the check into heartbeat_audit.py** instead of creating a new cron.

**Decision rules:**
| If the watchdog... | Then... |
|---|---|
| Checks memory, services, cron gaps, or gateway PIDs | Add a probe function to `heartbeat_audit.py` that returns `dict | None` |
| Alerts only when something is wrong | The probe fires a kanban triage task on threshold breach |
| Has a different cadence than ideal (e.g. every 4h but heartbeat runs every 2h) | Absorb it — the shorter cadence is better, and the probe costs near-zero |
| Needs external API access or MCP tools | CANNOT absorb (heartbeat is script-only infra). Keep as standalone. |

**Implementation pattern:**
```python
def my_new_probe_finding(now: datetime) -> dict | None:
    # 1. Run check
    # 2. If healthy → return None
    # 3. If threshold breached → return finding dict with title, body, assignee, priority, key
    ...
```

Append the probe call right before the `filed = []` line in `main()`. The existing filing loop handles prioritisation (max 3 filed per run) and deduplication (idempotency keys per date).

**Scripts absorbed via this pattern (2026-05-23):**
- `memory_watchdog.sh` → `memory_health_finding()` in heartbeat_audit.py (500MB threshold, parses `free -m`)
- `cron-gap-monitor.sh` → `cron_gap_finding()` in heartbeat_audit.py (checks last_run_at freshness for daily crons, 3h threshold)
- `services_health_watchdog.py` → `duplicate_gateway_finding()` in heartbeat_audit.py (checks pgrep PIDs > 16)

The original scripts remain on disk as reference but no cron fires them.

## Full Output Inventory (orphan + active + dead)

When Sahil asks "what output is reaching me and what's fallen off", run this 8-point audit before touching anything. It discovers active crons, unwired scripts, orphan output dirs, and dead platform connections in one pass.

### Step 1 — Root cron inventory
```bash
cronjob list
```
List all root-level jobs with name, schedule, deliver, last_run_at, last_status, no_agent, script.

### Step 2 — Profile cron inventory
```bash
find /home/kensei/.hermes/profiles -name "jobs.json"
```
Check every profile for scheduled jobs the root scheduler isn't running. Profile jobs with `last_run_at: null` but Discord delivery targets are a common source of "I never receive X" complaints.

### Step 3 — Cross-reference scripts against cron script: fields
```bash
ls /home/kensei/.hermes/scripts/
```
For each script, check if any active cron references it. Any script without a matching cron is **dead** (the code exists, nothing fires it). Track these separately — they're the pipeline that fell off.

### Step 4 — Check runbook directories for last-actual-run dates
```bash
ls -lt /home/kensei/.hermes/runbooks/<category>/ | head -3
```
A script may exist and a cron may be configured, but if the runbook output stopped 10 days ago, something broke silently (model failure, delivery error, scheduler skip). Compare last modified date against expected schedule frequency.

### Step 5 — Check for orphan cron output directories
```bash
ls /home/kensei/.hermes/cron/output/
```
Compare output dirs against active job IDs. Any dir that doesn't match an active job is an orphan — either a one-shot cleanup that ran once, or a cron that was deleted from jobs.json but left output behind.

### Step 6 — Verify platform connectivity
```bash
# For each messaging platform, check gateway logs
grep -i "connected\|disconnected" /home/kensei/.hermes/logs/gateway.log | tail -20
# Check env vars for disabled tokens
grep "TELEGRAM\|DISCORD" /home/kensei/.hermes/.env | grep -v "^#"
# Cross-check against processes
ps aux | grep -i "telegram\|discord\|gateway" | grep -v grep
```
A "migrated" platform may still show as connected in the gateway if the env var wasn't cleaned up. Look for `_DISABLED` suffixes, grep gateway.log for recent connection activity.

### Step 7 — Build the inventory table

| Cron Name | Schedule | Delivery Target | Last Run | Status | Content Type | Agent? | Notes |
|-----------|----------|----------------|----------|--------|-------------|--------|-------|
| | | | | | | | |

Then a separate table for **dead/unwired scripts**:

| Script | Purpose | Last Evidence | Status |
|--------|---------|--------------|--------|

### Step 8 — Flag gaps (morning pulse, end-of-day digest, blocker push)

Three patterns consistently missing in post-migration setups:
- No morning pulse consolidated from system_report + calendar + kanban snapshot
- No end-of-day kanban digest covering what completed vs what's still stuck
- No block-staleness detection that actually labels tasks as blocked and pushes them to Sahil

If all three are missing, state it explicitly in the findings — don't bury it.

See `references/output-inventory-2026-05-23.md` for a worked example that discovered:
- 11 active crons, 10+ dead scripts, 1 orphan output dir
- Zero blocked tasks (not because nothing's stuck — because nothing flags them)
- No morning pulse, no EOD digest, no blocker push
- Telegram confirmed dead with env var renamed to DISABLED


| Notification / Job | Purpose | Schedule / Trigger | Provider | Topic / Channel | Routing field | Output mode | Notes |
|---|---|---|---|---|---|---|---|

Checklist:
1. Include every active cron job and any obvious gateway/system notification source.
2. Show both the messaging provider and the exact topic/channel/thread where retrievable.
3. Flag stale routing explicitly, e.g. `Telegram Topic 20 (stale)` or `Discord #cron-outputs`.
4. Compare configured delivery with where Sahil expects to read the report. A successful Discord migration can still be wrong if every report lands in a dump channel instead of semantic channels like #calendar, #research-digest, #job-hunt, #content, or #ops.
5. Check execution freshness, not just delivery config: `last_run_at`, `next_run_at`, `last_status`, and `last_delivery_error`. Morning/report jobs can silently skip when scheduler state advances without a matching run.
6. Separate "delivery routing" problems from "scheduler execution" and "provider/model capacity" problems. Do not rewrite fallbacks or credentials unless Sahil explicitly approves.
7. After any cleanup, re-run representative jobs and verify the actual destination receives the message/attachment.

For Discord-specific cron delivery triage, use `hermes-cron-operations/references/discord-cron-delivery-diagnosis.md`.

Session-specific migration notes: `references/discord-notification-migration.md`

Discord output formatting audit and HTML-artifact pattern: `references/discord-output-format-validation.md`

## HTML Template

Base: `/home/kensei/.hermes/templates/cron-digest-template.html`. Content digest HTML generated via `generate_draft_html()` in `telegram_digest.py`.

**Content digest HTML structure:**
- Brand counts + content_type counts
- Numbered drafts with: brand, platform, content-type, pillar, body_text, visual_description
- **Engine recommendation hints for video content:** shows `video_engine` (auto/ai/hyperframes/ffmpeg) and suggests which engine fits the content type

**Radar digest HTML structure (`github-radar-daily`):**
- `<h1>` header with date
- Sections per label (ADOPT, EXTRACT, PLUGIN/SKILL, FORK/PRODUCT, INSPIRATION, NOISE)
- Full repo details per entry: name, stars, language, description, URL
- Total counts and filtering notes
- File path: `/home/kensei/.hermes/runbooks/github-radar/YYYY-MM-DD/github-radar-YYYY-MM-DD-HHMM.html`

**Critical patterns:**
1. Draft numbering sequential, 1-N.
2. Approval format: `A:1,2,3,6-28` or `A:all`
3. Reject format: `R:4,5,11,29`
4. Free-text notes allowed.
5. Radar label caps: ADOPT max 3, PLUGIN/SKILL max 3, EXTRACT max 3, FORK/PRODUCT max 2, INSPIRATION max 5 per run.
6. Save to `/home/kensei/.hermes/runbooks/content-digest/drafts-YYYYMMDD-HHMM.html`

## Legacy Delivery Mapping

Older Telegram topic mappings may still exist in historical files such as `references/delivery-mapping.txt` or `/home/kensei/.hermes/cron-delivery-mapping.txt`. Treat them as migration evidence, not current truth. For live routing, inspect active cron `deliver` fields and gateway defaults, then produce the Notification Delivery Audit table above.

## Evidence Gate — claims must be backed by verification (MANDATORY)

This applies to EVERY cron that reports system, pipeline, service, or task state: triage, content review, mailbox, ops/system health, kanban reconciliation, fork-integrity, and any monitor that posts a "what's wrong" message to Discord.

### The rule

Any **state or causal assertion** in delivered output MUST be backed by a cited observation made in this run. A state/causal assertion is any claim that something is broken, missing, stale, failing, unregistered, not running, or that names a cause. Examples that trigger the gate:

- "X is not registered" / "no scheduled jobs"
- "file/path/script does not exist"
- "CI is failing" / "tests failing"
- "service is down" / "process not found"
- "zero drafts today" / "pipeline produced nothing"
- "cron points to the wrong path"

For each such claim you must have actually run the check and you must show its provenance. Acceptable evidence is one of:

- the exact command run and a literal snippet of its output (e.g. `hermes cron list | grep content-engine` produced `content-engine-daily ... ok`)
- a file stat or listing (e.g. `ls -la /path` shows it exists, or `ls: No such file` shows it does not)
- an exit code (e.g. `gh run view <id> --json conclusion` returned `failure`)
- a query result against the authoritative datastore (e.g. a row count from the canonical DB)

### How to phrase it

Add an `Evidence:` marker to the finding carrying the proof. Pattern:

```
**What's happening:** content-engine cron is unregistered, zero drafts today.
Evidence: `hermes cron list` (base store) shows no content-engine job; `SELECT count(*) FROM drafts WHERE date(created_at)=date('now')` returned 0
```

### When you cannot verify

If you did not run the check, or the check was inconclusive, you MUST NOT assert a root cause. Phrase the line as:

```
**Symptom observed — cause unconfirmed:** <the raw symptom>. Not verified: <what you would need to check>.
```

Never present an inferred or plausible-sounding cause as fact. A guessed file path, a presumed "not registered", or a cause carried over from an upstream alert without re-checking is a contract violation.

### No action on unconfirmed findings

Do not auto-close, auto-escalate, file a fix task, or recommend a destructive/config/spend action on a finding that is `cause unconfirmed`. Unconfirmed findings may only be surfaced for human review, clearly labelled as unconfirmed.

### Authoritative sources (check these, not proxies)

- cron registration: `hermes cron list` in the **base store** (no `--profile`); a profile-scoped list is not authoritative and routinely shows "No scheduled jobs".
- CI status: `gh run list` / `gh run view` conclusion plus exit code, not an email notification. `action_required` means approval-gated, NOT failing.
- content freshness: row count in the canonical `content_engine.db`, not a stale in-prompt snapshot.
- file/script existence: an actual `ls`/stat of the path, not an assumed convention.
- service/process state: `ps`/`systemctl`/a health probe, not a single past error line.

### Regression guard

`/home/kensei/.hermes/scripts/cron-output-lint.py` flags strong causal assertions in visible output that lack a nearby evidence citation. Run it after any prompt/script change; it must return zero issues before a fix is called done.