---
name: hermes-bug-feature-reporting
description: "Hermes bug & feature reporting: Discord triage, then a verified GitHub issue. Enforces a pre-report gate (real evidence, no duplicates, no non-Hermes noise)."
version: 1.0.0
author: Schrauberhirn (NousResearch Discord), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Bug-Reporting, GitHub, Discord, Triage, Hermes, Verification-Gate]
    related_skills: [hermes-agent]
---

# Hermes Bug & Feature Reporting

## When to Use
Load this skill whenever the user wants to **report a Hermes bug, file a GitHub issue,
open a feature request, or triage an error**. It enforces a two-stage pipeline and a
mandatory verification gate so nothing unverified or duplicate gets posted upstream.

## Principle: two-stage pipeline
1. **Discord = maturation / triage.** Post errors there first to narrow the issue and
   confirm it is a real bug (not user error). Use `hermes debug share` for logs.
2. **GitHub = official record.** File the actual bug report or feature request only
   after the issue is confirmed/reified (or for clearly-official requests).

Never skip straight to GitHub with an unconfirmed error — Discord first to tighten the repro.

## Stage 1 — Discord support thread (triage)
- Support thread channel: https://discord.com/channels/1053877538025386074/1485307775444844625 (open a thread)
- Local-model help: https://discord.com/channels/1053877538025386074/1492640305881809068

### Required in every support thread
- what you were trying to do
- what happened instead
- how you installed Hermes
- OS / Docker / WSL / terminal app / Desktop version if relevant
- provider / model / platform or gateway
- what you already tried
- relevant logs

### Fastest log path
- `hermes debug share` → public paste links (usually agent.log, errors.log, gateway.log, gui.log, desktop.log when available)
- Private: `hermes debug share --nous` → redacted bundle sent to private viewer, deleted after 14 days
- Messaging platforms: `/debug`

### Redaction (CRITICAL)
Logs may include: recent conversation fragments, tool outputs, file paths, usernames, channel names, hostnames, local context. **Read generated pastes before posting. Do NOT post API keys, tokens, passwords, cookies, card details, OAuth secrets, or other private credentials.** Use synthetic token shapes (e.g. `sk-lm-ABC123:SECRETSUFFIX`) in any public repro.

### Manual log locations
- Local: `~/.hermes/logs/agent.log`, `errors.log`, `gateway.log`, `desktop.log`
- Profiles: `~/.hermes/profiles/<profile>/logs/`
- Docker: usually `/opt/data/logs/`
- Windows Desktop: `%LOCALAPPDATA%\hermes\logs\`
- Gateway issues → `gateway.log`
- Desktop boot → `desktop.log`
- Installer / first-launch / repair → `bootstrap-installer.log` or newest `bootstrap-*.log`; Desktop first-launch/repair also `desktop.log`
- Update → `update.log`
- Dashboard → `gui.log`
- Docker startup / profile restore → `container-boot.log`
- Provider / model / tool → `agent.log` + `errors.log`
- No installer log → full terminal output + the temp log path the installer printed

### Rules
- Threads without relevant logs may be deprioritized, locked, or closed.
- One user/issue per thread. Do not hijack others' threads.
- Billing / account / subscription / credits → `support@nousresearch.com` (NEVER public).
- **WE WILL NEVER DM YOU.** Random DM claiming to be staff → record account name + mutual servers, report to Discord. Do not ping staff about it.

## Stage 2 — GitHub official issue
- Issues: https://github.com/NousResearch/hermes-agent/issues
- Feature requests: https://github.com/NousResearch/hermes-agent/issues/new?template=feature_request.yml
- File here once the issue is mature (confirmed not user error, repro known) or for feature requests directly.
- Include the same substance as Discord (what / what instead / install / OS / provider / what tried / logs), plus:
  - clear step-by-step repro
  - expected vs actual behavior
  - reference the Discord thread if it helped narrow it down

## Second Brain (Obsidian) integration — OPTIONAL but recommended
Hermes may have an Obsidian "second brain" vault. If configured, use it BEFORE reporting
(to find prior art) and AFTER reporting (to persist the bug/feature + schedule a follow-up).
If NOT configured, skip these steps and note "Second Brain skipped — OBSIDIAN_VAULT_PATH unset / vault not found".

**Detect Second Brain:**
- Env `OBSIDIAN_VAULT_PATH` (from `${HERMES_HOME:-~/.hermes}/.env`); if unset, fallback `~/Documents/Obsidian Vault`.
- If neither resolves to an existing directory → Second Brain NOT available this session.

See skill `note-taking/obsidian` for the file-tool workflow (read_file / search_files / write_file / patch;
vault paths may contain spaces; never pass `$OBSIDIAN_VAULT_PATH` literally — resolve to a concrete absolute path first).

## Pre-report verification (MANDATORY gate)
Every bug/feature derived from the current session MUST pass this gate before any post.
Do NOT invent or guess — verify against real evidence.

### Step V0 — Second Brain check (if Second Brain available)
Before filing, search the vault for existing notes about the same bug/feature so you don't duplicate local knowledge:
- `search_files` target=content, file_glob=`*.md`, pattern = the error token / feature keyword (e.g. `exec request failed`, `DISCORD_HOME_CHANNEL_THREAD_ID`, `ControlMaster`).
- If a note exists: read it, reuse any root-cause/workaround already documented, and reference it in the report. Do NOT re-derive what is already captured.
- If no note exists: proceed (you will create one in the Post-report step).

### Step V1 — Self-evidence check (always)
- Bug: quote the EXACT error string / tool return captured in this session (paste verbatim, not paraphrased). No capture → not reportable, ask user for logs.
- Feature: state the concrete friction observed this session (command, failure mode, workaround used).
- If neither exists: stop. Nothing to report.

### Step V2 — GitHub verification (always)
Before opening a new issue, check it is not already filed.
**Search strategy (do NOT rely on a single method):**
1. `web_search` with `site:github.com/NousResearch/hermes-agent/issues <keyword>` — fast, but web_search often returns EMPTY for raw error strings / technical tokens (e.g. `lookup_sid 1332`, `CreateProcessW error 8`). Treat empty web_search result as "inconclusive", NOT "not found".
2. Preferred: `gh` CLI if authenticated (`gh search issues --repo NousResearch/hermes-agent "<kw>" --state open --limit 25`, `gh issue view <n>`, `gh pr view <n>` for merge status). Path on Windows: `C:\Program Files\GitHub CLI\gh.exe` (not on MSYS PATH).
   - **If the user blocks inline `powershell.exe -Command '... gh ...'` wrappers**, do NOT retry or rephrase — fall back to web_search/web_extract and state "gh check skipped (command blocked)".
   - Note: `gh search issues --state` only accepts `open` or `closed`, never `all` — run both states to cover everything.
3. `web_extract` on the specific issue URL once found — confirm state (open/closed), linked PRs, and whether a fix was MERGED (`gh pr view <n> --json merged` or the issue's "Development"/PR section).

**Interpreting results:**
- Duplicate Hermes issue found → do NOT open new; report existing URL + (optionally) add new repro as comment.
- Found RELATED but DIFFERENT issue (e.g. same component, different failure) → note it as context; still allowed to file a new, more specific issue if the symptom differs materially.
- **No Hermes issue matches, but root cause is an OS/third-party behavior (Windows OpenSSH token/SID, WSL, Docker) → this is a valid gate exit: "nothing to report upstream; document the Windows fix locally instead". Do NOT open a Hermes issue for a non-Hermes bug.**
- Inconclusive (all methods empty/blocked) → tell the user the check could not be completed and ask before filing.

**Upstream reporting conventions (from CONTRIBUTING.md):**
- File at GitHub Issues: https://github.com/NousResearch/hermes-agent/issues
- MUST include: **OS**, **Python version**, **Hermes version** (`hermes version`), **full error traceback**, **steps to reproduce**.
- Search FIRST (CONTRIBUTING "Before You Start: Search First"):
  - `gh search issues --repo NousResearch/hermes-agent "<terms>"`
  - `gh search prs --repo NousResearch/hermes-agent --state closed "<terms>"` ← also search MERGED/closed PRs; the issue tracker can lag the code, so a "requested feature" may already be implemented in-tree. Search the source too before proposing.
  - If an open PR already addresses it → review/improve that one instead of a competing duplicate.
- Security vulnerabilities → report PRIVATELY, not as a public issue.
- Contribution priorities (signal urgency in the report): 1 Bug fixes, 2 Cross-platform compat, 3 Security hardening, 4 Performance/robustness, 5 Skills, 6 Tools, 7 Docs.
- If the report matures into a PR: branch `fix/<desc>` / `feat/<desc>`, Conventional Commits (`fix(scope): ...`), PR description = What/Why + How-to-test + platforms tested + related issue.

### Step V3 — Discord verification (only if Discord access available this session)
Discord access exists when the Hermes gateway is bound (skill `lab/discord-hermes-lab`; ALLOWED_USERS in your profile's `.env`). When available:
- For errors of UNCERTAIN cause → post a triage thread FIRST (Channel: https://discord.com/channels/1053877538025386074/1485307775444844625), attach `hermes debug share` output, and wait for staff/user confirmation that it is a real bug before promoting to GitHub.
- For confirmed bugs / features → Discord is optional; GitHub is the official record.
If Discord access is NOT available this session: skip V3, note "Discord triage skipped — gateway not reachable in this session", and go straight to GitHub with full self-evidence.

### Gate outcome
- Passes V0(+V1)+V2 (and V3 if applicable) AND a Hermes-owned bug/feature exists → file per Stage 1/2, then run Post-report persistence.
- Root cause is NON-Hermes (OS/third-party) and no matching issue → valid exit "nothing to report upstream; keep the local fix/workaround in a skill or notes" (and still save to Second Brain if available, since the local fix is valuable knowledge).
- Fails any → do NOT post; tell the user what is missing.

## Post-report persistence (after a bug/feature is filed OR deemed non-Hermes but worth keeping)
If Second Brain is available, persist the finding so future sessions find it:
1. Create/update a vault note, e.g. `Bugs/Hermes-<topic>.md` (or `Features/Hermes-<topic>.md`), with:
   - GitHub issue/PR URL (if any) + issue number
   - verbatim error string / repro
   - root cause + workaround/fix
   - date filed, status (open/closed/wontfix)
   - wikilink to related notes
2. **Schedule a follow-up check** — propose ONE of:
   - Two reminder cron jobs (local, `deliver: local`): +1 week and +2 weeks after filing, each re-checks the issue via `gh issue view <n> --json state,closedAt` and reports status. Use `cronjob action=create` with a self-contained prompt that includes the issue URL/number and the check command.
   - OR GitHub-native notification: subscribe to the issue (`gh api -X PUT repos/NousResearch/hermes-agent/issues/<n>/subscriptions`) so GitHub emails on updates; note this to the user as the lower-maintenance option.
   - Recommend the GitHub-subscribe path by default (no cron needed); offer the cron reminders only if the user wants in-agent nudges.
3. If Second Brain is NOT available: tell the user the finding was NOT persisted locally and offer to save it to a skill/note instead.

## Decision tree
- Error, unsure if bug → Discord first (maturation, V3).
- Confirmed bug with repro → GitHub bug report (link Discord thread if useful), after V1+V2 gate.
- Want a feature → GitHub `feature_request.yml` template directly, after V1+V2 gate.
- Billing / account → email `support@nousresearch.com`, not public.

## GitHub label suggestions (use the repo's REAL labels)
Apply 1 `type/*` + 1+ `comp/*` or `platform/*` or `area/*` + optional `P0–P4`/`sweeper:*`.
Pull the live list first if unsure: `gh label list --repo NousResearch/hermes-agent --limit 200`.
Do NOT invent labels — only use ones that exist in the repo.

**Type (exactly one):**
- `type/bug` — something isn't working
- `type/feature` — new feature / request
- `type/docs`, `type/perf`, `type/refactor`, `type/test`, `type/security`

**Component (`comp/*`):**
- `comp/gateway` — gateway runner, session dispatch, delivery  ← cron/Discord delivery failures
- `comp/cron` — cron scheduler + job management
- `comp/desktop` — Electron desktop app (SSH settings dialog, etc.)
- `comp/cli`, `comp/agent`, `comp/tools`, `comp/tui`, `comp/plugins`, `comp/dashboard`, `comp/portal`, `comp/lsp`
- `backend/ssh` — SSH remote execution (agent-side SSH backend)
- `tool/*` (terminal, file, memory, skills, web, mcp, vision, …)

**Platform (`platform/*`):**
- `platform/discord` — Discord bot adapter  ← our Discord-delivery case
- `platform/windows` — native Windows breakage
- telegram / slack / matrix / whatsapp / email / webhook / etc.

**Area (`area/*`):**
- `area/config` — config system, migrations, profiles (e.g. `DISCORD_HOME_CHANNEL_THREAD_ID`)
- `area/auth`, `area/profiles`, `area/install-update`, `area/sessions`, `area/memory`, `area/streaming`

**Priority (`P0`–`P4`):**
- `P0` critical (data loss/security/crash loop) · `P1` high (broken, no workaround) · `P2` medium (workaround exists) · `P3` low/cosmetic · `P4` best-effort.

**Sweeper risk tags (add if the fix could regress others):**
- `sweeper:risk-platform-windows`, `sweeper:risk-message-delivery`, `sweeper:risk-session-state`, `sweeper:risk-compatibility`, `sweeper:risk-security-boundary`, plus `sweeper:blast-contained|moderate|broad|massive`.

**Worked example — the Discord cron-delivery 404 (issue #72731):**
`type/bug`, `comp/gateway`, `platform/discord`, `area/config`, `P2` (+ optional `sweeper:risk-message-delivery`).
If you later confirm it's purely a local misconfig, maintainers may relabel `invalid`/`wontfix` — that is expected for SKILLTEST issues.

**Lifecycle labels (maintainer-applied, don't self-set unless asked):** `duplicate`, `invalid`, `wontfix`, `question`, `needs-repro`, `blocked`, `stale`, `needs-decision`.

## Pitfalls
- **Redaction leaks:** a `Bearer` redactor that stops at `:` will leak the token suffix (see issue #74967). When quoting headers in a report, redact the FULL token and use synthetic shapes.
- **`gh search issues --state all` is invalid** — use `--state open` and `--state closed` separately.
- **Don't wrap `gh` in inline `powershell.exe -Command`** if the user blocks that pattern — fall back to web tooling.
- **Never post real credentials** in Discord threads or GitHub issues; the redaction rule is absolute.

## Verification
After filing, confirm the issue URL is reachable (`gh issue view <n>` or web_extract) and that
labels applied are real (none invented). If the gate exited "non-Hermes", state that clearly to
the user instead of opening an issue.
