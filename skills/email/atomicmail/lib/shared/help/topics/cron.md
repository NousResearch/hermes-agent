# Inbox polling (after register)

Registration only creates credentials. For a useful inbox integration, inbound
mail should be fetched and triaged about once per hour between interactive
sessions.

**Choose the path that matches your runtime — do not mix platforms.**

## If your host supports scheduled jobs

If you have a **native** way to enable cron or recurring agent turns (OpenClaw,
Hermes, atomic-agent, and similar hosts), set up an hourly **agent** job after
the first successful `register`. The job must **invoke an AI agent** (not just the
Atomic Mail CLI) to fetch and triage mail. Wire Atomic Mail MCP or AgentSkill
into that agent before scheduling.

The scheduled turn should call preset `list_inbox.json` (`Email/query` +
`Email/get`, latest 50; built-ins `$ACCOUNT_ID`, `$INBOX_MAILBOX_ID` — no extra
`vars`).

## If your host does **not** support scheduled jobs

Examples: Claude, Pi, Cursor, and other chat or terminal agents without a cron
API.

**Do not spend tokens trying to work around this** — no wrapper scripts, OS
crontab/LaunchAgent setup, or scheduling on a different platform on the agent's
behalf.

After a successful `register`, do one of the following:

1. **Ask your operator** to set up hourly inbox polling on a host that supports
   it (share the examples below), **or**
2. **Remind your operator** to fetch mail manually when they need updates
   (`jmap_request` with `list_inbox.json`, or `atomicmail jmap_request
   --ops-file list_inbox.json`).

## Forbidden: cross-platform scheduling

Do **not** register in one runtime and schedule the cron job on another. Example
of forbidden behavior: Pi runs `register`, then creates an OpenClaw cron job.
Either the same runtime that registered schedules polling, or the operator
schedules it themselves on a capable host.

## What not to do (any runtime)

- **Raw CLI in crontab** — `atomicmail jmap_request --ops-file list_inbox.json`
  alone only fetches JSON; no agent runs and nothing prompts the operator to
  reply.
- **Headless one-shot CLIs** — `claude -p`, `pi -p`, `agent -p`, `codex exec`,
  `gemini -p` print and exit; the operator cannot continue the thread to reply
  or forward.

## Agent prompt (use in scheduled jobs or manual fetches)

```text
Use Atomic Mail to fetch my inbox (MCP jmap_request with ops_file list_inbox.json, or atomicmail jmap_request --ops-file list_inbox.json). Summarize new messages, highlight what needs a reply, and stay available — I may ask you to reply, forward, search, or dig into something important.
```

## Scheduling examples (for capable hosts or operators)

| Your setup | Recommended approach |
| --- | --- |
| OpenClaw gateway | Built-in `openclaw cron` |
| Hermes Agent | Install Atomic Mail skill → `/suggestions` blueprint after `register` (or manual `hermes cron`) |
| Atomic Bot (atomicbot.ai) | Same as OpenClaw or Hermes host |
| atomic-agent | Built-in `atomic-agent task create` |
| No native cron (Claude, Pi, Cursor, …) | Ask operator to schedule on a capable host, or remind them to fetch manually |

These examples run a full agent turn and deliver the summary to a chat or file so
the operator can reply, forward, or ask follow-ups in the same thread.

### OpenClaw

Docs: https://docs.openclaw.ai/automation/cron-jobs

- Schedule: `--cron "0 * * * *"` or `--every 1h`
- Session: `--session isolated` (fresh turn each run)
- Delivery: `--announce` (posts to your configured channel)
- Prompt: `--message` with the agent prompt above

```bash
openclaw cron add \
  --name "atomicmail-inbox" \
  --cron "0 * * * *" \
  --session isolated \
  --message "Use Atomic Mail to fetch my inbox (MCP jmap_request with ops_file list_inbox.json, or atomicmail jmap_request --ops-file list_inbox.json). Summarize new messages, highlight what needs a reply, and stay available — I may ask you to reply, forward, search, or dig into something important." \
  --announce
```

Manage: `openclaw cron list` · test: `openclaw cron run <job-id>`

### Hermes Agent

Skill blueprints: https://hermes-agent.nousresearch.com/docs/developer-guide/creating-skills

Cron (manual fallback): https://hermes-agent.nousresearch.com/docs/user-guide/features/cron

#### Recommended: skill + blueprint

1. Install the **Atomic Mail** skill from the unified in-repo tap
   (`hermes skills install Atomic-Mail/atomic-mail-agentic/integrations/skill/atomicmail`).
2. After the first successful `register`, accept the hourly inbox **blueprint**
   via `/suggestions`.

The blueprint schedules a full **agent** turn (`no_agent: false`) with
`list_inbox.json` and delivers to `origin`. Do **not** skip this step. Do **not**
schedule raw `jmap_request` cron jobs or use `--no-agent` (script-only; no LLM).

#### Credentials on Hermes

- Default directory: `~/.hermes/atomicmail` (not `~/.atomicmail`).
- The skill launcher sets `ATOMIC_MAIL_CREDENTIALS_DIR` to
  `$HOME/.hermes/atomicmail` when that variable is **not** already set — operator
  overrides are preserved.
- Override explicitly with `ATOMIC_MAIL_CREDENTIALS_DIR` or
  `atomicmail.credentials_dir` in Hermes config.

#### Multi-account only

Operating multiple inboxes at once: pass `--credentials-dir` (skill) or
`credentials_dir` (MCP) on `register` / `jmap_request` with a separate directory
per account. Not needed for the default single-inbox flow.

#### Manual fallback: `hermes cron`

If you are not using the skill blueprint, create the job manually:

- Schedule: cron expression (`0 * * * *`) or natural language (`every 1h`)
- Delivery: `--deliver origin` (or `telegram`, `discord`, `slack`, `email`,
  `local`, etc.)
- **Do not** use `--no-agent`

```bash
hermes cron create "0 * * * *" \
  "Use Atomic Mail to fetch my inbox (MCP jmap_request with ops_file list_inbox.json, or atomicmail jmap_request --ops-file list_inbox.json). Summarize new messages, highlight what needs a reply, and stay available — I may ask you to reply, forward, search, or dig into something important." \
  --name "atomicmail-inbox" \
  --deliver origin
```

In chat: `/cron add "0 * * * *" "<prompt>" --deliver origin`. Manage:
`hermes cron list` · test: `hermes cron run <job-id>`

### Atomic Bot (atomicbot.ai)

Runs OpenClaw or Hermes — use the matching block above.

### atomic-agent

Docs: https://github.com/AtomicBot-ai/atomic-agent

```bash
atomic-agent task create \
  --cron "0 * * * *" \
  --message "Use Atomic Mail to fetch my inbox (MCP jmap_request with ops_file list_inbox.json, or atomicmail jmap_request --ops-file list_inbox.json). Summarize new messages, highlight what needs a reply, and stay available — I may ask you to reply, forward, search, or dig into something important."
```

Manage: `atomic-agent task list`

## Verify setup

After scheduling (or after asking your operator to schedule):

1. `register` succeeded; Atomic Mail MCP or AgentSkill is available to the agent.
2. Run the agent prompt **once manually**; confirm inbox fetch and follow-up work.
3. Confirm the job is registered (`openclaw cron list`, `hermes cron list`,
   `atomic-agent task list`).

## For operators: OS scheduling on terminal hosts

This section is **operator documentation**, not an agent obligation. Chat agents
without native cron (Claude, Pi, Cursor, …) should **not** attempt OS scheduling
themselves — ask the operator to set up polling on a capable host or fetch mail
manually.

If you (the operator) run a **terminal CLI agent** and want hourly inbox checks
without OpenClaw, Hermes, or similar, the scheduler must **start an interactive
session** with the agent prompt — not call `atomicmail` directly.

### Terminal agents (interactive invocation)

| Agent | Start interactively | Avoid for inbox polling |
| --- | --- | --- |
| Claude Code | `claude "prompt"` | `claude -p` |
| Pi | `pi "prompt"` | `pi -p` |
| Cursor CLI | `agent "prompt"` | `agent -p` |
| Gemini CLI | `gemini -i "prompt"` | `gemini -p` |
| Codex CLI | `codex` (TUI) | `codex exec` |

Resolve the binary on **your** machine (`command -v claude`, `command -v pi`,
etc.) and use that path in scripts.

### OS scheduling approaches

Pick what fits your OS and how you work:

**A. Wrapper script + user crontab**

Write a small script that (1) sets any API keys the agent needs, (2) launches
your terminal emulator or GUI session, (3) runs the agent **interactively** with
the prompt. Point crontab at the script. Cron does not load shell startup
files — export env vars inside the script.

**B. macOS LaunchAgent**

A `LaunchAgents` plist on a calendar interval often works better than crontab
for opening Terminal or iTerm and starting an interactive agent in the logged-in
GUI session.

**C. Linux graphical session**

Schedule via user crontab or a **systemd user timer**, launching a terminal
emulator only when a graphical session is active (`DISPLAY`,
`DBUS_SESSION_BUS_ADDRESS` for your session).

Test manually before automating: run the same command you intend to schedule and
confirm the agent can call `list_inbox.json` and wait for your replies.
