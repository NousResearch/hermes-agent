// Help topic: post-register inbox polling (agent invocation, not raw CLI cron).
/** Prompt for hourly inbox jobs — agent fetches mail and stays available for follow-up. */
export const INBOX_CRON_AGENT_PROMPT = "Use Atomic Mail to fetch my inbox (MCP jmap_request with ops_file " +
    "list_inbox.json, or atomicmail jmap_request --ops-file list_inbox.json). " +
    "Summarize new messages, highlight what needs a reply, and stay available — " +
    "I may ask you to reply, forward, search, or dig into something important.";
/** Short block — embed in MCP instructions, register tool text, overview workflow. */
export const postRegisterCronReminder = `\
AFTER REGISTER — INBOX POLLING
  Choose the path that matches your runtime (help topic "cron"):
  • Native cron (OpenClaw, Hermes, atomic-agent, …): schedule an hourly AGENT turn with list_inbox.json in the prompt; wire MCP or AgentSkill into that agent. Do NOT cron atomicmail jmap_request alone.
  • No native cron (Claude, Pi, Cursor, …): do NOT work around this (no wrapper scripts, OS crontab/LaunchAgent, or cross-platform scheduling). Ask your operator to set up polling on a capable host, or remind them to fetch mail manually when needed.
  • Forbidden: register in one runtime and schedule the cron job on another.`;
export const helpTopicCron = `\
# Inbox polling (after register)

Registration only creates credentials. For a useful inbox integration, inbound
mail should be fetched and triaged about once per hour between interactive
sessions.

**Choose the path that matches your runtime — do not mix platforms.**

## If your host supports scheduled jobs

If you have a **native** way to enable cron or recurring agent turns (OpenClaw,
Hermes, atomic-agent, and similar hosts), set up an hourly **agent** job after
the first successful \`register\`. The job must **invoke an AI agent** (not just the
Atomic Mail CLI) to fetch and triage mail. Wire Atomic Mail MCP or AgentSkill
into that agent before scheduling.

The scheduled turn should call preset **list_inbox.json** (\`Email/query\` +
\`Email/get\`, latest 50; built-ins \`$ACCOUNT_ID\`, \`$INBOX_MAILBOX_ID\` — no extra
\`vars\`).

## If your host does **not** support scheduled jobs

Examples: Claude, Pi, Cursor, and other chat or terminal agents without a cron
API.

**Do not spend tokens trying to work around this** — no wrapper scripts, OS
crontab/LaunchAgent setup, or scheduling on a different platform on the agent's
behalf.

After a successful \`register\`, do one of the following:

1. **Ask your operator** to set up hourly inbox polling on a host that supports
   it (share the examples below), **or**
2. **Remind your operator** to fetch mail manually when they need updates
   (\`jmap_request\` with \`list_inbox.json\`, or \`atomicmail jmap_request
   --ops-file list_inbox.json\`).

## Forbidden: cross-platform scheduling

Do **not** register in one runtime and schedule the cron job on another. Example
of forbidden behavior: Pi runs \`register\`, then creates an OpenClaw cron job.
Either the same runtime that registered schedules polling, or the operator
schedules it themselves on a capable host.

## What not to do (any runtime)

- **Raw CLI in crontab** — \`atomicmail jmap_request --ops-file list_inbox.json\`
  alone only fetches JSON; no agent runs and nothing prompts the operator to
  reply.
- **Headless one-shot CLIs** — \`claude -p\`, \`pi -p\`, \`agent -p\`, \`codex exec\`,
  \`gemini -p\` print and exit; the operator cannot continue the thread to reply
  or forward.

## Agent prompt (use in scheduled jobs or manual fetches)

\`\`\`text
${INBOX_CRON_AGENT_PROMPT}
\`\`\`

## Scheduling examples (for capable hosts or operators)

| Your setup | Recommended approach |
| --- | --- |
| OpenClaw gateway | Built-in \`openclaw cron\` |
| Hermes Agent | Install skill → \`/suggestions\` blueprint after \`register\` (or manual \`hermes cron\`) |
| Atomic Bot (atomicbot.ai) | Same as OpenClaw or Hermes host |
| atomic-agent | Built-in \`atomic-agent task create\` |
| No native cron (Claude, Pi, Cursor, …) | Ask operator to schedule on a capable host, or remind them to fetch manually |

### OpenClaw

Docs: https://docs.openclaw.ai/automation/cron-jobs

\`\`\`bash
openclaw cron add \\
  --name "atomicmail-inbox" \\
  --cron "0 * * * *" \\
  --session isolated \\
  --message "${INBOX_CRON_AGENT_PROMPT}" \\
  --announce
\`\`\`

Manage: \`openclaw cron list\` · test: \`openclaw cron run <job-id>\`

### Hermes Agent

Skill blueprints: https://hermes-agent.nousresearch.com/docs/developer-guide/creating-skills

Cron (manual fallback): https://hermes-agent.nousresearch.com/docs/user-guide/features/cron

**Recommended:** Install the Atomic Mail Hermes skill → after \`register\`, accept
the hourly inbox blueprint via \`/suggestions\` (\`no_agent: false\`,
\`list_inbox.json\`, \`deliver: origin\`). Do not cron raw \`jmap_request\` or use
\`--no-agent\`.

**Credentials:** Default \`~/.hermes/atomicmail\` (not \`~/.atomicmail\`). The skill
launcher sets \`ATOMIC_MAIL_CREDENTIALS_DIR\` when unset; override via env or
\`atomicmail.credentials_dir\` config. Use \`--credentials-dir\` only for
multi-account setups.

**Manual fallback:**

\`\`\`bash
hermes cron create "0 * * * *" \\
  "${INBOX_CRON_AGENT_PROMPT}" \\
  --name "atomicmail-inbox" \\
  --deliver origin
\`\`\`

Manage: \`hermes cron list\` · test: \`hermes cron run <job-id>\`

### atomic-agent

\`\`\`bash
atomic-agent task create \\
  --cron "0 * * * *" \\
  --message "${INBOX_CRON_AGENT_PROMPT}"
\`\`\`

Manage: \`atomic-agent task list\`

## Verify setup

1. \`register\` succeeded; Atomic Mail MCP or AgentSkill is available to the agent.
2. Run the agent prompt **once manually**; confirm inbox fetch and follow-up work.
3. Confirm the job is registered (\`openclaw cron list\`, \`hermes cron list\`,
   \`atomic-agent task list\`).

## For operators: OS scheduling on terminal hosts

This section is **operator documentation**, not an agent obligation. Chat agents
without native cron should **not** attempt OS scheduling themselves.

If you (the operator) run a **terminal CLI agent** without OpenClaw, Hermes, or
similar, the scheduler must **start an interactive session** with the agent
prompt — not call \`atomicmail\` directly.

| Agent | Start interactively | Avoid for inbox polling |
| --- | --- | --- |
| Claude Code | \`claude "prompt"\` | \`claude -p\` |
| Pi | \`pi "prompt"\` | \`pi -p\` |
| Cursor CLI | \`agent "prompt"\` | \`agent -p\` |

OS options: wrapper script + user crontab, macOS LaunchAgent, or Linux systemd
user timer — launching a terminal emulator with an interactive agent session.`;
