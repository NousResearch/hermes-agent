# All To One (A2O)

**All To One** is a project memory compression protocol for AI coding sessions.

It turns chaotic debugging, coding, deployment, migration, and configuration work into durable project memory:

- what changed
- why it changed
- what broke
- what fixed it
- what was verified
- what remains risky
- how a future human or agent can resume in 5-10 minutes

中文一句话：

> 把混乱的 AI 编程/排错过程，变成下次人或 agent 5-10 分钟能接回来的项目记忆。

## Why This Exists

AI can make a project work without leaving the user with real understanding.

Common failure mode:

```text
The app works now.
But nobody remembers why.
Next week, a new AI agent rereads the whole repo and chat history again.
The user spends another 30-60 minutes rebuilding context.
```

All To One fixes that by generating a reusable handoff document after each meaningful technical session.

## What A2O Produces

Default output:

```text
docs/all-to-one.md
```

It should be good enough to serve as:

1. future agent startup context
2. user learning notes
3. team handoff document
4. future debugging map

## Evidence Tags

A2O avoids fake certainty. Every important claim should use evidence tags:

- `[verified]` command/test/tool output proves it
- `[screenshot]` user screenshot proves it
- `[observed]` observed in conversation/session
- `[inferred]` reasoned but not directly verified
- `[unverified]` not checked yet
- `[blocked]` could not check due to blocker

## Modes

- `quick` — small bug/config fix, 10-25 lines
- `standard` — default full project memory
- `deep` — includes learning section and principles
- `handoff` — optimized for another agent/person to take over
- `archive` — long-term compressed project record

Example commands/prompts:

```text
All To One quick
总整理 deep
A2O handoff
生成项目记忆
下次重开不用重新读
```

## Cross-Agent Usage

A2O is designed to work in:

- Hermes Agent skills
- Codex CLI / Codex agents
- Claude Code
- Claude Desktop / Claude Projects
- Cursor / Copilot-style coding agents
- Any agent that can read instructions and inspect a repository

See:

- `templates/PROMPT.md` — portable prompt for any agent
- `templates/AGENTS_SNIPPET.md` — add to repo `AGENTS.md`
- `templates/CLAUDE_PROJECT_INSTRUCTIONS.md` — paste into Claude Desktop project instructions
- `templates/CLAUDE_CODE_SNIPPET.md` — add to `CLAUDE.md`
- `templates/standard.md` — default output template

## Recommended Repo Setup

For a repository that should always support A2O:

```text
AGENTS.md                      # include AGENTS_SNIPPET.md content
CLAUDE.md                      # include CLAUDE_CODE_SNIPPET.md content, if using Claude Code
docs/all-to-one.md             # generated project memory
```

Optional portable folder:

```text
.all-to-one/
├── SKILL.md
├── PROMPT.md
└── templates/
```

## Minimal Prompt

Paste this into any coding agent:

```text
Use All To One (A2O). Read all task-relevant context: conversation, git status/diff/log, README/AGENTS/CLAUDE/docs, key config, tests/build output, and relevant logs. Generate or update docs/all-to-one.md so a future human/agent can resume in 5-10 minutes. Mark key claims with [verified], [observed], [inferred], [unverified], or [blocked]. Preserve failed paths and root causes. Do not summarize prettily; create durable project memory.
```

## Quality Bar

A good A2O document lets a future agent answer:

- What was the original goal?
- What actually happened?
- What files/config/system state changed?
- What errors appeared?
- What was the root cause?
- Why does the final solution work?
- What was verified?
- What remains risky?
- How do I resume in 5-10 minutes?

If it cannot do that, it is not All To One. It is just a summary.
