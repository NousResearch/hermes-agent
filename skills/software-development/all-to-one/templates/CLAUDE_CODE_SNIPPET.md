# CLAUDE.md Snippet for All To One

Add this section to a repository's `CLAUDE.md` when using Claude Code.

```md
## All To One (A2O) Project Memory

When the user says "总整理", "All To One", "A2O", "handoff", "project memory", or asks to make the project easy to reopen later, use the All To One protocol.

Do not merely summarize the chat. Create durable project memory for future humans and agents.

Default output: generate or update `docs/all-to-one.md`.

Read all task-relevant context:

- current conversation/context
- `git status`, `git diff`, and recent commits
- README, AGENTS.md, CLAUDE.md, docs
- key source files and configuration
- test/build/lint/smoke outputs
- relevant logs and error messages
- existing project memory documents

Do not blindly read dependency folders, build artifacts, secrets, credentials, large binaries, or unrelated large files.

Use evidence tags:

- `[verified]` command/test/tool output proves it
- `[screenshot]` user screenshot proves it
- `[observed]` observed in conversation or session
- `[inferred]` reasoned but not directly verified
- `[unverified]` not checked yet
- `[blocked]` could not check due to blocker

Required sections:

1. One-sentence result
2. Background and goal
3. Final system state
4. Real timeline
5. Key changes
6. Bugs, pitfalls, and root causes
7. Plain-English principles
8. Verification record
9. Risks and future improvements
10. Resume in 5-10 minutes

For `A2O deep`, add:

- learning section: 3-5 principles
- wrong mental model vs correct mental model
- if starting from scratch, best route

For `A2O handoff`, add:

- first hour for the next maintainer/agent
- red lines
- rollback path

Preserve failed paths and wrong assumptions. Do not make the result look cleaner than reality.
```
