# AGENTS.md Snippet for All To One

Add this section to a repository's `AGENTS.md` if you want Codex or other coding agents to use All To One.

```md
## All To One Project Memory

When the user says "总整理", "All To One", "A2O", "handoff", "project memory", or asks to make the project easy to reopen later, use the All To One protocol.

Goal: generate or update `docs/all-to-one.md` so a future human/agent can resume in 5-10 minutes without rereading the entire chat or repository.

You may inspect all task-relevant context:

- current conversation/context
- `git status`, `git diff`, recent commits
- README, AGENTS.md, CLAUDE.md, docs
- key source files and configuration
- test/build/lint/smoke outputs
- relevant logs and error messages
- existing project memory documents

Do not blindly read dependency folders, build artifacts, secrets, credentials, large binaries, or unrelated large files.

Every important claim must use evidence tags:

- `[verified]` command/test/tool output proves it
- `[screenshot]` user screenshot proves it
- `[observed]` observed in conversation or session
- `[inferred]` reasoned but not directly verified
- `[unverified]` not checked yet
- `[blocked]` could not check due to blocker

The output must include:

1. one-sentence result
2. background and goal
3. final system state
4. real timeline
5. key changes
6. bugs/pitfalls/root causes
7. plain-English principles
8. verification record
9. risks and future improvements
10. resume in 5-10 minutes

Preserve failed paths and wrong assumptions. Do not make the result look cleaner than reality.
```