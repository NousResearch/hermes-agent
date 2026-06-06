# All To One Portable Prompt

You are using the All To One project memory compression protocol.

Your job is not to summarize chat. Your job is to create durable project memory for future humans and agents.

## Mission

Create a document that lets a future human or AI agent resume the project in 5-10 minutes without rereading the entire conversation or repository.

The document must explain:

1. What the original goal was.
2. What actually happened.
3. What changed in files, config, system state, data, services, or deployment.
4. What bugs/errors appeared.
5. The root causes.
6. Why the final solution works.
7. What was verified and what was not.
8. What risks remain.
9. How a future agent can resume in 5-10 minutes.

## Context Collection

Read all necessary context:

- Current conversation/context available to you.
- `git status`, `git diff`, and recent commits.
- README, AGENTS.md, CLAUDE.md, docs, deployment notes.
- Key configuration files.
- Test/build/lint/smoke outputs.
- Relevant logs and error messages.
- Existing project memory such as `docs/all-to-one.md`, `docs/handoff.md`, or `docs/PROJECT_MEMORY.md`.

Do not blindly read:

- dependency folders such as `node_modules`, `.venv`, `vendor`
- build artifacts such as `dist`, `.next`, `build`, `target`
- secrets, private keys, credentials, tokens
- large binaries or unrelated assets
- unrelated large files just to appear complete

If context is too large, first create a context index:

- Read sources
- Skipped sources
- Why skipped
- Possible blind spots

## Evidence Tags

Mark important claims with evidence tags:

- `[verified]` command/test/tool output proves it
- `[screenshot]` user screenshot proves it
- `[observed]` observed in conversation or session
- `[inferred]` reasoned but not directly verified
- `[unverified]` not checked yet
- `[blocked]` could not check due to blocker

Never turn `[inferred]` or `[unverified]` claims into `[verified]` facts.

## Output

Prefer writing/updating:

- `docs/all-to-one.md`

If the project already has a memory/handoff document, update it instead of creating fragments.

Use this structure by default:

```md
# All To One: {project/task name}

## 1. One-Sentence Result

## 2. Background and Goal

## 3. Final System State

## 4. Real Timeline

## 5. Key Changes

## 6. Bugs, Pitfalls, and Root Causes

## 7. Plain-English Principles

## 8. Verification Record

## 9. Risks and Future Improvements

## 10. Resume in 5-10 Minutes
```

Quality bar:

- A future agent can start from the document without rereading the whole repo.
- A human can explain why the fix works after reading it.
- Key commands, paths, files, configs, and services are named.
- Failed paths and wrong assumptions are preserved.
- Verified facts are separated from guesses.
- There is a clear restart path and clear red lines.