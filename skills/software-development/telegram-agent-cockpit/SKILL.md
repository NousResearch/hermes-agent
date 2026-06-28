---
name: telegram-agent-cockpit
description: "Use when operating long-running Hermes/Codex/agent tasks from Telegram topics. Enforces one-topic-one-workstream discipline, heartbeat/stall recovery, safety gates, media ingress handling, config-patch care, and concise final reporting."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [telegram, agent-ops, coding, safety, workstreams]
    related_skills: [systematic-debugging, skill-package-review, hermes-agent]
---

# Telegram Agent Cockpit

## Overview

Telegram work often arrives as short messages, screenshots, voice notes, files, and follow-up steering. This skill turns that stream into a reliable operating discipline for Hermes/Codex-style agent work.

**Core rule:** one Telegram topic or thread is one workstream. Keep the task bounded, keep the user informed when the work is long-running, protect secrets and configuration, verify real results, and finish with a compact operational report.

This is not an OpenClaw `/cas_*` implementation and does not install any third-party bridge. It captures the useful cockpit ideas as Hermes operating rules for Telegram: topic discipline, heartbeat, stall recovery, safety gates, media ingress, and final reporting.

## When to Use

Use when a Telegram task involves any of the following:

- coding, debugging, repository inspection, or GitHub operations;
- Hermes setup, gateway configuration, Telegram bindings, tools, skills, cron, or plugins;
- ZIP/skill/package review before installation;
- long-running commands, tests, builds, deployments, or background processes;
- screenshots, voice notes, logs, files, or other media as task input;
- possible secrets, tokens, credentials, private IDs, or sensitive logs;
- multi-step work where the user expects a finished result rather than advice.

Use a lighter response for simple one-shot questions that do not require tools, files, external state, or follow-up execution.

## Workstream Rules

1. **One topic = one workstream.** Treat the current Telegram topic/thread as the task boundary. Do not silently mix unrelated projects or old assumptions into the active workstream.
   - **Done when:** the active objective, scope, and next action are clear from the current thread.

2. **Escalate to a new topic/chat when context breaks.** If the task changes radically, two fixes fail in a row, or the thread becomes overloaded with unrelated context, recommend a fresh topic/chat before continuing.
   - **Done when:** the user has a clean continuation path instead of a confused mixed context.

3. **Use a visible task plan for 3+ step work.** Maintain an internal todo list for multi-step work and update it as steps complete, fail, or are replaced.
   - **Done when:** there is one active step, completed steps are marked, and blockers are explicit.

4. **Source-first, not memory-first.** If the user provides a live source, repo, URL, file, chat, or system identifier, inspect that source before relying on memory or previous sessions.
   - **Done when:** claims about current state are backed by tool output or the provided artifact.

## Status Card Discipline

For long, branching, or blocked workstreams, summarize state in this compact shape when it helps the user regain orientation:

```markdown
## Статус
- Цель:
- Сейчас:
- Сделано:
- Блокер:
- Следующий шаг:
```

Do not spam status cards for short work. Use them when a run is long, a blocker appears, the user steers mid-turn, or the task changes phase.

## Heartbeat and Long-Running Work

1. **Choose tracked execution.** For long-running bounded tasks, prefer `terminal(background=true, notify_on_complete=true)` or another tracked mechanism instead of an untracked shell background process.
   - **Done when:** Hermes can observe process completion and output.

2. **Do not disappear silently.** If work takes long enough that the user might reasonably think the agent stalled, provide a compact progress update or use process notifications.
   - **Done when:** the user either has a completion notification coming or a clear current status.

3. **Return blockers immediately.** If the next step needs user input, missing credentials, unavailable access, or an unsafe confirmation, stop and ask/offer choices instead of pretending progress continues.
   - **Done when:** the blocker is explicit and the user has 2-4 practical options when choices exist.

4. **Never fabricate completed work.** If a command, install, test, network call, or verification failed, report the failure and try a reasonable alternative before finalizing.
   - **Done when:** final claims match real tool output.

## Stall Recovery

When a command, process, or agent run appears stuck:

1. Check whether the process is still alive.
2. Read the latest available output/logs.
3. Classify the state: normal long run, silent hang, known prompt, missing dependency, permission issue, or crash.
4. If safe, recover: answer the prompt, retry with a tighter command, restart the process, or switch approach.
5. If recovery is not safe or not possible, report the blocker and the safest next step.

**Done when:** the workstream has either recovered with verified progress or the user has a clear blocker report.

## Safety Gates

Apply these gates before side effects:

| Action | Required discipline |
|---|---|
| Secrets/tokens/private logs may appear | Warn before exposing or operating on them; redact in summaries |
| Deleting files or destructive commands | Require explicit user confirmation unless operating only inside a disposable temp/sandbox path |
| Writing another Hermes profile's skills/plugins/cron/memory | Require explicit user direction |
| GitHub write actions | Use configured account only when requested; ask before dangerous actions |
| Full-access/yolo/danger modes | Do not enable by default; require explicit approval and explain risk |
| ZIP/skill/package install | Review read-only first; install only after explicit approval |
| Public/private split | Keep chat IDs, user IDs, private paths, bindings, prompts, and topology out of public artifacts |

## Config Patch Discipline

Configuration changes can silently break existing bindings when list-like fields are overwritten. For Hermes gateway, Telegram bindings, tools, providers, cron, skills, and plugin config:

1. Read the current config/state first.
2. Identify the exact field/array that will change.
3. Apply the smallest possible patch.
4. Verify old entries still exist after the patch.
5. Report what changed and what was preserved.

**Done when:** the changed setting works and unrelated bindings/settings were not removed.

## Media Ingress Rules

Telegram inputs are task context, not interruptions.

| Input | Handling rule |
|---|---|
| Screenshot/image | Inspect the image directly when vision is available; do not ask the user to retype it unless unreadable |
| Voice note/audio | Transcribe or use available STT when needed; treat transcript as user input |
| Text/log file | Read safely with pagination; search for root cause rather than dumping noise |
| ZIP/archive/skill | Inventory and security-review read-only before extraction/installation |
| Repo/link | Clone or inspect read-only first; record commit/hash when making a recommendation |
| Binary/large file | Identify type/size first and choose a safe extractor or ask for a smaller relevant slice |

## Review and Install Decision Matrix

When deciding whether to adopt a repo, plugin, skill, or package, end with a clear decision:

| Criterion | Question |
|---|---|
| Benefit | What capability or discipline does it actually add? |
| Fit | Does it fit the current Hermes runtime and Telegram workflow? |
| Safety | Does it execute shell, network, autostart, webhooks, destructive actions, or full-access modes? |
| Operational cost | What setup, maintenance, secrets, accounts, and monitoring are required? |
| Alternative | Can we keep the useful ideas as a rule/reference instead of installing code? |
| Decision | install now / keep as reference / postpone / reject / needs cleanup |

Prefer adopting principles over installing third-party code when the repo is only a compatibility alias, documentation, or a pattern that can be captured locally.

## Final Report Contract

For normal technical work, finish with:

```markdown
## Итог
<what is now true>

## Сделано
- ...

## Проверка
- <command/tool output/fact that verified it>

## Риски / нюансы
- ...

## Следующий шаг
<one concrete next step>
```

For incidents or failed workstreams, use an incident report:

```markdown
## Рапорт
1. Хронология событий
2. Корневая причина
3. Что исправлено
4. Safeguards / доказательство неповтора
```

Keep the final answer short enough for Telegram, but include the evidence needed to trust the result.

## Common Pitfalls

1. **Installing before reviewing.** A Telegram request to “look at this” is not approval to install. Review read-only first.
2. **Losing the topic boundary.** Pulling in unrelated old context causes wrong fixes. Keep the active workstream explicit.
3. **Silent long runs.** A tracked background process or a compact status beats disappearing.
4. **Config array overwrite.** Always read and verify list-like config sections before and after patches.
5. **Normalizing danger modes.** Full access may be convenient, but it is never the default for Telegram-driven work.
6. **Reporting plausible success.** The final report must be backed by real command/tool output or provided evidence.
7. **Dumping sensitive data.** Summaries should prove the point without exposing tokens, chat IDs, private paths, or credentials.

## References

- `references/v1-adoption-notes.md` — why this skill exists, what v1 adopted from the Telegram/Codex cockpit review, and the review heuristic for future cockpit-like repos.

## Verification Checklist

- [ ] The active Telegram topic/thread has one clear workstream.
- [ ] Multi-step work has a maintained todo/plan.
- [ ] Long-running commands are tracked or have a clear heartbeat path.
- [ ] Stalls/blockers are surfaced rather than hidden.
- [ ] Secrets and destructive actions passed the safety gate.
- [ ] Config changes used read → minimal patch → verify.
- [ ] Media/files/repos were inspected safely and source-first.
- [ ] Installation decisions separate “useful idea” from “safe to install”.
- [ ] Final report includes result, work done, verification, risks/nuances, and next step.
