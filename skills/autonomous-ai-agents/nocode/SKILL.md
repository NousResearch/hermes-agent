---
name: nocode
description: "Use when /nocode is typed: answer only, no code or installs."
version: 1.0.0
author: Bruno Barboza (@BrunoBza), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [answer-only, read-only, question-answering, agent-behavior, no-code]
    related_skills: [plan]
---

# Nocode Skill

Answer-only mode for the agent. When the user types `/nocode` followed by a
question, the agent answers directly with read-only research — it does NOT
write files, install packages, or mutate state. It is the mirror image of
plan mode: `/plan` produces a plan document, `/nocode` produces an answer.

## When to Use

- The user types `/nocode <question>` on the CLI or a messaging platform.
- The user asks for "answer only", "just answer, don't do anything",
  "só responde", "sem código", or an equivalent phrase.
- You are about to reach for a mutating tool (write, install, config change)
  in order to answer a question — stop and answer instead.

Don't use for:

- Requests that explicitly ask for action (edits, installs, deploys, tests) —
  those are normal turns.
- Planning work — use the `plan` skill instead.

## Prerequisites

- None. The mode uses only tools already available to the agent
  (`web_search`, `web_extract`, `read_file`, `session_search`,
  `skills_list`, read-only terminal inspection).

## How to Run

Type `/nocode <question>` in the CLI or any messaging platform. Installed
skills are exposed as dynamic slash commands on both surfaces, so the skill
loads as `/nocode`; the text after the command is the question.

## Quick Reference

`/plan` vs `/nocode` — both are mode skills that restrain the agent from
running off and doing things, but they produce different deliverables:

| | `/plan` | `/nocode` |
|---|---|---|
| Deliverable | A markdown plan document saved under `.hermes/plans/` | A direct answer in the conversation |
| Purpose | Prepare a task for later implementation | Answer a question now, without side effects |
| File writes | Writes exactly one file — the plan | Writes nothing, ever |
| Typical trigger | "Plan how to build X" | "Just answer, don't do anything" |
| After the turn | The plan is meant to be executed later | The answer is the end state; nothing remains |

Choose `/plan` when the user wants a roadmap for work that will happen later.
Choose `/nocode` when the user wants information and explicitly does not want
anything done. `/nocode` is the stricter mode: it permits only read-only
research, while `/plan` may inspect the repo and writes the plan file as its
deliverable.

## Procedure

1. **Answer directly in the same turn.** No preamble, no plan, no
   "I'll set that up for you".
2. **Research when it improves the answer** — anything read-only:
   `web_search`, `web_extract`, `read_file`, `session_search`,
   `skills_list`, read-only terminal inspection.
3. **Do not mutate anything** unless the user explicitly asks in the same
   message: no `write_file`/`patch`, no installs (`pip`, `apt`), no
   `git commit`/`push`, no `rm`/`mv`, no config changes or service
   restarts, no side-effecting `execute_code`, no cron/memory/skill
   mutations.
4. **If answering genuinely requires an action** (e.g. inspecting a live
   service): explain briefly what you would do and why, ask permission,
   and wait.

## Pitfalls

1. **Treating "answer only" as "no research".** Read-only research is the
   point of the mode — don't answer from memory when a quick `web_search`
   would verify the facts.
2. **Slipping in a "helpful" action.** The deliverable is the answer;
   artifacts, installs, and "I went ahead and..." are failures of this
   mode.
3. **Asking permission unnecessarily.** If you can answer, just answer —
   only gate when an action is truly required.

## Verification

The mode succeeded when:

- The user has a complete, direct answer.
- No files were created or edited.
- No packages were installed.
- No config, cron, memory, or service state changed.
- If an action was required, permission was requested before doing
  anything.
