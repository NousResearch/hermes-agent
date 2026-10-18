---
name: retrospective-bug-pattern-audit
description: Mine git history for recurring bugs, codify as rules.
version: 0.1.0
author: Ben Barclay (benbarclay), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [code-review, git-history, pattern-mining, lessons-learned, post-mortem]
    related_skills: [requesting-code-review, systematic-debugging]
---

# Retrospective Bug Pattern Audit

Mine a project's git history for recurring bug classes, codify each as an
imperative rule tied to the commit that proved it, and file the rules where
lesson-recall is automatic. Prevents re-discovering the same defect class in
code review after the fact. Does NOT replace live code review — it feeds it.

## When to Use

- A bug was fixed and you want sibling instances (\"fix one, sweep all\")
- Pre-merge audit of a branch with many fix commits
- Post-mortem after an incident: \"what pattern class did we miss?\"
- Trigger phrases: \"pattern mine the history\", \"recurring bug audit\",
  \"codify lessons\", \"what bugs keep coming back\"

## Prerequisites

- Git repository with the relevant history locally.
- A durable store for rules: project skill (for project-specific patterns) or
  Mnemosyne (for cross-project patterns). Prefer Mnemosyne for rules that apply
  beyond one repo (Go/SQL insert discipline, mock/test drift, config/code drift).

## How to Run

1. **Extract fix commits** through the `terminal` tool:
   ```bash
   git log --grep="fix" --format="%h %s%n%b---" --all
   ```
2. **Categorize** each fix by defect class: ID collision, FK violation,
   reconciliation double-count, test/mock drift, threshold guesswork,
   config/code drift, paper/live equivalence.
3. **Verify each class is real** — reproduce the original symptom against the
   pre-fix tree if possible, or ground it in the commit message + diff.
4. **Write one rule per class** as an imperative with the commit that proved it:
   ```
   When <trigger>, <imperative>. Evidence: <commit>.
   ```
5. **File rules** — project-specific into the project skill's `references/`;
   cross-project into Mnemosyne at importance >= 0.9.
6. **Sweep for siblings** — after codifying a rule, grep the repo for other
   instances of the same pattern and fix or flag them.

## Pitfalls

- **A fix commit is not a rule.** A commit that fixes a nil-check is a nil-check
  fix; the rule is \"check nil after every pointer deref.\" Distill the class,
  not the instance.
- **Don't codify without a proven incident.** A hypothetical pattern is a
  reference document; a codified rule needs a commit hash where it bit.
- **Filing is not the end.** A rule in a skill that no future session reads is
  ceremony. Verify the skill's description fires on the trigger prompts.

## Verification

After filing, walk one future trigger prompt (e.g. \"audit the close path\")
through skill recall — the codified rule should surface before you re-derive it.