# Blueprint + Plan — Compressed Agent Format

A build plan written for a human reads as prose and costs a subagent hundreds of
tokens to parse into intent. This is the machine-facing alternative: a single
fenced block, dense, unambiguous, and diffable. It is not a pretty format and it
is not meant to be — it is a wire format between agents.

Two artifacts, always in this order:

1. **BLUEPRINT** — the invariant truth about the codebase. Facts only, each with
   evidence. Written once per investigation, reused by every subagent.
2. **PLAN** — the mutable work queue derived from the blueprint. One line per
   job, directory-scoped, checkbox-tracked.

The blueprint is what stops a subagent re-deriving the repo from scratch. The
plan is what stops it doing work that is already done.

## Grammar

Field-tagged, newline-delimited, `|`-separated. No JSON (brace noise costs
tokens and invites truncation mid-object). No markdown tables (column padding is
pure waste). Unknown tags are ignored by readers, so the format extends without
breaking older consumers.

```
@<TAG> <field>|<field>|...
```

Rules an agent must follow when writing one:

- **Every claim carries evidence or it does not go in.** `path:line` or a command
  and its exit code. An assertion with no evidence marker is a guess and must be
  tagged `?` so downstream agents distrust it.
- **Absence is never asserted from a grep.** Use `@GAP` only after reading the
  files. A zero-match search is a hypothesis; write it as `@GAP ...|conf=low`.
- **One scope per job line.** If a change spans two directories it is two jobs.
- **Never restate the blueprint inside a job.** Jobs reference blueprint ids.

## Template

````markdown
```agentplan
@META id=BP-2026-0805-01|repo=<name>|head=<sha7>|branch=<branch>
@STACK <lang/runtime>|<pkg-mgr>|<test-runner>|<ci>
@GATE build=<cmd>|test=<cmd>|lint=<cmd>|types=<cmd>
@ENTRY <what runs first>|<path:line>
@FACT F1 <invariant that constrains the work>|<path:line>
@FACT F2 <second invariant>|<path:line>
@RISK R1 <what breaks if F1 is violated>|blast=<narrow|wide>
@GAP G1 <genuinely missing thing>|conf=<high|med|low>|checked=<paths read>
@DONE D1 <thing that looks missing but ships already>|<path:line>
@BAN <paths/actions no agent may touch>

@JOB J1 scope=<one dir>|goal=<imperative, one line>|needs=F1,G1|gate=<cmd>
@JOB J2 scope=<one dir>|goal=<...>|needs=J1|gate=<cmd>
@SEQ J1->J2|J3 par J4
@EXIT all JOB=[X] && GATE(test)=0 && GATE(types)=0
```
````

Status is mutated in place on the `@JOB` line: `[ ]` pending, `[/]` active,
`[X]` verified, `[!]` escalated. The plan block IS the state; there is no second
tracker to fall out of sync.

## Worked example

Real, from this repository, produced by the investigation that preceded this
file:

```agentplan
@META id=BP-2026-0805-01|repo=hermes-agent|head=36cb5ae|branch=main
@STACK python3.10+ts|uv+npm|pytest+vitest|github-actions
@GATE build=npm run check|test=scripts/run_tests.sh|lint=ruff check .|types=npx tsc --noEmit
@ENTRY skill loader reads skills/**/SKILL.md frontmatter|tools/skills_tool.py
@FACT F1 skill description hard-capped 60 chars by create gate|tools/skill_manager_tool.py
@FACT F2 skills/**/*.ts is outside every npm workspace, so CI never typechecks it|package.json:workspaces
@FACT F3 .gitignore patterns without a leading slash match at ANY depth|man gitignore
@RISK R1 unanchored generic filenames silently swallow future tracked docs|blast=wide
@GAP G1 no CI job typechecks skill-shipped TypeScript|conf=high|checked=.github/workflows/*.yml
@DONE D1 .pytest_cache/.mypy_cache/.ruff_cache already ignored|.gitignore:36-43
@BAN .env|*.pem|other profiles under ~/.hermes/profiles/

@JOB J1 [X] scope=./|goal=anchor generic ignore patterns with leading slash|needs=F3,R1|gate=git check-ignore -v
@JOB J2 [X] scope=skills/software-development/surgical-orchestration|goal=make refs compile strict and run|needs=F2|gate=npx tsc --noEmit --strict
@SEQ J1 par J2
@EXIT all JOB=[X] && GATE(types)=0
```

## Why each rule exists

| Rule | Failure it prevents |
|---|---|
| Evidence on every fact | Agent "remembers" an API that never existed and codes against it |
| `@DONE` section | Rebuilding a shipped feature because grep missed the renamed symbol |
| `conf=` on gaps | A low-confidence guess propagating as settled fact through three subagents |
| One scope per job | Cross-directory edits that no single verifier can review |
| `@GATE` as commands | "Tests pass" claimed without a command that returned 0 |
| `@BAN` | Secrets read, other profiles clobbered |
| Status in the plan block | A separate tracker that drifts from reality |

## Handing it to a subagent

Pass the blueprint block plus the one `@JOB` line. Nothing else. The subagent
gets scope, goal, prerequisites and its verification command in ~40 lines, and
has no path to context it should not have.

The `@BAN` line travels with every mission — it is the only part of the plan a
subagent may never override.
