---
name: festival
description: "Plan and execute multi-session work with fest and camp."
version: 1.0.0
author: Lance Rogers (lancekrogers)
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Planning, Workflow, Project-Management, Campaign, CLI, Long-Running]
    category: software-development
    related_skills: [subagent-driven-development]
---

# Festival Skill

Festival is a pair of external CLIs, `fest` and `camp`, that keep a plan for
long-running work in ordinary files inside a git repository, so a session can be
resumed by a different agent on a different day. This skill covers installing
them and driving the execution loop through the `terminal` tool. It does not
cover planning a project from scratch, which the `fest-planning` skill in the
Festival tap handles.

This skill file is MIT like the rest of this repository; the `fest` and `camp`
binaries it points at are Apache-2.0.

## When to Use

Use this skill when:

- Work spans more than one session and the plan has to survive the chat history.
- The user already has a campaign directory, or asks for one, and wants state
  tracked in files they own rather than in a transcript.
- A `festivals/` directory, a `fest.yaml`, or an `AGENTS.md` describing a
  campaign appears in the working tree.
- The user names `fest`, `camp`, a festival, or a campaign.

Do not use it for a single-session task. A plan that outlives its own execution
is overhead.

## Prerequisites

Install the binaries first. A skills tap teaches the vocabulary, it does not
install the CLIs. Pick one, invoked through the `terminal` tool:

```bash
brew install --cask Obedience-Corp/tap/festival
npm install -g @obedience-corp/festival
curl -fsSL https://raw.githubusercontent.com/Obedience-Corp/festival/main/install.sh | bash
```

Festival publishes 12 skills as a Hermes tap. This one is a pointer; the tap is
the full set:

```bash
hermes skills tap add Obedience-Corp/festival
hermes skills install Obedience-Corp/festival/skills/festival-intake --yes
hermes skills install Obedience-Corp/festival/skills/fest-planning --yes
hermes skills install Obedience-Corp/festival/skills/fest-execution --yes
hermes skills install Obedience-Corp/festival/skills/campaign-commit --yes
```

Full setup notes, including the commit guard hook and Hermes Desktop, are at
https://docs.fest.build/getting-started/agents/hermes/

## How to Run

Start the session at the campaign root, so Hermes picks up the campaign's
`AGENTS.md` from that git root:

```bash
hermes --in /path/to/my-campaign chat -q "Run fest next and do the task it prints"
```

`--in` is a global option and goes before the `chat` subcommand.

## Quick Reference

| Command | What it does |
| --- | --- |
| `camp init` | Create the campaign layout and its `AGENTS.md` |
| `fest intro` | Print the getting-started guide for the CLI |
| `fest next` | Print the next actionable task, with its context |
| `fest task completed --yes` | Mark the current task done without prompting |
| `fest task blocked --reason "..."` | Record a blocker and stop |
| `fest commit -m "..."` | Commit with the festival traceability trailer |
| `fest validate` | Score the festival structure |
| `fest status` | Show phase, sequence, and task progress |
| `fest workflow approve` | Human-only. Do not run it for the user |

## Procedure

1. Confirm both binaries answer (see Verification). If either is missing, stop
   and install rather than guessing at file layout.
2. From the campaign root, change into `festivals/active/<festival>`. `fest next`
   fails at the campaign root.
3. Run `fest next` and do exactly the task it prints, no more.
4. Run `fest task completed --yes`.
5. Run `fest commit -m "<what changed and why>"`.
6. Run `fest validate` and confirm the score did not drop.
7. Run `fest next` again and repeat from step 3.
8. When `fest next` returns a phase gate awaiting approval, report what was done
   and stop. Approval is the operator's.

## Pitfalls

- `fest next` only works inside a festival directory. At the campaign root it
  exits with `not inside a festival`. Navigate into
  `festivals/active/<festival>` and retry.
- Phase gates are human checkpoints. Submit the gate, report, and stop. Do not
  run `fest workflow approve` on your own work.
- Do not drive a campaign with `hermes -z`. Its terminal tool runs in the home
  directory and ignores both the working directory and `--in`, so every
  festival command lands in the wrong place. Use `hermes chat` instead.
- A `docker` terminal backend has neither `fest` nor `camp` on its PATH; the
  default image (`nikolaik/python-nodejs:python3.11-nodejs20`) fails every
  Festival command with `command not found`. Bake the binaries into your image
  or stay on the local backend.
- Leave `AGENTS.md` as the campaign's only context file. A `.hermes.md`
  outranks it and replaces it rather than layering on top, and the session then
  loses the campaign instructions entirely.
- A tap contributes nothing to `hermes skills browse` and nothing to
  `hermes skills search`. Install tap skills by their exact name; the full list
  is in the tap's own README at
  https://github.com/Obedience-Corp/festival/tree/main/skills

## Verification

```bash
fest --version && camp --version && fest validate
```

Both binaries print a version, and `fest validate` prints a score for the
current festival.
