# Prompt growth: what learning costs every later API call

Arm A ([#122544](https://github.com/NousResearch/hermes-agent/pull/122544)) measures what the learning loop
writes. `prompt_growth.py` measures what those writes cost afterwards. Every skill the background review writes
is listed in the skills index of every later session until the curator archives it, and `MEMORY.md` and
`USER.md` go into every session too. It is the cost side of
[#96704](https://github.com/NousResearch/hermes-agent/issues/96704).

It runs offline and is deterministic: no model calls, no network, no credentials. It drives Hermes' own code
against a throwaway `HERMES_HOME` and `HOME`, created per run and removed afterwards.

| step | code path |
|---|---|
| fresh install | `tools.skills_sync.sync_skills()` seeds the bundled skills |
| learning | `tools.skill_manager_tool.skill_manage(action="create")` with the write origin bound to the background-review fork, as `run_agent._spawn_background_review` binds it. Every write passes the fork's guards, the 60-character description budget and the ledger, and is recorded curator-managed (`created_by: agent`) |
| the next session | `agent.prompt_builder.build_skills_system_prompt()` for a default CLI session (`toolsets: [hermes-cli]`, plugins discovered before the tool snapshot as `agent/agent_init.py` does), in a fresh interpreter per checkpoint |
| memory | `tools.memory_tool_store.MemoryStore.add()` until the store refuses, then a fresh store's frozen system-prompt snapshot |

## Running it

From the repository root:

```bash
scripts/run-in-hermes-env python evals/skills/prompt_growth.py --out <dir>
uv run --no-project --with "tiktoken>=0.9,<1" python evals/skills/retokenize.py <dir>
```

The first command makes two full, independent runs (about five minutes on a 4-core Linux machine) and writes `results.json`, `results.md` and every
rendered block to `<dir>`. The second recounts those blocks with `o200k_base` in a throwaway environment, so the
Hermes environment is never modified. `--checkpoints` sets the learned-skill counts to measure (default
`0,10,25,50,100,200,400`).

## Results

Measured on `main` at `90d6b98f` (Ubuntu 24.04, x86_64, Python 3.14.9, a bare install):

| learned skills | index rows | `o200k_base` tokens | Hermes estimate | share of 32k | share of 128k |
|---:|---:|---:|---:|---:|---:|
| 0 (fresh install) | 39 | 1,015 | 1,084 | 3.1% | 0.8% |
| 10 | 49 | 1,211 | 1,278 | 3.7% | 0.9% |
| 25 | 64 | 1,474 | 1,566 | 4.5% | 1.1% |
| 50 | 89 | 1,944 | 2,048 | 5.9% | 1.5% |
| 100 | 139 | 2,875 | 3,000 | 8.8% | 2.2% |
| 200 | 239 | 4,914 | 4,980 | 15.0% | 3.7% |
| 400 | 439 | 8,989 | 8,940 | 27.4% | 6.9% |

Each learned skill adds **19.95 tokens** to the system prompt of every later session, on every API call of that
session (least squares on `o200k_base`, or 19.64 by Hermes' own estimate). The two counts agree within
7% at every checkpoint. At capacity, `MEMORY.md` and `USER.md` together cost **941 tokens** and stop
there: the store refuses the next write (after 18 and 11 entries of this length).

What this shows:

- Memory doesn't get more expensive as the agent learns. The character limits
  (`hermes_cli/config_defaults.py:1304-1305`) are hard caps on writes.
- The skills index does, linearly. Every learned skill stays listed in every new session
  (`agent/prompt_builder.py:1303`, "NEVER drop entries") until the curator archives it, by default after
  30 days unused (`hermes_cli/config_defaults.py:1488`). In practice the index tracks the skills used in the
  last month, so 400 is a stress point, not a typical profile.
- Where the prefix is cached this is mostly cache-read cost. Where it isn't (a cold cache, or a local server
  without prefix reuse) it is prefill on every call.
- A learned skill pays for itself only if the tokens and turns it saves in the sessions that use it exceed
  about 20 tokens times every API call of every session that doesn't. That is the break-even line any outcome
  result from #96704 has to clear.

What it doesn't show: whether a larger library makes the agent choose worse. That needs task outcomes.

## How the numbers are checked

- Two full runs in fresh sandboxes must agree exactly, or the script exits non-zero.
- Every run checks three invariants: every create succeeded, the index grew by exactly one row per learned
  skill, and every learned skill is curator-managed (`created_by: agent`, which `skill_manage` records only for
  the review fork's writes, `tools/skill_manager_tool.py:775`).
- Those invariants were shown to fail. With `skills.write_approval: true` every create "succeeds" while only
  staging the write, and the row invariant fails. With an `environments: [s6]` skill in the corpus, on a host
  without s6, the row invariant fails too, which is why the corpus excludes gated skills. Without the review
  fork's write origin the creates are recorded `created_by: learn`, and the curator-managed invariant fails.
- `tests/evals/test_skill_prompt_growth.py` runs a small version end to end (0, 2 and 5 learned skills, two
  repeats).
- Nothing is written outside the sandbox. The worker's `HOME` stays empty and the real Hermes home is never
  read or written.

## Conditions and limits

- A bare install: 20 tools in 11 toolsets, no provider keys or service credentials. Service-gated tools
  follow what a host has configured, and conditional skills follow them, so the number of listed bundled skills
  can differ by a few rows elsewhere. Here 39 of the 44 bundled skills are listed, and the others are
  macOS-only or need the kanban toolset. The per-skill slope doesn't depend on this.
- The default posture only. Focus mode (`agent/coding_context.py`) demotes categories to a names-only line,
  which costs less. It isn't measured.
- Learned skills are represented by the 106 shipped optional skills that carry no platform, environment, app
  or toolset gate (real names, real descriptions within the 60-character budget), cycled with a `-vN` suffix
  past 106. Names the agent writes may be longer or shorter. A row costs its name plus a description of at
  most 60 characters.
- Measured on the cold path. On current `main` a build served from the on-disk snapshot can list an
  `environments:`-gated skill that the cold scan hides. Three open PRs fix this
  ([#50316](https://github.com/NousResearch/hermes-agent/pull/50316),
  [#50335](https://github.com/NousResearch/hermes-agent/pull/50335),
  [#128038](https://github.com/NousResearch/hermes-agent/pull/128038)). The learned corpus holds no gated
  skill, so the numbers above are unaffected.
- The background review's own tokens (the fork that writes these skills) aren't measured here.
