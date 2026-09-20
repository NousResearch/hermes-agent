---
name: discrete-decision-gate
description: "Gate agent routing with typed decision models."
version: 0.1.0
author: Ron Malouin (rpmalouin), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [routing, classification, gates, delegation, decision, offline]
    category: autonomous-ai-agents
    related_skills: [hermes-agent, dynamic-workflow, subagent-driven-development]
---

# Discrete Decision Gate Skill

Asks a decision model exactly one typed question about program state and returns one label
(with confidence where the backend provides it). Three backends: TypeSafe Jev via
`/v1/systemone`, any OpenRouter chat model, or a local Ollama model as the offline fallback.
The gate **never fabricates a verdict** - a failed call returns `status: unavailable` and the
caller's own deterministic policy decides. It is not a general LLM call: single label, no text.

## When to Use

- Route an incoming task to an engine: `bash_direct` / `crg_first` / `direct_dsh`
- Score a code-review-graph blast radius as `low` / `high` before auto-applying a change
- Decide the next coding-harness loop action: `complete` / `retry` / `abort`
- Any binary/ternary "which bucket is this?" gate that should be cheaper and more predictable
  than a full chat turn

Don't use for: free-form text, summaries, code generation, multi-answer output, or decisions
where a wrong label costs less than the call itself.

## Prerequisites

- `python3` 3.10+ (`str | None` annotations). Standard library only - `urllib`, `json`,
  `argparse`. Nothing to install.
- One backend credential, resolved in this order by `--backend auto`:
  1. `OPENROUTER_API_KEY` - **TypeSafe Jev through OpenRouter** (`POST
     https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13`). Needs no TypeSafe
     account; this is the recommended default and what `--backend jev` selects.
  2. `OPENROUTER_API_KEY` - any chat model (default `openai/gpt-4o-mini`), the older fallback.
  3. Local Ollama (`http://localhost:11434`, default `qwen3:14b`) - no key, offline, slower.
- TypeSafe 1P (`TYPESAFE_API_KEY`, `POST https://api.typesafe.ai/v1/systemone`) is **opt-in only**:
  `--backend typesafe` for a single run, or `JEV_BACKEND=typesafe`. It is deliberately NOT in the
  auto chain - a free or rate-limited key in `~/.hermes/.env` must not silently become the gate
  path and throttle every verdict into fail-closed. `decision_gate.py doctor` lists the 1P backend
  only when its key is actually set, and an explicit pin with no key reports
  `TYPESAFE_API_KEY not set (backend pinned to typesafe)` rather than a generic no-keys message.
- Naming note: OpenRouter has **no** `typesafe/jev-latest` alias (`Model typesafe/jev-latest does
  not exist`); `typesafe/jev-1.13` is the only Jev id there and it serves the newest snapshot -
  every response names it (`typesafe/jev-1.13-20260917`). `--model latest`,
  `--model typesafe/jev-latest` and `JEV_MODEL=latest` all resolve to that pinned id, so the
  habitual spelling works instead of erroring.
- `scripts/jev_gate.py` speaks the Jev **decisions** protocol instead: `TYPESAFE_API_KEY`
  (`POST api.typesafe.ai/v1/systemone`) or `OPENROUTER_API_KEY`
  (`POST openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13` - no TypeSafe account
  needed). It deliberately has no chat-model fallback: a prose model answering a one-label
  question is a different tradeoff, and `decision_gate.py` covers that case.
- Keys normally live in `~/.hermes/.env`, **not** in the shell environment: a fresh `terminal`
  shows `OPENROUTER_API_KEY` empty. Load them before invoking:
  `set -a; . ~/.hermes/.env; set +a`.
- `references/jev-api.md` holds the Jev request/response contract and why it is not a
  `/chat/completions` model.

## How to Run

Run through `terminal` with the skill-relative script path, after loading keys:

```bash
set -a; . ~/.hermes/.env; set +a
python3 scripts/decision_gate.py triage --intent "rename a helper in one file"
python3 scripts/decision_gate.py blast-radius --summary "touched 6 files: models/auth.py, 31 dependents"
python3 scripts/decision_gate.py harness --exit-code 1 --tail "$(tail -c 1500 pytest.log)"
python3 scripts/decision_gate.py ask --state "..." --question "..." --choices a,b,c --describe 'a=...;b=...'
python3 scripts/decision_gate.py doctor
```

Bounded harness loop with the gate as judge (harness-agnostic - `{prompt}` is replaced by the
shell-quoted prompt, so it works for dsh, OpenCode, Codex, a make target, or a test script):

```bash
python3 scripts/gated_harness_loop.py --task "make the failing cart test pass" \
  --cmd 'node /path/to/checkout/apps/cli/lib/bin.js --profile headless {prompt}' \
  --workdir /path/to/repo --max-loops 2 --log /tmp/run.json
```

Loop result is the gate's, not the harness's: `complete` (0), `retry` re-runs with the failure
tail appended, `abort` (3) stops on an environment/credential failure, an unavailable gate stops
without declaring anything, and a harness command that is not found exits `4` before any call.
The per-attempt verdicts are appended to the log file.

`scripts/jev_gate.py` is the same gate in protocol shape - rules-first triage, blast radius,
diff-vs-intent, harness verdict - with `--json` for callers:

```bash
python3 scripts/jev_gate.py triage  --intent "clean up utils.py so it reads better"
python3 scripts/jev_gate.py blast   --summary "<code-review-graph impact radius output>"
git diff | python3 scripts/jev_gate.py diff --intent "rename _normalize to normalized"
python3 scripts/jev_gate.py harness --exit-code 0 --tail "$(tail -c 2000 pytest.log)" \
    --attempt 1 --max-attempts 3 --expect-file dist/report.json
```

Labels stay canonical (`bash_direct|direct_dsh|crg_first`, `low|high`, `match|mismatch`,
`complete|retry|abort`); exceptional causes ride in `error_kind`
(`no_key|transport|http|schema|empty_input|environment|artifact_missing|max_attempts`), so schema
drift is distinguishable from an outage instead of both looking like a permanent fail-closed
verdict. Four checks never spend a call: the rule tiers, the credential/harness failure list, the
bounded-retry ceiling, and the artifact check (exit 0 with a missing or empty `--expect-file` is
`retry`, not `complete`).

## Quick Reference

| Flag | Effect |
|---|---|
| `--backend auto\|typesafe\|openrouter\|ollama` | Pins a backend; `auto` walks typesafe -> openrouter -> ollama |
| `--model <id>` | Pins the model id (recommended before wiring a gate into automation) |
| `--timeout <s>` | Per-request timeout (default 20) |
| `--json` | Machine-readable result on stdout |
| `--quiet` | Exit code only |
| `--no-fallback-policy` | Omit the deterministic default from the result |

Exit codes: `0` verdict returned, `3` backend unavailable or off-schema answer. A `3` target on
stdout reads `UNAVAILABLE via <backend>: <reason>` plus `policy: use deterministic default '<x>'`.

Preflight with `python3 scripts/decision_gate.py doctor`: it prints, per backend, whether the key
is present, whether the endpoint answered, one live verdict, and the measured latency, and exits
`0` when at least one backend answers. It reads `~/.hermes/.env` itself, so a key that is missing
from the shell environment still counts.

`python3 scripts/jev_gate.py doctor` is the Jev-specific preflight: it asks OpenRouter which
provider snapshot it advertises for the pinned id (`TypeSafe | typesafe/jev-1.13-20260917`, price,
context, 1-day uptime) and then makes one live probe, printing the served model, latency and cost.
That resolved snapshot is what "latest" means on this route.

## Procedure

1. Load keys and run the preset that matches the decision (see How to Run). Done when: exit code
   is `0` with a label on stdout, or exit `3`.
2. Read confidence when the backend returns it (Jev answers with a full probability map). Treat
   `confidence < 0.5` as unavailable for high-stakes gates and route to a human instead of
   acting. Done when: every code path that acts on the label is either backed by
   `confidence >= 0.5` or by the staked default.
3. On exit `3`, apply the preset's deterministic default and record `backend` plus `reason`. The
   default is a policy choice, not a model verdict - never report it as one. Done when: the
   downstream action is attributable to the policy, not to a claimed model answer.
4. Before wiring a gate into automation (`cronjob_manage`, a harness loop, auto-apply), pin the
   model with `--model` and confirm one live `status: ok` call. Done when: the pinned call
   succeeded and the label matched the expected bucket for a known input.

## Presets and Failure Policy

| Preset | Choices | Deterministic default when the gate is unavailable |
|---|---|---|
| `triage` | bash_direct, crg_first, direct_dsh | `bash_direct` (low stakes: fail open, cheapest path) |
| `blast-radius` | low, high | `high` (high stakes: fail closed, needs signoff) |
| `harness` | complete, retry, abort | `abort` (high stakes: never auto-declare success) |

Rule: the default is chosen from the stakes, not from the label list. Never make the default
`low` or `complete` - a silent gate failure must not be able to auto-apply a change or end a
loop.

## Pitfalls

1. **Jev is a decisions model, not a chat model - and it IS reachable on OpenRouter.** Verified
   2026-09-20: `POST https://openrouter.ai/api/alpha/decisions` with a flat body
   `{model, state, questions}` answers 200 as `typesafe/jev-1.13-20260917` in ~0.25-0.43s for
   ~$1.6e-05 (see `references/jev-api.md`). The wrong paths, all measured:
   `/chat/completions` with `typesafe/jev` -> HTTP 400 `is not a valid model ID`;
   `/api/v1/decisions` -> HTTP 404 (the route lives under `/api/alpha`, not `/api/v1`);
   wrapping the body in `decisionsRequest` -> HTTP 400 `invalid_type ... path: ["model"]`.
   `GET /api/v1/models` (447 entries) contains NO `typesafe/*` id and
   `GET /api/v1/models/typesafe/jev-1.13` 404s even though the route works - so never preflight by
   looking the model up in the catalog; make a live call. Prefer TypeSafe 1P (`/v1/systemone`)
   when a `TYPESAFE_API_KEY` exists; OpenRouter is the working fallback.
   Related: **a `noul` answer is not `probabilities.true`.** A yes/no answer comes back as
   `{"type": "noul", "noul": 0.01}` - flat, no probability map, no confidence. Reading
   `answers.x.probabilities.true` always yields 0.0, which silently turns a diff-vs-intent gate
   into a permanent `mismatch`. Read `answers.x.noul`.
2. **The silent-failsafe anti-pattern.** `except Exception: return choices[0]` turns a total
   backend failure into confident-looking labels. Measured with a live key on the pasted version
   of this gate: every triage intent returned `bash_direct`, an auth-model change with 31
   dependents returned `risk: low` (auto-apply), and a run that died on
   `could not read Username for github.com` returned `verdict: complete` (loop ends). Return
   `unavailable` plus a staked default instead.
3. **Tiny `max_tokens` truncates to empty.** `max_tokens: 5` yields empty content on models that
   emit a preamble; the script uses 8 and requires an exact single-label match, so an ambiguous
   answer becomes `unavailable` rather than a coin flip.
4. **Reasoning models on Ollama** need `think: false` and **no** `format: json` (qwen3 returns a
   bare `{}` with a JSON schema attached). The script sets `think: false`.
5. **Model ids drift.** Free and renamed ids disappear; a 400 is a gate outage, not a label.
   Re-run the preflight in Procedure step 4 after any model change.
6. **Latency budget.** Measured: OpenRouter chat 0.5-1.4s, local `qwen3:14b` 0.06-3.5s, Jev
   claims 70-500ms. Do not put a chat-model gate in a per-item inner loop.

## Verification

- Run `triage` on two contrasting intents (a single-file rename vs. a refactor across 14
  modules). Completion: different labels. Identical labels for opposite inputs means the gate is
  failing silently - check the exit code, which will be `3` and not `0`.
- `--backend typesafe` with no `TYPESAFE_API_KEY` must print
  `UNAVAILABLE ... policy: use deterministic default 'high'` on a `blast-radius` run and exit `3`.
- `--backend openrouter --model typesafe/jev` must report the HTTP 400 in `reason` and exit `3` -
  never a label.
- `scripts/run_tests.sh tests/skills/test_discrete_decision_gate_skill.py -q` passes; the test
  suite mocks `urllib.request.urlopen`, so no live key or network is required.
- `python3 scripts/decision_gate.py doctor` exits `0` with at least one backend reporting
  `status: ok`; a backend with no key reports `unconfigured` instead of failing the run.
- `jev_gate.py` over the OpenRouter Decisions route, one battery with every assertion passing:
  rule tiers hit `bash_direct` / `direct_dsh` / `crg_first` with no call; an ambiguous intent
  resolves through Jev in 283-339ms; a leaf summary -> `low` and auth+schema with 31 dependents ->
  `high` (both confidence 1); a matching patch -> `match` (noul 0.96) while a patch renaming the
  wrong symbol -> `mismatch` (noul 0.35) and a patch that nukes CI + docs for a "fix the cart
  test" intent -> `mismatch` (noul 0.02); masked `1 failed` under exit 0 -> `retry`; a git
  credential failure -> `abort` (`error_kind: environment`) with no call; an empty diff or empty
  CRG summary -> the fail direction with `empty_input`; no keys at all -> each gate's own fail
  direction (`bash_direct` / `high` / `mismatch` / `abort`), never a fabricated label.
- Drive a harness whose command you know succeeds, e.g.
  `--cmd 'sh -c "echo all checks passed {prompt} >/dev/null"'`: the loop must exit `0` on a
  `complete` verdict and write one attempt to the log. Point `--cmd` at a nonexistent binary and
  it must exit `4` before asking the gate anything.
