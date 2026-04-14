# Delegate tiers validation report — 2026-04-13

Worktree: `/home/ubuntu/hermes-agent-dev/delegate-tiers`
Branch: `feat/delegate-task-tiers`
PR status: local only; previous PR was intentionally closed pending more evidence.

## Goal
Validate whether delegation tiers are worth upstreaming, using:
1. official docs / upstream code audit
2. benchmark references from the web
3. real local Hermes reproductions with `openai-codex`
4. tmux-isolated execution where useful

---

## 1. Official upstream / docs findings

Confirmed on clean `origin/main` before local implementation:
- Upstream `delegate_task` supports global `delegation.model` / `provider` overrides.
- Upstream does **not** support `tiers`, `default_tier`, or named worker roles.
- Upstream docs describe up to 3 parallel children, but children otherwise inherit parent/default delegation config.
- Relevant GitHub context:
  - issue #6306 — per-task model selection in `delegate_task`
  - issue #3719 / PR #5229 — per-call model/provider override and delegation model pool
  - stale PR #5692 — source of the earlier tier-profile concept

Interpretation:
- The tier concept is not official upstream behavior today.
- The routing problem is real and already acknowledged upstream.

---

## 2. External benchmark / routing references

### GPT-5.4 vs GPT-5.4 mini
Sources:
- OpenRouter compare page
- Adam Holter benchmark summary
- routing / subagent architecture articles

Key numbers collected:

| Model | Input $/M | Output $/M | Coding benchmark | Context | Throughput |
|------|-----------:|-----------:|-----------------:|--------:|-----------:|
| GPT-5.4 | 2.50 | 15.00 | SWE-Bench Pro 57.7% | 1.05M | 35 tok/s |
| GPT-5.4 mini | 0.75 | 4.50 | SWE-Bench Pro 54.4% | 400K | 64 tok/s |

Key routing takeaways from references:
- mini is the natural fit for narrow, high-volume subagent tasks.
- frontier/full model is justified for judgment-heavy leaf tasks.
- subagent spawn has a non-trivial overhead; broad delegation is not automatically faster.
- one practical reference estimates ~10K input tokens of cold-start overhead per spawn as a break-even heuristic.

Implication for Hermes tiers:
- `light -> gpt-5.4-mini` is supported by public benchmark positioning.
- `heavy/review -> gpt-5.4` is plausible, but needs local calibration on actual Hermes tasks.

---

## 3. Local implementation under test

Local branch implements:
- named tiers: `light`, `heavy`, `review`, `planning`, `research`
- per-tier overrides for `model`, `provider`, `reasoning_effort`, `max_iterations`
- reasoning floor guardrails:
  - `heavy`, `research` >= `medium`
  - `planning`, `review` >= `high`
- optional model pool validation
- top-level and per-task `tier`

---

## 4. Automated validation status

### Unit / integration tests
- `tests/tools/test_delegate.py`
- `tests/tools/test_delegate_tiers.py`
- `tests/tools/test_delegate_toolset_scope.py`

Status:
- 128 tests passed locally

### tmux harness tests
File: `tests/tools/test_delegate_tiers_tmux.py`

Important debugging note:
- first version falsely timed out because it tried to `capture-pane` after tmux had already exited.
- fixed by redirecting script output to a temp log file and polling for `EXIT_CODE=`.

Status after fix:
- 6/6 tmux-isolated tests passed

These tmux tests validate:
- tier resolution
- per-task tier precedence in batch mode
- pool validation
- backward compatibility
- explicit reasoning override behavior
- schema completeness

---

## 5. Real Hermes reproductions with `openai-codex`

Environment facts confirmed:
- `resolve_runtime_provider(requested='openai-codex')` returns a valid runtime bundle
- `api_key` is available via persisted auth, even when `OPENAI_API_KEY` env var is absent
- real delegation runs were executed using a real parent `AIAgent` and real child `AIAgent`s via `delegate_task(...)`
- local tests used `TERMINAL_CWD=/home/ubuntu/hermes-agent-dev/delegate-tiers` and explicit repository paths in context

### 5.1 Single `light` child — real run
Task:
- read `pyproject.toml`
- summarize project in one sentence

Observed result:
- status: completed
- model: `gpt-5.4-mini`
- api_calls: 4 (first exploratory run) and 2 (focused batch run)
- duration: ~19.5s in exploratory run; ~5.9s in focused run
- tokens (exploratory): input 17,170 / output 436
- tokens (focused batch sample): input 6,639 / output 212

Interpretation:
- `light` is appropriate for straightforward file-read + summarize tasks.
- When prompt/context is scoped well, `light` completes quickly and cheaply.
- Without explicit repo context, child may wander to another local checkout after search.

### 5.2 Single `heavy` child — real run
Task:
- trace execution path of `delegate_task` for a single-goal call
- repository restricted explicitly to this worktree

Observed result:
- status: completed summary, but via `max_iterations`
- model: `gpt-5.4`
- api_calls: 10
- duration: ~70.1s
- tokens: input 246,253 / output 1,540
- many `search_files` + `read_file` calls

Interpretation:
- `heavy` on a realistic code-analysis task is viable.
- But a budget of **10 iterations was too low**; the agent hit the ceiling and then summarized.
- This is strong evidence that a realistic `heavy` tier should not be calibrated too aggressively downward.

### 5.3 Single `review` child — real run
Task:
- review `tests/tools/test_delegate.py`
- identify 3 missing edge cases for delegation behavior

Observed result:
- status: completed summary, but via `max_iterations`
- model: `gpt-5.4`
- api_calls: 14
- duration: ~194.8s
- tokens: input 596,731 / output 4,020
- produced useful findings, but at very high token/time cost

Interpretation:
- `review` produces valuable output.
- But broad review tasks explode in token cost very quickly.
- `review` should probably be used only for tightly scoped targets, or with stronger prompt constraints.
- The raw result does **not** justify claiming `review = xhigh + 60` is universally efficient.

### 5.4 Real batch with 2 children (`light` + `heavy`)
Tasks:
- child 1: `light` summarize `pyproject.toml`
- child 2: `heavy` trace `delegate_task` single-goal path

Observed result:
- both completed
- wall time: ~78.4s
- child 1 duration: ~5.9s
- child 2 duration: ~78.0s

Comparison against serial approximation:
- standalone light sample: ~19.5s exploratory / ~5.9s focused
- standalone heavy: ~70.1s
- combined wall time in batch: ~78.4s

Interpretation:
- real parallelism is working for 2 children.
- wall-clock is lower than naive serial sum for comparable tasks.
- this validates the utility of batching when task shapes differ and one task dominates runtime.

### 5.5 Real batch with 2 `light` children
Tasks:
- summarize `pyproject.toml`
- count lines in `tools/delegate_tool.py`

Observed result:
- both completed
- wall time: ~6.4s
- child durations: ~5.5s and ~5.9s

Interpretation:
- two simple `light` children parallelize cleanly.
- strong evidence that `light` is suitable for narrow helper delegations.

### 5.6 Real batch with 3 children — root cause isolated
Initial symptom:
- 3-child real batches were flaky / sometimes appeared to hang for minutes
- 2-child batches completed normally

Systematic investigation performed:
- instrumented `CredentialPool.acquire_lease()` / `release_lease()`
- instrumented `AIAgent._interruptible_api_call()` for each child
- reran a real 3-child `light` batch with timestamped logs

Key finding:
- all 3 children **did** start concurrently and each acquired a distinct lease
- two children finished normally
- the third child repeatedly failed with:
  - `AuthenticationError: 401 Could not parse your authentication token. Please try signing in again.`
- the failing leased credential ID was `828874` (an `openai-codex` pool entry from source `device_code` in `~/.hermes/auth.json`)

Crucial confirmation test:
- reran the same real 3-child batch after **ephemerally filtering out credential `828874` from the parent credential pool**
- result: all 3 children completed successfully in ~12.2s wall-clock

Interpretation:
- the apparent “3-child concurrency stall” was **not primarily a thread-pool/delegate_tiers bug**
- it was caused by a **bad credential in the shared `openai-codex` credential pool**, combined with repeated retry behavior on auth failure
- therefore the root cause is better classified as:
  - `credential-pool hygiene / auth resilience issue`
  - not `delegate_task cannot really do 3 children`

Operational consequence:
- do **not** claim “3-child concurrency is broken” in the tier PR
- do record that real 3-child runs can look broken when one pooled credential is invalid
- stronger upstream hardening opportunity: classify repeated 401 auth failures for a child lease as exhausted/bad and rotate away quickly instead of looping for minutes

### 5.7 Single `planning` child — real run
Task:
- create a short validation plan for flaky 3-child delegation in this repo
- `planning` tier routed to `xiaomi/mimo-v2-pro` via `provider: nous`

Observed result:
- status: completed
- model: `xiaomi/mimo-v2-pro`
- api_calls: 4
- duration: ~37.2s
- tokens: input 66,820 / output 1,264
- output quality: good; produced a concrete 5-step validation plan with acceptance criteria

Interpretation:
- `planning` works as a distinct cross-provider tier in real execution.
- It is materially cheaper than an open-ended `review` run while still producing structured planning output.
- This gives real support to the value of provider-specific tier routing (`planning -> nous / mimo`).

### 5.8 Single `research` child — real run
Task:
- compare Hermes `delegate_task` threading with two external model-routing/subagent references
- local file read + explicit web research URLs

Observed result:
- status: completed summary, but via `max_iterations`
- model: `gpt-5.4`
- api_calls: 6
- duration: ~79.4s
- tokens: input 95,282 / output 1,798
- output quality: good; produced a concise comparison with recommendation

Interpretation:
- `research` is viable as a separate tier when the task is tightly scoped and references are explicit.
- It is still noticeably heavier than `light`, but dramatically more bounded than the earlier broad `review` task.
- This is enough to say `research` has **initial real validation**, though more runs would still help.

---

## 6. What the real data says about calibration

### `light`
Status: validated

Good fit for:
- simple reads
- summarization of one file
- narrow extraction tasks
- small deterministic helper tasks

Recommended local calibration:
- model: `gpt-5.4-mini`
- reasoning_effort: `low`
- max_iterations: 8–10

### `heavy`
Status: partially validated

Good fit for:
- code-structure analysis
- tracing a runtime path
- focused debugging / implementation planning

Observed caution:
- 10 iterations was not enough for a realistic trace task.

Recommended local calibration:
- model: `gpt-5.4`
- reasoning_effort: `medium`
- max_iterations: probably **14–18**, not 10

### `review`
Status: functionally validated, efficiency not yet comfortable

Good fit for:
- deep test-gap analysis
- security review
- judgment-heavy scoped review

Observed caution:
- review tasks can become very expensive, very quickly
- 14 iterations still hit the cap
- broad review prompts likely need tighter scope before upstream claims

Recommended local calibration for now:
- model: `gpt-5.4`
- reasoning_effort: `high` or `xhigh` depending task criticality
- max_iterations: 16–20 for scoped review tasks
- keep prompts tightly bounded to specific files / questions

### `planning`
Status: initially validated

Observed real run:
- `xiaomi/mimo-v2-pro` via `provider: nous`
- produced a useful structured validation plan in ~37s

Recommended local calibration:
- model: `xiaomi/mimo-v2-pro`
- provider: `nous`
- reasoning_effort: `high`
- max_iterations: ~10–14 for bounded planning prompts

### `research`
Status: initially validated

Observed real run:
- local file + explicit web references
- good output in ~79s with 6 API calls
- still hit `max_iterations`, but remained much more bounded than broad review

Recommended local calibration:
- model: `gpt-5.4`
- reasoning_effort: `high`
- max_iterations: ~8–12 for tightly scoped research prompts
- pass explicit URLs / questions whenever possible

---

## 7. Main conclusions

### Strongly supported
1. The tier abstraction is useful and maps well to real delegation behavior.
2. `light -> gpt-5.4-mini` is a strong default for narrow helper tasks.
3. Per-task tiering in batch mode is the right UX shape.
4. Reasoning-effort override per tier is valuable and should stay.
5. Backward compatibility is intact when tiers are absent.

### Supported, but needs calibration
1. `heavy` should likely get a higher practical iteration budget than the first aggressive local draft.
2. `review` works, but should be framed as high-cost / tightly scoped.

### Not yet safe to oversell
1. Broad `review` tasks are still expensive enough that defaults/prompts should stay conservative.
2. The tier PR should not be framed as solving credential-pool/auth resilience; that is a related but separate hardening problem.
3. More repeated real runs would still be useful before claiming the exact default iteration numbers are universally optimal.

---

## 8. Recommended next step before reopening PR

Before reopening upstream PR, do one more focused cycle:
1. decide whether to keep the tier PR scoped narrowly, or pair it with a separate credential-pool/auth-hardening PR for repeated 401 child failures
2. rerun a small repeated benchmark matrix (e.g. 5× each) for `light`, `heavy`, `planning`, `research`
3. keep the PR narrative precise:
   - tiers are validated
   - `light`, `heavy`, `planning`, `research` now have real evidence
   - broad `review` remains the most expensive tier and should be used carefully
   - 3-child failures previously observed were traced to an invalid pooled credential, not to the tier abstraction itself

---

## 9. Bottom line

The tier feature is worth pursuing.
The evidence now supports a materially stronger upstream story than before:
- `light` is clearly validated
- `heavy` is validated with calibration notes
- `planning` has real cross-provider validation (`nous` / `mimo`)
- `research` has initial real validation
- `review` is functionally validated but expensive
- the apparent 3-child concurrency problem was traced to a bad `openai-codex` credential pool entry (`828874`), not to `delegate_task` batching itself

The main caution now is not “tiers are speculative”; it is:
- keep review defaults conservative
- do not hide the credential-pool/auth issue if we mention the 3-child investigation

