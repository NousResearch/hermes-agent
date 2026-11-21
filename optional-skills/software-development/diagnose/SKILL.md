---
name: diagnose
license: MIT
version: 1.5.0
author: afonsoft
platforms: [linux, macos, windows]
description: "Root-cause analysis for hard bugs and regressions."
metadata:
  hermes:
    tags: [Debugging, Root-Cause, Troubleshooting]
  version: "1.5.0"
  visibility: public
  author: afonsoft
  url: https://github.com/afonsoft/skills
---

# Diagnose

A discipline for hard bugs and regressions. Skip phases only when explicitly justified. All questions and findings reported to the user must be in **Portuguese (pt-BR)**.

When exploring the codebase, use the project's domain glossary from `.claude/CONTEXT.md` (if it exists) to get a clear mental model of the relevant modules, and check `docs/architecture/` for decisions and ADRs in the area you are touching.

**Re-validation loop**: after every hypothesis, fix, or change, re-run the reproduction and the regression checks before declaring the bug resolved.

**Iron rule:** NO FIXES WITHOUT A RED LOOP AND A CONFIRMED ROOT CAUSE. A change made before Phase 1 produced a red signal and Phase 4 confirmed a hypothesis is a guess — and guessing under pressure is how one bug becomes three. This holds ESPECIALLY under time pressure, when the bug "looks simple", and after a previous fix already failed: those are exactly the moments the process pays off. Violating the letter of this process is violating the spirit of debugging.

## Pipeline

```text
Phase 0  Frame the bug          → symptom, expected vs actual, env, frequency, scope
Phase 1  Build a feedback loop  → ONE tight, red-capable command (the core of this skill)
Phase 2  Reproduce + minimise   → smallest scenario that still goes red
Phase 3  Hypothesise            → 3–5 ranked, falsifiable predictions
Phase 4  Instrument             → one probe per prediction, one variable at a time
Phase 5  Fix + regression test  → test at a correct seam, watch red → green
Phase 6  Cleanup + re-validate  → remove probes, re-run the original repro

   ↺ If the bug survives or mutates, return to Phase 3.
   ↺ If every hypothesis dies, see "Stuck — when hypotheses run out".
```

Entry points for non-code failures:

- The bug is the agent session itself → [Agent Self-Debug](#agent-self-debug--introspection-use-when-the-failure-is-the-agent-itself)
- The misbehaving system is an agent/AI workflow → [AI Workflow Diagnostic](#ai-workflow-diagnostic)
- The code "works" but quietly produces wrong/absent effects → [Silent-Failure Hunt](#silent-failure-hunt)

## Redact

This skill has you show commands, outputs, and captured artifacts. **Redact every secret first**: write `<REDACTED>` in its place. Build loops against env vars so credentials stay in the environment, not in what you show. Captured artifacts carry auth headers — quote only the lines that carry the signal.

If the redacted output is not enough to diagnose the bug, say so and ask the user (in Portuguese).

## When to Use

- User asks or mentions this skill in English (e.g., "use /diagnose", "run diagnose").
- O usuário pede ou menciona esta skill em português (ex.: "use /diagnose", "execute diagnose").
- The failure is the agent session itself (loops, context drift, repeated tool calls) → jump to [Agent Self-Debug](#agent-self-debug--introspection-use-when-the-failure-is-the-agent-itself).
- The misbehaving system is an agent or AI workflow → run the [AI Workflow Diagnostic](#ai-workflow-diagnostic).

## Stack playbooks

When the target codebase matches one of these stacks, load the matching playbook from `references/` before building the loop — they contain ready-made reproduction commands, debugger/profiler workflows, instrumentation snippets, and log-query cheat sheets:

- `references/dotnet-debugging.md` — .NET / ASP.NET Core / EF Core (dotnet-counters, dotnet-stack, dotnet-dump, dotnet-trace, EF `LogTo`/`TagWith`, journalctl/jq queries)
- `references/angular-debugging.md` — Angular (Angular DevTools profiler, `ng.*` console APIs, RxJS `tap`, change-detection profiling)
- `references/python-debugging.md` — Python (`breakpoint()`/`pdb`, pytest `--pdb`, py-spy, faulthandler, tracemalloc, asyncio debug)
- `references/react-debugging.md` — React (React DevTools profiler, why-did-you-render, re-render/effect bug patterns)

## Phase 0 — Frame the bug

Before building anything, pin down what "broken" means. A fuzzy symptom produces a fuzzy loop, and a fuzzy loop wastes every phase after it.

Capture — from the user, the bug report, or the evidence:

- [ ] **Symptom, verbatim** — the exact error message, wrong value, or timing. Quote it; do not paraphrase.
- [ ] **Expected vs actual** — what should happen vs what does happen.
- [ ] **Environment** — local / CI / staging / prod; OS, runtime, versions, config profile.
- [ ] **Since when** — first sighting; the deploy, commit, or data change it correlates with; or "always".
- [ ] **Frequency** — every run, flaky (roughly how often?), or once.
- [ ] **Scope** — one user / input / environment, or everything.
- [ ] **Recent changes** — code, dependencies, config, data, infrastructure.

If the report is thin, interview the reporter before touching code (in Portuguese):

```text
Antes de investigar, preciso enquadrar o bug:

1. O que deveria acontecer vs o que aconteceu? (mensagem de erro exata, se houver)
2. Onde acontece — local, CI, staging, produção?
3. Desde quando? Relacionado a algum deploy ou mudança recente?
4. Com que frequência — sempre, às vezes (quantas?), ou uma vez só?
5. Atinge um caso específico ou tudo?
```

**Output of Phase 0**: a one-sentence bug statement — *"In \<env\>, \<input/action\> produces \<actual\> instead of \<expected\>, since \<when\> (\<frequency\>)."* If you cannot write that sentence, keep interviewing — do not start Phase 1.

## Phase 1 — Build a feedback loop

**This is the skill.** Everything else is mechanical. If you have a **tight** pass/fail signal for the bug — one that goes red on *this* bug — you will find the cause; bisection, hypothesis-testing, and instrumentation all just consume that signal. If you don't have one, no amount of staring at code will save you.

Spend disproportionate effort here. **Be aggressive. Be creative. Refuse to give up.**

### Ways to construct one — try them in roughly this order

1. **Failing test** at whatever seam reaches the bug — unit, integration, e2e.
2. **Curl / HTTP script** against a running dev server.
3. **CLI invocation** with a fixture input, diffing stdout against a known-good snapshot.
4. **Headless browser script** (Playwright / Puppeteer) — drives the UI, asserts on DOM/console/network.
5. **Replay a captured trace.** Save a real network request / payload / event log to disk; replay it through the code path in isolation.
6. **Throwaway harness.** Spin up a minimal subset of the system (one service, mocked deps) that exercises the bug code path with a single call.
7. **Property / fuzz loop.** If the bug is "sometimes wrong output", run 1000 random inputs and look for the failure mode.
8. **Bisection harness.** If the bug appeared between two known states (commit, dataset, version), automate "boot at state X, check, repeat" so you can `git bisect run` it. For "which test creates this stray file/state?", `scripts/find-polluter.sh` runs test files one-by-one until the polluter appears.
9. **Differential loop.** Run the same input through old-version vs new-version (or two configs) and diff outputs.
10. **HITL bash script.** Last resort. If a human must click, drive *them* with `scripts/hitl-loop.template.sh` so the loop is still structured; captured output feeds back to you.

Build the right feedback loop, and the bug is 90% fixed.

### Tighten the loop

Treat the loop as a product. Once you have *a* loop, tighten it:

- **Faster?** Cache setup, skip unrelated init, narrow the test scope.
- **Sharper signal?** Assert on the specific symptom, not "didn't crash".
- **More deterministic?** Pin time, seed RNG, isolate filesystem, freeze network.

A 30-second flaky loop is barely better than no loop; a 2-second deterministic one is a debugging superpower.

### Non-deterministic bugs

The goal is not a clean repro but a **higher reproduction rate**. Loop the trigger 100×, parallelise, add stress, narrow timing windows, inject sleeps. A 50%-flake bug is debuggable; 1% is not — keep raising the rate until it is debuggable.

If the flake lives in a test that guesses timing with `sleep`/`setTimeout`, the flaky wait may *be* the bug — replace guessed delays with condition polling (`references/condition-based-waiting.md`).

### When you genuinely cannot build a loop

Stop and say so explicitly, in Portuguese. List what you tried. Ask for: (a) access to whatever environment reproduces it, (b) a redacted captured artifact (HAR file, log dump, core dump, screen recording with timestamps), or (c) permission to add temporary production instrumentation. Do **not** proceed to hypothesise without a loop.

```text
Preciso de um exemplo mínimo que reproduza o erro. Você consegue me fornecer:

1. O comando ou ação que dispara o problema.
2. A saída ou mensagem de erro exata.
3. O ambiente (local, CI, staging, produção).
4. Se possível, um artefato capturado (HAR, dump de log, gravação de tela) com segredos redigidos.

➡️ Se não tiver, vou tentar construir um caso de reprodução sozinho.
```

### Completion criterion: a tight loop that goes red

Phase 1 is done when you can name **one command** (script path, test invocation, curl) that you have **already run at least once** — show the invocation and its output, redacted — and that is:

- [ ] **Red-capable** — drives the actual bug code path and asserts the user's exact symptom, so it can go red on *this* bug and green once fixed. "Runs without erroring" is not enough; it must be able to *catch this specific bug*.
- [ ] **Deterministic** — same verdict every run (for flaky bugs: a pinned, high reproduction rate).
- [ ] **Fast** — seconds, not minutes.
- [ ] **Agent-runnable** — you can run it unattended; a human only via `scripts/hitl-loop.template.sh`.

If you catch yourself reading code to build a theory before this command exists, **stop — jumping straight to a hypothesis is the exact failure this skill prevents.** No red-capable command, no Phase 2.

## Phase 2 — Reproduce + minimise

Run the loop. Watch it go red as the bug appears. Confirm:

- [ ] The loop produces the failure mode the **user** described, not a different failure that happens to be nearby. Wrong bug = wrong fix.
- [ ] The failure is reproducible across multiple runs (or, for non-deterministic bugs, at a high enough rate to debug against).
- [ ] You captured the exact symptom (error message, wrong output, slow timing) so later phases can verify the fix actually addresses it.

### Minimise

Shrink the repro to the **smallest scenario that still goes red**. Cut **one thing at a time**, re-running the loop after each cut, and keep only what is load-bearing for the failure:

1. Remove **code** until the bug disappears.
2. Remove **data**.
3. Remove **configuration**.
4. Remove **dependencies**.
5. Remove **environment**.

Why bother: a minimal repro shrinks the hypothesis space in Phase 3 (fewer moving parts to suspect) and becomes the clean regression test in Phase 5.

**Done when every remaining element is load-bearing** — removing any one of them makes the loop go green. Do not proceed until you have reproduced **and** minimised.

## Phase 3 — Hypothesise

Generate **3–5 ranked hypotheses before testing any of them**. Single-hypothesis generation anchors on the first plausible idea.

Each hypothesis must be **falsifiable** — state the prediction it makes:

> "If \<X\> is the cause, then \<changing Y\> will make the bug disappear / \<changing Z\> will make it worse."

If you cannot state the prediction, the hypothesis is a vibe — discard or sharpen it.

Sources for hypotheses — cheap checks first:

1. **Recent diffs** — `git log -p --since="<first sighting>"` on the touched area, plus dependency and config changes. Bugs correlate strongly with fresh changes; "it worked before" is a bisection waiting to happen.
2. **Working vs broken path** — find a sibling flow that *does* work and diff the two paths (inputs, config, call chain, data shape).
3. **Environment delta** — diff env vars, config files, feature flags, and dependency versions between an environment that works and the one that doesn't.
4. **Boundary check** — is the bad value already wrong at the entry boundary, or does it appear mid-pipeline? Each boundary that verifies clean halves the suspect zone.
5. **Backward trace** — when the error surfaces deep in the call stack, the throw site is a symptom, not the cause. Trace the bad value back to its original trigger before forming hypotheses (`references/root-cause-tracing.md`).

Where the bug may live, in roughly this order:

1. **The code** — is the bug in the code you are looking at, the code it calls, or the code calling it?
2. **The docs** — is the documented behaviour even correct?
3. **The logs** — are they telling the truth, or masking the real error?
4. **The tests** — do they cover the real path?
5. **The codebase** — are there other call sites with the same pattern?

**Show the ranked list to the user before testing** — domain knowledge re-ranks instantly ("we just deployed a change to #3", "#2 is already ruled out"). Cheap checkpoint, big time saver. Don't block on it: proceed with your ranking if the user is AFK.

```text
Hipóteses ranqueadas:

1. [HIPOTESE_1] — previsão: [PREDICAO_1]
2. [HIPOTESE_2] — previsão: [PREDICAO_2]
3. [HIPOTESE_3] — previsão: [PREDICAO_3]

Vou validar a nº [N] com [ACAO_EXPERIMENTAL]. Se confirmar, o próximo passo é [PROXIMO_PASSO].

Concorda, ou quer que eu teste outra hipótese primeiro?
```

### Diagnostic journal

For anything non-trivial, keep a running journal — it prevents re-testing a rejected hypothesis and survives context drift/compaction:

```markdown
## Diagnostic Journal — <one-sentence bug statement>
- Loop: `<the Phase 1 command>` → currently red
- Min repro: <smallest red scenario>

| Hypothesis | Prediction | Test run | Result |
|---|---|---|---|
| H1: ... | ... | ... | rejected / confirmed |

- Root cause: <filled at the end>
```

## Phase 4 — Instrument

Each probe must map to a specific prediction from Phase 3. **Change one variable at a time.**

Tool preference:

1. **Debugger / REPL inspection** if the environment supports it — one breakpoint beats ten logs.
2. **Targeted logs** at the boundaries that distinguish hypotheses.
3. **Metrics / traces** to measure the suspect behaviour and follow the request path.
4. **Assertions** that fail fast on invariants.
5. Never "log everything and grep".

**Tag every debug log** with a unique prefix, e.g. `[DEBUG-a4f2]` — cleanup at the end becomes a single grep. Untagged logs survive; tagged logs die.

**Probe placement.** Put probes at the boundaries that bisect the hypothesis space: function entry/exit, before and after each transformation, each external call (network, DB, filesystem), and every state transition. One probe that discriminates between H1 and H2 beats ten that don't.

**Multi-component sweep.** When the bug could sit anywhere in a pipeline (CI → build → deploy, API → service → DB, frontend → gateway → backend), do ONE evidence run before drilling in: log what enters and what exits *each* component boundary — data shape, env/config propagation, state — in a single pass. That run usually reveals which layer breaks; then concentrate probes on that layer instead of instrumenting the whole pipeline.

**Correlation.** Stamp a request/trace id as early as possible and log it at every probe — otherwise concurrent flows interleave and your evidence is ambiguous.

**Perf branch.** For performance regressions, logs are usually wrong. Establish a baseline measurement (timing harness, profiler, query plan), then bisect. Measure first, fix second.

## Phase 5 — Fix + regression test

Write the regression test **before the fix**, but only if there is a **correct seam** for it — a seam where the test exercises the real bug pattern as it occurs at the call site. If the only available seam is too shallow (a unit test that cannot replicate the chain that triggered the bug), a regression test there gives false confidence.

**If no correct seam exists, that itself is a finding.** The architecture is preventing the bug from being locked down — flag it for Phase 6 and for `improve-codebase-architecture`.

If a correct seam exists:

1. Turn the minimised repro into a failing test at that seam.
2. Watch it fail.
3. Apply the smallest, safest change that removes the root cause — no band-aids.
4. Watch it pass.
5. Re-run the Phase 1 feedback loop against the original (un-minimised) scenario.
6. If the root cause was invalid data or a violated invariant that reached deep into the system, harden the layers it passed through — a single check fixes *this* bug; validation at each layer makes the *class* of bug impossible (`references/defense-in-depth.md`).

If the fix touches many files, present the plan to the user in Portuguese before editing:

```text
Raiz do problema: [RAIZ].
Correção proposta: [DESCRICAO_DA_CORRECAO].
Arquivos afetados: [LISTA].

Posso aplicar a correção e depois rodar os testes?
```

## Phase 6 — Cleanup + re-validate

Required before declaring done:

- [ ] Original repro no longer reproduces (re-run the Phase 1 loop).
- [ ] Regression test passes (or absence of a correct seam is documented).
- [ ] Affected test layer run (unit, integration, e2e) plus a lightweight regression on neighbouring flows.
- [ ] All `[DEBUG-...]` instrumentation removed (`grep` the prefix).
- [ ] Throwaway prototypes deleted (or moved to a clearly marked debug location).
- [ ] The hypothesis that turned out correct is stated in the commit / PR message, so the next debugger learns.

Report the result in Portuguese:

```text
Correção aplicada em [ARQUIVOS].

- Reprodução original: [não reproduz mais / ainda falha]
- Teste de regressão: [PASS/FAIL]
- Testes afetados: [PASS/FAIL]
- Instrumentação removida: [SIM/NÃO]

O bug está resolvido. Quer que eu abra uma Issue para documentar a causa raiz com /create-issues?
```

## Stuck — when hypotheses run out

**Three-strike rule.** Count applied fixes, not hypotheses: if 3 fixes failed and the bug survives or mutates — each fix revealing a new problem in a different place, or each fix needing a workaround to land — the pattern itself is suspect. STOP and question the architecture with the user (in Portuguese) before attempting fix #4: is this design fundamentally sound, or are we fixing symptoms of a wrong structure? That conversation is `improve-codebase-architecture` territory, not another patch.

If every hypothesis is rejected, the failure is upstream of your assumptions:

1. **Question the loop** — is it really red-capable on *this* bug? Re-check the Phase 1 completion checklist; a loop that cannot go red makes every later phase theater.
2. **Question the frame** — re-run Phase 0. The reported symptom may be downstream of the real one (e.g. the user sees a timeout, but the root cause is a deadlock ten seconds earlier).
3. **Widen the search** — the cause may sit outside the code: data, config, infrastructure, a dependency's changed behaviour. Diff the whole environment, not just the source diff.
4. **Bring evidence to the user** — report the diagnostic journal in Portuguese: what was tried, what each test proved or disproved, and what you need next (access, an artifact, or a decision).
5. **"No root cause" is a last resort, not a conclusion** — most "environmental / unexplainable" verdicts are incomplete investigations. If the evidence genuinely points outside the system (timing, external service, hardware), document what was ruled out, add handling at the boundary (retry, timeout, explicit error), and add durable monitoring via `observability-and-instrumentation` so the next occurrence brings evidence with it.

## Agent Self-Debug / Introspection (use when the failure is the agent itself)

If the bug is not in the codebase but in the agent session — repeated tool-call failures, loops, context drift, or mismatch between expected and actual filesystem state — apply the agent-introspection loop.

### Phase A — Failure Capture
Before trying to recover, record:
- error type, message, and stack trace when available
- last meaningful tool call sequence
- what the agent was trying to do
- current context pressure: repeated prompts, oversized pasted logs, duplicated plans, or runaway notes
- current environment assumptions: cwd, branch, relevant service state, expected files

```markdown
## Failure Capture
- Session / task:
- Goal in progress:
- Error:
- Last successful step:
- Last failed tool / command:
- Repeated pattern seen:
- Environment assumptions to verify:
```

### Phase B — Root-Cause Diagnosis
Match the failure to a known pattern:

| Pattern | Likely Cause | Check |
| --- | --- | --- |
| Maximum tool calls / repeated same command | loop or no-exit observer path | inspect the last N tool calls for repetition |
| Context overflow / degraded reasoning | unbounded notes, repeated plans, oversized logs | inspect recent context for duplication and low-signal bulk |
| `ECONNREFUSED` / timeout | service unavailable or wrong port | verify service health, URL, and port assumptions |
| `429` / quota exhaustion | retry storm or missing backoff | count repeated calls and inspect retry spacing |
| file missing after write / stale diff | race, wrong cwd, or branch drift | re-check path, cwd, git status, and actual file existence |
| tests still failing after "fix" | wrong hypothesis | isolate the exact failing test and re-derive the bug |

Diagnosis questions:
- is this a logic failure, state failure, environment failure, or policy failure?
- did the agent lose the real objective and start optimizing the wrong subtask?
- is the failure deterministic or transient?
- what is the smallest reversible action that would validate the diagnosis?

### Phase C — Contained Recovery
Prefer these interventions in order:
1. Restate the real objective in one sentence.
2. Verify the world state instead of trusting memory.
3. Shrink the failing scope.
4. Run one discriminating check.
5. Only then retry.

Bad pattern: retrying the same action three times with slightly different wording.
Good pattern: capture failure → classify the pattern → run one direct check → change the plan only if the check supports it.

### Phase D — Self-Debug Report
End with:

```markdown
## Agent Self-Debug Report
- Session / task:
- Failure:
- Root cause:
- Recovery action:
- Result: success | partial | blocked
- Token / time burn risk:
- Follow-up needed:
- Preventive change to encode later:
```

## AI Workflow Diagnostic

When the misbehaving system is an agent or AI workflow (not a code path), audit it across 5 dimensions. Score each 1–5 and report specific findings in Portuguese.

### Dimension 1 — Prompt Quality
Structure (role, context, instructions, output zones); explicit output schema; instruction clarity; edge-case handling; anti-patterns (wall of text, contradictions, implicit format).

### Dimension 2 — Context Efficiency
Context budget allocation (planned vs. ad-hoc); attention gradient (critical info at start/end); context window utilisation; state management (explicit vs. implicit); memory strategy appropriate for conversation length.

### Dimension 3 — Tool Health
Tool count (3–7 ideal, 13+ problematic); description quality; error handling; schema completeness (input/output/error); idempotency. **Scope attribution**: distinguish project-configured tools from agent-level/built-in tools — only flag overhead the project can actually control.

### Dimension 4 — Architecture Fitness
Topology appropriateness (single vs. multi-agent justified); agent boundaries (clear vs. overlapping); handoff protocols (structured vs. ad-hoc); observability; cost awareness.

### Dimension 5 — Safety & Reliability
Input validation; output filtering (PII, content policy — scope contextually: user's own frontend→backend is lower risk than external services); cost controls; error recovery; evaluation strategy (golden tests vs. "it seems to work").

### Report format

```text
╔══════════════════════════════════════╗
║          WORKFLOW DIAGNOSTIC         ║
╠══════════════════════════════════════╣
║ Prompt Quality       ████░  4/5      ║
║ Context Efficiency   ███░░  3/5      ║
║ Tool Health          ██░░░  2/5      ║
║ Architecture         ████░  4/5      ║
║ Safety & Reliability ██░░░  2/5      ║
╠══════════════════════════════════════╣
║ Overall Score:       15/25           ║
╚══════════════════════════════════════╝

ACHADOS CRÍTICOS:
1. [Problema mais severo — ação imediata]
2. [Segundo mais severo]
3. [Terceiro]

AÇÕES RECOMENDADAS:
1. [Correção específica para o achado nº 1]
2. [Correção específica para o achado nº 2]
3. [Correção específica para o achado nº 3]
```

### Scoring guide

| Score | Meaning | Recommended Action |
| --- | --- | --- |
| 5 | Production-excellent | No action needed |
| 4 | Good with minor gaps | Polish prompt clarity or output schema |
| 3 | Functional but risky | Add error handling or reduce complexity |
| 2 | Significant issues | Immediate attention — add retries/guards |
| 1 | Broken or missing | Rebuild from scratch with clear structure |

## Silent-Failure Hunt

When the code "works" but misbehaves quietly, hunt for silent failures before declaring the bug resolved.

### Hunt Targets
1. **Empty Catch Blocks** — `catch {}`, ignored exceptions, errors converted to `null` / empty arrays with no context.
2. **Inadequate Logging** — logs without enough context, wrong severity, log-and-forget handling.
3. **Dangerous Fallbacks** — default values that hide real failure, `.catch(() => [])`, graceful-looking paths that make downstream bugs harder to diagnose.
4. **Error Propagation Issues** — lost stack traces, generic rethrows, missing async handling.
5. **Missing Error Handling** — no timeout or error handling around network/file/db paths, no rollback around transactional work.

### Output Format
For each finding:
- location
- severity
- issue
- impact
- fix recommendation

## Re-Validation Loop

Diagnosis is iterative. After every change, re-run the reproduction. If the bug moves or changes, go back to Phase 3. Do not declare the bug fixed until the reproduction passes and the regression suite is green.

## Red Flags — STOP and re-enter the pipeline

If you catch yourself thinking any of these, you are off the process — return to the phase you skipped:

- "Quick fix for now, investigate later."
- "Just try changing X and see if it works."
- "It's probably X — let me fix it." (a fix without a confirmed hypothesis is a guess)
- "Bundle a few changes and run the tests." (you won't know what worked)
- "Skip the repro, I'll verify manually."
- "I don't fully understand it, but this might work."
- "The bug looks simple — no need for the whole process." (simple bugs have root causes too; the loop is *fast* for simple bugs)
- "Emergency — no time for process." (systematic beats thrashing, especially under pressure)
- "One more fix attempt" — after 2+ failures. Three failed fixes means wrong architecture, not wrong hypothesis (see Stuck).

User redirections are the same signal from the outside: "is that really happening?", "did you verify?", "stop guessing", "we're going in circles". When you hear one, stop and return to the phase you skipped.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Investigating a vague symptom | Phase 0 first — one-sentence bug statement or keep interviewing. |
| Fixing without a reproduction first | Build a red-capable loop before changing code. |
| Hypothesising before a tight loop exists | Phase 1 completion criterion first — no red command, no theory. |
| Testing the first plausible hypothesis only | Generate 3–5 ranked, falsifiable hypotheses. |
| Skipping minimisation | Minimise first, or you fix symptoms, not the cause. |
| Asserting "didn't crash" instead of the symptom | Sharpen the signal — assert the exact user-reported failure. |
| Regression test at a seam too shallow | Use a correct seam, or document that none exists. |
| Removing instrumentation too early | Keep `[DEBUG-...]` logs until the fix is verified, then grep-clean. |
| Not adding a regression test | Every fixed bug deserves a test. |
| Re-testing a rejected hypothesis | Keep the diagnostic journal; record every result. |
| Assuming recent changes are innocent | Diff first — fresh code is the prime suspect. |
| Ambiguous evidence from concurrent flows | Stamp a correlation id and log it at every probe. |
| Debugging the wrong environment | Verify env matches the Phase 0 report before trusting results. |
| Declaring done without re-validation | Re-run the reproduction and the suite. |
| Fixing where the error is thrown | Trace the bad value back to its origin (`references/root-cause-tracing.md`). |
| Fixing only the immediate cause | Add validation at every layer the bad data crosses (`references/defense-in-depth.md`). |
| Flaky test fixed by bumping a `sleep` | Wait on the actual condition, not guessed time (`references/condition-based-waiting.md`). |
| Instrumenting a whole pipeline at once | One sweep across all component boundaries first, then drill into the failing layer. |

## References

- `scripts/hitl-loop.template.sh` — human-in-the-loop reproduction driver
- `scripts/find-polluter.sh` — bisects test files to find which one creates stray files/state
- `references/root-cause-tracing.md` — trace a bad value backward through the call stack to its origin
- `references/defense-in-depth.md` — validate at every layer so the bug class becomes impossible
- `references/condition-based-waiting.md` — replace guessed sleeps with condition polling in flaky tests
- `references/dotnet-debugging.md` — .NET diagnostics tools and log queries
- `references/angular-debugging.md` — Angular DevTools, change-detection and RxJS debugging
- `references/python-debugging.md` — pdb/pytest/py-spy debugging and log queries
- `references/react-debugging.md` — React DevTools and re-render debugging
- `qa-analyst` — for test planning and bug reporting
- `write-specs` — for producing specs when the bug reveals missing requirements
- `improve-codebase-architecture` — when the diagnosis reveals missing seams or structural problems
- `observability-and-instrumentation` — when the fix needs durable logging/metrics/tracing, not throwaway probes
- `create-issues` — to track the documented root cause
- `agent-introspection-debugging` — when the failure is the agent session itself (loops, context drift, repeated tool calls)
- `silent-failure-hunter` — when the code works but misbehaves quietly
