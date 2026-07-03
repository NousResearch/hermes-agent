# Evals

Evaluation + validation harnesses for the **next-best-questions ranker** (formerly information-gain). Findings live in
`../references/{benchmark-findings,evsi-validation-findings}.md`. These run on the host against
`localhost:11434` (immune to container restarts). The end-to-end **wrapper** A/B harness moved with the
loop into the sibling **`investigator`** skill (`../../investigator/evals/validate_wrapper.py`).

| script | what it does | findings |
|---|---|---|
| `testbank.py` | 34-prompt / 17-category bank (LIFE control + agentic BANK) + `REALIZED_SUBSET`. Imported by the others. | — |
| `benchmark.py` | prompt × config × rep matrix; usage + adjudicated scores per run. | `benchmark-findings.md` |
| `adjudicator.py` | LLM judge for a single run (framing/relevance/value/diversity/calibration). | — |
| `score_scan.py` | cheap value-structure scan across the bank (U/EVSI/value/stakes/Δ/derivable, per category). No realized_change. Default pool = agentic BANK; `--include-life` adds the LIFE control. `--families [--premortem on\|off\|auto]` runs the families layer (lens-tagged rows) for the #25 two-arm scan. | `evsi-validation-findings.md` §Domain sensitivity |
| `saturation_scan.py` | breadth sweep. Default: distinct-target **coverage** (generation-only). `--scored`: full-pipeline **value** saturation (max_value + #≥floor per breadth). | §Stop + breadth calibration |
| `compare_domains.py` | life vs agentic side-by-side of the value distributions. | §Domain sensitivity |
| `validate_evsi.py` | inject projected answer → re-derive → judge **realized change** (+ realized **stakes**). `--source bucket\|all_scored`. `--ab` A/Bs absolute-vs-pairwise elicitation on one shared realized set (`--elicit-model` to set a host-local judge). `--ab-probs` A/Bs sampled-vs-stated P(a) (#26; the run samples, the stated arm is a free re-score; realized shared over the union of each arm's top-N). `--ab-solution` A/Bs the solution-space Δplan judge (#27; `--answer-prob-mode sampled` pins the #26 winner). `--families [--premortem …]` runs the families layer; rows carry `lens`/`family`. | §P1a, §Agentic realized calibration, §Comparative elicitation |
| `analyze_evsi.py` | post-hoc calibration + formula ablations. On any `--ab*` run, prints the **A/B gate** (#24/#26/#27 — same decisive rule): per-method within-task ρ + adopt/keep verdict vs the control (`absolute`/`stated`). On a `--families` run, prints **per-lens attribution** (#25). | §P1a / §P1c / §Comparative elicitation |
| `analyze_validity.py` | de-confounded per-regime analysis (stakes-judge calibration, regret). | §realized-stakes instrument |
| `outcome_bank.py` | 20 ambiguous-but-executable tasks (hidden spec + hidden asserts), each verified against a reference impl. | — |
| `outcome_eval.py` | the OBJECTIVE tier: strict user simulator + arms (baseline/nbq/zeroshot/prompt-evsi/nbq-derive) + sandboxed hidden-test runner + paired sign-test analysis + q_value→Δpass anchor. | §Objective-outcome validation |

*(End-to-end wrapper A/B — `validate_wrapper.py` — now lives in the `investigator` skill's `evals/`.)*

## Headline results

- **Ranking validated** where it matters: in the agentic domain, value/EVSI predict the *clean*
  realized-change signal (per-answer ρ 0.64; question-level value-vs-realized-change 0.66) — vs a
  near-null on the generic "life" prompts, which turned out to be a degenerate corner.
- **Value structure is a 3-regime spectrum** (ask-user / go-find-out / just-do-it); `U` is **not**
  inert in the target domain (it discriminates ask-vs-find-out), so the life-only "drop U" was an
  artifact. Absolute thresholds mis-fire across regimes → selection is top-K **by rank**.
- **Stakes resists absolute post-hoc measurement** (collapse / central-tendency) → comparative/pairwise
  is the path if ever needed; the Δ-half is the validated part. **Comparative elicitation (#24) is now
  BUILT as an off-by-default, A/B-gated experiment** (`value_judge_mode=pairwise`): forced-choice
  comparisons → Bradley-Terry (`scripts/pairwise.py`), anchored FLOOR/CEILING so between-task scale is
  preserved. **Powered 12-prompt A/B verdict: #24 CLOSED — keep `absolute`.** The gate ranks on
  `realized_regret` (realized EVSI); with power, pairwise is **slightly worse** on every target (regret
  abs +0.360 vs pw +0.204, loses 9/12) — comparative elicitation doesn't help projected Δ/stakes.
  Documented negative result; the realized-pairwise judge is **NOT built** (pointless). The n=6
  sub-stories (change "dead", pairwise edge, saturation) were small-sample noise. **Positive:** the p1c
  ablation vs regret ranks `√(U·EVSI)` best (+0.360) above every component → the frozen formula is
  validated *within*-task too. The live default is untouched.
- **Sampled P(a) (#26) is also a powered null — keep `stated`.** The BED-LLM/OPEN calibration
  critique doesn't transfer: forced-choice frequencies moved P on 79% of pairs and q_value on 76%,
  yet within-task ranking didn't improve (regret Δρ +0.010, wins 4/12). Bonus: the run's ablation
  re-validates `√(U·EVSI)` as best on all three realized targets (regret +0.356 ≈ prior +0.360).
  **Re-confirmed under deepseek elicit+judge (Δρ +0.058, 5/12 — still a null).**
  See `evsi-validation-findings.md` §Sampled P(a) (#26).
- **Solution-space Δplan (#27) is decisively worse — keep `absolute`.** ATD-style "which of K
  sampled solutions does the answer invalidate" collapses to near-binary (69% of deltas exactly 0)
  and within-task ρ goes negative (regret −0.047 vs absolute +0.360, Δρ −0.343, loses 7/10). The
  absolute judge carries strictly more within-task signal here. **Re-confirmed ALL-deepseek:
  granularity partially recovers (mass-at-0 53%) and the method is worse (Δρ −0.369, 1/10) — the
  collapse is inherent to the framing, not a fast-model floor.** See `evsi-validation-findings.md`
  §Solution-space Δplan (#27) and §Deepseek re-adjudication.
- **Objective-outcome tier (P3-P6):** `outcome_bank.py` (20 micro tasks + 8 AGENTIC script tasks
  with sandbox fixtures, `--bank micro|agentic|both`) + `outcome_eval.py` (strict simulator, arms
  baseline/nbq/zeroshot/prompt-evsi/nbq-derive/nbq-behavior/nbq-derive-behavior, hidden-test pass
  rate). First ground-truth verdicts: clarification WORKS (zeroshot +0.317, p=0.002); the plain
  skill is out-asked by the naive baseline (Δplan = text-volume bias; the realized proxy shares
  the blindness); **derive-or-ask triples the skill's end-to-end benefit** (+0.067→+0.183);
  q_value→Δpass anchor ρ 0.432. See findings §Objective-outcome validation.
- **#28 behavior-Δ judge: NO ADOPT** (the first objectively-gated experiment): paired vs absolute
  +0.064 at 6W/5L (broad-win guard failed), unanswerable 65% vs the ~60% bar; directionally right
  (agentic Δ tripled) but zeroshot still leads by +0.157 — the residual is GENERATION altitude,
  not judging. `--value-judge-mode behavior` stays built for re-testing.
- **#29 reach lens: ADOPTED (auto, like vantage).** Tier-1: survives buckets exactly on
  access/systems tasks, prunes to zero on read-only. Tier-2 realized: regret 0.351 ≈ vantage's
  0.362 — adds signal, not noise.
- **The within-task ρ ceiling is the task, not the judge.** Same-response fast↔deepseek judge
  agreement ρ 0.814; q_value's realized-change link 0.353 (fast) → 0.398 (deepseek); within-task ρ
  under deepseek stays in the fast band (+0.24–0.35, never jumping past 0.5). Judge/elicit models
  default to `deepseek` now that cloud is reachable; pin `fast` only for cloud-outage-immune repro.
- **Wrapper end-to-end is task-dependent** (de-confounded 1-1 at k=1): helps where a clarification
  shapes the work, redundant where a capable agent self-investigates. Distinctive value = user-only
  constraints. The grounded answerer's **cwd** must be the user's project.

## Running the tests

```bash
python3 tests/run.py            # basic suite (DEFAULT): all mocked classes — offline, ~seconds
OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes python3 tests/run.py live   # model-calling classes only
OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes python3 tests/run.py all
```

Direct runs (`python3 tests/test_infogain.py` etc.) are basic-by-default too — the live classes
(`TestLive`, `TestEvalLive`) skip unless `INFOGAIN_TEST_LIVE=1` is set (run.py sets it for
`live`/`all`).

Judge/elicit models are **preflighted** (one trivial call, `validate_evsi.preflight_model`) before
any rows are produced: reasoning-channel models (e.g. `gpt-oss:20b`) return empty
`message.content` via `raw_chat` and would silently null every judged value — the run aborts with
exit 2 naming the model instead of finishing hours later all-null.

## Coverage (2026-07-03 hygiene audit)

Line coverage of the 191-test basic suite, measured with stdlib trace (no `coverage` module on
host/container): `python3 -m trace --count --missing --coverdir=/tmp/nbq-cov tests/run.py`, then
count `N:`-prefixed vs `>>>>>>` lines per `.cover` file.

| tier | files (executed/executable) |
|---|---|
| live path | voi 97.7% · infogain 93.7% · pipeline 89.4% · pairwise 96.8% |
| eval instruments | adjudicator 97.4% · validate_evsi 89.5% · analyze_evsi 89.2% · score_scan 87.7% · outcome_eval 72.2% · outcome_bank/testbank 100% |
| **archival** (declared) | rejudge 55% · analyze_validity 39% · compare_domains 32% · run_evals 28% · saturation_scan 26% |

**Archival policy:** benchmark, compare_domains, analyze_validity, saturation_scan, and the
rejudge/run_evals CLI drivers produced findings that are already recorded; they are not on any
live path. Their pure MATH helpers are pinned by tests (`TestArchivalHelpers` — knee detection,
reorderings, P′-weighting); their print/CLI drivers are deliberately untested rather than
theater-tested. Remaining uncovered lines in the live tier are render cosmetics, live-only
network checks (`ollama_reachable`), and `pragma`-style defensive branches — reviewed and
accepted in the 2026-07-03 audit (see git history for the audit reports).

## Run examples

```bash
# host
OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes python3 evals/score_scan.py --out /tmp/scan.json
OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes python3 evals/validate_evsi.py --source all_scored --out /tmp/ve.json
python3 evals/analyze_evsi.py /tmp/ve.json

# value-saturation (does the HIGH-value signal plateau before coverage does?)
OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes python3 evals/saturation_scan.py --scored --out /tmp/sat.json

# comparative-elicitation A/B gate (#24) — host-local judge so cloud isn't needed; both arms share it
OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes python3 evals/validate_evsi.py \
  --ab --source all_scored --gen-model fast --elicit-model fast --judge-model fast \
  --prompt-ids add-auth gmail-triage deploy-app --out /tmp/ab.json
python3 evals/analyze_evsi.py /tmp/ab.json   # prints the per-method within-task ρ + verdict

# sampled-P(a) A/B gate (#26) and solution-space Δplan A/B gate (#27) — same shape, same gate;
# powered runs use the full n=12 REALIZED_SUBSET
OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes python3 evals/validate_evsi.py \
  --ab-probs --source all_scored --gen-model fast --elicit-model fast --judge-model fast \
  --prompt-ids add-auth gmail-triage deploy-app --keep-responses --out /tmp/ab26.json
OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes python3 evals/validate_evsi.py \
  --ab-solution --source all_scored --gen-model fast --elicit-model fast --judge-model fast \
  --prompt-ids add-auth gmail-triage deploy-app --keep-responses --out /tmp/ab27.json

# in-container (wrapper end-to-end), pinned to a real project to de-confound
docker exec -e OLLAMA_URL=http://host.docker.internal:11434/api/chat -e HERMES_HOME=/opt/data hermes \
  /opt/hermes/.venv/bin/python <skill>/evals/validate_wrapper.py \
  --ids add-auth --k 1 --cwd /opt/data/projects/<proj> --responder-tools file --out /opt/data/wv.json
```
