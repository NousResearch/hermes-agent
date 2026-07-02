# Evals

Evaluation + validation harnesses for the **information-gain ranker**. Findings live in
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
| `validate_evsi.py` | inject projected answer → re-derive → judge **realized change** (+ realized **stakes**). `--source bucket\|all_scored`. `--ab` A/Bs absolute-vs-pairwise elicitation on one shared realized set (`--elicit-model` to set a host-local judge). `--families [--premortem …]` runs the families layer; rows carry `lens`/`family`. | §P1a, §Agentic realized calibration, §Comparative elicitation |
| `analyze_evsi.py` | post-hoc calibration + formula ablations. On an `--ab` run, prints the **#24 gate**: per-method within-task ρ + adopt/keep verdict. On a `--families` run, prints **per-lens attribution** (#25). | §P1a / §P1c / §Comparative elicitation |
| `analyze_validity.py` | de-confounded per-regime analysis (stakes-judge calibration, regret). | §realized-stakes instrument |

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

# in-container (wrapper end-to-end), pinned to a real project to de-confound
docker exec -e OLLAMA_URL=http://host.docker.internal:11434/api/chat -e HERMES_HOME=/opt/data hermes \
  /opt/hermes/.venv/bin/python <skill>/evals/validate_wrapper.py \
  --ids add-auth --k 1 --cwd /opt/data/projects/<proj> --responder-tools file --out /opt/data/wv.json
```
