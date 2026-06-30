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
| `score_scan.py` | cheap value-structure scan across the bank (U/EVSI/value/stakes/Δ/derivable, per category). No realized_change. | `evsi-validation-findings.md` §Domain sensitivity |
| `compare_domains.py` | life vs agentic side-by-side of the value distributions. | §Domain sensitivity |
| `validate_evsi.py` | inject projected answer → re-derive → judge **realized change** (+ realized **stakes**). `--source bucket\|all_scored`. | §P1a, §Agentic realized calibration |
| `analyze_evsi.py` | post-hoc calibration + formula ablations from a validate_evsi run. | §P1a / §P1c |
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
  is the path if ever needed; the Δ-half is the validated part.
- **Wrapper end-to-end is task-dependent** (de-confounded 1-1 at k=1): helps where a clarification
  shapes the work, redundant where a capable agent self-investigates. Distinctive value = user-only
  constraints. The grounded answerer's **cwd** must be the user's project.

## Run examples

```bash
# host
OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes python3 evals/score_scan.py --out /tmp/scan.json
OLLAMA_URL=http://localhost:11434/api/chat HERMES_HOME=~/.hermes python3 evals/validate_evsi.py --source all_scored --out /tmp/ve.json
python3 evals/analyze_evsi.py /tmp/ve.json

# in-container (wrapper end-to-end), pinned to a real project to de-confound
docker exec -e OLLAMA_URL=http://host.docker.internal:11434/api/chat -e HERMES_HOME=/opt/data hermes \
  /opt/hermes/.venv/bin/python <skill>/evals/validate_wrapper.py \
  --ids add-auth --k 1 --cwd /opt/data/projects/<proj> --responder-tools file --out /opt/data/wv.json
```
