# Summary and original-evidence study: fixture reproduction

All **546 contexts / 1,092 answers** completed and passed the recorded input ceilings.
All 14 public histories, fixed exams, gold answers and original evidence units match the prior frozen inputs.
For the experimental setting and limitations, see [STUDY.md](../STUDY.md).

## Results

Fields are scored by exact JSON types and values. Each task has equal weight; repeated answers and rounds are not independent tasks. Input tokens are actual answering API usage, excluding shared summary and ranking preparation.

### Four primary tasks

| Method | Field accuracy | All fields correct | Mean answer input tokens |
| --- | ---: | ---: | ---: |
| Full | 100.0% | 100.0% | 4,014 |
| S | 82.0% | 8.3% | 2,481 |
| H | 79.0% | 25.0% | 2,435 |
| S + Jev Top 1 | 83.5% | 12.5% | 2,754 |
| S + Jev Top 2 | 91.6% | 50.0% | 2,976 |
| S + Jev Top 3 | 97.2% | 83.3% | 3,234 |
| S + Jev Top 4 | 95.7% | 70.8% | 3,496 |
| S + Jev Top 5 | 96.5% | 75.0% | 3,639 |
| H + Jev Top 1 | 86.5% | 41.7% | 2,708 |
| H + Jev Top 2 | 93.5% | 58.3% | 2,930 |
| H + Jev Top 3 | 97.4% | 79.2% | 3,188 |
| H + Jev Top 4 | 96.9% | 75.0% | 3,450 |
| H + Jev Top 5 | 96.9% | 75.0% | 3,593 |

### Ten short-source or query-only controls

| Method | Field accuracy | All fields correct | Mean answer input tokens |
| --- | ---: | ---: | ---: |
| Full | 100.0% | 100.0% | 2,727 |
| S | 71.9% | 41.7% | 2,318 |
| H | 79.6% | 45.0% | 2,309 |
| S + Jev Top 1 | 87.2% | 70.0% | 2,496 |
| S + Jev Top 2 | 86.9% | 80.0% | 2,577 |
| S + Jev Top 3 | 93.3% | 90.0% | 2,674 |
| S + Jev Top 4 | 92.5% | 90.0% | 2,725 |
| S + Jev Top 5 | 92.5% | 90.0% | 2,725 |
| H + Jev Top 1 | 80.7% | 60.0% | 2,488 |
| H + Jev Top 2 | 93.9% | 83.3% | 2,568 |
| H + Jev Top 3 | 95.6% | 90.0% | 2,665 |
| H + Jev Top 4 | 95.6% | 90.0% | 2,716 |
| H + Jev Top 5 | 95.1% | 88.3% | 2,716 |

The four primary tasks reproduce the improvement direction:
- S: **82.0% → 97.2%** with Top 3 (+15.2 percentage points).
- H: **79.0% → 97.4%** with Top 3 (+18.4 percentage points).
- Top 3 mean answering inputs are **19.4% / 20.6% shorter** than Full for S/H.

The earlier frozen study reported S 82.0% → 99.5% and H 79.4% → 97.9%. These fresh generations reproduce the qualitative effect, not identical predictions. Temperature zero does not guarantee immutable service weights or deterministic responses.

Top 4–5 did not improve on Top 3 in the primary cohort. This is a point estimate on four synthetic primary tasks, not an optimal-N claim. The experiment supports original evidence supplementing these frozen summaries; without other selectors or an equally long summary control, it does not isolate how much of the gain comes specifically from Jev ranking.

## Run and audit

- Hermes source at run start: `5ea04f26f9a9cb74613abf4efb9df628b004b2b8`; an implementation snapshot and hashes are retained with the local artifacts.
- Text model: `qwen3.8-max`, through Hermes auxiliary compression routing; thinking disabled, temperature zero.
- Jev: requested `jev-latest`, all 54 responses resolved to `jev-1.13.0`.
- 42 S summaries accepted after 45 length-check attempts; 42 H summaries accepted after 42 attempts. No answer-based retries.
- All 546 frozen contexts were checked for exact summary reuse, strict nested evidence prefixes and identical S/H selections.
- Every recorded answer was regraded offline; all grades and aggregates matched. No configured API keys were found in artifacts.
- Shared preparation plus answers: 1,179 Qwen calls, 3,197,729 input / 60,735 output tokens. The separate routing probe is excluded.
- Jev preparation: 54 calls, 77,799 input / 5,634 output tokens. No dollar-cost estimate is inferred from token counts.
- All 42 H-alone conditions were shorter than paired Full. Mean reductions do not imply that every fusion condition shortened its paired input.

Validation: **11 focused tests** passed through `scripts/run_tests.sh`; all **12 `scripts/check` groups** passed. The full repository test suite was not run. Raw histories, requests and responses remain outside Git.
