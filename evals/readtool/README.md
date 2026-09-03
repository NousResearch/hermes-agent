# Read-Tool Eval

A/B harness measuring how `read_file` engineering choices affect real agent
runs. Motivated by Command Code's read-tool writeup (Aug 2026), which
benchmarked ten harnesses on hostile-file handling — and whose Hermes column
contained several errors (we already ship a per-line clamp, did-you-mean
suggestions, notebook/docx/xlsx extraction, PDF conversion, and a device-path
blocklist). This eval tests the failure shapes for real, through the real
`AIAgent`, instead of trusting anyone's capability table.

## What it measures

Every task runs the full Hermes agent (file + terminal + search toolsets)
against a deterministic hostile workspace:

| fixture | shape | tasks |
|---|---|---|
| `package-lock.json` | 80K lines, 2.7MB — token tarpit | `lockfile_version` |
| `src/app.min.js` | one 600KB line matching greps | `minified_backoff` |
| `logs/server.log` | 150K lines, one ERROR near tail | `log_error_hunt` |
| `data/report.txt` | 412 lines — past-EOF probe | `past_eof` |
| `config/overrides.yaml` | empty file | `empty_config` |
| `notes/Meeting…PM.txt` | NFD + U+202F + U+2019 filename | `unicode_filename` |
| `AGENTS.md` vs `AGENT.md` | near-miss filename | `near_miss_filename` |
| `logs/live.pipe` | FIFO — blocks naive reads | `fifo_hang` |
| `data/data.txt` | PNG bytes behind a .txt name | `lying_extension` |

Metrics per task: **accuracy** (substring/regex graders against planted
ground truth), **api_turns**, **tool_calls**, **read_file_calls**,
**total_tokens**, **wall_s**. Efficiency aggregates are per-task means,
never sums.

## Running

```bash
# Baseline (3 reps, both models)
python3 evals/readtool/runner.py --model anthropic/claude-opus-4.8 \
    --provider openrouter --reps 3 --label baseline
python3 evals/readtool/runner.py --model qwen/qwen3.8-max \
    --provider openrouter --reps 3 --label baseline

# After a feature change, re-run with a new label:
python3 evals/readtool/runner.py --model qwen/qwen3.8-max \
    --provider openrouter --reps 3 --label feat-stat-guard

# Compare
python3 evals/readtool/report.py --labels baseline feat-stat-guard
```

Rules of engagement (from hermesbench discipline):

- **3 reps minimum**; single-run deltas within ±3% are noise, not wins.
- Never edit `tools/` while a run is in flight — the runner imports the
  live tree.
- Two models on purpose: a frontier model (opus) that can absorb sloppy
  reads, and a strong open model (qwen-max) where harness quality shows.
  A feature that only helps qwen still counts — that's the population the
  hardening serves.
- Errored task-runs score 0 and stay in the accuracy denominator but are
  excluded from efficiency means.

## Results layout

```
results/<label>/<model_slug>/rep<N>.json
```

`results/` is gitignored except for `SUMMARY.md`, which records the
verdict + numbers for each feature evaluated.
