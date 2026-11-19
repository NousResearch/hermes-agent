# Measurement Protocol — v1 (Frozen)

## Environment
- OS: Windows 10/11 (native CMD)
- Python: 3.14.6
- Hermes baseline commit: d1b46d718ffe675427928eff7e5fff69bcb4519b (`hermes-baseline-d1b46d7`)
- Evolving artifact: Instruction-policy bundle (`evolve/artifacts/candidate-XXX`)

## Tokens & Latency
- `tokens`: task execution tokens only (`prompt_tokens + completion_tokens` from Hermes `--usage-file` ledger). Learning tokens are tracked separately by B2.
- `time_seconds`: wall-clock seconds from task process start to exit.

## Repeats & Splits
- Minimum 3 repeats per headline measurement (`--repeat 3`), reporting mean and spread (min/max/std).
- `train` split: visible to variation agent (`bench/config/train_manifest.json`).
- `heldout` split: sealed (`bench/config/heldout_manifest.json`), gated by `HERMES_BENCH_ALLOW_HELDOUT=1`.
