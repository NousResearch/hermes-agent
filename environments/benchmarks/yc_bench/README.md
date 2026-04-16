# YC-Bench: Long-Horizon Agent Benchmark

[YC-Bench](https://github.com/collinear-ai/yc-bench) by [Collinear AI](https://collinear.ai/) is a deterministic, long-horizon benchmark that tests LLM agents' ability to act as a tech startup CEO. The agent manages a simulated company over 1-3 years, making compounding decisions about resource allocation, cash flow, task management, and prestige specialisation across 4 skill domains.

Unlike TerminalBench2 (which evaluates per-task coding ability with binary pass/fail), YC-Bench measures **long-term strategic coherence** — whether an agent can maintain consistent strategy, manage compounding consequences, and adapt plans over hundreds of turns.

## Setup

```bash
# Install yc-bench (optional dependency)
pip install "hermes-agent[yc-bench]"

# Or install from source
git clone https://github.com/collinear-ai/yc-bench
cd yc-bench && pip install -e .

# Verify
yc-bench --help
```

## Running

```bash
# From the repo root:
bash environments/benchmarks/yc_bench/run_eval.sh

# Or directly:
python environments/benchmarks/yc_bench/yc_bench_env.py evaluate \
    --config environments/benchmarks/yc_bench/default.yaml

# Override model:
bash environments/benchmarks/yc_bench/run_eval.sh \
    --openai.model_name anthropic/claude-opus-4-20250514

# Quick single-preset test:
bash environments/benchmarks/yc_bench/run_eval.sh \
    --env.presets '["fast_test"]' --env.seeds '[1]'
```

## How It Works

### Architecture

```
HermesAgentLoop (our agent)
  -> terminal tool -> subprocess("yc-bench company status") -> JSON output
  -> terminal tool -> subprocess("yc-bench task accept --task-id X") -> JSON
  -> terminal tool -> subprocess("yc-bench sim resume") -> JSON (advance time)
  -> ... (100-500 turns per run)
```

The environment initialises the simulation via `yc-bench sim init` (NOT `yc-bench run`, which would start yc-bench's own built-in agent loop). Our `HermesAgentLoop` then drives all interaction through CLI commands.

### Simulation Mechanics

- **4 skill domains**: research, inference, data_environment, training
- **Prestige system** (1.0-10.0): Gates access to higher-paying tasks
- **Employee management**: Junior/Mid/Senior with domain-specific skill rates
- **Throughput splitting**: `effective_rate = base_rate / N` active tasks per employee
- **Financial pressure**: Monthly payroll, bankruptcy = game over
- **Deterministic**: SHA256-based RNG — same seed + preset = same world

### Difficulty Presets

| Preset | Employees | Tasks | Focus |
|-----------|-----------|-------|-------|
| tutorial  | 3         | 50    | Basic loop mechanics |
| easy      | 5         | 100   | Throughput awareness |
| **medium**| 5         | 150   | Prestige climbing + domain specialisation |
| **hard**  | 7         | 200   | Precise ETA reasoning |
| nightmare | 8         | 300   | Sustained perfection under payroll pressure |
| fast_test | (varies)  | (varies) | Quick validation (~50 turns) |

Default eval runs **fast_test + medium + hard** × 3 seeds = 9 runs.

### Scoring

```
composite = 0.5 × survival + 0.5 × normalised_funds
```

- **Survival** (binary): Did the company avoid bankruptcy?
- **Normalised funds** (0.0-1.0): Log-scale relative to initial $250K capital

## Configuration

Key fields in `default.yaml`:

| Field | Default | Description |
|-------|---------|-------------|
| `presets` | `["fast_test", "medium", "hard"]` | Which presets to evaluate |
| `seeds` | `[1, 2, 3]` | RNG seeds per preset |
| `max_agent_turns` | 200 | Max LLM calls per run |
| `run_timeout` | 3600 | Wall-clock timeout per run (seconds) |
| `survival_weight` | 0.5 | Weight of survival in composite score |
| `funds_weight` | 0.5 | Weight of normalised funds in composite |
| `horizon_years` | null | Override horizon (null = auto from preset) |

## Cost & Time Estimates

Each run is 100-500 LLM turns. Approximate costs per run at typical API rates:

| Preset | Turns | Time | Est. Cost |
|--------|-------|------|-----------|
| fast_test | ~50 | 5-10 min | $1-5 |
| medium | ~200 | 20-40 min | $5-15 |
| hard | ~300 | 30-60 min | $10-25 |

Full default eval (9 runs): ~3-6 hours, $50-200 depending on model.

## References

- [collinear-ai/yc-bench](https://github.com/collinear-ai/yc-bench) — Official repository
- [Collinear AI](https://collinear.ai/) — Company behind yc-bench
- [TerminalBench2](../terminalbench_2/) — Per-task coding benchmark (complementary)
