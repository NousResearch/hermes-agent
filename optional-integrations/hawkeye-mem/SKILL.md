---
name: hawkeye-mem
description: HawkEye-Mem integration — lightweight Rust-based memory/GPU/thermal sensor for AI Agents via MCP. Use when the user asks about system memory pressure, GPU status, thermal throttling, cache strategy, agent concurrency, or token-cost optimization.
version: 1.0.0
author: ligl0325
license: Apache-2.0
platforms:
  - linux
  - macos
  - windows
metadata:
  hermes:
    tags:
      - mcp
      - monitoring
      - memory
      - gpu
      - performance
    category: optional-integration
triggers:
  - check system memory
  - memory pressure
  - GPU status
  - thermal status
  - cache strategy
  - agent concurrency
  - token audit
  - HawkEye-Mem
toolsets:
  - terminal
  - mcp
---

# HawkEye-Mem Integration

HawkEye-Mem (https://github.com/qiuhaomem/HawkEye-Mem) is a Rust-based sensor for AI Agents. It exposes system state (memory, CPU, GPU, disk, temperature, agent processes) as JSON and MCP tools.

## Quick Install

```bash
# From source
git clone https://github.com/qiuhaomem/HawkEye-Mem.git
cd HawkEye-Mem && cargo build --release
sudo -S -p '' cp target/release/hawk-eye-mem /usr/local/bin/

# Verify
hawk-eye-mem --json | head -c 200
```

## MCP Server Setup

Add to Hermes `config.yaml`:

```yaml
mcp:
  servers:
    hawkeye-mem:
      command: hawk-eye-mem
      args: ["mcp", "serve"]
      transport: stdio
```

Then restart: `hermes gateway restart` or `/reset` in CLI.

## Available Tools (15 total)

| Tool | Purpose |
|------|---------|
| `get_memory_status` | Full memory/swap + agent action guidance |
| `get_memory_metric` | Single metric: total/used/available/pressure |
| `get_gpu_status` | VRAM, temp, power, utilization |
| `get_thermal_status` | CPU/GPU temperatures + pressure level |
| `get_network_status` | NICs, throughput, latency |
| `get_agent_processes` | Other AI agents running on host |
| `get_cache_strategy` | Dynamic cache mode (aggressive/balanced/conservative) |
| `get_concurrency_suggestion` | Safe parallel task count |
| `get_environment_fingerprint` | Hardware/OS hash for env matching |
| `get_trend_report` | Memory/CPU/disk trends over time |
| `run_onboarding_showcase` | One-shot capability demo |
| `report_cache_hit` | Log cache hit/miss for cost tracking |
| `start_remote_server` | Start remote monitoring daemon |
| `reset_environment_fingerprint` | Force re-calibration |
| `run_token_audit` | Token consumption audit |

## Integration Patterns

### Context Budget Guard
```yaml
agent:
  max_turns: 90
  # HawkEye-Mem can dynamically suggest lower values when memory is high
```

### Cache Strategy
```yaml
compression:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20
  # Swap to 'conservative' when pressure is high
```

### Concurrency Control
```yaml
delegation:
  max_concurrent_children: 3
  # Reduce when system memory is constrained
```

## Use Cases

1. **Memory-aware compression** — reduce context before OOM
2. **Token-cost tracking** — report cache hits to monitor API spend
3. **Multi-agent coordination** — limit parallel tasks based on available RAM
4. **Thermal awareness** — throttle background work when CPU/GPU is hot

## License

HawkEye-Mem is Apache-2.0. This integration note is MIT.
