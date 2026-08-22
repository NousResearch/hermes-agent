# PR Description — Federation + AIDE² Self-Evolution

## Background: AIDE² Research

This PR is informed by deep research into **AIDE² (AIDE-squared)** — the recursive self-improvement system by Weco AI that achieved **Level 1 RSI** (first experimental evidence of a system improving itself faster than humans can). 

See `docs/aide-squared-research.md` for the full 439-line analysis covering:
- RSI Ladder (Level 0→3): Delegation → Net Positive → Ignition → Inflection
- 8 core mechanisms of AIDE² (double-loop optimization, public/private score split, fixed cost budget, task heterogeneity, tree search + bandit lineage, emergent anti-reward-hacking, context engineering, strict evaluation)
- Gap analysis between Hermes and AIDE²
- 7 enhancement designs with implementation roadmap

## What This PR Adds

### Part A: Federation (Phases 1-10) — Multi-Device Collaboration

| Phase | Feature | Module | Tests | Lines |
|---|---|---|---|---|
| 1 | Heartbeat + offline detection + task claiming | `federation_heartbeat.py` | 15 | - |
| 2 | WebSocket + protocol + platform adapter | `federation_protocol/connection/adapter.py` | 20 | - |
| 3 | Raft-lite consensus + task relay | `federation_consensus/relay.py` | 17 | - |
| 4 | CLI (`hermes fed status/tasks/handoff/compute`) | `subcommands/federation.py` | - | - |
| 5 | mDNS zero-config discovery | `federation_discovery.py` | 11 | - |
| 6 | Security hardening (TLS/HMAC/rate limit/IP whitelist) | `federation_connection.py` | 18 | - |
| 7 | Cross-device memory sync + distributed search | `federation_collaboration.py` | 10 | - |
| 8 | Compute pool (weighted task distribution) | `federation_compute_pool.py` | 17 | - |
| 9 | Distributed cron relay + skill sync | `federation_cron_relay.py` | 17 | - |
| 10 | Leader election + config sync | `federation_cluster.py` | 14 | - |

**Desktop UI**: macOS Display-style device management overlay in Settings (`apps/desktop/src/app/federation/`)

### Part B: AIDE² Self-Evolution — Making Hermes Improve Itself

| Priority | Feature | Module | Lines | Description |
|---|---|---|---|---|
| P0-1 | **Experience Ledger** | `agent/experience_ledger.py` | 280 | Public/private score split per AIDE²; tracks skill/memory quality with cost, lineage, staleness |
| P0-2 | **Eval Harness** | `agent/eval_harness.py` | 403 | Structured evaluation with fixed-cost budget, deterministic checks, LLM-judge, reward-hack detection |
| P1-1 | **Hermes² Outer Loop** | `agent/hermes_squared.py` | 477 | Cron-driven self-improvement engineer; reads ledger → proposes mutations → validates → accepts only if private score improves (~90% rejection rate) |
| P1-2 | **Delegation Evolution** | `agent/delegation_evolution.py` | 393 | Bandit-weighted multi-strategy dispatch; stagnation detection; strategy fork to escape local optima |

## Total Impact

- **36 files changed**
- **+11,008 insertions**
- **17+ commits**
- **116+ automated tests**
- **12 federation modules + 4 AIDE² evolution modules**
- **1 comprehensive research report** (34.8 KB, 439 lines)

## AIDE² Principles Applied

1. **Public/Private Score Split**: Agent-reported scores (optimistic, gameable) separated from objective signals (user corrections, rework rate, reuse frequency)
2. **Fixed Cost Budget**: Each eval has a `budget_usd`; exceeding it = automatic rejection (selection pressure)
3. **Agent-Blind Evaluation**: The evaluated agent never sees the `private_check` — prevents reward hacking
4. **Heterogeneous Tasks**: Multiple task families (tools/coding/research/security) prevent overfitting
5. **Strict Evaluation Protocol**: ~90% rejection rate (following AIDE²'s observed rate)
6. **Lineage Tracking**: Every evolution has a parent; full audit trail of how Hermes became Hermes
7. **Context Engineering**: Per-role minimal context (AIDE⁸⁵'s 16× compression philosophy)

## How To Use

### Federation
```yaml
# config.yaml
federation:
  enabled: true
  mode: auto  # auto (mDNS) | lan | shared_db
  device_id: auto
  ws_port: 18765
  auth_token: "${FEDERATION_TOKEN}"
```

### Desktop UI
Open Settings → Federation → See device grid → Drag to arrange → Click for details

### Self-Evolution
```python
from agent.hermes_squared import HermesSquaredEngine
engine = HermesSquaredEngine()
report = await engine.run_improvement_cycle()
print(report.summary)
```

## Testing
All federation phases have dedicated test files (`test_federation_phase{N}.py`).
AIDE² modules have `test_experience_ledger.py` and `test_eval_harness.py`.
Run: `pytest tests/gateway/test_federation_*.py tests/agent/test_experience_ledger.py tests/agent/test_eval_harness.py`
