---
name: hermes-agentic-trader
description: Autonomous Base/EVM trading agent using defi-trading-mcp (paper mode by default).
version: 0.6.0
author: Hermes Agentic Trader
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Trading, DeFi, Base, EVM, MCP, Crypto, Agentic]
    related_skills: []
---

# Hermes Agentic Trader (P0–P5)

Autonomous market scanning and trade **recommendations** on Base and EVM chains via the `defi-trading` MCP server. **Paper mode** is default (read/quote only). **Live execution** requires `mode: live`, a signed `mandate.json`, and passing the deterministic risk gate.

## When to Use

- User asks to scan Base/EVM markets, check DeFi portfolio, find trending or new pools, or get swap quotes.
- User wants an agentic crypto trader on Hermes with MCP execution (not CEX API keys).
- Scheduled cron job: "Run Base market scan and summarize opportunities."

## Prerequisites

1. **MCP server** (catalog install):
   ```
   hermes mcp install defi-trading
   ```
   Or preset:
   ```
   hermes mcp add defi-trading --preset defi-trading
   ```

2. **Credentials** in `~/.hermes/.env` (prompted at install):
   - `USER_ADDRESS` — EVM wallet `0x...`
   - `USER_PRIVATE_KEY` — local signing only
   - `COINGECKO_API_KEY` — market data
   - `ALCHEMY_API_KEY` — optional premium RPC

3. **Trader config** — copy `config/hermes_trader.example.yaml` to `~/.hermes/trader/hermes_trader.yaml`

4. **Node.js** on PATH (for `npx defi-trading-mcp`)

## Mode Rules (mandatory)

### Paper mode (default)
- **Allowed MCP tools:** portfolio reads, pool discovery, OHLCV, token data, `get_swap_quote`, `get_gasless_quote`, chain list, unit converters.
- **Blocked at transport layer:** `execute_swap`, `submit_gasless_swap` — enforced by `hermes_trader.hooks.pre_trade`.
- Respond with: "Paper mode — no transaction submitted."

### Live mode (P1)
1. Set `mode: live` in `~/.hermes/trader/hermes_trader.yaml`
2. Sign `~/.hermes/trader/mandate.json` (HMAC over wallet + timestamp)
3. Run every trade intent through `RiskGate.evaluate()` before execution
4. Never bypass the gate — LLM reasoning is advisory only

**Kill switch:** `HERMES_TRADER_KILL_SWITCH=1` or touch `~/.hermes/trader/KILL_SWITCH` — halts all write tools immediately.

Default chain: **Base** (`primary_chain: base` in config).
Always normalize MCP outputs into `MarketState` before reasoning (see `hermes_trader.market_state`).

## Standard Scan Procedure

Run each step via MCP tools on server `defi-trading`:

### 1. Load config

Read `~/.hermes/trader/hermes_trader.yaml` (or defaults: paper mode, chain base).

### 2. Perceive (read-only)

On `primary_chain` (default `base`):

1. `get_portfolio_tokens` — current holdings
2. `get_trending_pools` — volume leaders
3. `get_new_pools` — recent launches (memecoin discovery)
4. For top 1–2 candidates: `get_pool_ohlcv` + `get_token_data`

Build a mental `MarketState` summary: portfolio, trending, new pools, liquidity filters (`min_pool_liquidity_usd` from config).

### 3. Reason

Produce a **Trade Recommendation** (not execution in P0):

```json
{
  "action": "hold | buy | sell | watch",
  "chain": "base",
  "token_address": "0x...",
  "size_usd": 0,
  "confidence": 0.0,
  "reasoning": "...",
  "strategy_tag": "memecoin_momentum | rebalance | watchlist"
}
```

- If `action` is `buy` or `sell`, include `get_swap_quote` for sizing estimate only.
- If confidence < `min_confidence` from config → `action: hold`.

### 4. Output

Report:
- Portfolio snapshot
- Top 3 opportunities with liquidity/volume
- Trade recommendation JSON
- Explicit "PAPER MODE — no transaction submitted"

## Tool Allowlist Reference

Paper tools are defined in `hermes_trader.tools.PAPER_MODE_READ_TOOLS`. The MCP catalog installs this allowlist by default via `tools.default_enabled` in `optional-mcps/defi-trading/manifest.yaml`.

## Windows Note

On native Windows, if MCP connection fails with "Connection closed", ensure the catalog uses `npx` (already configured). Use `cmd /c npx ...` only if you manually add the server outside the catalog.

## Related Files

| Path | Purpose |
|------|---------|
| `hermes_trader/config.py` | Load trader YAML |
| `hermes_trader/market_state.py` | Normalize MCP payloads |
| `hermes_trader/tools.py` | Paper vs live tool policy |
| `hermes_trader/risk/gate.py` | Deterministic risk gate |
| `hermes_trader/risk/mandate.py` | Mandate signing + validation |
| `hermes_trader/hooks/pre_trade.py` | MCP write-tool transport gate |
| `hermes_trader/loop/scheduler.py` | Trading cycle orchestration |
| `hermes_trader/loop/executor.py` | Gate-approved MCP execution |
| `hermes_trader/loop/intent_schema.json` | TradeIntent JSON schema |
| `hermes_trader/memory/episodes.py` | SQLite trade episode ledger |
| `hermes_trader/memory/retrieval.py` | Top-K episode retrieval |
| `hermes_trader/memory/strategic_rules.yaml` | Distilled heuristics |
| `hermes_trader/reflection/judge.py` | LLM-as-Judge scoring |
| `hermes_trader/reflection/distill.py` | Strategic rule distillation |
| `hermes_trader/reflection/calibration_report.py` | Confidence calibration |
| `optional-mcps/defi-trading/manifest.yaml` | MCP catalog entry |
| `config/hermes_trader.example.yaml` | Example config |

## Signing a Mandate (live mode)

```python
from hermes_trader.risk.mandate import sign_mandate, save_mandate
import os
mandate = sign_mandate(os.environ["USER_ADDRESS"])
save_mandate(mandate)  # writes ~/.hermes/trader/mandate.json
```

Requires `USER_PRIVATE_KEY` or `HERMES_TRADER_MANDATE_SECRET` in `~/.hermes/.env`.

## Risk Gate (before any live trade)

```python
from hermes_trader.risk import RiskGate, TradeIntent
from hermes_trader.config import load_trader_config

gate = RiskGate(config=load_trader_config())
intent = TradeIntent.from_mapping({...})
decision = gate.evaluate(intent, market_state=state)
if decision.approved:
    # use decision.order — never call execute_swap without APPROVE
    pass
```

Reject reason codes: `KILL_SWITCH`, `PAPER_MODE`, `CHAIN_DENIED`, `ROLLOUT_CHAIN`, `ROLLOUT_CAP`, `OVERSIZE`, `DAILY_LIMIT`, `LOW_LIQUIDITY`, `SLIPPAGE`, `NO_MANDATE`, `LOW_CONFIDENCE`, `HOLD`.

## P2 — Autonomous Trading Cycle

Programmatic closed loop (`hermes_trader.loop`):

```python
from hermes_trader.loop import run_trading_cycle

result = run_trading_cycle(mcp_call=my_mcp_callable)
# result.market_state → result.intent → result.decision → result.execution
```

### Hermes cron (LLM-driven)

```python
from hermes_trader.loop import build_cron_job_spec
from cron.jobs import create_job

spec = build_cron_job_spec()
create_job(**spec)
```

Or CLI:

```bash
hermes cron add --id trader-scan \
  --schedule "*/15 * * * *" \
  --skill hermes-agentic-trader \
  --prompt "Run Base market scan cycle. Output TradeIntent or hold."
```

### Script cron (no agent)

```bash
python -m hermes_trader.loop.scheduler
```

Audit log: `~/.hermes/trader/cycles.jsonl`

## P3 — Episodic + Strategic Memory

Every trading cycle is persisted to SQLite (`~/.hermes/trader/trade_episodes.db`). Before reasoning, the agent retrieves:

1. **Strategic rules** from `~/.hermes/trader/strategic_rules.yaml` (bundled defaults if missing)
2. **Top-K similar episodes** (default 3) — same `strategy_tag`, liquidity band, gate outcome
3. **Working memory** — current portfolio/pool counts

Episodic memory is injected as **untrusted context** — it must never override `RiskGate` or `mandate.json`.

```bash
python -m hermes_trader.memory.migrate
```

```python
from hermes_trader.memory import EpisodeStore, retrieve_similar_episodes
store = EpisodeStore()
episodes = retrieve_similar_episodes(market_state, store, limit=3)
```

## P4 — Post-Trade Reflection

After a position closes (or 24h mark-to-market review), run the reflection pipeline:

```bash
python -m hermes_trader.reflection.pipeline --episode-id <id> --pnl-usd -8.5
python -m hermes_trader.reflection.pipeline --weekly
```

```python
from hermes_trader.reflection import run_reflection, run_weekly_calibration

result = run_reflection(episode_id, pnl_usd=-8.5)
print(result.score.overall)  # 1–5 judge dimensions
print(run_weekly_calibration())  # markdown calibration table
```

- **LLM-as-Judge** scores thesis/timing/sizing/execution (not just PnL)
- **Distillation**: score &lt; 3 or loss → `negative_constraints`; score ≥ 4 + profit → `positive_heuristics`
- **Calibration**: confidence buckets → suggested `min_confidence` tightening

Weekly cron:

```python
from hermes_trader.reflection import build_weekly_reflection_cron_spec
from cron.jobs import create_job
create_job(**build_weekly_reflection_cron_spec())
```

## P5 — Production Hardening

- **Secrets hygiene:** `USER_PRIVATE_KEY` only in `~/.hermes/.env` — never in prompts or logs. Addresses redacted at info level (`hermes_trader.audit.redact`).
- **MCP audit log:** `~/.hermes/trader/mcp_audit.jsonl` — `{correlation_id, tool, params_hash, result_status, latency_ms}` on every defi-trading call.
- **Write rate limit:** max `max_write_tools_per_hour` (default 10) enforced in `pre_trade` hook; successful writes recorded after MCP response.
- **Alerts:** `evaluate_alerts()` after each cycle — kill switch, consecutive losses, gate spike, failed tx → `~/.hermes/trader/alerts.jsonl`.
- **Rollout stages:** `rollout_stage` in config (`paper` → `canary` → `limited` → `steady`) caps capital and chains inside `RiskGate`.
- **Size modifier (optional):** `enable_size_modifier: true` applies reduce-only multiplier 0.5–1.0 before final order sizing.

```yaml
rollout_stage: canary
max_write_tools_per_hour: 10
consecutive_loss_alert_count: 3
gate_reject_spike_threshold: 10
enable_size_modifier: false
```

## Roadmap

- **P1:** ✅ Deterministic risk gate + `mandate.json` for live mode
- **P2:** ✅ Scheduled perceive→reason→gate→execute loop
- **P3:** ✅ Episodic memory (ReasoningBank-style) + strategic rules
- **P4:** ✅ Post-trade reflection + LLM-as-Judge + calibration report
- **P5:** ✅ Audit logging, rate limits, alerts, rollout stages, optional size modifier