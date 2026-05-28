# Decision Dossier: Gateway Choice
# Status: COMPLETE — Board Review Concluded
# Date: 2026-06-02

## Question
Use `hermes-cli` or `run_bridge.py` as the primary gateway entry point for Hermes Agent on Linux (.114)?

## Context
- The gateway (`gateway/run.py`) is already wired with 4 lifecycle calls to context_orchestrator
- `run.py` line 57 imports `from hermes_cli.fallback_config import get_fallback_chain` — hard dependency on hermes-cli
- All architecture diagrams, board review documents, and deployment docs designate hermes-cli as the gateway
- There's a known dual-gateway conflict between `hermes-cli` and `run_bridge.py` (PID 33241 on Linux)
- SSH to Linux .114 is confirmed working as of June 2
- Linux box is secondary hardware (was for Ollama), Mac is primary pipeline

## Options

### Option A: `hermes-cli` (Unified CLI Entry Point) ✅ RECOMMENDED
**Pros:**
- Single entry point for all operations (CLI + gateway)
- Consistent across platforms (Mac, Linux, Docker, WSL2)
- Already has systemd unit (`ai.hermes.gateway.plist`) and PID 18596 running on Linux
- The fallback chain mechanism (`get_fallback_chain()`) lives exclusively in `hermes_cli/fallback_config.py`
- Avoids dual-gateway conflict entirely by consolidating on one tool
- Better integration with the rest of the agent stack (run_agent.py, cli.py)
- Easier to maintain one code path

**Cons:**
- Heavier startup footprint for just running the gateway
- Coupled to CLI infrastructure even in headless/gateway mode
- More complex to run as a systemd service (mitigated by existing unit file)

### Option B: `run_bridge.py` (Dedicated Bridge Runner) ❌ NOT RECOMMENDED
**Pros:**
- Lightweight, purpose-built for gateway/bridge operation
- Already integrated with bridge/signals infrastructure
- Simpler to run as a background service on Linux
- Decoupled from CLI — can run independently

**Cons:**
- Separate code path from CLI — divergence risk
- Less feature-rich for debugging/administration
- Dual-gateway conflict unresolved (hermes-cli vs run_bridge.py)
- Would require rewriting import chain across 18,323 lines of run.py
- Treated as non-primary in all architecture docs

## Board Review Chain Result
| Model | Role | Verdict | Confidence |
|-------|------|---------|------------|
| DeepSeek v4 | Draft analyst | Option A | High |
| Grok-4.20 | Review | Option A | High |
| Ring-2.6-1t | Quality gate | Option A | 98/100 |
| Kimi K2 | Board member | Option A | Consensus |

**Final Board Confidence: WATER CLEAR (98/100)**

## Recommendation
**Option A: `hermes-cli`** — Unified entry point reduces maintenance surface, aligns with "Mac is primary pipeline" architecture, avoids the dual-gateway conflict entirely by killing run_bridge.py (PID 33241) and consolidating on one tool. The import chain (`run.py` → `hermes_cli.fallback_config`) makes this the only structurally sound choice.

## Decision
```
DECISION: Gateway entry point — hermes-cli
DATE: 2026-06-02
QUESTION: Use hermes-cli or run_bridge.py as primary gateway entry point?
OPTIONS:
  A) hermes-cli — unified CLI, consistent across platforms ✓ WATER CLEAR
  B) run_bridge.py — separate bridge runner, divergence risk
BOARD RECOMMENDATION: Option A (hermes-cli) — Confidence: 98/100
GERALD SIGN-OFF: Pending final confirmation
RATIONALE: run.py imports from hermes-cli at line 57. The fallback chain lives exclusively in hermes_cli/fallback_config.py. All architecture docs designate hermes-cli as the gateway. Killing run_bridge.py resolves the dual-gateway conflict (P0 blocker).
IMPACT: Gateway runs via unified entry point. run_bridge.py process killed on Linux. No import chain changes needed — already wired.
REVERSIBLE: Yes — run_bridge.py can be restarted, but would re-create the dual-gateway conflict.
```