# Decision Dossier: Gateway Choice
# Status: IN BOARD REVIEW
# Date: 2026-06-02

## Question
Use `hermes-cli` or `run_bridge.py` as the primary gateway entry point for Hermes Agent on Linux (.114)?

## Context
- The gateway (`gateway/run.py`) is already wired with 4 lifecycle calls to context_orchestrator
- There's a known dual-gateway conflict between `hermes-cli` and `run_bridge.py`
- SSH to Linux .114 is confirmed working as of June 2
- Linux box is secondary hardware (was for Ollama), Mac is primary pipeline
- Gerald's directive: "awaiting two remaining decisions: gateway choice and quality gate enforcement"

## Options

### Option A: `hermes-cli` (Unified CLI Entry Point)
**Pros:**
- Single entry point for all operations (CLI + gateway)
- Consistent across platforms (Mac, Linux, Docker)
- Better integration with the rest of the agent stack (run_agent.py, cli.py)
- Easier to maintain one code path
- Auto-discovery of plugins and skills built in

**Cons:**
- Heavier startup footprint for just running the gateway
- Coupled to CLI infrastructure even in headless/gateway mode
- More complex to run as a systemd service

### Option B: `run_bridge.py` (Dedicated Bridge Runner)
**Pros:**
- Lightweight, purpose-built for gateway/bridge operation
- Already integrated with bridge/signals infrastructure
- Simpler to run as a background service on Linux
- Decoupled from CLI — can run independently

**Cons:**
- Separate code path from CLI — divergence risk
- Less feature-rich for debugging/administration
- Dual-gateway conflict unresolved (hermes-cli vs run_bridge.py)
- May lack some CLI conveniences for troubleshooting

## Recommendation
**Lean: Option A (`hermes-cli`)** — Unified entry point reduces maintenance surface, aligns with "Mac is primary pipeline" architecture, and avoids the dual-gateway conflict entirely by consolidating on one tool.

## Confidence: MEDIUM (needs board chain confirmation)