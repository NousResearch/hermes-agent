---
name: antseed-smart-delegate
description: "AntSeed P2P status, dashboard, and smart delegation. Use when user asks about antseed status, balance, delegation, P2P models, or wants to run tasks through AntSeed network."
version: 2.0.0
author: Hermes Agent Community
license: MIT
metadata:
  hermes:
    tags: [antseed, p2p, delegation, status, dashboard]
    requires_toolsets: [terminal]
    prerequisites:
      commands: [antseed]
    required_environment_variables:
      - name: ANTSEED_WALLET_KEY
        prompt: AntSeed wallet private key or funded address
        help: See https://antseed.ai/docs
        required_for: full functionality
---

# AntSeed Smart Delegate

Quick status, dashboard, and task delegation through AntSeed P2P network.

## ⚡ Quick Reference — What to Run

**DO NOT read script files. DO NOT patch scripts. Just run them.**

| User says | Run this | Expected output |
|----------|----------|-----------------|
| "статус", "status", "баланс" | `bash SKILL_DIR/scripts/status-bar.sh --icon` | One-line: 🐝 AntSeed \| $1.00 \| model \| reqs \| peer \| state |
| "dashboard", "подробно" | `bash SKILL_DIR/scripts/dashboard.sh --compact` | Full table: wallet, balance, peer, models |
| "health", "чек" | `bash SKILL_DIR/scripts/preflight.sh` | ✓/✗ checks with details |
| "делегируй X", "run X on antseed" | `bash SKILL_DIR/scripts/auto-delegate.sh TASK_TYPE "prompt"` | Full result from P2P model |
| "расходы", "cost" | `bash SKILL_DIR/scripts/cost-report.sh` | Spending report |

Where `SKILL_DIR` = path to this skill directory (use the skill's own path).

Task types for auto-delegate: `chat`, `code`, `research`, `vision`

## How to Use

### Step 1: Check Status (always start here)

```bash
bash <skill_path>/scripts/status-bar.sh --icon
```

If this shows errors → run preflight.sh to diagnose:

```bash
bash <skill_path>/scripts/preflight.sh
```

### Step 2: Show Dashboard (when user wants details)

```bash
bash <skill_path>/scripts/dashboard.sh --compact
```

### Step 3: Delegate Task (when user wants AI work done)

```bash
bash <skill_path>/scripts/auto-delegate.sh chat "user's prompt here"
```

The script handles everything: preflight → model selection → execution → fallback → result extraction.

## Error Handling

| Error | What it means | Fix |
|-------|---------------|-----|
| `command not found: antseed` | CLI not installed | Tell user to run `npm install -g @antseed/cli` |
| `Proxy not responding` | Buyer proxy not running | Tell user to run `antseed buyer start` |
| `No deposits` | Wallet empty | Tell user to fund wallet with USDC on Base |
| `All models failed` | Peer offline or no matching service | Check peer connection, try different task type |
| HTTP 400/404 | Model not available on peer | Script auto-fallbacks to next model |

## Pitfalls

1. **Always run scripts, never read them** — they are tested and working. Reading wastes tokens.
2. **Scripts use ANSI-colored output** from `antseed buyer status`. The tbl_val() helper strips colors automatically. Do NOT re-implement parsing.
3. **Auto-delegate takes 30-60s** — it calls real LLMs over P2P network. Warn user about wait time.
4. **GLM models return thinking in reasoning_content field** — extraction handles this transparently.
5. **Disk/RAM tight on small servers** — AntSeed proxy + Hermes need ~500MB combined.
6. **Wallet address is sensitive** — status-bar shows truncated version only.
