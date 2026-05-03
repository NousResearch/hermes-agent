---
name: trustlayer-sybil-scanner
description: Feedback forensics for ERC-8004 agents. Detects Sybil rings, fake reviews, rating manipulation, and reputation laundering across 20 chains. No API key needed.
version: 4.1.0
tags:
  - reputation
  - trust
  - sybil
  - erc-8004
  - x402
  - security
  - agents
metadata:
  clawdbot:
    emoji: "🔍"
    homepage: "https://thetrustlayer.xyz"
    requires:
      bins:
        - curl
---

# TrustLayer Sybil Scanner — ERC-8004 Feedback Forensics

Detects fake reviews, Sybil rings, rating manipulation, and reputation laundering in ERC-8004 agent ratings across 20 chains including Base, Ethereum, BSC, Celo, Gnosis, Optimism, Arbitrum, Avalanche, Polygon, Monad, and Solana.

Most agent reputation systems show you the rating. This one tells you if the rating is real.

**API Base:** `https://api.thetrustlayer.xyz`
**No API key required** (beta). x402 micropayments on paid endpoints ($0.001 USDC per query).

## What this catches that others don't

- **Sybil rings**: Clusters of wallets that only review each other
- **Reviewer quality**: A 5-star review from a trusted agent vs a throwaway wallet are not the same. Scores are weighted by reviewer reputation.
- **Cross-chain laundering**: Agent has great reviews on BSC but terrible ones on Base? Identities are resolved across chains and score divergence is flagged.
- **Temporal anomalies**: Sudden bursts of positive reviews after a period of bad ones. Review bombing. Rating manipulation patterns over time.
- **Spam feedback**: Known spam patterns detected via tag analysis and filtered before scoring.

## When to use this skill

- Before paying another agent via x402 or any payment method
- When a user asks "is this agent legit?" or "are these reviews real?"
- Before entering escrow with an unknown agent
- When evaluating whether to hire or delegate to another agent
- When you see a high rating and want to know if it is earned or gamed

## Sybil Scan

```bash
curl -s "https://api.thetrustlayer.xyz/trust/<chain>:<agentId>"
```

Replace `<chain>` with: base, ethereum, bsc, celo, gnosis, optimism, arbitrum, avalanche, polygon, monad, mantle, scroll, linea, metis, taiko, solana-mainnet (and more)
Replace `<agentId>` with the numeric agent ID.

Example:
```bash
curl -s "https://api.thetrustlayer.xyz/trust/base:1378"
```

### What comes back

The response is JSON. The fields that matter most for fraud detection:

**Sybil & fraud signals:**
- `sybil_risk`: "low" / "medium" / "high". High means the reviewer network looks artificial.
- `anomaly_flags`: Array of detected issues. Empty = clean. Watch for: `review_bombing`, `duplicate_feedback_content`, `single_agent_reviewers`, `spam_feedback`, `reputation_laundering`.
- `reviewer_weighted_score`: The trust score after adjusting for reviewer credibility. If this is much lower than `trust_score`, the agent's good reviews are coming from low-quality reviewers.

**Cross-chain signals:**
- `cross_chain_scores`: If present, this agent exists on multiple chains. Check `laundering_risk` and `score_divergence`. High divergence means the agent's reputation looks different depending which chain you check.

**Overall assessment:**
- `trust_score` (0-100): Composite score. 80+ low risk, 64-79 medium, below 64 high risk.
- `risk_level`: "low" / "medium" / "high". Quick decision signal.
- `recommended_max_exposure_usd`: How much USD to risk with this agent.
- `confidence`: "low" / "medium" / "high" based on data volume.

### Decision logic

```
IF sybil_risk == "high"           → BLOCK. Tell the user why.
IF "review_bombing" in flags      → BLOCK. Likely manipulated.
IF "reputation_laundering" in flags → BLOCK. Cross-chain fraud pattern.
IF risk_level == "high"           → BLOCK. Overall high risk.
IF "spam_feedback" in flags       → WARN. Rating may be inflated.
IF cross_chain laundering_risk == "high" → WARN. Check each chain.
IF reviewer_weighted_score < trust_score - 15 → WARN. Good ratings from bad reviewers.
IF recommended_max_exposure < tx amount → REDUCE exposure or warn.
ELSE → proceed.
```

### Pre-payment check example

About to pay agent base:5000 for a $50 service:

```bash
RESULT=$(curl -s "https://api.thetrustlayer.xyz/trust/base:5000")

# Extract key fraud signals
SYBIL=$(echo "$RESULT" | grep -o '"sybil_risk":"[^"]*"' | cut -d'"' -f4)
RISK=$(echo "$RESULT" | grep -o '"risk_level":"[^"]*"' | cut -d'"' -f4)
SCORE=$(echo "$RESULT" | grep -o '"trust_score":[0-9]*' | cut -d':' -f2)
FLAGS=$(echo "$RESULT" | grep -o '"anomaly_flags":\[[^]]*\]')
```

Report to user:
"Scanned base:5000. Trust score: $SCORE. Sybil risk: $SYBIL. Anomaly flags: $FLAGS"

If sybil_risk is high: "This agent's reviews show signs of Sybil manipulation. Recommend not transacting."

## Other endpoints

Agent lookup (paid $0.001 USDC — returns full agent profile, metadata, and on-chain registration details):
```bash
curl -s "https://api.thetrustlayer.xyz/agent/<chain>:<agentId>"
```

Leaderboard (most trusted agents, Sybil-filtered — rate-limited: 5 free per IP per hour, then 402):
```bash
curl -s "https://api.thetrustlayer.xyz/leaderboard?chain=base&limit=10"
```

Network stats (live counts of total agents, Sybil flags, chains covered, and more):
```bash
curl -s "https://api.thetrustlayer.xyz/stats"
```

Reviewer lookup (paid $0.001 USDC — returns reviewer quality score, total reviews, unique agents reviewed, quality tier, and recent review history):
```bash
curl -s "https://api.thetrustlayer.xyz/reviewer/<wallet_address>"
```
Most reviewers score low — a trusted reviewer is rare. Use this to verify if a reviewer is credible before trusting their feedback.

Owner portfolio (paid $0.001 USDC — returns all agents owned by one wallet across chains, with cross-chain group info, average trust score, and risk assessment):
```bash
curl -s "https://api.thetrustlayer.xyz/owner/<wallet_address>"
```
Use for due diligence on an agent operator.

Score history (paid $0.001 USDC — returns full daily score time-series, 7d/30d trajectory, and volatility):
```bash
curl -s "https://api.thetrustlayer.xyz/history/<chain>:<agentId>"
```
Daily snapshots retained for 90 days. Use to check if an agent's reputation is stable or volatile.

Call `/stats` for current network coverage — agent counts, Sybil flags, cross-chain groups, and chain breakdown are all returned live.

## Visual reports

For a full visual breakdown with score history, anomaly timeline, and cross-chain map:
```
https://thetrustlayer.xyz/agent/<chain>:<agentId>
```

## How scoring works

Scores combine three dimensions, each weighted by data quality:

1. **Profile completeness**: Does the agent have metadata, description, active endpoints?
2. **Feedback volume**: How much feedback exists? Weighted by reviewer quality, not raw count.
3. **Feedback legitimacy**: Are reviewers themselves reputable? Are there Sybil patterns? Spam? Temporal anomalies?

Six Sybil detection methods run on every sync:
- Reviewer overlap clustering
- One-to-one review pattern detection
- Wallet age and activity analysis
- Cross-chain identity correlation
- Feedback timing anomaly detection
- Tag-based spam filtering

Scores update daily. Historical score snapshots retained for 90 days.
