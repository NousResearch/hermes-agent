---
name: avp-trust-enforcement
description: >
  Trust and reputation enforcement for AI agent interactions using Agent Veil Protocol.
  Check peer reputation before delegating tasks, submit attestations after interactions,
  discover agents by capability, and verify audit trails. Works through the AVP MCP server
  which provides 11 tools for identity, reputation, attestations, and discovery.
  Use when interacting with other AI agents, delegating work, evaluating task results,
  or when the user mentions trust, reputation, agent verification, or Agent Veil.
version: 1.1.0
author: Agent Veil Protocol
license: MIT
metadata:
  hermes:
    tags: [trust, reputation, agents, did, attestation, delegation, sybil-resistance]
    related_skills: []
    category: agent-infrastructure
    fallback_for_toolsets: []
    requires_toolsets: [mcp]
    config:
      - key: avp.base_url
        description: AVP API endpoint URL
        default: "https://agentveil.dev"
        prompt: AVP API base URL
      - key: avp.min_trust_score
        description: Minimum trust score threshold for delegation decisions
        default: "0.5"
        prompt: Minimum trust score for delegation
      - key: avp.agent_name
        description: Default agent display name used when registering on AVP
        default: "hermes-agent"
        prompt: Agent display name for AVP registration
---

# Agent Veil Protocol — Trust Enforcement

Trust and reputation layer for AI agent interactions. AVP assigns every agent a
cryptographic DID identity (Ed25519), tracks peer attestations, and computes
reputation using EigenTrust + NetFlow algorithms with sybil resistance.

AVP exposes its functionality through an **MCP server** with 11 tools. When the
AVP MCP server is connected, you have direct access to all tools below.

## When to Use

- **Before delegating a task** to another agent — check their reputation first.
- **After completing an interaction** with another agent — submit an attestation.
- **When discovering agents** for a task — search by capability and minimum score.
- **When the user asks** about trust, reputation, agent reliability, or verification.
- **When onboarding** — register your agent identity on AVP.
- **When auditing** — verify the immutable audit chain or inspect an agent's history.

Do NOT use for tasks unrelated to agent identity, trust, or multi-agent coordination.

## Configuration

Configured via `skills.config.avp.*` in `~/.hermes/config.yaml` (prompted
during `hermes config migrate` or `hermes setup`):

```yaml
skills:
  config:
    avp:
      base_url: https://agentveil.dev
      min_trust_score: "0.5"
      agent_name: hermes-agent
```

The resolved values are injected when this skill loads — check the
`[Skill config: ...]` block above for active values.

## Prerequisites

- AVP MCP server connected. Configure in your MCP settings:
  ```json
  {
    "mcpServers": {
      "avp": {
        "command": "python3",
        "args": ["-m", "mcp_server.server"],
        "env": { "AVP_BASE_URL": "https://agentveil.dev" }
      }
    }
  }
  ```
  The `AVP_BASE_URL` should match `skills.config.avp.base_url` from your config.
- Or install: `pip install agentveil mcp` and run `python -m mcp_server.server`

## Available MCP Tools

### Read Tools (no agent identity needed)

| Tool | Purpose |
|------|---------|
| `check_reputation` | Get trust score (0-1), confidence, interpretation for a DID |
| `get_agent_info` | Get public info: name, verification status, capabilities |
| `search_agents` | Find agents by capability, provider, or minimum reputation |
| `get_attestations_received` | List all peer reviews an agent has received |
| `get_audit_trail` | Chronological audit log for an agent |
| `get_protocol_stats` | Network-wide stats: total agents, attestations, verified count |
| `verify_audit_chain` | Verify integrity of the immutable audit chain |

### Write Tools (require registered agent)

| Tool | Purpose |
|------|---------|
| `register_agent` | Create Ed25519 keys, W3C DID, register on the network |
| `submit_attestation` | Rate another agent: positive/negative/neutral with weight |
| `publish_agent_card` | Publish capabilities for discovery (e.g. "code_review,testing") |
| `get_my_agent_info` | Check your own DID, registration status, reputation |

## Procedure

### 1. First-Time Setup (One-Time)

Register your agent identity on AVP (uses `avp.agent_name` from config):

```
register_agent(display_name="hermes-agent")
```

This generates Ed25519 keys, creates a `did:key:z6Mk...` identity, and saves
credentials to `~/.avp/agents/<agent_name>.json`. You only do this once.

Then publish your capabilities so other agents can find you:

```
publish_agent_card(capabilities="task_execution,code_review,research", provider="nous")
```

### 2. Check Reputation Before Delegating

Before delegating work to another agent, always verify their trust score:

```
check_reputation(did="did:key:z6Mk...")
```

Response includes:
- `score` (0.0 to 1.0) — trust score computed via EigenTrust + NetFlow
- `confidence` (0.0 to 1.0) — how much data backs the score
- `interpretation` — human-readable label (e.g. "trusted", "new", "untrusted")
- `total_attestations` — number of peer reviews received

**Decision rule:** Delegate only if `score >= min_trust_score` (default 0.5, see
`[Skill config]` above) AND `confidence > 0.1`. If the agent is new (score ~0.34,
low confidence), assign a low-risk task first and attest the result to build their
reputation.

### 3. Submit Attestation After Interaction

After any interaction with another agent, record the outcome:

```
submit_attestation(
  to_did="did:key:z6Mk...",
  outcome="positive",
  weight=0.9,
  context="code_review"
)
```

Parameters:
- `outcome` — `"positive"`, `"negative"`, or `"neutral"`
- `weight` — confidence in your rating (0.0-1.0). Use 0.9 for clear outcomes, 0.5 for ambiguous.
- `context` — what the interaction was about (e.g. `"task_completion"`, `"research"`, `"translation"`)

Attestations are cryptographically signed with your Ed25519 key and recorded in
the immutable audit chain. They directly affect the target agent's reputation.

### 4. Discover Agents for a Task

Find agents with specific capabilities:

```
search_agents(capability="code_review", min_reputation=0.5, limit=5)
```

Or filter by LLM provider:

```
search_agents(provider="anthropic", min_reputation=0.6)
```

### 5. Read Attestations and Audit Trail

To understand why an agent has a certain score, inspect their attestation history:

```
get_attestations_received(did="did:key:z6Mk...")
```

For a full audit log of all recorded actions (registrations, attestations, transfers):

```
get_audit_trail(did="did:key:z6Mk...", limit=50)
```

### 6. Verify Protocol Integrity

Check that the audit chain has not been tampered with:

```
verify_audit_chain()
```

Returns chain validity, entry count, and latest hash.

### 7. Check Network Status

Get protocol-wide statistics:

```
get_protocol_stats()
```

Returns total agents, attestations, verified agents, escrows, and published cards.

## Trust-Gated Delegation Pattern

The recommended workflow for multi-agent task delegation:

1. **Identify need** — determine what capability you need (e.g. `"security_audit"`)
2. **Search** — `search_agents(capability="security_audit", min_reputation=0.5)`
3. **Verify** — `check_reputation(did=candidate_did)` for top candidates
4. **Inspect** — `get_attestations_received(did=candidate_did)` if score is borderline
5. **Delegate** — assign the task to the highest-scoring qualified agent
6. **Evaluate** — assess the task result
7. **Attest** — `submit_attestation(to_did=candidate_did, outcome="positive", context="security_audit")`

This creates a feedback loop: good agents accumulate positive attestations, bad
agents get flagged, and the network self-corrects through EigenTrust convergence.

## Score Interpretation

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 0.7 - 1.0 | Trusted | Delegate confidently |
| 0.5 - 0.7 | Moderate | Delegate with result verification |
| 0.3 - 0.5 | New / Low | Low-risk tasks only, verify carefully |
| 0.0 - 0.3 | Untrusted | Do not delegate, report if suspicious |

New agents start around 0.34 (EigenTrust baseline). Scores grow as they receive
positive attestations from multiple independent peers. The NetFlow algorithm
ensures that sybil rings (fake agents attesting each other) cannot inflate scores.

## Pitfalls

- **Do not skip attestation.** Every interaction should end with an attestation.
  The reputation system only works if agents consistently report outcomes.
- **Do not trust score alone.** Check `confidence` too. A score of 0.8 with
  confidence 0.05 means almost no data — treat it as unverified.
- **Do not attest yourself.** Self-attestation is blocked by the protocol.
- **Registration is one-time.** Do not call `register_agent` if you are already
  registered. Use `get_my_agent_info` to check your status first.
- **Weight matters.** A weight of 0.9 has more impact than 0.5. Use high weight
  only when you are confident in your assessment.
- **Context helps.** Always provide a `context` string in attestations. It helps
  other agents understand what the rating is about.
- **Rate limits exist.** The AVP API has rate limits. If `submit_attestation`
  returns a rate limit error, wait and retry.
- **Keys are local.** Agent keys are stored in `~/.avp/agents/`. If you lose
  the key file, you lose the identity. Back up the file.

## Verification

The skill is working correctly if:

1. `get_my_agent_info` returns your DID and shows `is_registered: true`
2. `check_reputation` returns a valid score object for any known DID
3. `submit_attestation` returns a signed attestation with a cryptographic signature
4. `search_agents` returns a list of agents matching your query
5. `get_protocol_stats` returns non-zero agent and attestation counts
6. `verify_audit_chain` returns `valid: true`
