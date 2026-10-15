# Agentic Services and x402 Profile

Decompose opportunities into bounded actions, authority requirements, exception paths, and machine-purchasable capabilities.

## Triggers

agent workflow, delegated outcome, automation, x402, machine services, capability market.

## Runes

- `RUNE.NG.AGENT.ACTION_DECOMPOSE`
- `RUNE.NG.AGENT.AUTHORITY_GATE`
- `RUNE.NG.AGENT.OUTCOME_CONTRACT`
- `RUNE.NG.X402.SCAN`
- `RUNE.NG.X402.CAPABILITY_GRAPH`
- `RUNE.NG.X402.MISFIT_CHECK`

## Action decomposition

For each action: actor (human/agent), input, output, authority class, failure mode, exception path, verification, cost/price surface.

## Autonomy gates

| Gate | Meaning |
|------|---------|
| `AUTO_ALLOWED` | Safe to automate within declared policy |
| `POLICY_ALLOWED` | Allowed if policy engine confirms |
| `USER_CONFIRMATION_REQUIRED` | Human must confirm each run |
| `QUALIFIED_HUMAN_REQUIRED` | Specialist human required |
| `PROHIBITED` | Never automate / never execute here |

Default for Neon Genie drafts: no packet elevates above drafting; execution remains downstream.

## x402 positive test

Machine-addressable, bounded, explicitly priced, verifiable, frequent/dynamic, substitutable, policy-bounded, privacy-safe.

## x402 reject (Gate F)

Reject x402 when subscription, account billing, or a persistent relationship is superior — see `evals/cases/x402-misfit.json`.

## Outputs

- `AgenticServiceGraph`
- `NeonGenieRunReceipt`

Schema: `schemas/agentic-service-graph.schema.json`
