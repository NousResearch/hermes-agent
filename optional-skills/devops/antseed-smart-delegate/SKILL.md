---
name: antseed-smart-delegate
description: "Use when delegating LLM calls through AntSeed P2P network. Auto peer selection by task type, cost-aware routing, fallback on failure. Requires funded wallet + buyer proxy."
version: 2.0.0
author: "Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
required_environment_variables:
  - name: ANTSEED_IDENTITY_HEX
    prompt: AntSeed buyer identity (64 hex chars, no 0x prefix)
    help: "Run: antseed buyer wallet create  →  cat ~/.antseed/identity.key"
    required_for: opening payment channels
prerequisites:
  commands: [antseed]
metadata:
  hermes:
    tags: [antseed, p2p, delegation, smart-routing, peer-selection, fallback]
    related_skills: []
    requires_toolsets: [terminal]
---

# AntSeed Smart Delegate

> **Prerequisite:** Funded AntSeed wallet + running buyer proxy. See `references/setup.md`.

DO NOT read script files. DO NOT patch scripts. Just run them.

All model/peer data is fetched **live** from the AntSeed network — no hardcoded catalogs.

## When to Use

- User asks to delegate via AntSeed
- Previous AntSeed delegation failed (502/timeout/402)
- User wants best peer+model for a specific task type
- Setting up AntSeed delegation on a new machine

**Don't use for:** Direct LLM calls through OpenAI/Anthropic. Tasks not needing model inference.

## Quick Reference

| Command | When |
|---------|------|
| `bash ${HERMES_SKILL_DIR}/scripts/models.sh` | List all live models grouped by category |
| `bash ${HERMES_SKILL_DIR}/scripts/models.sh --json` | Same as JSON for parsing |
| `bash ${HERMES_SKILL_DIR}/scripts/best-peer.sh research` | Research/deep-thinking task |
| `bash ${HERMES_SKILL_DIR}/scripts/best-peer.sh code` | Code generation task |
| `bash ${HERMES_SKILL_DIR}/scripts/best-peer.sh vision` | Image/multimodal task |
| `bash ${HERMES_SKILL_DIR}/scripts/best-peer.sh chat` | General conversation |
| `bash ${HERMES_SKILL_DIR}/scripts/best-peer.sh cheap` | Minimum cost routing |
| `bash ${HERMES_SKILL_DIR}/scripts/best-peer.sh any` | Any available model |
| `delegate_task(provider="antseed", model="<result.model>", goal="...")` | Delegate with recommended model |

## Procedure

1. **Check proxy** — `curl -sf http://127.0.0.1:8377/v1/models | head`. If down → `antseed buyer start`
2. **Find best peer** — Run `best-peer.sh <task_type>`. Read `recommended.model` and `recommended.peer_id` from JSON output.
3. **Pin peer** — `antseed buyer connection set --peer <peer_id>`
4. **Delegate** — `delegate_task(provider="antseed", model="<model>", goal="...")`
5. **On failure** — Use `fallback_chain` from `best-peer.sh` output. Max 3 retries, then alert user.

## Error Handling

| Error | Meaning | Fix |
|-------|---------|-----|
| `proxy_down` | Buyer proxy not running | `antseed buyer start` |
| `no_peer` | No peer pinned | Run `best-peer.sh` → pick → pin |
| `no_funds` | Deposits = 0 | `antseed buyer deposit 1` |
| 502/timeout | Peer unreachable | Try next peer in fallback chain |
| 400 "model not found" | Model catalog drifted | Re-run `best-peer.sh` |
| 402 | Insufficient deposits | Alert user — needs more funds |

## Pitfalls

- **Unicode tables:** AntSeed CLI uses `│` (U+2502), NOT ASCII `|`. Scripts use Python for robust parsing.
- **openai-responses protocol** requires streaming — not suitable for auxiliaries. `best-peer.sh` gives `chat_completions` a +10 score bonus.
- **Reserve ceiling ≠ price:** Peer may require $1 reserve even for cheap model.
- **Model catalog drift:** Peers add/remove models anytime. Re-run `best-peer.sh` on 400 errors.
- **No real data in examples:** Use placeholders only (`0x1234...abcd`, `<peer-id>`).

## Verification

- [ ] `bash ${HERMES_SKILL_DIR}/scripts/test.sh` passes all checks
- [ ] `antseed buyer balance` shows available deposits
- [ ] `curl -sf http://127.0.0.1:8377/v1/models` returns model list

## References

- `references/setup.md` — CLI install, wallet, config wiring
