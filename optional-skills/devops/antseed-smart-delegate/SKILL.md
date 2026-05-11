---
name: antseed-smart-delegate
description: "Smart delegation through AntSeed P2P network — auto peer selection by task type, cost tracking, fallback on failure. Use when user wants to delegate via AntSeed or when delegate_task through antseed provider fails."
version: 1.3.0
author: "Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [antseed, p2p, delegation, smart-routing, peer-selection, fallback]
    related_skills: [antseed]
    requires_toolsets: [terminal]
---

# AntSeed Smart Delegate

> **Important:** This skill requires a funded AntSeed wallet. See `references/setup.md`.

DO NOT read script files. DO NOT patch scripts. Just run them.

## ⚡ Quick Reference — What to Run

| Command | When |
|---------|------|
| `bash scripts/best-peer.sh research` | Research/deep-thinking task |
| `bash scripts/best-peer.sh code` | Code generation task |
| `bash scripts/best-peer.sh vision` | Image/multimodal task |
| `bash scripts/best-peer.sh chat` | General conversation |
| `bash scripts/best-peer.sh cheap` | Minimum cost routing |
| `bash scripts/best-peer.sh any` | Any available model |
| `delegate_task(provider="antseed", model="<result.model>", goal="...")` | Delegate with recommended model |

## When to Use

- User asks to delegate via AntSeed
- Previous AntSeed delegation failed (502/timeout/402)
- User wants best peer+model for specific task type
- Setting up AntSeed on new machine

**Don't use for:** Direct LLM calls through OpenAI/Anthropic. Tasks not needing model inference.

## Error Handling

| Error | Meaning | Fix |
|-------|---------|-----|
| `proxy_down` | Buyer proxy not running | `antseed buyer start` |
| `no_peer` | No peer pinned | Run `best-peer.sh` → pick → pin |
| `no_funds` | Deposits = 0 | `antseed buyer deposit 1` |
| 502/timeout | Peer unreachable | Try next peer in fallback chain |
| 400 "model not found" | Model catalog drifted | Re-run `best-peer.sh` |
| 402 | Insufficient deposits | Alert user — needs more funds |

Max 3 retries across fallback peers. On 3 failures — alert user.

## Pitfalls

- **Unicode tables:** AntSeed CLI uses `│` (U+2502), NOT ASCII `|`. Parse with `split('\u2502')` in Python.
- **`$` in grep:** Prices contain `$`. Use glob `[[ "$x" == *'$'* ]]` or awk instead of grep.
- **openai-responses protocol** requires streaming — not suitable for auxiliaries. `best-peer.sh` prefers `chat_completions`.
- **Reserve ceiling ≠ price:** Peer may require $1 reserve even for cheap model.
- **Model catalog drift:** Peers add/remove models anytime. Re-run `best-peer.sh` on 400 errors.
- **Subshell variable loss:** `while read` in pipes loses vars. Script uses temp files.
- **No real data in examples:** Use placeholders only (`0x1234...abcd`, `<peer-id>`).

## Verification Checklist

- [ ] Frontmatter valid: name, description ≤1024, version, author, license, platforms
- [ ] `scripts/best-peer.sh` passes `bash -n` and is executable
- [ ] `scripts/test.sh` passes all checks
- [ ] No hardcoded wallets, IPs, tokens, usernames
- [ ] `references/setup.md` and `references/model-catalog.md` present

## References

- `references/setup.md` — CLI install, wallet, config wiring
- `references/model-catalog.md` — full model list + selection logic