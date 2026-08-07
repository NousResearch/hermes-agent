---
name: actual-setup
description: Set up Actual Computer (actual.inc) inference in Hermes.
version: 2.0.0
author: shl0ms + Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [actual, actual-inc, provider, local-inference, relay, gguf, setup]
    category: devops
---

# Actual Computer Setup Skill

Sets up [actual.inc](https://actual.inc) (Actual Computer) as a Hermes inference
provider. Actual turns the user's own hardware into a private inference cluster
and exposes an OpenAI-compatible API two ways: a hosted end-to-end-encrypted
relay at `https://api.actual.inc` (authenticated with an `ac_` key), and a local
on-device daemon at `http://127.0.0.1:8080` (no auth on loopback). This skill
does not install the Actual daemon for the user — device authorization requires
a human in a browser.

## When to Use

- User wants to add actual.inc as an inference provider (cloud relay or local).
- User has an `ac_` key and wants Hermes routed through their Actual cluster.
- User wants fully-local, on-device inference via the Actual daemon.
- Troubleshooting: Actual requests failing with cryptic 400s or empty streams.

## Prerequisites

- Hermes has **first-class `actual` provider support** (provider id `actual`,
  aliases `actual-computer`, `actualcomputer`, `aci`). Do NOT configure Actual
  as a `custom_providers` / `providers.actual.*` entry on current Hermes — the
  built-in provider owns the name and handles base-url normalization, the
  Responses transport, and local no-auth automatically.
- Relay mode: an Actual account and an `ac_` inference key from
  https://actual.inc/user/keys.
- Local mode: the user has installed the daemon
  (`curl -fsSL "https://actual.inc/install" | bash`) and completed device
  authorization by running `actual` once and opening the printed
  `https://actual.inc/device?code=...` URL in a browser. Relay that URL to the
  user and WAIT — never invent an email or authorize on their behalf. Codes
  expire in 5 minutes; re-run `actual` for a fresh one.

## How to Run

### Relay / API mode

1. Put the key in `.env` (secrets only — never config.yaml):
   append `ACTUAL_API_KEY=ac_...` to `~/.hermes/.env`.
2. Verify the key and discover models with `terminal`:
   ```bash
   curl -s https://api.actual.inc/v1/models -H "Authorization: Bearer $ACTUAL_API_KEY"
   ```
3. Select provider + model:
   ```bash
   hermes config set model.provider actual
   hermes config set model.default "MODEL_ID_FROM_DISCOVERY"
   ```
4. Verify end-to-end:
   ```bash
   hermes chat -Q -q "Reply with exactly: ACTUAL_OK" --provider actual -m MODEL_ID
   ```

### Local mode

1. Human has installed + authorized the daemon (see Prerequisites).
2. Download and load a model (scriptable once authorized):
   ```bash
   actual models search "qwen2.5 0.5b instruct gguf" --limit 8 --no-prompt
   # Downloads REQUIRE an explicit quantization (409 ambiguous_model_download otherwise):
   actual models download "Qwen/Qwen2.5-0.5B-Instruct-GGUF/Q4_K_M"
   actual models list        # note the INSTALLED name (differs from download id)
   actual models load "qwen2.5-0.5b-instruct-q4_k_m"   # load by installed name
   ```
3. Point Hermes at the daemon. `ACTUAL_BASE_URL` with a loopback host flips the
   built-in provider into local no-auth mode automatically — no key needed:
   append `ACTUAL_BASE_URL=http://127.0.0.1:8080` to `~/.hermes/.env`, then:
   ```bash
   hermes config set model.provider actual
   hermes config set model.default "INSTALLED_MODEL_NAME"
   ```
4. Verify (reduced toolset — see context-window pitfall below):
   ```bash
   hermes chat -Q -q "Reply with exactly: LOCAL_OK" --provider actual -m INSTALLED_NAME -t file,web
   ```

## Quick Reference

| Thing | Value |
|---|---|
| Hosted relay | `https://api.actual.inc/v1` (normalized from bare host automatically) |
| Local daemon | `http://127.0.0.1:8080/v1` (no auth on loopback) |
| Key env var | `ACTUAL_API_KEY` (`ac_...`) |
| Base URL env var | `ACTUAL_BASE_URL` (loopback host ⇒ local no-auth mode) |
| Provider id / aliases | `actual` / `actual-computer`, `actualcomputer`, `aci` |
| Transport | Responses API (`codex_responses`) — built-in, do not override |
| Cluster pinning | `X-Cluster-ID` header via `providers.actual.extra_headers` in config.yaml |
| Model size guide | 0.5B Q4_K_M ~470MB (toy), 7-8B Q4_K_M ~4.5GB (daily driver), 32B ~20GB |

## Pitfalls

1. **reasoning_effort trap (handled by Hermes since the first-class provider).**
   Actual's SGLang/vLLM backends accept only `none/low/medium/high/max`;
   `xhigh`/`ultra` used to fail with a cryptic
   `Expecting value: line 1 column 1 (char 0)` (a wrapped HTTP 400). The
   built-in provider clamps `xhigh→high` and `ultra→max` on the wire. If a
   request still 400s this way on an old Hermes, set a per-model cap:
   `agent.reasoning_overrides.<model>: high` in config.yaml.
2. **Context-window overflow on small local models.** Hermes' default toolset
   is ~26k tokens of schemas plus a ~9k-token system prompt. A model loaded
   with a 32k context overflows before the first turn, and llama.cpp-family
   servers emit a bare `data: [DONE]` — Hermes reports
   `Provider returned an empty stream with no finish_reason`. This is NOT an
   SSE bug. Fixes: restrict tools (`-t file,web`), load the model with a
   larger `n_ctx`, or pick a >=64k-context model for the full toolset.
   Upstream tracking: #51448 (do not file new issues; add evidence there).
   Related but distinct: #65631 (HTTP-200 SSE carrying a 400), #56516
   (reasoning-only streams).
3. **Download ids vs installed names.** `actual models download` takes
   `repo/QUANT` and 409s without an explicit quantization;
   `actual models load` takes the INSTALLED name from `actual models list`.
4. **Reasoning models returning empty content.** GLM/Qwen reasoning variants
   emit thinking in a separate `reasoning` field and can burn a small
   `max_tokens` entirely on reasoning. Give generous max_tokens before
   assuming failure.
5. **Do not create a custom provider named `actual`.** Older setup guides
   (pre first-class support) wrote `providers.actual.*` config blocks. On
   current Hermes the built-in provider wins the name; stale custom blocks
   are ignored or conflict. Remove them and use the env vars + model.provider
   flow above.

## Verification

```bash
# Relay:
hermes chat -Q -q "Reply with exactly: ACTUAL_OK" --provider actual -m MODEL
# Local (small model — reduced toolset):
hermes chat -Q -q "Reply with exactly: LOCAL_OK" --provider actual -m MODEL -t file,web
# Provider status (local no-auth shows key_source=local-offline):
hermes status
```

For other OpenAI-compatible clients (e.g. OpenCode), see
`references/opencode.md`.
