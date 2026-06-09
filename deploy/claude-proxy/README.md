# Claude Code Proxy

`claude_proxy.py` turns the local `claude -p` CLI (OAuth-authenticated Claude
Code) into an **OpenAI-compatible** `/v1/chat/completions` endpoint, so Hermes
can use `claude-sonnet-4-6` as its default model through a `custom` provider.

```
Hermes gateway ──HTTP──▶ claude_proxy.py (127.0.0.1:11435) ──subprocess──▶ claude -p (OAuth) ──▶ Anthropic
                              │
                          fallback: deepseek (when Claude errors/empties out)
```

## ⚠️ This file is NOT baked into the image

The proxy runs from the **persistent data volume**, not the application image:

| Where | Path |
|-------|------|
| Host  | `/opt/hermes/data/proxy/claude_proxy.py` |
| In container | `/opt/data/proxy/claude_proxy.py` (mount: `/opt/hermes/data → /opt/data`) |
| Supervised by | s6 service `claude-proxy` (`/run/service/claude-proxy`) |
| Logs | `/opt/hermes/data/logs/claude-proxy/current` |
| Cache | `/opt/hermes/data/proxy/.cache.json` |

The CI/image deploy (`git reset --hard origin/main` into `/opt/hermes/app`) does
**not** touch the data volume, so this copy is the version-controlled
**source of truth** — it is deployed manually (see below). A data-volume reset
would otherwise silently lose the live fix.

## The bug this fixes (poison cache)

`call_claude` cached **every** non-error result for `CACHE_TTL` (24h) — including
responses where `claude -p` returned an empty `result` (a silent/tool-use-only
turn) or a claude-reported error. The cache key is `(tier, system, prompt)`.

It also checked `result.get("error")` — a key `claude -p` **never emits** (claude
uses `is_error`) — so even genuine errors were cached as "success."

Consequence: once an empty response for a given prompt was cached, every Hermes
retry replayed the cached empty (`[cache] HIT`), all 3 in-turn retries were
useless, and the turn failed over to DeepSeek — **for 24 hours**, until the TTL
expired. `agent/conversation_loop.py` was behaving correctly toward a
misbehaving provider; the bug was here.

## The fix

- **`_is_bad_result()`** — empty/whitespace `result`, `is_error: true`,
  non-`success` `subtype`, or a proxy error envelope.
- **Never cache bad results in the 24h cache** (`cache_set` guards on
  `_is_bad_result`; `_cache_load` drops any pre-fix poisoned entries on load).
- **Negative cache** (`NEG_CACHE_TTL`, default 60s) for bad results — absorbs
  conversation_loop's in-turn burst retries (so we don't respawn `claude` 3×
  within seconds) while letting a fresh attempt run again moments later. No
  more 24h poison.
- **One in-proxy retry** (`EMPTY_RETRIES`, default 1) on a bad result — recovers
  *transient* empties so Claude stays primary instead of failing over on a
  one-off.
- **Circuit breaker** — trips only when **many distinct** prompts return bad
  within a window (= broad Claude outage, e.g. auth/quota), short-circuiting to
  the fallback for a cooldown. A single poisoned prompt cannot trip it (the
  negative cache absorbs its retries).
- **Structured `claude_call:` log** on every call: tier, prompt/system size,
  cache decision, attempts, has_text, is_error, subtype, stop_reason,
  duration_ms, decision, elapsed_ms.
- Genuine claude errors (`is_error` / non-success subtype) now map to an OpenAI
  **error envelope** instead of a misleading empty completion.
- The existing 300s per-call subprocess timeout is the hang watchdog.

## Env knobs (all optional, sensible defaults)

| Var | Default | Meaning |
|-----|---------|---------|
| `CLAUDE_CACHE_TTL` | `86400` | 24h good-result cache; `0` disables |
| `CLAUDE_NEG_CACHE_TTL` | `60` | negative cache TTL for bad results; `0` disables |
| `CLAUDE_EMPTY_RETRIES` | `1` | in-proxy retries on empty/error |
| `CLAUDE_BREAKER_ENABLED` | `1` | circuit breaker on/off |
| `CLAUDE_BREAKER_THRESHOLD` | `8` | distinct bad prompts in window to trip |
| `CLAUDE_BREAKER_WINDOW` | `120` | breaker window (s) |
| `CLAUDE_BREAKER_COOLDOWN` | `90` | breaker open duration (s) |
| `CLAUDE_BIN` | auto | path to the `claude` binary (else `which` → node_modules glob) |
| `CLAUDE_PROXY_WORKDIR` | `/opt/data/proxy/workdir` | sandbox cwd, created `0700` at startup |
| `CLAUDE_PROXY_BACKEND` | `cli` | `cli` (OAuth subprocess) or `anthropic` (direct API) |
| `CLAUDE_PROXY_ALLOW_ANON` | `1` | serve keyless requests as tenant `anon`; `0` → 401 |
| `CLAUDE_PROXY_KEYS_FILE` | `/opt/data/proxy/keys.json` | key → tenant map (mtime-reloaded) |
| `CLAUDE_PROXY_DAILY_TOKEN_BUDGET` | `5000000` | per-tenant daily tokens; over → 429; `0` disables |
| `ANTHROPIC_API_KEY` | — | required only when `CLAUDE_PROXY_BACKEND=anthropic` |
| `ANTHROPIC_MODEL_HAIKU/SONNET/OPUS` | `claude-haiku-4-5` / `claude-sonnet-4-6` / `claude-opus-4-8` | tier→model map for the API backend |

`GET /health` reports backend, auth (allow_anon + keys_loaded), per-tenant
metering (`tenants`), daily-budget spend, cache + negative-cache + breaker state.

## Hardening (proxy lane)

### Sandbox (Phase 0)

Both the main completion subprocess and the Haiku classifier run `claude -p`
with **no tool access**: `--tools ""` (disable all tools) plus an explicit
`--disallowedTools` list of every dangerous tool (`Bash Read Write Edit Glob
Grep WebFetch WebSearch`), and **without** `--permission-mode bypassPermissions`
— non-interactive `-p` denies tool calls by default, so default-deny is the
safety net even if a flag is ignored. The cwd is a private `0700` workdir off
the shared data volume. Cache writes are atomic (temp file + `os.replace`) and
debounced to at most once per ~30s.

### Tenant identity + auth (Phase 1)

`hermes-provision-client` mints a random per-client proxy key (idempotent) and
records it in `<hermes_root>/proxy/keys.json` (`key -> client_name`, `0600`).
The gateway sends it as `Authorization: Bearer <key>`; the proxy reloads the map
on mtime change (no restart) and resolves the tenant (anonymous → `anon`).
Cache keys are tenant-scoped (good **and** negative cache); the circuit breaker
stays **global** (it measures Claude health, not tenant behavior). Every
`claude_call:` log line and the `/health` `tenants` block carry the tenant id —
the usage-metering seed.

> **Deploy safety:** `CLAUDE_PROXY_ALLOW_ANON` defaults to `1`, so a deploy does
> **not** take existing prod bots dark — keyless requests are still served.
> Tighten to `0` (requiring a valid key, else 401) only **after** keys are
> rolled out to every live bot.

### Pluggable backend + budgets (Phase 4)

`CLAUDE_PROXY_BACKEND` selects the generation backend; it **defaults to `cli`**
(prod behavior unchanged). `anthropic` is opt-in and calls the Anthropic
Messages API directly (stdlib `urllib`, no new deps), mapping the response into
the same result shape so cache/breaker/conversion are agnostic. The classifier
is **keyword-first** (free) and only falls back to a Haiku subprocess for
genuinely ambiguous prompts, caching the tier by prompt hash so repeats never
re-spawn it. A per-tenant daily token budget returns a `429` when exceeded so
the gateway's fallback chain takes over.

## Tests

Pure stdlib + pytest; no network or `claude` binary needed (subprocess mocked):

```bash
python3 -m pytest deploy/claude-proxy/test_claude_proxy.py -v
```

> Requires Python 3.10+ (the module uses `X | None` annotations).

## Deploy (manual — not via CI)

```bash
# from a checkout with the updated file:
scp deploy/claude-proxy/claude_proxy.py root@<server>:/tmp/claude_proxy.py

ssh root@<server> '
  TS=$(date +%s)
  cp /opt/hermes/data/proxy/claude_proxy.py /opt/hermes/data/proxy/claude_proxy.py.bak-$TS
  cp /tmp/claude_proxy.py /opt/hermes/data/proxy/claude_proxy.py
  chown 1001:1001 /opt/hermes/data/proxy/claude_proxy.py   # must be readable by the hermes user
  chmod 600 /opt/hermes/data/proxy/claude_proxy.py
  docker exec hermes /command/s6-svc -r /run/service/claude-proxy   # graceful restart (handles SIGTERM)
  sleep 5
  docker exec hermes /command/s6-svstat /run/service/claude-proxy   # expect: up
  docker exec hermes curl -s http://127.0.0.1:11435/health
'
```

> The deployed file **must be owned by the `hermes` user (uid 1001)** — the s6
> `run` script execs it via `s6-setuidgid hermes`, so a root-owned `600` file
> makes the proxy fail to start (exit 2, "permission denied").
