# Model Doctor 🩺

Proactive LLM health monitor for [Hermes Agent](https://hermes-agent.nousresearch.com).

Hermes already *rotates* models when one fails (round-robin). Model Doctor is the
missing **diagnostic layer**: it watches every provider API request, records
per-model uptime and latency, classifies failures, and answers the question
round-robin never does — **which models are sick, and why**.

## What it does

- **Passive monitoring** — hooks into `post_api_request` / `api_request_error`;
  zero extra network calls, works with any provider, no credentials.
- **Failure diagnosis** — classifies every failure:
  - `AUTH` → HTTP 401/403 (key invalid, expired, or **truncated** — a real-world
    config bug where YAML quoting cuts keys to 13 chars)
  - `RATE` → HTTP 429 (quota / rate limit)
  - `UPSTREAM` → HTTP 5xx (provider down)
  - `EMPTY` → 200 but zero content (stalled model or `max_tokens` too small)
  - `SLOW` → OK but >60s
- **Health verdicts** — `healthy / flaky / sick / auth-broken` per model with
  ok-ratio and rolling average latency.
- **State survives restarts** — JSON snapshot under `<HERMES_HOME>/model-doctor/`.

## Tools

| Tool | What it gives the agent |
|---|---|
| `model_doctor_status` | Full health table for every model seen |
| `model_doctor_ping` | Same view, re-sorted (no live network call) |

## Install

```bash
hermes plugins install <this-repo>
# or for development, symlink into the plugin dir:
ln -s /path/to/model-doctor ~/.hermes/plugins/model-doctor
```

## Example

> "why does my chat keep dying?"
> → agent calls `model_doctor_status`
> → `AUTH on provider 9router model bai/hy3 — key invalid/expired/truncated`
> → root cause found in one tool call.

## Design

- **Observer, not router.** Complements round-robin; never intercepts traffic.
- **Passive only.** Health comes from real traffic — no synthetic pings, no
  extra cost, no credentials.
- **Fail-open.** Any plugin error is logged and Hermes continues.

MIT License.
