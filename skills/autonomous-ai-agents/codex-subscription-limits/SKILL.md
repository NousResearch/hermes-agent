---
name: codex-subscription-limits
description: "Use when the user asks for OpenAI Codex subscription remaining limits/quota: 5h and weekly/7d windows, reset times, or a publishable implementation that mirrors the Codex app's Rate limits remaining panel."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Codex, OpenAI, Quota, Rate-Limits, Subscription]
    related_skills: [codex]
---

# Codex Subscription Limits

## Overview

Use this skill to answer or implement: "How much Codex subscription quota do I have left?" It targets the same data shown by the Codex app's **Rate limits remaining** panel: a short rolling window (usually **5h**) and a long window (**Weekly** / **7d**), both expressed as remaining percentages with reset times.

This is **subscription/OAuth quota**, not OpenAI API-key billing usage. Do not use the OpenAI dashboard billing endpoints for this task.

The recommended implementation is:

1. Prefer the local Codex App Server JSON-RPC method `account/rateLimits/read`.
2. Fall back to the ChatGPT backend endpoint `https://chatgpt.com/backend-api/wham/usage` when App Server is unavailable.
3. Convert used percentages to remaining percentages: `remaining_percent = 100 - used_percent`.
4. Print only normalized quota data; never print tokens, raw auth JSON, account IDs, or bearer tokens.

A reusable helper is included at `scripts/codex_limits.py`.

## When to Use

- User asks for current Codex remaining limits, quota, usage, or reset time.
- User attaches a Codex screenshot showing **Rate limits remaining** and asks how to retrieve the same values.
- You are building a Hermes command, CLI utility, cron watchdog, status-line widget, menu-bar widget, or dashboard card for Codex subscription usage.
- You need a publishable, auditable implementation pattern for GitHub distribution.

Do **not** use this skill for:

- OpenAI API-key billing spend, credits, or hard/soft limits.
- OpenRouter credits or provider-level API quota.
- Local token accounting for a single conversation.

## Known References

### `Easy-codex-limit-check`

- GitHub: <https://github.com/Elon-H/Easy-codex-limit-check>
- Scope: macOS menu-bar widget using local Codex login.
- Primary source: Codex App Server over stdio.
- JSON-RPC method: `account/rateLimits/read`.
- Fallback source: `https://chatgpt.com/backend-api/wham/usage`.
- Example display: `5h 69% 03:05 | W 95% Apr 29`.

This is the best public implementation reference because it uses the local Codex App Server first, which is less brittle than scraping sessions or relying only on an undocumented backend endpoint.

### `@kmiyh/pi-codex-plan-limits`

- GitHub: <https://github.com/Kmiyh/pi-codex-plan-limits>
- npm: <https://www.npmjs.com/package/@kmiyh/pi-codex-plan-limits>
- Scope: Pi coding-agent extension.
- Source: `https://chatgpt.com/backend-api/wham/usage` using Pi's `openai-codex` OAuth credentials.
- Reads `rate_limit.primary_window` and `rate_limit.secondary_window`.
- Computes `100 - used_percent`.

### `@slkiser/opencode-quota`

- GitHub: <https://github.com/slkiser/opencode-quota>
- npm: <https://www.npmjs.com/package/@slkiser/opencode-quota>
- Scope: OpenCode quota plugin/CLI.
- Example output includes `OpenAI Pro 5h 100%, 7d 100%`.

## Quick Check from Hermes

Use the bundled script directly. It is safe to run: it prints normalized quota only and never prints raw credentials.

```python
terminal(
    command="python3 skills/autonomous-ai-agents/codex-subscription-limits/scripts/codex_limits.py --pretty",
    workdir="/path/to/hermes-agent",
    timeout=30,
)
```

For JSON output suitable for a command, dashboard, or cron job:

```python
terminal(
    command="python3 skills/autonomous-ai-agents/codex-subscription-limits/scripts/codex_limits.py --json",
    workdir="/path/to/hermes-agent",
    timeout=30,
)
```

If the user is not inside the Hermes repo checkout, copy or reference the script path from the installed skill directory.

## Source 1: Codex App Server JSON-RPC

Preferred method:

```bash
codex app-server --listen stdio://
```

Send JSON Lines to stdin:

```json
{"method":"initialize","id":0,"params":{"clientInfo":{"name":"codex-subscription-limits","title":"Codex Subscription Limits","version":"1.0.0"}}}
{"method":"initialized","params":{}}
{"method":"account/rateLimits/read","id":1,"params":null}
```

Expected response shapes observed in public references:

```json
{
  "rateLimits": {
    "limitId": "codex",
    "limitName": "Codex",
    "planType": "pro",
    "primary": {
      "usedPercent": 25,
      "resetsAt": 1770000000,
      "windowDurationMins": 300
    },
    "secondary": {
      "usedPercent": 5,
      "resetsAt": 1770500000,
      "windowDurationMins": 10080
    }
  },
  "rateLimitsByLimitId": {
    "codex": { "...": "..." },
    "codex_some_model": { "...": "..." }
  }
}
```

Mapping:

- `primary` -> usually 300 minutes -> **5h**.
- `secondary` -> usually 10080 minutes -> **Weekly** / **7d**.
- `usedPercent` -> convert to `remaining_percent = clamp(100 - usedPercent, 0, 100)`.
- `resetsAt` -> Unix timestamp seconds.
- `limitName` -> display name. For the base `codex` limit, display `Rate limits remaining` to match the Codex panel.

## Source 2: ChatGPT `wham/usage` Fallback

Fallback endpoint:

```text
GET https://chatgpt.com/backend-api/wham/usage
```

Headers:

```text
Authorization: Bearer <access_token>
Accept: application/json
Content-Type: application/json
chatgpt-account-id: <account_id>   # include when known
```

Expected response shape:

```json
{
  "plan_type": "pro",
  "rate_limit": {
    "primary_window": {
      "used_percent": 24,
      "limit_window_seconds": 18000,
      "reset_at": 1770000000
    },
    "secondary_window": {
      "used_percent": 7,
      "limit_window_seconds": 604800,
      "reset_at": 1770500000
    }
  }
}
```

Potential local auth files:

- Codex CLI: `~/.codex/auth.json`.
- Hermes-managed Codex OAuth: `~/.hermes/auth.json` after `hermes auth add openai-codex`.

Auth JSON layouts change. The helper script recursively searches common fields such as `access_token`, `access`, `account_id`, and `accountId` without printing them.

## Output Contract

For publishable tools, normalize to this shape:

```json
{
  "source": {
    "provider": "app_server",
    "captured_at": "2026-05-15T10:30:00Z"
  },
  "rate_limits": [
    {
      "name": "Rate limits remaining",
      "plan_type": "pro",
      "five_h": {
        "remaining_percent": 75.0,
        "used_percent": 25.0,
        "reset_at": "2026-02-02T08:00:00Z",
        "window_minutes": 300
      },
      "week": {
        "remaining_percent": 95.0,
        "used_percent": 5.0,
        "reset_at": "2026-02-08T03:00:00Z",
        "window_minutes": 10080
      }
    }
  ]
}
```

Plain-text display should be concise:

```text
Codex limits remaining (app_server)
Rate limits remaining: 5h 75% reset 11:00 · Weekly 95% reset Feb 08
```

## Implementation Recipe

1. Check tool availability:

   ```bash
   command -v codex || true
   codex --version || true
   ```

2. Try App Server:

   ```bash
   python3 scripts/codex_limits.py --provider app-server --pretty
   ```

3. If App Server fails, try fallback:

   ```bash
   python3 scripts/codex_limits.py --provider wham --pretty
   ```

4. For production status surfaces, use auto mode:

   ```bash
   python3 scripts/codex_limits.py --provider auto --json
   ```

5. If using cron/watchdog, keep output silent unless threshold-based notifications are needed. For example, alert only when `five_h.remaining_percent < 20` or `week.remaining_percent < 20`.

## Publishing Checklist

For a GitHub-ready repository or package:

- Include a clear README explaining that this is subscription/OAuth quota, not API billing.
- State that App Server is preferred and `wham/usage` is a compatibility fallback.
- Never log or store raw bearer tokens.
- Include `--json` output for automation.
- Include `--pretty` output for humans.
- Include fixture tests for both App Server and `wham/usage` shapes.
- Include troubleshooting for expired login: run Codex login again or refresh Hermes `openai-codex` auth.
- Mention that backend fields may change and normalization should tolerate missing secondary windows.

## Common Pitfalls

1. **Using API billing endpoints.** They do not return Codex subscription 5h/weekly remaining limits.
2. **Reading `~/.codex/sessions`.** Session transcripts are not the source of subscription quota.
3. **Assuming only one bucket.** App Server can return `rateLimitsByLimitId` with model-specific buckets.
4. **Assuming secondary exists.** Some plans or failures may return primary only; show partial data instead of failing hard.
5. **Leaking credentials.** Do not print `~/.codex/auth.json`, `~/.hermes/auth.json`, bearer tokens, account IDs, or full HTTP headers.
6. **Treating `wham/usage` as stable public API.** It is an internal ChatGPT backend endpoint; keep it behind App Server and handle failures gracefully.
7. **Forgetting timezone.** Store JSON timestamps in UTC ISO-8601; format human output in local time unless the user requests otherwise.

## Verification Checklist

- [ ] `python3 scripts/codex_limits.py --fixture app-server --json` prints normalized `five_h` and `week` data.
- [ ] `python3 scripts/codex_limits.py --fixture wham --json` prints normalized `five_h` and `week` data.
- [ ] `python3 scripts/codex_limits.py --fixture app-server --pretty` prints a concise human-readable line.
- [ ] Live App Server probe works when `codex` is installed and logged in.
- [ ] Fallback probe fails safely when auth is missing or expired.
- [ ] No command output contains raw tokens or full auth files.
