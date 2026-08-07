# actual.inc as an OpenCode provider

Verified end-to-end 2026-07 (OpenCode 1.18.3, macOS). Adds Actual's relay/GLM
cluster to OpenCode as a custom OpenAI-compatible provider.

## Design: secret in auth.json, config in opencode.json

OpenCode auto-injects a credential when the provider **id** in `opencode.json`
matches a credential **id** in `~/.local/share/opencode/auth.json`. So put the
key in auth.json and NOTHING sensitive goes in opencode.json. This is more robust
than `options.apiKey: "{env:ACTUAL_API_KEY}"`, because `{env:...}` only resolves
if the var is exported in the shell OpenCode launches from — and the Actual key
is typically only in `~/.hermes/.env`, not a shell profile, so the env form
breaks outside an inheriting terminal.

### 1. Add the credential to auth.json

File: `~/.local/share/opencode/auth.json`. Shape (preserve existing entries):
```json
{
  "anthropic": { "type": "api", "key": "..." },
  "actual":    { "type": "api", "key": "ac_..." }
}
```
Do this with a read-modify-write (json load, add the `actual` key, dump) so the
other credentials stay intact — don't overwrite the file.

### 2. Add the provider to opencode.json

File: `~/.config/opencode/opencode.json` (or `~/.opencode.json`). Add under
`provider` alongside anything already there. NO `apiKey` field — it comes from
auth.json by id match.
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "actual": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Actual (GLM cluster)",
      "options": {
        "baseURL": "https://api.actual.inc/v1",
        "headers": {
          "X-Cluster-ID": "<cluster-id-hash>"
        }
      },
      "models": {
        "glm-5.2-nvfp4": {
          "name": "GLM-5.2 (b300x8)",
          "limit": { "context": 1048576, "output": 65536 }
        }
      }
    }
  }
}
```
- `npm`: `@ai-sdk/openai-compatible` for `/v1/chat/completions`. Use
  `@ai-sdk/openai` only if the model needs `/v1/responses`.
- `options.headers.X-Cluster-ID`: pin to a specific cluster (optional; omit to
  let the relay route). Get the hash from the Actual console URL
  (`console/computers?cluster=<hash>`).
- `models.<id>`: the id MUST match what `GET /v1/models` returns. Discover it
  first: `curl -s https://api.actual.inc/v1/models -H "Authorization: Bearer ac_..." -H "X-Cluster-ID: <hash>"`.
- `limit`: lets OpenCode track remaining context (custom providers don't get
  this from models.dev). GLM-5.2 context = 1_048_576.

### 3. Verify live (headless)

```bash
opencode run -m actual/glm-5.2-nvfp4 "Reply with exactly this text: OPENCODE_ACTUAL_OK"
```
OpenCode DOES use the `provider/model` slash form on the CLI (unlike Hermes,
where the slash form 404s custom providers). Expect the exact reply. Run a second
reasoning check (e.g. "What is 17 * 23?") since GLM-5.2 is a reasoning model.

## Why no reasoning_effort trap here

The Actual relay rejects `reasoning_effort: xhigh` with an HTTP 400 (see the
`hermes-custom-providers` skill, pitfall 2). Hermes hits this because it forwards
its global `agent.reasoning_effort`. OpenCode's ai-sdk does NOT send that param,
so Actual + OpenCode works with zero reasoning config. No `reasoning_overrides`
equivalent needed.

## Gotchas

- `auth.json` is the same store `/connect` writes; editing it directly is fine
  and equivalent. `opencode auth list` should then show `actual` under
  Credentials.
- If discovery/models don't appear, confirm the provider id in opencode.json
  EXACTLY matches the auth.json credential id (`actual` == `actual`).
- No git-tracking risk on the default config dir (`~/.config/opencode` is not a
  repo), but still keep the key in auth.json, not opencode.json, as the habit.
