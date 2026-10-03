---
name: litellm-gateway
description: "Use when routing Hermes through a local LiteLLM gateway."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [LiteLLM, Gateway, Routing, Caching, Cost-Tracking, Providers]
    related_skills: [hermes-agent]
---

# LiteLLM Gateway for Hermes

Run a profile-local [LiteLLM](https://github.com/BerriAI/litellm) proxy in front of an OpenAI-compatible Hermes provider. The gateway can normalize providers, return cost metadata, cache identical requests, and add routing/fallback policies without changing Hermes core.

Hermes already supports LiteLLM through its `custom` provider interface. This skill supplies the audited installation and verification workflow; it does not add a core tool or vendor-specific plugin.

## When to Use

- The user explicitly asks to integrate LiteLLM.
- A deployment needs an OpenAI-compatible routing layer, short-lived exact-response caching, cost headers, or multiple upstream deployments.
- Hermes must point at a locally managed gateway without exposing upstream credentials to the endpoint client.

Do not use LiteLLM merely to proxy one model when the user does not need gateway features; direct provider configuration is simpler and removes a failure point.

## Tested Baseline

| Item | Value |
|---|---|
| Repository | `BerriAI/litellm` |
| Release | `v1.94.0` |
| Commit | `38f2e023f1179d06a199f3d5f02702c89c1a8a58` |
| Python | `>=3.10,<3.15` |
| License | MIT outside `enterprise/`; enterprise directory separately licensed |
| Hermes transport | named custom provider |

Review the current upstream release and security advisories before changing the pin. The tested baseline intentionally removes optional `diskcache==5.6.3` because `PYSEC-2026-2447` had no fixed release at integration time; process-local memory caching works without it.

## Architecture

```text
Hermes Agent
  provider: custom:litellm
  base URL: http://127.0.0.1:4000/v1
  key: LITELLM_MASTER_KEY
          |
          v
LiteLLM proxy (profile-local isolated venv)
  message logging disabled
  local-memory exact cache, 600-second TTL
  cost and usage response headers
          |
          v
Upstream OpenAI-compatible provider
  key: LITELLM_UPSTREAM_API_KEY
  URL: LITELLM_UPSTREAM_BASE_URL
```

Keep the upstream key distinct from the gateway master key. Never replace the upstream key until it has been copied to `LITELLM_UPSTREAM_API_KEY`.

## Install

1. Resolve the active profile home from `$HERMES_HOME`. If unset, use the path returned by `hermes config path`, not a hardcoded default-profile path.
2. Inspect the upstream release, license, dependency manifest, and security posture.
3. Run the installer:

```bash
bash skills/autonomous-ai-agents/litellm-gateway/scripts/install.sh
```

The script creates `$HERMES_HOME/integrations/litellm/.venv`, installs `litellm[proxy]==1.94.0` and `pip-audit`, removes the vulnerable optional disk cache, and audits only the isolated site-packages directory.

4. Copy `templates/config.yaml` into the integration directory.
5. Add these secrets to the active profile `.env` without printing their values:

```dotenv
LITELLM_UPSTREAM_API_KEY=<existing upstream provider key>
LITELLM_UPSTREAM_BASE_URL=<existing upstream base URL>
LITELLM_MASTER_KEY=<new random gateway key>
```

Generate the master key with `python3 -c 'import secrets; print("sk-litellm-" + secrets.token_urlsafe(32))'` and write it directly to `.env`; do not paste it into chat, logs, YAML, launchd, systemd, or git.

## Configure the Model

Edit the copied LiteLLM config's model alias and upstream provider prefix as needed. The supplied template maps the public alias `hermes-default` to the example upstream `openai/gpt-5` and uses the profile secrets above.

For GPT-5.x reasoning models with Hermes function tools, configure the named custom provider with `codex_responses`; `/v1/chat/completions` rejects reasoning plus tool calls for these models.

```bash
hermes config set providers.litellm.base_url http://127.0.0.1:4000/v1
hermes config set providers.litellm.key_env LITELLM_MASTER_KEY
hermes config set providers.litellm.model hermes-default
hermes config set providers.litellm.api_mode codex_responses
```

Do not set `model.provider` globally until the direct gateway smoke test and a Hermes one-shot test both pass.

## Start and Verify

Start the gateway in a tracked background process. The launcher parses `.env` as data instead of sourcing it as shell code:

```bash
HERMES_HOME="${HERMES_HOME:-$(dirname "$(hermes config path)")}"
"$HERMES_HOME/integrations/litellm/.venv/bin/python" \
  skills/autonomous-ai-agents/litellm-gateway/scripts/run_proxy.py
```

Verify readiness and the model catalog with the master key. Then run:

```bash
"${HERMES_HOME}/integrations/litellm/.venv/bin/python" \
  skills/autonomous-ai-agents/litellm-gateway/scripts/smoke_test.py
```

The smoke test sends the same deterministic request twice. Success requires:

- both responses return HTTP 200 and the expected text;
- the model catalog contains the configured alias;
- the second response includes `x-litellm-cache-key` or completes at local-cache latency;
- usage and `x-litellm-response-cost` metadata are present.

Test the actual Hermes transport before switching globally:

```bash
hermes -z 'Reply with exactly LITELLM_HERMES_OK' \
  --provider custom:litellm -m hermes-default \
  -t safe --ignore-rules
```

Then activate:

```bash
hermes config set model.provider custom:litellm
hermes config set model.default hermes-default
```

Tool/provider changes apply to new sessions. Do not mutate the active conversation's provider mid-turn.

## Persistence

Use the OS service manager only after foreground verification.

- macOS: create a user LaunchAgent whose program invokes `run_proxy.py`; never place secrets in the plist.
- Linux: create a user systemd service whose `ExecStart=` invokes `run_proxy.py`; the launcher reads the profile `.env` as data.
- Windows requires an adapted PowerShell installer and service wrapper; this POSIX workflow is not declared supported.

Bind only to `127.0.0.1` unless remote access has a separate authenticated TLS boundary.

## Budgets and Cost Tracking

LiteLLM returns request-level cost and usage headers for known models without a database. Persistent spend dashboards, virtual keys, team budgets, and durable budget enforcement require the documented database-backed proxy configuration.

Do not claim persistent budget enforcement from an in-memory single-process setup. On LiteLLM `v1.94.0`, the model-level budget limiter emitted a non-blocking logging error for the tested custom GPT-5.x Responses route; omit that guardrail until upstream behavior is verified. Hermes `--usage-file` remains available for per-run accounting.

## Security

- Install into an isolated venv; never modify Hermes' runtime venv.
- Pin the tested LiteLLM version.
- Run `pip-audit` against the isolated site-packages directory, not the ambient Hermes environment.
- Disable message logging unless the user explicitly enables a reviewed observability sink.
- Keep the proxy loopback-only.
- Do not commit `.env`, generated master keys, logs, caches, or launchd/systemd files containing absolute personal paths.
- Treat third-party callbacks, database-backed admin UI, and remote proxy exposure as separate security reviews.

## Rollback

```bash
hermes config set model.provider openai-api
# restore the previous model.default if it changed
```

Stop the LiteLLM process/service. Keep the integration directory until direct-provider operation is verified, then remove it and delete only the three LiteLLM-specific variables from the profile `.env`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| HTTP 400: tools with `reasoning_effort` unsupported | Hermes used chat completions for GPT-5.x | Set named provider `api_mode` to `codex_responses` |
| `custom_llm_provider is required` from budget limiter | Model-level budget bug on custom Responses route | Remove model `max_budget`; use request cost headers or DB-backed budgets |
| First request succeeds, cache never hits | Payload differs or cache disabled | Reuse byte-equivalent request; verify `cache: true` and local type |
| Proxy starts but Hermes gets 401 | Gateway key not resolved | Set `providers.litellm.key_env LITELLM_MASTER_KEY` |
| Upstream 401 | Upstream key or URL not copied correctly | Verify `LITELLM_UPSTREAM_*` variables without printing values |
| Port 4000 busy | Existing proxy/process | Inspect the listener and stop the stale process; do not launch a second instance |

## Verification Checklist

- [ ] Canonical release, commit, license, and dependency audit recorded
- [ ] Isolated venv passes `pip-audit`
- [ ] Proxy binds only to loopback
- [ ] Message logging disabled
- [ ] Direct model and cache smoke tests pass
- [ ] Hermes one-shot through `custom:litellm` passes
- [ ] Provider activation occurs only after tests
- [ ] Rollback path is documented and tested
