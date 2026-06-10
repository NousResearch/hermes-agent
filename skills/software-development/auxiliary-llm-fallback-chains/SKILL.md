---
name: auxiliary-llm-fallback-chains
description: "Use when configuring or debugging per-task LLM fallback chains for Hermes auxiliary calls (vision, web_extract, compression, skills_hub, etc.) and when the main model has a fallback but aux tasks don't."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes, auxiliary, fallback, llm, providers, reliability, rate-limit]
    related_skills: [hermes-agent, systematic-debugging, test-driven-development]
---

# Auxiliary LLM Fallback Chains

## Overview

Hermes runs two parallel provider systems: the **main chat model** (governed by `model:` and `fallback_providers:` in `config.yaml`) and the **auxiliary client** (governed by `auxiliary.<task>.*` in `config.yaml` — used for vision, web extraction, context compression, session search, skills_hub, MCP, title generation, triage, kanban, profile description, curator, etc.).

The two systems have **independent fallback chains.** A `fallback_providers:` entry at the top of `config.yaml` only protects the main chat loop. It does **not** automatically apply to auxiliary tasks. If your main provider hits a 429/402/connection error, the main loop will fall back, but your compression/vision/web_extract calls will keep hammering the failing provider until the auxiliary call raises and the whole task fails.

This skill covers how to configure a per-task `fallback_chain` for the auxiliary client, the silent-failure mode that hides broken chains, and the end-to-end verification recipe.

## When to Use

Use when:
- The main chat model has a working `fallback_providers:` chain but auxiliary tasks (vision, compression, web extract, etc.) still fail with 429/402 errors
- The user wants to extend a multi-provider setup to side-tasks (e.g. "if MiniMax quota runs out, fall back to StepFun for vision too")
- An auxiliary task is logging `Auxiliary <task>: ... and all fallbacks exhausted (fallback_chain + main agent model). Raising original error.` but the configured chain is non-empty in `config.yaml`
- Adding any new provider to an existing auxiliary task (vision, web_extract, etc.) and wanting graceful degradation

Do NOT use for:
- Main chat fallback configuration — that's a different code path (`agent_init.py:885-893` + `chat_completion_helpers.py:try_activate_fallback`); see `hermes-agent` skill
- Plugin-registered auxiliary tasks — those use `defaults` from `PluginContext.register_auxiliary_task` layered under `auxiliary.<task>`, the chain still works the same way
- Adding providers that have no API key in `.env` — the chain will silently skip them

## Quick Reference

| Field | Where | Effect |
|---|---|---|
| `model.provider` | `config.yaml` top | Main chat model provider |
| `fallback_providers:` | `config.yaml` top (list) | Main chat fallback chain (independent of aux) |
| `auxiliary.<task>.provider` | per-task block | Aux task's primary provider |
| `auxiliary.<task>.fallback_chain` | per-task block (list) | Aux task's fallback chain (independent of main) |

The 12 built-in auxiliary tasks (as of June 2026): `vision`, `web_extract`, `compression`, `skills_hub`, `approval`, `mcp`, `title_generation`, `triage_specifier`, `kanban_decomposer`, `profile_describer`, `curator`, `session_search`. Plugin-registered tasks appear in the same section.

## Config Shape

```yaml
auxiliary:
  vision:
    provider: minimax
    model: MiniMax-M3
    timeout: 120
    extra_body: {}
    fallback_chain:
      - provider: stepfun
        model: step-3.7-flash
        base_url: https://api.stepfun.ai/step_plan/v1
        # api_key omitted → resolver pulls from env (e.g. STEPFUN_API_KEY)
  web_extract:
    provider: minimax
    model: MiniMax-M2.7-highspeed
    timeout: 360
    fallback_chain:
      - provider: stepfun
        model: step-3.5-flash-2603
        base_url: https://api.stepfun.ai/step_plan/v1
  compression:
    provider: minimax
    model: MiniMax-M2.7-highspeed
    timeout: 120
    fallback_chain:
      - provider: stepfun
        model: step-3.5-flash-2603
        base_url: https://api.stepfun.ai/step_plan/v1
  # ... repeat for skills_hub, approval, mcp, title_generation,
  #     triage_specifier, kanban_decomposer, profile_describer,
  #     curator, session_search
```

Each chain entry accepts: `provider`, `model`, `base_url` (optional), `api_key` (optional — pulls from env if omitted). Flow YAML inline mapping syntax is fine for compact chains (`{ provider: stepfun, model: step-3.7-flash, base_url: ... }`).

## How the Fallback Fires

`agent/auxiliary_client.py:call_llm()` runs the auxiliary call through this cascade when the primary provider fails:

1. **Same-provider credential-pool recovery** (line ~5115) — rotates the credential pool for the same provider. Doesn't help if the entire provider is down.
2. **`_try_configured_fallback_chain(task, failed_provider, reason)`** (line ~2904) — reads `auxiliary.<task>.fallback_chain` from config, iterates each entry, calls `_resolve_single_provider()` to build a client. Skips entries that match `failed_provider` (line 2933) to avoid loops.
3. **`_try_main_agent_model_fallback(...)`** (line ~2855) — last-resort safety net that uses the main chat provider+model. Skipped if the failed provider IS the main provider (line 2879) — common case when both are the same.
4. Re-raise original error with a `WARNING` log if all three layers return None.

Capacity errors (`_is_payment_error` for 402/credit-exhaustion and `_is_connection_error` for transport failures) bypass the explicit-provider gate at line 5201, so fallback fires even when the user explicitly chose the failing provider. **Rate-limit errors (429) do NOT bypass the gate** — that is intentional, but means a `provider: minimax` + MiniMax 429 will not fall back. Workaround: switch to `provider: auto` or add `_is_rate_limit_error` to `is_capacity_error` at line 5200.

## Common Pitfalls

1. **Silent TypeError in `_resolve_single_provider` (the bug this skill exists to document).** Pre-fix version of `agent/auxiliary_client.py:2973-2979` called `resolve_provider_client(base_url=..., api_key=...)` but the real signature uses `explicit_base_url=...` / `explicit_api_key=...`. Every call raised `TypeError`, the surrounding `except Exception: fb_client = None` swallowed it, and the chain silently appeared empty. Symptom: log line `Auxiliary <task>: ... and all fallbacks exhausted (fallback_chain + main agent model)` even though `auxiliary.<task>.fallback_chain` is non-empty. Fix: use the explicit-prefixed kwarg names. Regression test in `tests/agent/test_auxiliary_client.py:TestConfiguredFallbackChainResolvesClient`.

2. **Top-level `fallback_providers:` does NOT cascade to auxiliary tasks.** This is the #1 user mistake. The main loop reads `agent._fallback_chain` (set from `fallback_providers:` in `agent_init.py:885-893`) but the auxiliary fallback cascade reads only `auxiliary.<task>.fallback_chain`. Two independent configs.

3. **`api_key: ''` in chain entry ≠ use env.** The chain reader does `str(entry.get("api_key", "")).strip() or None` (line 2937), so an empty string falls through to `None` and `_resolve_single_provider` passes `explicit_api_key=None`, which causes the resolver to pull from env (`STEPFUN_API_KEY` etc.). If the env var is missing, the chain entry returns `None` silently. Always confirm the env var is loaded with `os.environ["STEPFUN_API_KEY"]` or via `get_env_value("STEPFUN_API_KEY")` from `hermes_cli.config`.

4. **`base_url` must match the provider registry default or the chain silently uses the registry default.** This is actually convenient for StepFun (its registry default IS `https://api.stepfun.ai/step_plan/v1`), but breaks for custom providers, Azure, or any provider where the chain's `base_url` differs from the registry. After the TypeError fix above, the resolver will honor the chain's `base_url` explicitly.

5. **`fallback_chain` entry whose `provider` matches the failed provider is silently skipped** (line 2933). This is correct behavior (avoiding the same-backend loop) but means a `provider: minimax` entry in the chain won't catch a MiniMax failure. Use a different provider in the chain.

6. **The main-model safety net (`_try_main_agent_model_fallback`) is a no-op when the main and aux providers are the same.** If you set `model.provider: minimax` and `auxiliary.*.provider: minimax`, the main-model fallback is skipped because the failure IS the main model. This is the default setup; the per-task chain is the only thing protecting you.

7. **`timeout` on a fallback entry is not honored.** The chain entry's `timeout` field is read by `_get_task_timeout` (which reads `auxiliary.<task>.timeout`, NOT the chain entry) — chain entries have no per-entry timeout override. If the chain provider is slower than the primary, raise `auxiliary.<task>.timeout` to cover the chain's expected latency.

8. **`fallback_providers:` rate-limit gate is different from auxiliary's.** Main loop gates 429 fallback on `_pool_may_recover_from_rate_limit` and a 60s cooldown. Auxiliary's `_try_configured_fallback_chain` doesn't pool — it tries each entry until one works. The two behaviors can desync (main is in cooldown, aux is still trying) but that's intentional.

## End-to-End Verification Recipe

After editing `~/.hermes/config.yaml`, verify the chain actually works **before** trusting it in production. The chain may parse correctly but resolve to `None` due to a silent bug — always run this recipe:

### 1. Confirm config parses and reads the chain

```python
import yaml
with open('/Users/ashuaria/.hermes/config.yaml') as f:
    cfg = yaml.safe_load(f)
for task, tc in cfg['auxiliary'].items():
    chain = tc.get('fallback_chain') if isinstance(tc, dict) else None
    fb0 = chain[0] if chain else None
    print(f"  {task:20s} {tc.get('provider')}/{tc.get('model'):24s} chain[0]={fb0}")
```

### 2. Confirm `_try_configured_fallback_chain` returns a real client

```python
import os, sys
sys.path.insert(0, '/Users/ashuaria/.hermes/hermes-agent')
from unittest.mock import patch
from hermes_cli.config import get_env_value
os.environ["STEPFUN_API_KEY"] = get_env_value("STEPFUN_API_KEY") or ""

import agent.auxiliary_client
from agent.auxiliary_client import _try_configured_fallback_chain

with patch("hermes_cli.config.load_config", return_value=cfg):
    for task in sorted(cfg['auxiliary'].keys()):
        c, m, l = _try_configured_fallback_chain(
            task=task, failed_provider="minimax", reason="verify")
        ok = "✓" if c else "✗"
        print(f"  {task:20s} {ok} {l or 'NONE':30s} {m or ''}")
```

If any task prints `✗ NONE`, the chain resolved to nothing — re-read the pitfalls section, especially #1 (the silent TypeError bug).

### 3. Confirm the live model actually responds

```python
# After step 2 gave you a real client
resp = c.chat.completions.create(
    model=m,
    messages=[{"role": "user", "content": "What is 2+2? One number."}],
    max_tokens=200,
    timeout=30,
)
print(resp.choices[0].message.content)
# Should print "4" (or a reasoned version ending in 4)
```

A non-empty response confirms: provider is reachable, API key is valid, model slug exists on the endpoint. If the model returns empty content, bump `max_tokens` — small values get consumed by reasoning traces on some providers.

### 4. Restart the gateway (only required after CODE changes, not config changes)

Config changes pick up live (the auxiliary client reads config per call). Code changes (like the `_resolve_single_provider` fix) require:

```bash
cd ~/.hermes/hermes-agent
venv/bin/pip install -e .
hermes gateway restart
```

Verify the new code is loaded by checking the running process started after your install (`ps -o etime,pid,command -p $(pgrep -f 'hermes_cli.main gateway')`).

## Verification Checklist

- [ ] `config.yaml` `auxiliary.<task>.fallback_chain` is set for every task whose primary provider is rate-limited or has credit exposure
- [ ] Each chain entry has a `provider` + `model` (and `base_url` if the provider has multiple endpoints, e.g. Azure, custom proxies)
- [ ] API key for each chain provider is in `~/.hermes/.env` and loaded by `get_env_value()` (NOT in `config.yaml` per the user's preference)
- [ ] End-to-end test (steps 1-3 above) returns ✓ for all 12 tasks
- [ ] At least one live HTTP call to the chain's model succeeds
- [ ] Gateway has been restarted since the last source code edit (only relevant if you patched `auxiliary_client.py`)
- [ ] `hermes gateway status` shows the running PID; `LastExitStatus` is 0 or a recent fresh start
- [ ] After a real failure, `tail ~/.hermes/logs/agent.log` shows the `Auxiliary <task>: configured fallback to fallback_chain[0](<provider>) (<model>)` log line (NOT the silent `all fallbacks exhausted` line)

## Related Code Pointers

- `agent/auxiliary_client.py:4844` — `call_llm()` main entry
- `agent/auxiliary_client.py:5186-5250` — fallback cascade (payment / connection / rate-limit)
- `agent/auxiliary_client.py:2904-2960` — `_try_configured_fallback_chain` (reads `auxiliary.<task>.fallback_chain`)
- `agent/auxiliary_client.py:2855-2901` — `_try_main_agent_model_fallback` (last-resort main-model safety net)
- `agent/auxiliary_client.py:2963-2988` — `_resolve_single_provider` (the helper that contained the silent-TypeError bug)
- `agent/auxiliary_client.py:4589-4630` — `_get_auxiliary_task_config` (where chain entries are actually read from disk)
- `agent/auxiliary_client.py:2266-2330` — `_is_payment_error` / `_is_rate_limit_error` (gates which errors trigger fallback)
- `tests/agent/test_auxiliary_client.py:TestConfiguredFallbackChainResolvesClient` — 5 regression tests pinning the kwarg names and the warning log
