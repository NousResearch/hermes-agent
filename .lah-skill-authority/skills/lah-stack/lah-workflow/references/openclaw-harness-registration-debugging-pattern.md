# OpenClaw Harness Registration Debugging Pattern

## When to use

The running OpenClaw gateway throws:

  `Requested agent harness "<X>" is not registered.`

or the Control UI shows a model as "Off" / "Unavailable" and fails before any reply.

This pattern applies when the error is **configuration-driven**: the config references a harness that no plugin registers.

It does NOT replace upstream-bug debugging (see `references/upstream-package-runtime-patching-pattern.md`). Use this pattern when the fix lives in the JSON config, not in source code.

---

## 1. Identify the running artifact

The gateway runs from an npm package, not from the local checkout:

```bash
ps aux | grep openclaw | grep gateway
# e.g. node .../node_modules/openclaw/dist/index.js gateway --port 18789
```

Confirm: `readlink -f /proc/<PID>/exe` and `cat /proc/<PID>/cwd`

**Key insight:** Dist files are the source of truth for runtime behavior. The local checkout (`lah-openclaw-mvp/`) may differ from what's actually loaded.

---

## 2. Find the config

The gateway loads `openclaw.json` from `~/.openclaw/openclaw.json` by default:

```bash
ls -la ~/.openclaw/openclaw.json
cat ~/.openclaw/openclaw.json | jq '.models.providers'
```

Check: `systemctl cat openclaw-gateway.service` for env overrides that change the config path.

---

## 3. Trace the error source

Search the dist files for the error message to find the throwing class:

```bash
grep -rl 'is not registered' /path/to/openclaw/dist/ | grep -v '.d.ts'
```

The `MissingAgentHarnessError` class lives in `model-fallback-<hash>.js`:
```javascript
class MissingAgentHarnessError extends Error {
  constructor(harnessId) {
    super(`Requested agent harness "${harnessId}" is not registered.`);
    this.name = "MissingAgentHarnessError";
    this.harnessId = harnessId;
  }
}
```

---

## 4. Find where the harness is selected

Look in `selection-CVIPXpKT.js` (or similar hash) for `selectAgentHarnessDecision`.

The selection flow:
1. `resolveAgentHarnessPolicy(params)` — resolves runtime from config
2. If runtime === "openclaw" — use built-in harness (always works)
3. If runtime === "auto" — search registered plugin harnesses
4. **If runtime is anything else** — look for plugin harness by that id — throw `MissingAgentHarnessError` if not found

The function `resolveAgentHarnessPolicy` is imported from `harness-runtimes--<hash>.js`.

---

## 5. Trace runtime resolution

In `harness-runtimes--<hash>.js`, the resolver `resolveAgentHarnessPolicy` calls `resolveModelRuntimePolicy` from `model-runtime-policy-BS47zw0g.js`.

The resolution chain:
```
resolveModelRuntimePolicy(params)
  → resolveAgentModelEntryRuntimePolicy(params)  — checks agent's models{} map
  → resolveModelConfig(params)                    — checks providerConfig.models[]
    → found model in provider
      → model has agentRuntime.id = "<id>"
  → returns { policy: { id: "<id>" }, source: "model" }
```

The **agentRuntime.id** field on a model config explicitly sets what harness to use for that model. If the value does not match a registered plugin harness id, the selection fails.

---

## 6. Verify no plugin registers the harness

Check which plugins are enabled:

```bash
jq '.plugins.entries | to_entries[] | select(.value.enabled) | .key' ~/.openclaw/openclaw.json
```

Search for the harness id in dist:
```bash
grep -rl '"<id>"' /path/to/openclaw/dist/
# Empty = no harness with that id exists anywhere
```

---

## 7. Root cause classification

| Scenario | Diagnosis | Fix |
|----------|-----------|------|
| Model's `agentRuntime.id` points to non-existent harness | `jq '.models.providers.<X>.models[].agentRuntime' config.json` reveals mismatch | Remove `agentRuntime` block — falls back to embedded harness |
| Plugin that should register the harness is disabled | Enabled plugin list missing the provider plugin | Enable the plugin or install it |
| Agent's `model` field uses non-existent provider/model ref | `agents.list[].model` or `agents.defaults.model.primary` references provider/model not in `models.providers` | Fix the model ref to a real provider/model |
| Harness name changed (rename drift) | Agent config uses old harness id, but plugin registers new id | Update config to match registered id |
| No harness needed (provider is API-only) | The model works as a regular API call, no special harness needed | Remove the `agentRuntime` block |

---

## 8. Verify the fix — diagnostic CLI test

After changing the config (e.g. removing `agentRuntime`), restart the gateway and verify the fix using the `openclaw agent` CLI with `--local --json`:

```bash
systemctl --user restart openclaw-gateway.service
sleep 4
openclaw agent --agent <agent-id> --message "salut" --local --json
```

**Why `--local`:** The `openclaw agent` command goes through the gateway WebSocket, which requires the client to hold `operator.write` scope. The `--local` flag runs the **embedded agent directly** — it always works for diagnostics and always prints detailed `provider-transport-fetch` and `model-fallback` events to stderr (see section 9 for interpretation).

**Before vs after comparison:**

| Before (broken) | After (fixed) | Meaning |
|-----------------|---------------|---------|
| `MissingAgentHarnessError: Requested agent harness "X" is not registered.` | `FailoverError: The provider returned an HTML error page...` | Harness selection works; request now reaches the provider. Provider integration is a separate issue. |
| No `provider-transport-fetch` event | `[provider-transport-fetch] start provider=cloe api=openai-completions model=brain url=http://127.0.0.1:4000/...` | Built-in openclaw harness correctly identified the provider, model, and endpoint. |

**Critical diagnosis markers (after fix):**

The output **must** show:
- `provider=cloe` (the CLOE provider, not an OpenAI fallback)
- `model=brain` (the model, unchanged from config)
- `url=http://127.0.0.1:4000/...` (reaches the configured provider port)

If a different provider or model appears in the diagnostic output, routing is still misconfigured. If no `provider-transport-fetch` event appears at all, the harness selection is still broken.

**Gateway health check:**

```bash
curl -s http://127.0.0.1:<port>/health
# Expected: {"ok":true,"status":"live"}
```

The Control UI may still show the model as "Off" / "Unavailable" after the fix. This is expected when only the harness error is resolved but the provider endpoint does not serve an OpenAI-compatible API — see section 10 for what "Off" actually means.

---

## 9. Interpret embedded agent diagnostic output

When running `openclaw agent --local --json`, the diagnostic events printed to stderr tell the full story of what happened after the harness was selected. The three most important event types:

### provider-transport-fetch

```
[provider-transport-fetch] start provider=cloe api=openai-completions model=brain
  method=POST url=http://127.0.0.1:4000/chat/completions
[provider-transport-fetch] response provider=cloe api=openai-completions model=brain
  status=404 elapsedMs=32 contentType=text/html; charset=utf-8
```

The `start` line confirms the harness made a real HTTP request to the provider. The `response` line shows the result. A `status=404` with `contentType=text/html` means the provider doesn't expose that endpoint (provider integration issue, not harness issue). A `status=200` with `contentType=application/json` means the provider is working.

### model-fallback/decision

```
[model-fallback/decision] model fallback decision:
  decision=candidate_failed requested=cloe/brain candidate=cloe/brain
  reason=model_not_found next=none
```

This event fires when the model is unreachable on the provider. `reason=model_not_found` means the provider returned a non-200 status. `next=none` means there is no fallback model configured.

### embedded run failover decision

```
[agent/embedded] embedded run failover decision:
  runId=... stage=assistant decision=surface_error
  reason=model_not_found from=cloe/brain profile=-
```

This is the agent-level decision after the model fails. `decision=surface_error` means the error will be shown to the user. There is no automatic fallback to another provider.

**Telling harness failure from provider failure:**

| Symptom | Root cause |
|---------|------------|
| No `provider-transport-fetch` event at all | Harness selection failed before reaching the provider |
| `provider-transport-fetch` shows the wrong provider/model | Routing/mapping misconfiguration |
| `provider-transport-fetch` shows correct provider/model but non-200 status | Provider integration issue (missing endpoint, auth failure, etc.) |
| `provider-transport-fetch` shows correct data but times out | Network or provider capacity issue |

---

## 10. Related patterns

- `references/upstream-package-runtime-patching-pattern.md` — when the fix is in source code, not config
- `references/gateway-routing-fix-pattern.md` — gateway routing/deployment fixes
- `references/cloe-canonical-pipeline-e2e-certification-pattern.md` — CLOE pipeline diagnostics
