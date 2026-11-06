# Switching the default model / provider

The class of request: "make <model> the default", "can we run <model> through <provider>". It is a config
write plus an end-to-end check — not a `/model` switch, which only pins the session you are talking in.

## Procedure

1. **Resolve the exact model id from the vendor's live `GET /v1/models` first.** The id on the wire is
   not always the spelling in the picker or in a test fixture (a family id can carry a version marker
   that a curated row drops, or the reverse). Never set a default from a remembered or inferred id.
2. **Confirm the provider's credential is present**, by key *name* only:
   `grep -o '^[A-Z_]*' ~/.hermes/.env | sort -u` (never print values). Providers whose picker row is
   live-merged from the vendor list appear in `hermes status`'s provider line only when configured.
3. **Write the two keys, provider first:**
   ```bash
   hermes config set model.provider <provider>
   hermes config set model.default  <model-id>
   hermes config get model          # read-back
   ```
   `~/.hermes/config.yaml` is agent-write-protected — `hermes config set` / `unset` are the only way in.
   `hermes config unset <key>` exists for removal.
4. **Clear a stale route inherited from the previous provider.** `model.base_url` and `model.api_mode`
   describe *one* provider's endpoint and transport; `base_url` when set **overrides** the provider
   profile, so a switch that leaves it behind sends the new provider's key to the old vendor's URL (or
   the old vendor's key to a bare 401). The `model.provider` setter usually clears them itself when it
   recognizes the old route as belonging to another provider and says so in its output — always confirm
   with `hermes config get model` that neither key survived, and `hermes config unset` them by hand if
   they did. Leaving `api_mode` behind is the quieter failure: it can pin a transport the new provider
   does not serve.
5. **Verify in two rungs — the second is the one that matters:**
   ```bash
   cd ~/.hermes/cache/scratch
   hermes --ignore-rules -z "Reply with exactly one word: pong" -m <model-id> --provider <provider>  # route reachable
   hermes --ignore-rules -z "Reply with exactly one word: pong"                                     # config DEFAULT
   ```
   Rung 1 proves the model answers on that provider; rung 2 is the acceptance test for "as the default",
   because it resolves everything from `config.yaml`. Run both from a scratch dir so a repo `AGENTS.md`
   and its skills do not slow the probe down. A nested one-shot proves the *config* route only — it is
   not the path the running gateway takes, so do not cite it as proof about the live session.
6. **Sweep the routes that do not follow `model.default`** and report them explicitly: `delegation.model`
   / `provider` (subagents), `fallback_providers`, `auxiliary.*` (vision/compression/session_search),
   and any cron job carrying its own `model`/`provider`. A "make it the default" request leaves all of
   these on the old route; say so rather than silently leaving them.
7. **Report the revert path**, since `hermes config set` keeps no backup: the two `config set` commands
   with the previous values. Also state the cost shape when it differs from the old route (a metered
   plan with rolling/weekly/monthly windows visible via `/usage` vs pay-per-token API billing) — the user
   is choosing a billing model, not just a model.

## Auditing an existing route ("is it connected / showing all models / is this the default?")

Cheapest-to-strongest evidence, in this order — stop when a step contradicts:

1. Credential present, by key *name* only (`grep -c '^OPENCODE_GO_API_KEY=' ~/.hermes/.env`); never print the value.
2. Live catalog: `curl -s <base_url>/models -H "Authorization: Bearer $KEY"` → HTTP 200 and an id count.
3. **The list the picker actually builds — resolve it in code instead of opening the interactive picker:**
   ```bash
   cd ~/.hermes/hermes-agent && .venv/bin/python - <<'EOF'
   import sys; sys.path.insert(0, '.')
   from hermes_cli.models import _profile_live_catalog
   rows = _profile_live_catalog('opencode-go')
   print(len(rows)); print('\n'.join(rows))
   EOF
   ```
   Use the checkout's `.venv/bin/python` (the Hermes runtime), not the system `python3`.
4. Config read-back: `model.default`, `model.provider`, and no stray `base_url` / `api_mode`.
5. **Proof of service, not of config:** `hermes --ignore-rules --usage-file u.json -z "Reply with exactly one word: pong"`, then read `model`, `provider` and `failed` out of `u.json`. The reply text alone only proves *something* answered.
6. Side routes: `delegation.*`, `fallback_providers` (probe the fallback vendor's `/models` too — a fallback that 401s is not a fallback), `auxiliary.*`, and any cron job carrying its own `model`/`provider`. No `model_override` in `sessions.json` means the config default governs every session, the current one included.

**Live-first providers merge the vendor list ahead of the curated floor.** `opencode-go`, `opencode-zen` and `meta-ai` (`_LIVE_FIRST_PICKER_PROVIDERS`) put live ids first and append every curated id the live endpoint omitted, so the picker legitimately shows more rows than the vendor serves. When answering "does it show all available models", report the split (live count vs merged count) and probe any tail row you might recommend instead of calling the extras a defect.

**A provider profile's `default_aux_model` is not necessarily what aux `auto` uses.** The profile constant can point at an id the vendor has already retired (`400 … Model is unavailable`) while every auxiliary task in the same session resolves to the main chat model. The runtime log states the real route: `agent.auxiliary_client: Auxiliary <task>: using <provider> (<model>)`; the usage file's `auxiliary.by_task` block corroborates. Read one of those before treating the profile constant as live or declaring aux broken.

## Health audit: "is provider X actually working well?"

A different question from "what is the default". Answer it with a live probe **plus** a failure profile
drawn from the logs — a config read-back proves nothing about service, and the answer must separate
"works right now" from "what has bitten it recently".

1. **Live probe** — `/models` → HTTP 200 + id count, then one streamed chat completion, timed. A 200 on
   both, sub-2 s, is the "yes" half of the answer.
2. **Failure profile, scoped to that provider:**
   ```bash
   cd ~/.hermes/logs
   grep -h "provider=<provider>" errors.log | grep -oE "error_type=[A-Za-z]+" | sort | uniq -c
   grep -hi "usage limit exceeded" errors.log | awk '{print $1}' | sort | uniq -c   # quota hits per day
   ```
   Report the date of the last failure, not just totals: an old cluster in the same file does not make a
   currently-clean route unhealthy, and a bare "N errors" with no dates reads as a live problem.
3. **Cache proof from the live session** — `grep "API call #" ~/.hermes/logs/agent.log | tail` carries
   `latency=` and `cache=hit/total`. A relay returning 90%+ cache on the second call of a session is
   routing well; 0% every call is a real finding worth reporting.
4. **Validate the fallback credential with a real chat completion**, not
   `auth.json → credential_pool.<p>[0].last_status`. A pool entry can read `ok` from long ago while the
   key 401s today, and a `fallback_providers` row naming a dead vendor is worse than none — it hides the
   problem until the primary hits its quota.

### 429 mechanics on metered plans (OpenCode Go and similar)

- Plan quotas answer `429` with a structured error type (e.g. `GoUsageLimitError`) and
  `limitName: monthly | 5 hour`. Hermes classifies it as a rate limit, so `compute_error_backoff`
  (`agent/turn_recovery.py`) waits out the reset window with an adaptive policy — observed 600 s per
  attempt across 3 attempts, i.e. a ~20-minute silent stall that reads to the user as a hang.
- Failover to `fallback_providers` fires on a non-retryable verdict, after retries are exhausted, or via
  `free_tier_cooldown_ends_turn` (attended free-model sessions only). Do not promise instant failover:
  the user-visible symptom of an exhausted plan is a long stall, then either the fallback model or an
  error.
- The two levers to offer are a shorter retry budget for rate-limit errors and a stronger fallback chain
  (a second provider *and* a non-flash model), because the cheap fallback silently degrades the session.

### Probe pitfalls

- **A Cloudflare-fronted provider rejects a bare `urllib` request with HTTP 403 `error code 1010`.** That
  is a User-Agent check, not a bad key or a dead endpoint — `opencode.ai` is one. Send an SDK-looking UA
  (`User-Agent: OpenAI/Python 2.x`) with the bearer token and re-probe before reporting anything broken.
- **Confirm a 401 came from the key currently in `.env`** — compare the last four characters DeepSeek-style
  error bodies echo back, and re-run once. The runtime can rewrite `.env` between probes, so a lone 401
  is not proof the credential is dead.
- **Scope log greps to `~/.hermes/logs/*.log` or a named file.** Grepping `~/.hermes` recursively lands in
  large cache/session artifacts and buries the lines that matter.

## Scope: config default vs. live session

- A new gateway/CLI session picks up the new default. An existing chat keeps its persisted per-session
  `model_override` in `~/.hermes/sessions/sessions.json`, so the change does not disturb the chat you are
  in — and an override already pointing at the same route just becomes redundant.
- Auditing what is pinned where (read-only):
  ```bash
  python3 - <<'EOF'
  import json; d=json.load(open('/Users/<user>/.hermes/sessions/sessions.json'))
  def walk(o,p=''):
      if isinstance(o,dict):
          if o.get('model_override'): print(p, {k:v for k,v in o['model_override'].items() if k!='api_key'})
          for k,v in o.items(): walk(v,p+'/'+str(k))
  walk(d)
  EOF
  ```
- No gateway restart is needed for a default change; the running gateway re-reads `config.yaml` when it
  builds the next session. Never bounce the gateway from inside a gateway session anyway.

## Probing a provider endpoint by hand

Load the env without ever printing a secret, then reference the variable in the request:

```bash
set -a; . ~/.hermes/.env >/dev/null 2>&1; set +a
curl -s -m 25 <base_url>/models -H "Authorization: Bearer $PROVIDER_API_KEY"
```

- **OpenCode family (Go and Zen) rejects a bare chat completion with HTTP 400
  `MissingSessionID`** and requires an `x-opencode-session: <any-id>` request header for routing. Hermes
  sends it automatically on every OpenCode request (`agent/opencode_affinity.py`), so this only bites
  hand-rolled `curl` probes — add the header rather than concluding the key or model is bad.
- **A probe that returns empty `content` is not a failure on a thinking model.** With a tiny `max_tokens`
  the whole budget can go to reasoning (`completion_tokens_details.reasoning_tokens`) and the response
  still has no text. Judge the probe by `http=200` + the echoed `model` id + a non-zero `usage`, not by
  the content string; give the probe room on `max_tokens` if you want visible text.
- Providers with rolling/weekly/monthly plans expose usage through `hermes status`/`/usage` rather than a
  dollar figure — quote windows, not a made-up cost.

## Pitfalls

- Setting `model.default` without `model.provider` (or vice versa) leaves a mixed route that resolves
  against the wrong endpoint; set both, then read back.
- A provider picker row that lists a model is not proof the vendor still serves it, and a vendor's
  `/v1/models` entry is not proof the picker offers it — check the same id on both sides before
  switching a default onto it. For live-first providers the gap is systematic: the curated floor is
  merged in behind the live list and only refreshes with a Hermes update, so the tail rows are where
  retired ids collect.
- Do not edit `~/.hermes/config.yaml` with file tools, and do not "fix" a Hermes source file to make a
  route resolve — that is an update/user-side decision, and hand-patches get stashed by the next update.
