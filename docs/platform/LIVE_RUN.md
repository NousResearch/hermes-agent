# Live worker run — what only running it could find

An end-to-end run of the `quarterly-refund-audit` objective against real worker processes:
the real dispatcher, real profile resolution, real subprocess spawn, real plugin loading,
real tool dispatch. **Core patches: still 1.** Four defects found, all four in NOVA, all
four fixed and regression-tested.

---

## What could and could not be real

No model provider exists in this environment — the AWS credentials are placeholders
(`InvalidClientTokenId` from STS) and there is no Anthropic API key. So the model is a
**scripted OpenAI-compatible server** the runtime talks to over its real client, with real
SSE streaming and real tool-call accumulation.

That is not a workaround, it is the better instrument. What is scripted is the model's
*choice* of which tool to call — and a real model cannot be made to attempt a denied tool on
demand. "Does NOVA's policy actually block it inside a live worker" is the question worth
answering, and only a scripted model can ask it reliably.

Everything between NOVA and the model is genuine: tool schemas serialised and parsed, the
policy hook, the knowledge plugin, the dispatcher, the profile boundary.

---

## The four defects

Each subsystem was correct in isolation. Every defect was in the **seam** between it and the
runtime, and no unit test could have found any of them — because each test called NOVA's own
code directly, which is exactly what the runtime was failing to do.

### 1. The policy plugin was never enabled

`hermes plugins list` in a materialized profile:

```
nova-knowledge   not enabled
nova-policy      not enabled
Plugins are opt-in by default — only 'enabled' plugins load.
```

NOVA installed the plugin files and never wrote `plugins.enabled`. Since Phase 2 the
enforcement plugin had been **installed, correct, and never consulted** — the most dangerous
state a governance control can be in, because it passes review by inspection.

*Fixed:* `materialize.plugins_section()` writes the `plugins` block for whichever plugins
this materialization installs, with `allow_tool_override: false` stated explicitly.

### 2. The policy hook was never registered

```
Plugin 'nova-policy' has no register() function
```

`enforcement.py` defined `pre_tool_call` at module level and declared `hooks: [pre_tool_call]`
in its manifest. But the manifest field is documentation; the loader calls `register(ctx)`
and the plugin registers its own callbacks. So even once enabled, it registered nothing.

Two independent reasons the same control did nothing. Fixing either alone would have left it
inert.

*Fixed:* `enforcement.register(ctx)` calls `ctx.register_hook("pre_tool_call", pre_tool_call)`.

### 3. NOVA's policy blocked NOVA's own knowledge tool

With 1 and 2 fixed, the policy started enforcing — and immediately refused the tool NOVA had
just installed:

```json
{"error": "BLOCKED by NOVA policy: knowledge_search is not granted to this agent,
           and this agent runs under an allow-list"}
```

`customer-support` runs under an allow-list built from its permissions. `knowledge_search`
was in neither the permissions nor the baseline, so granting an agent a corpus produced an
agent that could not search it. The trap only fires under an allow-list, where the failure
reads as a knowledge bug rather than a policy one.

*Fixed:* declaring `knowledge.sources` grants `knowledge_search` in `compile_policy`. The
grant **is** the authorization; requiring a second, separate permission is a trap. An
explicit `tools.deny` still overrides it, so a tenant can revoke it.

### 4. A deferred tool the model never hears about

`knowledge_search` registered correctly and was still absent from the 37 tools offered to the
model. The runtime defers plugin tools by design
(`tools/tool_search.py::is_deferrable_tool_name` — anything outside `_HERMES_CORE_TOOLS` and
two GUI toolsets), reaching them through `tool_search` / `tool_call` instead. There is no
supported config to exempt one plugin toolset short of disabling deferral for everything.

Reachable, then — but a model that does not know a knowledge base exists will not go looking
for one. A granted agent would quietly answer from memory: the exact failure the capability
was built to remove.

*Fixed:* `knowledge_briefing()` names the granted corpora in the agent's `SOUL.md` and says
how to reach the tool. `SOUL.md` is part of the static system prompt, so this costs nothing
per turn and cannot disturb prompt caching.

---

## The run, after the fixes

```
tick  1  spawned=[pull-ledger -> operations, handbook-thresholds -> customer-support]
                                              {'running': 2, 'todo': 2}
tick  2  spawned=[compare -> operations]      {'done': 2, 'running': 1, 'todo': 1}
tick  3  spawned=[write-findings -> operations] {'done': 3, 'running': 1}
tick  4  settled                              {'done': 4}
```

Tick 1 spawned both independent steps **at once, to the two different agents NOVA routed them
to** — the declared parallelism, executing. Ticks 2 and 3 released the dependent steps only
as their parents completed.

`nova objective status` → `done (4/4 steps done)`.

The audit log from the run, end to end:

| Event | Count |
|---|---|
| `agent.materialized` intent/committed | 2 / 2 |
| `identity.applied` intent/committed | 1 / 1 |
| `knowledge.indexed` intent/committed | 2 / 2 |
| `work.submitted` intent/committed | 1 / 1 |
| `knowledge.search` intent/committed | 2 / 2 |
| `policy.decision` | 4 |

All four policy decisions were `terminal` denied — one by explicit deny, three by allow-list.
The knowledge searches returned cited passages (`refunds.md:1-20 (Refund Policy)`).

---

## Verified live

| Claim | How |
|---|---|
| The policy plugin enforces | `terminal` denied in a worker, decision in the audit log |
| An explicit deny and an allow-list both fire | 1 + 3 decisions, distinct reasons |
| The knowledge tool works | search returned a cited passage from the corpus |
| Scope is real | grant resolved from the profile, not from the model's request |
| Routing reaches the worker | `handbook-thresholds` ran as `customer-support` |
| Parallelism is real | both independent steps spawned on tick 1 |
| Dependencies gate | `compare` spawned only after both parents finished |
| Collection is real | `done (4/4)` derived from completed tasks |

---

## Found but not fixed: provider config has no home

NOVA compiles `model.provider` per agent, but a provider's **endpoint** lives in
`custom_providers` — and a worker runs with `HERMES_HOME` set to its profile, so it reads
only that profile's `config.yaml` and does not inherit the runtime home's. NOVA owns and
rewrites that file on every apply, so an operator who adds `custom_providers` there loses it
on the next `nova apply`.

For this run the provider block was injected manually after each apply.

This is a real gap with no good workaround today, and it is deliberately **not** fixed here:
it needs a designed seam — an operator-owned passthrough block that NOVA writes verbatim and
never invents, carrying endpoints and `${ENV_VAR}` references but **never secrets**, which
remain out of scope by standing constraint. That is a phase, not a patch.

---

## Reproducing

```bash
uv pip install --python .venv/bin/python -e .          # the runtime and its deps
python /tmp/e2e/model_server.py 8904 &                 # the scripted model
export HERMES_HOME=/tmp/live NOVA_HOME=/tmp/live NOVA_MODEL_PROVIDER=scripted
python -m nova apply nova/examples/acme
python -m nova knowledge ingest nova/examples/acme
# inject custom_providers into each profile config — see the gap above
python -m nova objective submit nova/examples/acme quarterly-refund-audit
python /tmp/e2e/dispatch.py                            # real dispatcher ticks
python -m nova objective status nova/examples/acme
```

The scripted model and dispatcher driver live outside the repo deliberately: they are test
instruments, not product, and shipping them would invite someone to mistake one for a
supported provider.
