# PR #18565 — draft review responses

Do **not** post these until Ben explicitly approves. Paste as PR comments / review replies on https://github.com/NousResearch/hermes-agent/pull/18565 after the implementation commit lands.

---

## 1. Reply to review body (teknium1 / hermes-sweeper)

> Thanks for isolating the built-in-memory and provider-initialization concerns… Suggested: explicit per-job provider-tools-only opt-in…

**Draft reply:**

```markdown
Thanks — agreed on all three problems with the original patch shape. We reworked the design instead of landing the global `skip_memory_provider=False` for every cron run.

### Addressing the review

1. **Init surface** — `AIAgent` now forwards both `memory_provider_mode` and legacy `skip_memory_provider` into `agent/agent_init.py` (the live initializer). Mode resolution lives in `agent/memory_provider_mode.py`.

2. **Not tools-only by accident** — providers expose three modes:
   - `off` — no provider init (cron default)
   - `tools` — provider loaded + tools injected; **no** automatic system-prompt context, prefetch, per-turn `sync_turn`, or session-end retain
   - `full` — normal interactive lifecycle

   Lifecycle gates:
   - prompt block: `agent/system_prompt.py` (`_memory_provider_prompt_context`)
   - prefetch / on_turn_start: `agent/turn_context.py` (`_memory_provider_prefetch`)
   - turn sync: `run_agent.py::_sync_external_memory_for_turn` (`_memory_provider_auto_sync`)
   - session-end retain: `shutdown_memory_provider` / `commit_memory_session` (same auto-sync flag)

   Built-in `MEMORY.md` / `USER.md` still always skip under cron (`skip_memory=True`).

3. **Per-job opt-in (not global)** — cron jobs carry `memory_provider: off|tools|full`. Scheduler maps that to `AIAgent(memory_provider_mode=...)`. There is no `cron.skip_memory=false` global switch (too broad vs #4052). Recommended path for maintenance jobs is `tools`.

4. **Legacy mapping** — `skip_memory_provider=True` → `off`; `False` → `tools` (never silently implies `full`).

5. **Tests** — unit coverage for mode resolution, init flags (no-prefetch / no-sync in `tools`), and cron constructor kwargs mapping the job field through the scheduler.

Docs updated in this revision: `AGENTS.md` cron section, `website/docs/user-guide/features/cron.md`, `memory-providers.md`, and `website/docs/guides/cron-troubleshooting.md` (new Memory section).
```

---

## 2. Inline: `run_agent.py` init param (discussion_r3567113045)

**Draft reply:**

```markdown
Fixed. Constructor now accepts `memory_provider_mode` + legacy `skip_memory_provider` and forwards both into `agent/agent_init.py`, where the provider is gated with `provider_tools_enabled(mode)` and lifecycle flags are set from `provider_lifecycle_enabled(mode)`.
```

---

## 3. Inline: `cron/scheduler.py` global enable (discussion_r3567113046)

**Draft reply:**

```markdown
Agreed — the original `skip_memory_provider=False` on every cron job was the wrong shape (full auto-sync / Hindsight retain).

Cron still always uses `skip_memory=True`. Providers default to `off`. A job must set `memory_provider="tools"` (tools-only/no-sync) or `"full"` (explicit lifecycle) to load the configured external provider. `tools` is the recommended opt-in for curator/maintenance jobs.
```

---

## 4. Reply to @alt-glitch (duplicate of #9802 / #9763)

**Draft reply:**

```markdown
Thanks for the pointer — same problem space as #9763 / #9802 (external providers dead under cron because `skip_memory=True` also skipped provider init).

This PR is no longer the original "always enable provider for cron" approach. After review feedback (and #4052 history), the direction is:

1. keep built-in memory skipped for all cron
2. default external providers to **off** under cron
3. per-job opt-in with `memory_provider=tools|full`, where `tools` is tools-only/no-auto-sync

Happy to coordinate / close-as-duplicate once that lands if #9802 is still the preferred vehicle — or keep this PR as the mode + docs vehicle if it supersedes the simpler bool.
```

---

## 5. Reply to @jeffla (per-job opt-in vs global)

**Draft reply:**

```markdown
Exactly the shape we landed on:

1. cron skips built-in memory by default (always)
2. provider tools can be exposed in tools-only / no-sync mode (`memory_provider=tools`)
3. full lifecycle / hot memory only via explicit per-job opt-in (`memory_provider=full`)

No global `cron.skip_memory=false` — that would re-open the #4052 corruption path for every scheduled job.
```

---

## 6. Suggested PR description rewrite (for force-push / edit body)

```markdown
## Problem

Cron always passes `skip_memory=True` so built-in MEMORY.md/USER.md are not corrupted by cron system prompts (#4052). That same flag also prevented external memory **providers** (Hindsight, Honcho, mem0, …) from initializing, so provider tools appeared missing during scheduled jobs (#9763).

Earlier revisions that set `skip_memory_provider=False` for **every** cron job were too broad: provider init also enables automatic turn sync / session retain (e.g. Hindsight retain), which is not tools-only behavior.

## Solution

Decouple built-in memory from external providers with an explicit mode:

| `memory_provider_mode` | Built-in | Provider tools | Auto prompt / prefetch / sync / session retain |
|---|---|---|---|
| `off` (cron default) | skipped when `skip_memory` | no | no |
| `tools` | independent | yes | **no** |
| `full` (interactive default) | independent | yes | yes |

- `agent/memory_provider_mode.py` — resolution + helpers
- `agent/agent_init.py` + `run_agent.py` forwarder — init + lifecycle flags
- gates in system prompt, turn context, turn sync, session-end
- per-job cron field `memory_provider` (scheduler maps → mode); built-in still always skipped
- legacy `skip_memory_provider`: True→off, False→tools

## Docs

- `AGENTS.md` cron section
- user guide: cron + memory-providers
- `guides/cron-troubleshooting.md` Memory section

## Tests

- mode resolution / isolation flags
- cron constructor kwargs for the job field

Closes / relates: #9763, #4052; supersedes the global-on approach in early #18565 / overlaps #9802.
```
