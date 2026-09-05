# Claude Agent SDK Runtime

> **Audience:** Gateway developers and maintainers
> **Source files:** `agent/claude_sdk_runtime.py`, `agent/transports/claude_agent_sdk_session.py`, `agent/turn_runtime_handoff.py`, `tools/approval_sdk_gateway.py`, `gateway/run_background_results.py`, `gateway/run_turn_runner.py`, `gateway/run_turn.py`, `gateway/run_agent_cache.py`
> **Last updated:** 2026-08-16

## Overview

The `claude-agent-sdk` provider is not an API client. It spawns the **Claude Code
CLI as a subprocess** and drives it over stream-json. That single fact produces
most of this lane's surprises: the runtime owns a real child process with its own
lifecycle, its own context window, and its own compaction — none of which Hermes
controls.

Every other provider is a stateless HTTP call. This one is a long-lived process
tree.

Two modules matter:

- `agent/claude_sdk_runtime.py` — `run_claude_agent_sdk_turn()`, the turn loop.
  Owns prompt assembly, the compaction status edges, and budget enforcement.
- `agent/transports/claude_agent_sdk_session.py` — `ClaudeAgentSdkSession`,
  the process/option layer. Owns option construction, the environment handed to
  the child, hooks, and teardown.

---

## 1. Configuration

All keys live under `agent.claude_agent_sdk` in `config.yaml`.

Dependency floor: the `[claude-agent-sdk]` extra pins `claude-agent-sdk>=0.2.140` — the first release
whose `mcp` range admits 2.x — together with the same `mcp==2.0.0` the `[mcp]` extra pins, so the
`hermes-tools` stdio server (`mcp.server.MCPServer`) and the SDK share one `mcp` major in every extra
combination. The stdio server also falls back to mcp 1.x's `FastMCP` so an older resolution still starts.

| Key | Type | Default | Effect |
|---|---|---|---|
| `streaming` | bool | off | Stream partial output rather than delivering at turn end. |
| `permission_mode` | str | *(SDK default)* | Passed to the SDK **verbatim**; validated against the installed SDK's literals (`default`, `acceptEdits`, `plan`, `bypassPermissions`, `dontAsk`, `auto`). An invalid value is rejected rather than guessed. |
| `setting_sources` | list | *(none)* | Which on-disk setting sources the CLI may read. Empty by default — opt in explicitly with `["user"]`. Unknown entries are dropped with a warning. |
| `append_file` | path | *(none)* | Operator persona/guidance file appended to the system prompt. Set-but-unreadable warns rather than silently continuing. |
| `append_total_max_chars` | int | 22,000 | Whole system-prompt append budget in characters. Blocks are packed whole; evictions warn with internal content-free labels. Positive integer overrides are accepted, while invalid values fall back to 22,000. |
| `allow_metered_key` | bool | false | Explicit "bill me metered" opt-in. Disables the credential scrub and the child-reported API-key/Extra-Usage refusal (§5). |
| `deliver_background_results` | bool | false | Deliver results produced by background work. |
| `max_budget_usd` | float | *(none)* | Forwarded to the SDK's `max_budget_usd`; the query stops with `error_max_budget_usd` once exceeded. Non-numeric, non-positive, and boolean values are ignored with a warning — a `0` cap would fail every turn instantly, and YAML `true` would `float()` to a nonsense `1.0`. |
| `max_buffer_size` | int | 10 MiB | Maximum size of one CLI NDJSON message. Hermes sets this explicitly because the SDK's 1 MiB default can terminate a turn on a large tool result. Positive integer overrides are accepted; invalid values warn and fall back to 10 MiB. The pinned SDK currently measures Unicode code points despite documenting bytes. |
| `env` | mapping | `{}` | Arbitrary environment passed to the CLI subprocess (§3). |

---

## 2. Context is owned by the CLI, not by Hermes

**Hermes does not compact this lane.** `conversation_compression` short-circuits
when `api_mode == "claude_agent_sdk"`, because Hermes summarizing its own copy of
the transcript cannot shrink the context the CLI is actually sending. The gateway
logs the skip:

```
Session hygiene: skipping compression for <session>; the claude-agent-sdk lane
compacts inside the CLI, so Hermes compaction cannot shrink it
```

Two consequences that have each caused real incidents:

**Hermes' own token estimate is wrong here** — it has over-reported by ~10×
(1.5–2.4M for a ~111k transcript). Never size a decision on it. Ask the CLI:
`ClaudeAgentSdkSession.context_usage()` returns the CLI's ground truth
(`maxTokens`, `contextWindow`, `autoCompactThreshold`, `isAutoCompactEnabled`).

**Routine hygiene must not run here.** A no-op compression pass that still
evicted the cached agent cost ~273k cache-write tokens *per turn* — pure waste,
invisible in logs.

---

## 3. Effective context window and autocompact knobs (`env`)

The CLI reads operational knobs from its environment that the SDK exposes no
typed option for. `agent.claude_agent_sdk.env` is a generic passthrough:

```yaml
agent:
  claude_agent_sdk:
    env:
      CLAUDE_CODE_AUTO_COMPACT_WINDOW: '300000'
```

**Select the intended Opus ceiling before applying a clamp.** Fresh local probes
against `claude-agent-sdk 0.2.120` observed these effective child values:

| Requested model | `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | `maxTokens` | threshold | source |
|---|---:|---:|---:|---|
| `claude-opus-5` | unset | 200,000 | 167,000 | `auto` |
| `claude-opus-5[1m]` | unset | 1,000,000 | 967,000 | `auto` |
| `claude-opus-5[1m]` | `300000` | 300,000 | 267,000 | `env` |

Thus `[1m]` establishes the observed 1M-capable ceiling, while
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` can deliberately clamp that ceiling downward.
A bare Opus identifier did not rise from 200k when given a 300k clamp in the
measured path. Always treat the child’s `context_usage()["maxTokens"]` as the
per-session authority, not model metadata or a configuration name.

An independent Max-plan report on 2026-08-19, using Claude Code 2.1.220,
observed both bare `claude-sonnet-5` and bare `claude-opus-5` at 1M. That
conflicts with the local observation above and is evidence that CLI/account
behavior changes across environments, not a reason to replace one static table
with another. The runtime's `context_usage()["maxTokens"]` probe remains the
only supported authority for the live session.

A public Anthropic tracker issue reports a similar bare-vs-suffixed split on an
earlier Opus generation, but it is user-reported evidence rather than a
maintainer-documented contract. It corroborates the observation; it does not
explain every session’s historical measurements.

**Recorded operator observation, distinct from a causal claim.** Before trying
to lower the context, the operator was certain Hermes displayed Opus at a 1M
window. After experimenting with a 300k context limit and then a 300k
autocompact window, removing those settings left bare Opus at 200k; `[1m]` was
then required to obtain the larger window. That sequence is evidence worth
preserving, but it does not by itself establish whether the change was caused by
the knob experiments, account/session state, CLI behavior, or another factor.

Most plausible alternative knobs did nothing in the same tested path:

| Variable | observed result |
|---|---|
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS=300000` | inert |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50` | inert |
| `CLAUDE_CODE_DISABLE_AUTO_COMPACT` | disables native autocompaction |

These knobs are undocumented enough to shift between CLI builds. Verify every
setting with `context_usage()` rather than trusting its name.

**The default 1M threshold is 967,000 — 96.7% of that window, not the ~80% one
might assume.** With the intentional 300k clamp, the measured threshold is
267,000.

**Shrinking the window desynchronises Hermes' own sizing.** Hermes resolves
`context_length` from model metadata (`claude-opus-5` → 1,000,000), which is not
what the CLI is running with once this knob is set. The runtime footer read
**16% while the CLI was at 53%** of its real window and one turn from
autocompacting; because compaction fires at 267,000 against a denominator of
1,000,000, the gauge could never have exceeded ~27% — the whole scale squashed
into its bottom quarter. Gateway session hygiene sizes off the same value.

`_sync_context_length_from_cli()` corrects it from `context_usage()["maxTokens"]`
once per session (the value is fixed at spawn, and the query is a real
round-trip to the child). The compressor's `context_length` setter invalidates
the derived budgets, so the threshold follows the window rather than stranding a
threshold above `maxTokens` that can never be reached.

Lowering the window is a real trade. Measured on a live session, dropping to
300k moved `cache_read` from 678,625 to 36,516 per request. But 36k is the
*post-compaction floor*, climbing back toward the threshold — the honest
steady-state saving is ~75–80%, not 95%. The cost is a ~2 minute stall per
compaction, arriving ~3.6× more often (267k of growth per cycle instead of 967k).

---

## 4. Child process lifecycle

`close()` must outlive the SDK's own shutdown ladder. `_SDK_DISCONNECT_TIMEOUT_S`
is **25.0s** for exactly this reason: a shorter timeout than the SDK's internal
~20s ladder abandons the child mid-shutdown and strands a ~260 MB process that
GC can never reap.

If disconnect still fails, teardown uses psutil's cross-platform ladder:
`terminate()` → 5s → `kill()`. Two guards apply before either operation:

- `_is_own_sdk_child(pid)` — the PID must be a live child of *this* process.
  Guards against PID reuse killing an unrelated process.
- A **zombie counts as already dead** — it holds no RSS and needs no signal.

Prefer `release_clients()` (soft) over `close()` (hard) where the sandbox,
browser, and background processes should survive.

---

## 5. Billing safety

The lane is a subscription lane. `_scrubbed_sdk_env()` blanks every metered
billing vector present in the parent environment (`ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, metered-shaped `ANTHROPIC_TOKEN`, the Bedrock/Vertex
switches, AWS credentials, `GOOGLE_APPLICATION_CREDENTIALS`). A subscription-
shaped setup/OAuth `ANTHROPIC_TOKEN` is preserved. Only keys **actually
present** are blanked — writing `""` for absent ones can itself confuse
credential chains.

The child then supplies stronger, post-start evidence. The init
`SystemMessage.data["apiKeySource"]` reports whether an API key was selected,
and the pinned SDK's typed `RateLimitEvent` carries `isUsingOverage` plus
`overageStatus`. With the default guard, any non-`none` API-key source or
enabled/active subscription Extra Usage interrupts the turn, retires the
session, and stops the run as a durable account/configuration error. This also
prevents accounting from labeling a reported metered turn as
`subscription_included`. `allow_metered_key: true` is the one explicit escape
hatch for both classes; admitted metered turns are labeled
`sdk_reported_metered` and their reported cost is persisted.

`allow_metered_key: true` is the operator's explicit opt-in and disables the
scrub, since the documented escape hatch would otherwise hand the CLI a blanked
key.

**Ordering matters.** Configured `env` is applied *after* the scrub so deliberate
knobs win over defaults — but a plain `update()` would let
`env: {ANTHROPIC_API_KEY: ...}` overwrite the scrub's `""` and silently re-arm
metered billing behind `allow_metered_key: false`, from a file that looks like it
only holds tuning knobs. `_sdk_env_overrides()` therefore drops denylisted keys
with a warning unless the metered opt-in is set.

That merge lives in a module-level function rather than inline in
`build_option_fields()` specifically so the guard is testable — see
`tests/agent/test_claude_sdk_configured_env.py`.

> Separately: the SDK serializes the stdio MCP config — env included — onto the
> child's argv, readable by any local user via `ps`. That env is a strict
> allowlist and must never carry a secret.

---

## 6. Compaction visibility

Because Hermes does not compact here, a turn can stall for two minutes inside a
CLI compaction with nothing to show the user. Two signals bracket it: the SDK's
`PreCompact` hook on the way in, and the CLI's `compact_boundary` stream message
on the way out.

`_build_compaction_hooks()` registers it **only when `on_compaction` is wired**,
so the default option set is unchanged for callers that do not want it. The hook:

- announces `auto` triggers only — a manual `/compact` is the user's own action
  and already has feedback;
- **always returns `{}`**, because refusing a hook can block the compaction
  itself;
- reuses `COMPACTION_STATUS` / `COMPACTION_DONE_STATUS` from
  `agent/conversation_compression.py`. This is not stylistic: the gateway's
  Telegram noise filter is **built from those same constants**, so a re-inlined
  string is silently dropped on chat surfaces.

**The completion edge is a stream message, not a hook.** The SDK offers no
`PostCompact`, which originally led to firing the completion at the end of the
turn — "a completed turn is the terminal edge". That reasoning is sound and the
result was still useless: end-of-turn is exactly where progress cleanup deletes
the message, so the notice was created and destroyed in the same instant.

The CLI does announce completion, as a plain `system` message with
`subtype="compact_boundary"` carrying `compact_metadata` (`trigger`,
`preTokens`, `durationMs`). The SDK's message parser routes unknown subtypes
through a generic `SystemMessage` fallback, so it arrives on the ordinary
message stream mid-turn. `_handle_compact_boundary()` fires the completion
there. Measured 2026-08-16: boundary at 07:41:16 vs the deferred emit at
07:42:17 — **61 seconds late, and invisible**.

The end-of-turn emit survives as a fallback for a CLI that stops streaming the
boundary; whichever fires first clears `_sdk_compaction_pending`, so the notice
is emitted exactly once.

> **Do not "fix" an invisible notice by making its status durable.** Cleanup is
> not the bug — every other provider's compaction runs mid-turn, so the notice
> is naturally visible for the rest of the turn and cleaned up with the other
> progress messages. Diverging here would leave a permanent bubble per
> compaction on one lane and paper over the wrong edge.

On chat surfaces these are gated behind `compression.progress_notices: true`.

---

### Compaction gets its own status key

Start and completion both emit under `COMPACTION_STATUS_KEY` (`"compaction"`)
via `_emit_status_kind(kind, message, origin=...)` (`agent/status_output.py`),
rather than sharing the generic `"lifecycle"` key. That is what lets a keyed consumer — the Telegram adapter —
**replace** the existing notice in place instead of stacking a second bubble, so
the user sees one status that resolves rather than two that accumulate.

`_emit_status_kind` is the general form; `_emit_status` delegates to it with
`"lifecycle"`, so existing callers are unchanged. The TUI translates the key
back into the wire protocol's `compacting`/`compacted` edges
(`tui_gateway/server.py::_status_update`).

Every emit site is guarded (`getattr` + `try`, or `contextlib.suppress`). Not
every object reaching the compression path is a full `AIAgent` — test doubles
and agent shims reach it too — and an unguarded `agent._emit_status_kind(...)`
raises `AttributeError` mid-compression for anything lacking the method. That failure is real, not
hypothetical: it was reproduced against the deployed tree.

### The watchdog must not kill a compacting turn

Between `PreCompact` and `compact_boundary` the CLI emits **nothing**.
`_TurnWatch` cannot distinguish that silence from a wedge, so the
`post_tool_quiet` rule (90s) tripped, interrupted the CLI mid-compaction, and
the turn's terminal `ResultMessage` never arrived. It surfaced on the next turn
as `discarding N stale unsolicited text(s)` and, to the user, as a turn that
silently died at a compaction — with the "compressing" status reappearing on
their next message.

Measured (CEST):

| Time | Event | Gap | Outcome |
|---|---|---|---|
| 03:57:17 | compaction started | — | |
| 03:58:48 | `no compact_boundary seen` | 91s | **died** |
| 03:59:15 | compaction started | — | |
| 04:01:02 | `compact_boundary` | 107s | survived |
| 05:04:07 | compaction started (post-fix) | — | |
| 05:06:12 | `compact_boundary` | 125s | survived |

91s is one `post_tool_quiet` timeout after the last tool result. The *longer*
compaction survived, so duration is not the trigger — what matters is whether
compaction begins while `post_tool_armed` is set, i.e. right after a tool call.
That is why it hit tool-heavy turns and looked intermittent.

`PreCompact` now calls `_TurnWatch.compaction_begin()` and `compact_boundary`
calls `compaction_end()`, suspending both the quiet and budget rules in between.
Three details are load-bearing:

- **Bounded** by `_COMPACTION_MAX_SUSPEND` (600s). `compact_boundary` is not
  guaranteed — the 03:57 case never produced one — so an unbounded gate would
  trade a killed turn for a hung one.
- **Earliest start wins** on re-entrant `PreCompact`, so a repeated hook cannot
  extend the ceiling indefinitely.
- **`compaction_end()` restamps** `last_activity`. Without that, a turn that
  compacted for 91s would resume already 91s idle and trip on the very next
  poll — the same kill, one poll later.

The hook is wired even when no status callback is set (§6), and both watch
lookups use `getattr` — an `AttributeError` raised on the message drain would
break the very turn the suspension exists to protect.

### Routed self-curation remains safe

The SDK runtime does not start a background memory/skill review when that review
would inherit `api_mode="claude_agent_sdk"`: a fresh SDK turn used to consume a
subscription turn while lacking the writable loop state it needs. That remains
an explicit skip.

When `auxiliary.background_review` resolves to a *different* runtime, the same
review is safe and should run there. The resolver is consulted at the nudge
boundary; only a confirmed `routed=True` result starts the background review.
Resolver or spawn failures fail closed and leave the user turn intact.

The Claude SDK MCP profile also exposes `skill_manage` alongside its existing
bounded inspection tools. It is scoped to that profile—not the Codex default
surface—so the skill nudge can retain durable procedure knowledge without
widening unrelated runtime capabilities.

## 7. Agent cache interaction

The gateway caches one agent per session key. Every **eviction** path releases
what it pops, but a plain cache **overwrite** originally released nothing,
dropping the displaced agent's provider session to GC. On this lane that means an
orphaned CLI subprocess. Measured: 13 turns produced 11 SDK sessions but only 2
closes — 11 orphans holding 2.9 GB.

It was self-reinforcing. Orphans pushed RSS past `memory_high_mb`, triggering
memory-pressure sweeps, which displaced more agents.

`_release_displaced_agent()` (`gateway/run_agent_cache.py`, called from the
cache write in `gateway/run_turn_runner.py`) now handles this. It skips `None` and the pending
sentinel, **skips mid-turn agents** (their own completion path owns teardown —
releasing one kills a live turn), releases on a daemon thread with contained
exceptions, and falls back to an inline release when a thread cannot start
(interpreter shutdown).

---

## 8. Observability

> Both `_sweep_agent_cache_under_pressure` and `_evict_cached_agent`
> (`gateway/run_agent_cache.py`) contain **zero `logger.` calls.** That is why the leaks above hid for so long, and why
> the tests are the only regression guard.

The gateway logs inbound messages but **never logs an outbound send**
(`grep -c "outbound\|sending message\|sent message" gateway.log` → `0`). A path
that talks to chat and writes nothing to disk is unfalsifiable after the fact:
after a real compaction it was impossible to determine whether the completion
notice had reached the user, because the only witness was someone watching the
screen. Both compaction emit paths now log, including the silent-return branch
where `status_callback` is absent — that is the branch that actually loses the
notice.

When adding a user-visible edge on this lane, log it. Chat delivery is not
evidence.

### Triage

| Symptom | First check |
|---|---|
| Idle `claude` subprocesses accumulating | Orphan reap (§4) and cache displacement (§7) |
| Context never compacts | `context_usage()` — threshold depends on the effective child window: 167,000 bare Opus, 967,000 for `[1m]`, or 267,000 with the intentional 300k clamp (§3) |
| Compaction knob has no effect | It is probably inert (§3); confirm with `context_usage()` |
| Notice never appeared | `compression.progress_notices`; then grep `compact_boundary` — a notice emitted at turn end is deleted by cleanup (§6) |
| Footer % looks far too low | `context_usage()["maxTokens"]` vs `context_compressor.context_length` (§3) |
| Cache-write cost per turn | Hygiene should be skipped on this lane (§2) |

---

## 9. Testing

| File | Covers |
|---|---|
| `tests/agent/test_claude_sdk_child_reap.py` | Disconnect timeout, PID-reuse guard, zombie handling |
| `tests/agent/test_claude_sdk_context_usage.py` | CLI ground-truth context reporting |
| `tests/agent/test_claude_sdk_compaction_status.py` | PreCompact hook, trigger forwarding, status single-sourcing, emit logging, compact_boundary completion edge |
| `tests/agent/test_claude_sdk_context_length.py` | CLI `maxTokens` overriding metadata, per-session caching, degradation |
| `tests/agent/test_claude_sdk_configured_env.py` | `env` passthrough, stringification, metered-denylist guard |
| `tests/agent/test_claude_sdk_aux_routing.py` | Subscription-only one-shot routing, tool isolation, child env scrub, terminal-error handling |
| `tests/agent/test_system_prompt_restore.py` | Effective SDK prompt snapshot survives continuing turns |
| `tests/gateway/test_agent_cache_displacement.py` | Displaced-agent release, mid-turn protection |
| `tests/gateway/test_sdk_background_result_delivery.py` | Direct outbound delivery of background results, transcript projection, orphan fallback |
| `tests/agent/test_claude_sdk_runtime.py` | Runtime glue, streaming/turn lifetime, session identity, approval bridge, canonicalization hardening |
| `tests/agent/test_hermes_hybrid_mcp.py` | Hybrid MCP bridge registration and tool exposure |
| `tests/agent/transports/test_hermes_tools_mcp_server_shims.py` | Stateless memory / session_search shims |

**Known gap:** the compaction *start* and *completion* closures both sit inside
`run_claude_agent_sdk_turn()` and cannot be exercised without standing up a full
turn. The transport-side halves they depend on (`_build_compaction_hooks`,
`_handle_compact_boundary`) are covered directly; the closures themselves are
left to production verification rather than covered by a source-text assertion,
which would claim coverage without evidence the line ever runs.
