# CaMeL Guard Architecture

This document explains the current Hermes CaMeL design in terms of the exact constraints raised during PR review.

## Design Goals

The implementation is optimized for four properties:

- security: untrusted tool output must not silently authorize sensitive side effects
- cost efficiency: the guard must not add a model call on every API turn
- maintainability: the runtime diff must stay narrow and understandable
- operator control: the feature must be opt-in and observable before enforcement

## What Changed From The Original PR

### 1. Prompt caching is preserved

The original PR appended a dynamic security envelope to the system prompt on every turn.
That broke prompt caching.

The current design does not mutate the system prompt.

Instead:

- trusted/untrusted state lives in the runtime policy layer
- tool decisions are made before tool execution
- response hijack handling is evaluated on the final assistant text
- internal bookkeeping is stripped before provider API calls

Result:

- stable system prompt prefix
- no per-turn cache busting
- lower provider cost

## 2. Guard is opt-in

The runtime now ships with these defaults:

- `camel_guard.enabled: false`
- `camel_guard.mode: monitor`
- `camel_guard.wrap_untrusted_tool_results: false`

Modes:

- `off` / `legacy`: no CaMeL enforcement
- `monitor` / `on`: trace policy decisions without blocking tools
- `enforce`: block unauthorized sensitive tool calls

This keeps the default Hermes experience unchanged.

## 3. Capability authorization is model-based, not regex-based

The original regex approach was too broad for a security boundary.

The current design uses a small isolated classifier call with:

- trusted user text only
- recent trusted user turns only
- no tools
- no memory
- no untrusted tool output in the classifier context
- strict JSON output

The classifier returns:

- `goal_summary`
- `allowed_capabilities`
- `denied_capabilities`
- `rationale`

This keeps the security boundary legible while avoiding brittle surface-verb matching.

## 4. The classifier is lazy

The classifier does not run at `begin_turn()` anymore.

It runs only when all of the following are true:

- CaMeL is enabled
- a tool is policy-gated
- untrusted context is present
- the current trusted plan has not already been classified or cached

This matters because many Hermes turns are:

- pure chat
- read-only
- tool-free
- or sensitive-tool calls without any untrusted context

For those turns, the classifier never fires.

Result:

- lower auxiliary-model spend
- lower latency
- tighter alignment with the actual threat model

## 5. Tool outputs stay raw

The original PR wrapped untrusted tool results in an internal protocol.

The current design does not rewrite tool outputs.

Instead:

- the guard infers untrusted lineage from assistant tool calls plus tool results
- legacy wrapped history is still understood for backward compatibility
- new tool results are left untouched

This keeps the runtime interoperable with the normal Hermes message flow.

## 6. The `run_agent.py` surface is intentionally small

The runtime integration remains narrow:

- initialize `CamelGuard`
- begin a guarded turn
- evaluate sensitive tool calls
- save trace artifacts
- evaluate final-response hijack markers

The current `run_agent.py` diff is small enough to review directly and avoids a large control-flow fork around the tool dispatch loop.

## 7. Failure mode is conservative

If the classifier is unavailable or returns invalid output:

- the plan falls back to read-only
- `planner_status` becomes `fallback_read_only`
- the trace records why

This favors safety under degraded conditions.

## 8. Traceability is first-class

CaMeL Trace records:

- trusted operator request
- planner status
- untrusted sources
- suspicious flags
- tool decisions
- response-hijack decisions

That gives an operator enough evidence to tune the guard in `monitor` mode before moving to `enforce`.

## Testing Strategy

The current test/benchmark story covers three layers:

1. unit semantics in `tests/agent/test_camel_guard.py`
   - lazy classification
   - cache reuse
   - fallback behavior
   - follow-up context handling

2. runtime integration in `tests/test_run_agent.py`
   - legacy/off parity
   - monitor-mode non-blocking behavior
   - enforce-mode blocking behavior
   - automatic memory flush gating

3. deterministic attack benchmarks in `tests/agent/test_camel_benchmark.py`
   - hidden terminal exfiltration
   - hidden external messaging
   - hidden memory writes
   - hidden browser steering
   - explicit user denial
   - explicit safe authorization

## Practical Configuration

If you want the cheapest viable classifier path, use a small fast auxiliary model under:

```yaml
auxiliary:
  camel_guard:
    provider: auto
    model: ""
```

That keeps the main model and guard model decoupled.

## Non-Goals

This design is not trying to become:

- a general-purpose user-policy engine for every turn
- a replacement for provider-side safety models
- a full reproduction of the CaMeL paper benchmark harness

It is a narrow runtime trust boundary for Hermes.
