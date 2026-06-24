# ADR-0041: Cooperative Model Routing ("Sandwich Model")

**Status:** Draft  
**Authors:** JustinOhms  
**Date:** 2026-06-17  
**Depends on:** PR #48018 (`pre_failover_decision` hook)  
**Addresses:** #23739, #46991

---

## Context

Hermes currently has two model-selection mechanisms:

1. **Static routing** — user-configured `model.routing` graph with keyword-based
   cheap-vs-strong classification (built into core).
2. **Fallback chain** — sequential provider failover on error (built into retry loop).

Neither mechanism is plugin-extensible. Community plugins that need per-turn model
routing (topic-based, quota-aware, cost-optimized) must monkey-patch internal agent
methods — a pattern that breaks across releases.

Issue #23739 requests that `pre_llm_call` gain the ability to override
model/provider/system_prompt at runtime. Issue #46991 requests a dedicated
`resolve_route` hook for quota-aware subscription balancing. Both describe the same
underlying need: a plugin-extensible per-turn model selection surface.

Meanwhile, the existing heuristic routing (complexity scoring, interaction-mode
detection, de-escalation) is valuable but should not be privileged over plugin-driven
routing — it should participate as one voice among many.

---

## Decision

We introduce a **cooperative routing pipeline** structured as a sandwich:

```
┌──────────────────────────────────────────────────┐
│  TOP BUN: Availability Filter                    │
│  Defines which models/providers are reachable    │
├──────────────────────────────────────────────────┤
│  FILLINGS: Routing Plugins (0 or more)           │
│  Rank available models by preference             │
├──────────────────────────────────────────────────┤
│  BOTTOM BUN: Aggregator                          │
│  Chooses winner, builds context metadata         │
└──────────────────────────────────────────────────┘
```

### Core Principles

1. **Top bun and bottom bun are mandatory.** If none are configured, built-in
   defaults are used. A sandwich without both buns is invalid.
2. **Fillings are optional.** Zero or more routing plugins can participate.
3. **Anyone can replace the buns.** Custom availability filters and aggregators are
   first-class plugins with the same interface as the defaults.
4. **The aggregator always runs.** No plugin can bypass it — the aggregator is
   responsible for building the context metadata that gets injected into the prompt.
5. **Order in config matters for fillings.** Later plugins see the rankings of
   earlier plugins and can endorse (boost scores) or further filter the available
   model list.
6. **A single plugin can implement multiple roles.** A plugin may be both the
   availability filter and the aggregator (a "border sandwich" or "bagel").

---

## Three Plugin Interfaces

### 1. `AvailabilityFilter` (Top Bun)

Runs first. Receives all configured models/providers from the routing graph.
Returns the subset that is currently reachable/available.

```python
class AvailabilityFilter(ABC):
    """Top bun: constrain models/providers to what's actually available."""

    @abstractmethod
    def filter_available(
        self,
        models: List[ModelInfo],
        providers: List[str],
        *,
        user_message: str,
        session_id: str,
        platform: str,
    ) -> FilterResult:
        """
        Args:
            models: All models from routing graph (model name + provider + metadata).
            providers: All unique provider names from graph.

        Returns:
            FilterResult with `models` and `providers` lists narrowed to available.
        """
        ...
```

**Default implementation** (`hermes-availability`):
- Checks local model readiness (llama-server health endpoint)
- Checks rate-limit cooldown (`_rate_limited_until`)
- Checks quota exhaustion (if quota tracking is enabled)
- Returns filtered list

### 2. `RoutingPlugin` (Filling)

Runs in config order between the buns. Receives available models + rankings from
previous fillings. Returns preference rankings, further model filtering, or a
forced decision.

```python
class RoutingPlugin(ABC):
    """Filling: rank available models or force a decision."""

    @abstractmethod
    def route(
        self,
        *,
        user_message: str,
        conversation_history: List[Dict],
        available_models: List[ModelInfo],
        available_providers: List[str],
        current_rankings: List[Ranking],
        session_id: str,
        turn_id: str,
        platform: str,
        current_model: str,
        current_provider: str,
    ) -> Optional[RouteResult]:
        """
        Returns:
            None — no opinion this turn.
            RouteResult with any combination of:
              - rankings: List[Ranking] — preference scores for models
              - models: List[ModelInfo] — further-filtered model list
              - providers: List[str] — further-filtered provider list
              - force: ForceDecision — binding decision (stops pipeline)
        """
        ...
```

**Example implementations:**
- `hermes-smart-routing` — complexity scoring, interaction-mode detection
- `topic-router` — domain classification → model mapping
- `quota-router` — subscription headroom scoring

### 3. `Aggregator` (Bottom Bun)

Runs last. Always. Receives all rankings, available models, and any force decision
from the pipeline. Chooses the winner and builds the context/metadata string.

```python
class Aggregator(ABC):
    """Bottom bun: choose winner and build context metadata for prompt injection."""

    @abstractmethod
    def aggregate(
        self,
        *,
        available_models: List[ModelInfo],
        available_providers: List[str],
        rankings: List[Ranking],
        force_decision: Optional[ForceDecision],
        user_message: str,
        session_id: str,
        turn_id: str,
        platform: str,
    ) -> Optional[AggregateResult]:
        """
        Args:
            rankings: All rankings from all fillings.
            force_decision: If a filling forced a decision, it's here.

        Returns:
            AggregateResult with:
              - model: str — chosen model
              - provider: str — chosen provider
              - context: str — metadata string injected into prompt
            Or None — no override, use current model.
        """
        ...
```

**Default implementation** (`hermes-default-aggregator`):
- If force decision exists → use it, build metadata context
- Otherwise → sum scores per model across all rankings, pick highest
- Build context string explaining: chosen model, why, alternatives considered

---

## Data Types

### `ModelInfo`

```python
@dataclass
class ModelInfo:
    model: str           # e.g. "claude-opus-4"
    provider: str        # e.g. "bedrock" or "anthropic"
    position: str        # e.g. "upper", "fast_fallback"
    base_url: str = ""
    api_mode: str = ""
    is_local: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `Ranking`

```python
@dataclass
class Ranking:
    model: str
    provider: str
    score: float         # 0.0–1.0
    reason: str          # Human-readable explanation
    plugin_name: str     # Which plugin produced this ranking
    priority: float = 0.5  # 0.0–1.0, default 0.5
    sticky_turns: Optional[int] = None      # Request: keep this decision for N turns
    sticky_timeout_s: Optional[int] = None  # Request: keep this decision for N seconds
```

### `ForceDecision`

```python
@dataclass
class ForceDecision:
    model: str
    provider: str
    reasons: List[str] = field(default_factory=list)
    context: Optional[str] = None  # Optional pre-built context string
    sticky_turns: Optional[int] = None      # Request: keep this decision for N turns
    sticky_timeout_s: Optional[int] = None  # Request: keep this decision for N seconds
```

### `RouteResult`

```python
@dataclass
class RouteResult:
    rankings: Optional[List[Ranking]] = None
    models: Optional[List[ModelInfo]] = None   # Further-filtered
    providers: Optional[List[str]] = None      # Further-filtered
    force: Optional[ForceDecision] = None      # Binding — stops pipeline
```

### `FilterResult`

```python
@dataclass
class FilterResult:
    models: List[ModelInfo]
    providers: List[str]
```

### `AggregateResult`

```python
@dataclass
class AggregateResult:
    model: str
    provider: str
    context: str    # Injected into prompt as metadata
    reasons: List[str] = field(default_factory=list)
```

---

## Pipeline Execution

```python
def resolve_model_sandwich(agent, user_message, turn_id, session_id, platform):
    """Per-turn model resolution pipeline."""

    # ── Initialize from routing graph ──
    all_models = get_graph_models(agent)
    all_providers = get_graph_providers(agent)

    # ── TOP BUN: Availability Filter ──
    availability = get_availability_plugin()  # Default if none configured
    filter_result = availability.filter_available(
        models=all_models,
        providers=all_providers,
        user_message=user_message,
        session_id=session_id,
        platform=platform,
    )
    available_models = filter_result.models
    available_providers = filter_result.providers

    # ── FILLINGS: Routing Plugins ──
    rankings = []
    force_decision = None

    for plugin in get_routing_plugins():  # Config order
        result = plugin.route(
            user_message=user_message,
            conversation_history=get_history(agent),
            available_models=available_models,
            available_providers=available_providers,
            current_rankings=rankings,
            session_id=session_id,
            turn_id=turn_id,
            platform=platform,
            current_model=agent.model,
            current_provider=agent.provider,
        )

        if result is None:
            continue

        if result.models is not None:
            available_models = result.models
        if result.providers is not None:
            available_providers = result.providers
        if result.rankings is not None:
            rankings.extend(result.rankings)
        if result.force is not None:
            force_decision = result.force
            break  # Stop fillings — aggregator still runs

    # ── BOTTOM BUN: Aggregator (always runs) ──
    aggregator = get_aggregator_plugin()  # Default if none configured
    agg_result = aggregator.aggregate(
        available_models=available_models,
        available_providers=available_providers,
        rankings=rankings,
        force_decision=force_decision,
        user_message=user_message,
        session_id=session_id,
        turn_id=turn_id,
        platform=platform,
    )

    if agg_result is None:
        return None, ""  # No override — use current model

    return agg_result, agg_result.context
```

---

## Config Validation Rules

The plugin system enforces these invariants at startup:

1. **Exactly one availability filter must be active.** If zero are configured, the
   default (`hermes-availability`) is auto-registered. If more than one is
   configured, startup fails with an error.

2. **Exactly one aggregator must be active.** If zero are configured, the default
   (`hermes-default-aggregator`) is auto-registered. If more than one is configured,
   startup fails with an error.

3. **If a plugin implements `AvailabilityFilter`, it must be first.** Config
   validation rejects a plugin with `role: availability` that appears after a
   `role: routing` plugin.

4. **If a plugin implements `Aggregator`, it must be last.** Config validation
   rejects a plugin with `role: aggregator` that appears before a `role: routing`
   plugin.

5. **A plugin implementing both `AvailabilityFilter` and `Aggregator` must be the
   only routing-related plugin** (the "bagel" rule). If other routing plugins are
   configured alongside it, startup fails with an error explaining the constraint.

6. **Priority field (0.0–1.0) defaults to 0.5.** Priority is metadata for the
   aggregator to use when scoring — it does NOT affect execution order. Execution
   order is strictly config file order.

### Error Messages

```
ERROR: Plugin 'my-availability' has role 'availability' but is not first in the
       routing plugin chain. Move it above all 'routing' plugins in config.yaml.

ERROR: Plugin 'my-aggregator' has role 'aggregator' but is not last in the routing
       plugin chain. Move it below all 'routing' plugins in config.yaml.

ERROR: Plugin 'my-bagel' implements both AvailabilityFilter and Aggregator but other
       routing plugins are also configured. A bagel plugin must be the only routing
       plugin. Either disable 'hermes-smart-routing' or split 'my-bagel' into
       separate availability and aggregator plugins.

ERROR: Multiple availability filters configured: 'hermes-availability', 'my-filter'.
       Only one availability filter is allowed.
```

---

## Context Injection

The aggregator is solely responsible for building and returning the context string.
This string is injected into the prompt (appended to the user message) so the model
knows what routing decisions were made and why.

**Only the aggregator injects context.** Individual routing plugins do NOT inject
context — they contribute rankings and reasons, which the aggregator uses to build
a unified context string.

### Context String Format (Default Aggregator)

```
--- Model Selection ---
Selected: claude-opus-4 (bedrock)
Reason: high complexity (score=0.87), oversight escalation
Available: claude-opus-4 (bedrock), claude-sonnet-4 (bedrock), qwen3-30b (local)
Alternatives considered:
  - claude-sonnet-4 (bedrock): score=0.62 [complexity=medium]
  - qwen3-30b (local): score=0.31 [low complexity, de-escalation candidate]
-----------------------
```

Custom aggregators may use different formats.

---

## Turn Scope and Restoration

Model overrides from the routing pipeline are **turn-scoped**:

- Override is applied via `switch_model()` at the start of the turn.
- At the start of the *next* turn, `restore_primary_runtime()` restores the
  original model/provider (same mechanism used by fallback recovery today).
- The `_fallback_activated` flag (or a new `_routing_override_active` flag) triggers
  restoration.

This ensures a specialist route for one turn does not leak into subsequent turns.

---

## Relationship to `pre_failover_decision` (PR #48018)

The two hooks are complementary:

| Aspect | `resolve_model` (this ADR) | `pre_failover_decision` (#48018) |
|--------|---------------------------|----------------------------------|
| **When** | Start of turn (before LLM call) | During retry loop (after error) |
| **Purpose** | Choose which model handles this turn | Choose what to do when the model fails |
| **Trigger** | Every turn | Only on error |
| **Scope** | Happy path | Error path |

A complete routing plugin registers both:
- `resolve_model` (via `RoutingPlugin` interface) → proactive turn routing
- `pre_failover_decision` (via existing hook) → reactive error recovery

---

## Relationship to Existing `pre_llm_call`

`pre_llm_call` remains unchanged — it handles **context injection** only (appending
ephemeral text to the user message). It does NOT gain model override capabilities.

The separation:
- `pre_llm_call` → "What extra context should the model see?"
- `resolve_model` pipeline → "Which model should see it?"

A plugin can register both hooks if it needs to inject context AND influence model
selection.

---

## Configuration Example

### Standard Sandwich (Separate Buns + Fillings)

```yaml
plugins:
  routing:
    availability:
      plugin: hermes-availability    # Top bun (default)
      enabled: true

    fillings:                        # Config order = execution order
      - plugin: hermes-smart-routing
        enabled: true
        priority: 0.5               # Default
        config:
          complexity_threshold: 0.7
          de_escalation: true

      - plugin: quota-router
        enabled: true
        priority: 0.5
        config:
          window_minutes: 60

    aggregator:
      plugin: hermes-default-aggregator  # Bottom bun (default)
      enabled: true
      config:
        score_threshold: 0.5             # Minimum score to override
```

### Bagel (Single Plugin Does Everything)

```yaml
plugins:
  routing:
    availability:
      plugin: my-omnibus-router
      enabled: true
    fillings: []
    aggregator:
      plugin: my-omnibus-router
      enabled: true
```

### Minimal (Defaults Only — No Custom Routing)

```yaml
# No routing plugins configured → defaults auto-register
# Availability: hermes-availability (passes everything)
# Fillings: none
# Aggregator: hermes-default-aggregator (no rankings → no override → current model)
```

---

## Migration Path

### Phase 1: Hook Infrastructure (PR)

- Define the three interfaces (`AvailabilityFilter`, `RoutingPlugin`, `Aggregator`)
- Add `resolve_model` to `VALID_HOOKS`
- Implement pipeline runner (`resolve_model_sandwich`)
- Implement default availability filter and default aggregator
- Config validation
- Wire into `build_turn_context` (fires after `restore_primary_runtime`, before
  `pre_llm_call`)
- Tests

### Phase 2: Null Router (PR)

- Default routing plugin that ships with Hermes when no other fillings are configured
- Does nothing — returns the currently configured model/provider unchanged
- Serves as the reference implementation for the `RoutingPlugin` interface
- Ensures the pipeline always has a well-defined pass-through behavior
- Validates that the sandwich infrastructure works end-to-end with a trivial filling
- Backward compatible — existing users see no behavior change

### Phase 3A: Smart Routing Plugin (PR)

- Port existing `agent/routing/` logic into a `RoutingPlugin` implementation
- Complexity scoring, interaction-mode detection, de-escalation
- Registers as a filling plugin
- Config-driven (reads `model.routing` section)
- Backward compatible — existing `model.routing` configs work unchanged
- Also registers `pre_llm_call` for context injection (routing metadata)
- Also registers `pre_failover_decision` for error-path recovery

### Phase 3B: Topic Router Plugin (PR)

- Domain/topic classification of user messages
- Maps detected topic to preferred model (e.g., coding → strong model, casual → fast)
- Returns rankings based on topic confidence scores
- Reference: `hermes-arc` community plugin (ShockShoot/hermes-arc)
- Addresses issue #23739

### Phase 3C: Quota-Aware Router Plugin (PR)

- Tracks rolling-window token usage per subscription/provider
- Scores models by remaining quota headroom
- Routes to provider with most remaining capacity
- Supports multiple subscriptions of the same model (e.g., two Claude Max accounts)
- Addresses issue #46991

### Phase 3D: Availability Filter Plugin (PR)

- Replace the default availability filter with a production-grade implementation
- Local model health checks (llama-server `/health` endpoint polling)
- Rate-limit cooldown awareness (`_rate_limited_until`)
- Quota exhaustion detection (integrates with 3C's tracking data if present)
- Provider API status checks (optional, configurable)
- Registers as top bun, replaces the default pass-through filter

---

## Consequences

### Positive

- **Plugin extensibility** — anyone can write routing logic without monkey-patching
- **Cooperative** — multiple plugins contribute rankings, aggregator resolves
- **Transparent** — context metadata explains every routing decision
- **Backward compatible** — existing routing configs still work (Phase 2 plugin)
- **Testable** — each interface is independently testable
- **Addresses #23739 and #46991** — both use cases implementable as plugins

### Negative

- **Per-turn overhead** — pipeline runs every turn (mitigated: skip if no routing
  plugins configured beyond defaults)
- **Complexity** — three interfaces + config validation is more complex than a
  single hook
- **Plugin ordering matters** — users must understand config order semantics
- **Single aggregator constraint** — prevents composing multiple aggregation
  strategies (acceptable for v1)

### Neutral

- Existing `pre_llm_call` and `pre_failover_decision` hooks are unaffected
- Existing fallback chain logic is unaffected (orthogonal)
- `switch_model()` is reused as-is for the actual runtime swap

---

## Resolved Questions

1. **Where does `model.routing.graph` come from?**
   It already exists in Hermes config (`config.yaml` → `model.routing.graph`). Each
   entry is a named position (e.g., `fast_fallback`, `interactive_lower`, `upper`)
   with model, provider, tier, and profile metadata. The pipeline reads this as its
   canonical source for `ModelInfo` — plugins filter/rank within it but cannot
   register models outside it.

2. **Does the aggregator need full conversation history?**
   No. The aggregator receives only: available models, rankings, force decision,
   current user message, session_id, turn_id, platform. Individual routing plugins
   that need history (e.g., topic classification) receive it in the `route()` call,
   but the aggregator's job is purely selecting a winner from already-scored
   candidates — it doesn't need to re-read the conversation.

3. **Should we support async availability checks?**
   Yes — the availability filter and routing plugins run async with a per-plugin
   timeout (configurable, default 2s). Additionally, see "Pipeline Invocation
   Strategy" below for when the pipeline fires at all.

4. **Is the default aggregator's score threshold configurable?**
   Yes — via aggregator config section (`score_threshold: 0.5`).

---

## Pipeline Invocation Strategy

Running the full sandwich on every turn is wasteful. Most turns don't need re-routing
— the current model is already correct. The pipeline should have mechanisms to skip
or cache:

### Skip Conditions (Pipeline Not Invoked)

- **No routing plugins configured** (only default null-router filling) — short
  circuit, return current model immediately.
- **Sticky decision still valid** — if the pipeline produced a decision N turns ago
  and no invalidation signal has fired, reuse the cached decision.
- **Turn is a continuation** — tool-call → result → model-response cycles within
  a single logical turn don't re-invoke routing (only the initial user message does).

### Invalidation Signals (Force Re-evaluation)

- **User message arrives** — first message after idle always re-evaluates.
- **Availability change** — model goes down, rate limit hits, quota exhausted.
- **Explicit plugin request** — a plugin can set `invalidate_after_turns: N` to
  force periodic re-evaluation.
- **Session model override** — user explicitly sets a model via `/model` command;
  routing pipeline is bypassed until session ends or user clears override.

### Caching and Persistence

Routing decisions can be **sticky** — once the pipeline runs, the result persists
for a window determined by the winning plugin, bounded by system-level min/max:

```yaml
plugins:
  routing:
    invocation:
      mode: "on_change"       # "every_turn" | "on_change" | "sticky"
      sticky_min_turns: 1     # Floor: plugin can't request less than this
      sticky_max_turns: 20    # Ceiling: plugin can't request more than this
      sticky_min_timeout_s: 30    # Floor for time-based stickiness
      sticky_max_timeout_s: 600   # Ceiling for time-based stickiness
```

The **winning plugin** (the one whose ranking/force was selected by the aggregator)
declares how long it wants the decision to stick:

```python
@dataclass
class Ranking:
    model: str
    provider: str
    score: float
    reason: str
    plugin_name: str
    priority: float = 0.5
    sticky_turns: Optional[int] = None    # "Keep me for N turns"
    sticky_timeout_s: Optional[int] = None  # "Keep me for N seconds"
```

```python
@dataclass
class ForceDecision:
    model: str
    provider: str
    reasons: List[str] = field(default_factory=list)
    context: Optional[str] = None
    sticky_turns: Optional[int] = None
    sticky_timeout_s: Optional[int] = None
```

**Resolution logic:**

1. Aggregator picks the winner (from rankings or force decision).
2. If the winner declared `sticky_turns` or `sticky_timeout_s`, use it — clamped
   to the system min/max from config.
3. If the winner didn't declare stickiness (`None`), fall back to system default
   (1 turn = no stickiness beyond current turn).
4. Whichever expires first (turns or timeout) triggers re-evaluation.

**Examples:**

- Smart routing says "this is a coding task, keep upper model for 10 turns" →
  clamped to `min(10, sticky_max_turns)` → pipeline won't re-fire for 10 turns
  (assuming max ≥ 10).
- Quota router says "I'm routing to provider B because A is exhausted, stick for
  60s" → next user message within 60s skips the pipeline.
- A plugin returns `sticky_turns=100` but max is 20 → clamped to 20.
- A plugin returns `sticky_turns=None` → default behavior (re-evaluate next turn
  if an invalidation signal fires).

| Mode | Behavior |
|------|----------|
| `every_turn` | Pipeline fires on every user message (ignores stickiness) |
| `on_change` | Pipeline fires on invalidation signals; stickiness suppresses re-eval between signals |
| `sticky` | Pipeline fires once, then respects plugin-requested stickiness within bounds |

### Warm Cache for Availability

The availability filter maintains a warm cache of model health status (polled
in the background on a configurable interval, e.g., every 30s). When the pipeline
fires, availability checks are instant (read from cache, not live-probed). Cache
invalidation is event-driven: a failed LLM call immediately marks that model
unavailable in the cache.

This means even `on_change` mode is cheap — the availability filter doesn't make
network calls during pipeline execution, only during background refresh.

---

## References

- PR #48018: `pre_failover_decision` hook (error-path routing)
- Issue #23739: `pre_llm_call` model override request
- Issue #46991: `resolve_route` quota-aware routing request
- `hermes-arc` plugin: community topic-routing plugin (monkey-patches today)
- `agent/routing/` (fork): existing heuristic routing implementation
