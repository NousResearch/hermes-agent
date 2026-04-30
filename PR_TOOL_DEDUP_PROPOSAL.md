# PR Proposal: Agent-Level Tool Call Deduplication / Loop Prevention

## Problem Statement

The Hermes agent can enter expensive loops where it executes the same tool call (e.g., `git log`, `search_files`) multiple times in a row with identical arguments, producing identical output each time. This wastes tokens, burns context window space, and delays actual work.

**Real-world example from 2026-04-30:**
The agent ran `git log --all --oneline --grep="tokens_per_sec\|tok/s\|token.*sec\|speed" -- cli.py run_agent.py` 30+ times consecutively with slightly different grep patterns, all returning the same 7 unrelated commits. The agent never recognized it had already confirmed the feature didn't exist upstream.

## Why Existing PRs Don't Fully Solve This

- **PR #16641** (tool-call loop guardrails): Detects repeated *failing* or *non-progressing* tool calls within a single turn and injects warnings. Our bug was repeated *successful* calls across multiple turns that returned the same information.
- **PR #3006 / #8126** (tool result caching): Caches results for identical calls. This would help but doesn't address the root cause — the model shouldn't generate the redundant calls in the first place.

## Proposed Solution

Add an agent-level **tool call deduplication and progress tracking layer** that sits between the model's output and actual tool execution.

### Core Mechanism

1. **Per-turn tool call registry**: Hash (tool_name + normalized_args) → output
2. **Before executing any tool**: Check if this exact call was already made this turn
3. **If duplicate**: Return cached result with a `duplicate: true` flag and append a system note: `"Note: This tool call was already executed in this turn with identical arguments. Result was: ..."`
4. **If similar but not exact** (same tool, slightly different args, same output): Flag as "no new information" after N repeats
5. **Cross-turn tracking** (optional): Maintain a short LRU cache of recent tool calls to catch loops that span multiple turns

### Implementation Sketch

```python
# agent/tool_dedup.py

class ToolCallRegistry:
    """Tracks tool calls within a turn to prevent redundant execution."""
    
    def __init__(self, max_history=100):
        self._history = {}  # hash -> (args, output, timestamp)
        self._max_history = max_history
    
    def check(self, tool_name: str, args: dict) -> tuple[bool, Any]:
        """Returns (is_duplicate, cached_output) if this exact call was already made."""
        key = self._hash(tool_name, args)
        if key in self._history:
            return True, self._history[key][1]
        return False, None
    
    def record(self, tool_name: str, args: dict, output: Any):
        """Record a tool call result for deduplication."""
        key = self._hash(tool_name, args)
        self._history[key] = (args, output, time.time())
    
    def _hash(self, tool_name: str, args: dict) -> str:
        # Normalize args (sort dict keys, handle lists)
        normalized = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(f"{tool_name}:{normalized}".encode()).hexdigest()[:32]
    
    def reset(self):
        """Clear history at the start of each turn."""
        self._history.clear()
```

### Integration Points

- **run_agent.py**: Instantiate `ToolCallRegistry` at turn start, check before each tool execution
- **cli.py**: Add config option `tool_deduplication.enabled` (default: true)
- **Config**: Add to `hermes_cli/config.py` and `cli-config.yaml.example`

### Config Defaults

```yaml
tool_deduplication:
  enabled: true
  # How many identical calls before we force a cache hit
  exact_duplicate_threshold: 1  # Always dedup exact duplicates
  # How many similar calls (same tool, different args, same output) before warning
  no_progress_threshold: 3
  # Whether to append a system note when returning cached results
  append_system_note: true
```

### Test Plan

1. **Unit tests**: `tests/agent/test_tool_dedup.py`
   - Exact duplicate detection
   - Args normalization (dict order, list order)
   - Registry reset behavior
   - Hash collision safety

2. **Integration tests**: `tests/run_agent/test_tool_dedup_runtime.py`
   - Agent makes same `read_file` call twice in one turn → second returns cached
   - Agent runs `git log` with different grep patterns → all tracked separately
   - Turn reset: new turn can re-run same tool with fresh result

3. **Regression tests**:
   - Ensure normal multi-tool workflows still work
   - Ensure deliberate re-checks (e.g., polling `process(action="poll")`) aren't broken

## Scope

- **Files to modify**: ~5 files
  - `agent/tool_dedup.py` (new)
  - `run_agent.py` (wire into tool execution loop)
  - `hermes_cli/config.py` (add defaults)
  - `cli-config.yaml.example` (document)
  - `tests/agent/test_tool_dedup.py` (new)
  - `tests/run_agent/test_tool_dedup_runtime.py` (new)

- **Risk**: Low. This is a pure optimization — it only prevents redundant execution, never blocks new calls.

## Alternative: Simpler Approach

Instead of full deduplication, we could add a **"no new information" detector**:
- After any tool call, hash the output
- If the same tool was called recently with the same output, append a note: `"This produced the same result as the previous call. Consider if you already have the answer."`
- This is lighter weight but less aggressive about preventing the redundant execution itself.

## Recommendation

Implement the full deduplication approach. The cost is low (one dict lookup per tool call), the benefit is high (prevents the exact loop we hit), and it generalizes to any idempotent tool.

## Related PRs

- #16641: Tool-call loop guardrails (warning-first, single-turn)
- #3006: RAM-backed tool result cache
- #8126: Opt-in result memoization for idempotent tools

This PR would complement those by addressing the execution layer, not just the result caching or warning layer.
