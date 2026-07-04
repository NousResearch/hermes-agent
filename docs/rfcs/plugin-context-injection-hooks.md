# Plugin Context Injection Hooks — Design Proposal

**Branch:** `feat/plugin-context-injection-hooks`
**Target:** `agent/prompt_builder.py`, `hermes_cli/plugins.py`
**Concrete consumer:** kanban-advanced (worker context, board state, plan metadata)

---

## Summary

Today plugins have two paths to inject context into the agent's conversation:
1. **Skills** — explicit opt-in loads via `/skill` or `register_skill()`
2. **`pre_llm_call` hooks** — inject context into the user message each turn

Neither path can add stable context to the **system prompt** without breaking
prompt caching (the system prompt must be byte-identical across turns).
Plugins also can't extend the environment hints block
(`build_environment_hints()`) that tells the agent about its execution context.

This proposal adds two extension points that give plugins controlled access to
the system prompt and environment hints while preserving the cache invariant.

---

## Proposal 1: `ctx.register_system_prompt_section()`

### Current state
Plugin-provided context (kanban board state, plan context, governance rules)
must be loaded as explicit skills. If the agent doesn't load the skill, it has
no awareness of the plugin's context. `pre_llm_call` hooks can inject into user
messages, but this is per-turn and ephemeral — not suitable for persistent
context that should be available every turn.

### Proposed API

```python
class PluginContext:
    def register_system_prompt_section(
        self,
        key: str,              # stable key for cache pinning
        content: str | Callable[[], str],
        *,
        priority: int = 0,     # ordering among sections (higher = later)
        enabled: bool = True,  # can be toggled per-session
    ) -> None:
        """Register a stable section of the system prompt.

        The content is evaluated once at session start and remains byte-stable
        for the session lifetime. Changing the key, priority, or content
        invalidates the prompt cache — this is documented behavior.

        When *content* is a callable, it's called once at session start with
        no arguments. The result is cached. This lets plugins provide dynamic
        context (e.g. current board state) without breaking the cache for
        the rest of the session.

        Sections are rendered between the core system prompt and the
        <available_skills> block, ordered by priority (lowest first).
        """
```

### Cache safety

The system prompt cache is keyed on the concatenation of all registered
sections in priority order. If any section changes (key, priority, or
content), the cache is invalidated for the NEXT session — never mid-session.
This follows the same contract as toolset changes.

### Concrete use case (kanban-advanced)

```python
# In kanban-advanced register(ctx):
def _board_context() -> str:
    """Dynamic context evaluated once at session start."""
    board = os.environ.get("HERMES_KANBAN_BOARD", "default")
    task_id = os.environ.get("HERMES_KANBAN_TASK", "")
    plan_id = os.environ.get("KANBAN_PLAN_ID", "")
    return f"""## Kanban Context
- Board: {board}
- Current task: {task_id}
- Plan: {plan_id}
- Governance: Load kanban-advanced:kanban-worker for full SOP."""

ctx.register_system_prompt_section(
    key="kanban_context",
    content=_board_context,
    priority=100,
)
```

---

## Proposal 2: `build_environment_hints` hook

### Current state
`agent/prompt_builder.py::build_environment_hints()` hardcodes the OS/shell/
backend hints. Plugins can't add their own execution-context information
without monkey-patching. For kanban workers, critical context (board name,
plan ID, workspace path) is only available via env vars that the system prompt
may or may not reference.

### Proposed hook

```python
# VALID_HOOKS addition: "build_environment_hints"
# Fired during system prompt construction, once per session.
# Callbacks receive the current hints dict and may return additional
# (key, value) tuples to append.
# Return shape: {"hints": [("Key", "Value"), ...]}
#
# The core environment hints block renders as:
#   Host: Windows (10)
#   User home directory: C:\Users\Owner
#   ...
# Plugin hints are appended after core hints in registration order.

def on_build_environment_hints(hints: dict, **kwargs) -> dict | None:
    return {
        "hints": [
            ("Kanban board", os.environ.get("HERMES_KANBAN_BOARD", "default")),
            ("Kanban plan", os.environ.get("KANBAN_PLAN_ID", "")),
        ]
    }
```

### Fire site

In `agent/prompt_builder.py::build_environment_hints()`:
```python
# After building the core hints block, before returning:
from hermes_cli.plugins import invoke_hook
plugin_hints = invoke_hook("build_environment_hints", hints=core_hints)
for result in plugin_hints:
    if isinstance(result, dict) and "hints" in result:
        for key, value in result["hints"]:
            lines.append(f"{key}: {value}")
```

### Cache safety
Environment hints are built once at session start. Plugin hooks fire during
that single construction pass. The resulting block is cached for the session
lifetime — identical to current behavior.

---

## Non-goals

- **Dynamic per-turn context** — `pre_llm_call` hooks already handle this.
- **Mid-session prompt modification** — would break prompt caching. Rejected.
- **Automatic skill loading** — skills remain explicit opt-in to preserve
  the user's control over context budget.

---

## Related

- PR #58541 — kanban lifecycle hooks
- PR #58542 — plugin config & state bridge
- AGENTS.md: "Per-conversation prompt caching is sacred"
