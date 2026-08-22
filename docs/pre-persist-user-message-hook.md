# `pre_persist_user_message` hook

Agent-path ingress seam. Fires once per user turn in `build_turn_context`, **before**
the inbound user message is appended and persisted. It is the agent-path analogue of
the gateway's `pre_gateway_dispatch` (which mutates `event.text`): a plugin can fold
context into the user turn on paths that never go through the gateway (CLI, subagent,
agent-to-agent). Unlike `pre_llm_call` — whose returns are ephemeral — returns here
reach **both the wire and the persisted session-DB row**.

## Kwargs

The hook is called with:

```
session_id, task_id, turn_id, user_message, conversation_history, platform, sender_id
```

## Return contract

Each registered plugin returns one of:

| Return | Effect |
|--------|--------|
| `None` | ignored |
| `{"context": str}` / bare `str` | appended at the tail |
| `{"user_message": str, "data": {"priority": int}}` | replaces the body |

Composition (`_compose_pre_persist_returns`) is a pure function over all plugin
returns:

- **Append** — every `{"context"}`/`str` fragment is joined by blank lines and
  appended at the tail, in plugin-iteration order. Several plugins each adding a note
  all land, no coordinator needed.
- **Replace** — among `{"user_message": ...}` returns the highest `data.priority`
  wins (ties → first, stable); a warning is logged if more than one plugin replaces.
  This is the escape hatch for a plugin that owns the *whole* body (e.g. it has placed
  one fragment before the user text and another after). Symmetric to
  `transform_llm_output` on the egress side.
- **The two compose** — a winning replace sets the body, and append fragments still
  stack at the tail *on top of it*. Replace competes only with other replaces, never
  with appends, so a body-owning plugin and a simple note-appender coexist.

The composed result is also re-applied to `_persist_user_message_override` so an
injected block survives into replayed history instead of vanishing next turn.

Additive and off by default: with no plugin registered, the compose is a no-op.

## Example 1 — minimal append

```python
# plugin __init__.py
"""Append a one-line banner to the persisted user turn ({"context"} shape)."""

def register(ctx):
    def _on_pre_persist(**wire):
        if not (wire.get("user_message") or "").strip():
            return None
        return {"context": "[example] seen this turn"}
    ctx.register_hook("pre_persist_user_message", _on_pre_persist)
```

```yaml
# plugin.yaml
name: pre-persist-example
version: "0.1.0"
kind: standalone
description: Minimal example of the pre_persist_user_message hook (append shape).
provides_hooks: [pre_persist_user_message]
provides_tools: []
```

## Example 2 — an ordering hub (why the replace path exists)

A ~25-line hub built on top of the hook. Other plugins register a
`(priority, producer)` with it; the hub runs them, sorts fragments by priority, and
returns **one** composed body via the replace path. Append alone cannot do this — it
can never place a fragment above the user text — which is why the contract needs
`user_message` replace.

```python
# plugin __init__.py
"""Order multi-plugin fragments into one body via the replace path."""

_SUBS = {}  # name -> (priority, producer(wire) -> str | None)

def subscribe(name, priority, producer):
    """Any plugin calls this (plain import) to contribute a fragment."""
    _SUBS[name] = (priority, producer)

def register(ctx):
    def _compose(**wire):
        base = wire.get("user_message") or ""
        frags = []
        for name, (prio, produce) in _SUBS.items():
            try:
                out = produce(wire)          # one bad producer never sinks the rest
            except Exception:
                continue
            if out:
                frags.append((prio, name, out))
        if not frags:
            return None
        frags.sort(key=lambda t: (t[0], t[1]))          # by priority, then name
        body = base + "\n\n" + "\n\n".join(f[2] for f in frags)
        return {"user_message": body, "data": {"priority": 100}}   # own the body
    ctx.register_hook("pre_persist_user_message", _compose)
```

```yaml
# plugin.yaml
name: inject-ordering-hub
version: "0.1.0"
kind: standalone
description: Example hub — orders multi-plugin injects into one body via the replace path.
provides_hooks: [pre_persist_user_message]
provides_tools: []
```
