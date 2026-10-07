# LCM recall policy

## Scope: the current session's DAG, nothing else

LCM recall reads a session-scoped SQLite store:

- every raw message the engine retained for this session, and
- every summary node plus its lineage edges.

It does **not** read other sessions, other profiles, or the session database.

That is deliberate. Hermes already has `session_search` for broad history
across sessions. Making LCM recall a second, weaker cross-session search would
create two answers to "what did we decide weeks ago" and no way to tell which
is authoritative. So the split is:

| Question | Tool | Scope |
|---|---|---|
| "What exactly did the message that got compacted out of *this* context say?" | `lcm_expand` / `lcm_search` | this session |
| "Where did we discuss X, in any session?" | `session_search` | host-wide history |

## Bounds are contractual, not advisory

| Bound | Meaning |
|---|---|
| `page_size` | Maximum lineage edges (or messages) returned per call. Configurable down, clamped up. |
| `max_page_size` | Hard ceiling on `page_size`. Default 50. |
| `max_search_results` | Maximum matches per search. Default 8, ceiling 25. |
| `body_chars` | Per-message body cap. Default 2000, ceiling 8000. |

These are enforced in the storage layer, so no caller (model, CLI, or config)
can widen them past the ceiling. A request above the ceiling is clamped, not
honored and not errored: the caller gets a valid page plus `has_more: true`.

## What "never loads an unbounded transcript" means concretely

1. There is no tool argument meaning "the whole session". `lcm_page` takes a
   cursor and returns at most `page_size` messages.
2. A node expansion returns lineage for **one page** of edges. The node's
   `source_count` and the response's `total` tell you how many more pages exist
   without materializing them.
3. Message bodies are truncated to `body_chars` and the response says
   `truncated: true`, so a caller can ask for a targeted higher cap (still
   clamped) rather than being silently handed a partial body.

## Cost and cache

- Recall tools are explicit calls. The engine does **not** implement
  per-turn `select_context()` replacement, so an LCM session's prompt prefix is
  byte-identical to a non-LCM one for a given transcript: installing the engine
  does not, by itself, change prompt-cache behaviour. What changes the prefix is
  compaction, and compaction is the feature.
- Expanding a node does not write to the transcript. Reads never mutate the
  session or the DAG.

## Failure behaviour

Every recall path is fail-open. If the store cannot be opened, a tool returns a
JSON error object and the turn continues. If the engine cannot be loaded at all,
the host falls back to the built-in `ContextCompressor` and the `lcm_*` tools do
not appear. There is no state to reconcile when the plugin is disabled. The
store is a read-only record of what already happened.
