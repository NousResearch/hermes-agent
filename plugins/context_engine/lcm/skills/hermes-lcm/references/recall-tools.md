# LCM recall tool reference

All four tools are injected by the engine at startup. They only exist while
`context.engine: lcm` is active.

## `lcm_search`

Find a retained message in the current session.

| Arg | Type | Notes |
|---|---|---|
| `query` | string | Required. Tokens are matched conjunctively. Operator characters are neutralised. |
| `limit` | integer | Optional. Clamped to `max_search_results`. |

Returns `{session_id, query, returned, results[], bounded, has_more, note}`.
Each result is `{message_id, body, truncated, match}` where `match` is `fts`
(index) or `like` (fallback when the FTS index is unavailable).

## `lcm_expand`

Descend a summary node, or read one stored message.

| Arg | Type | Notes |
|---|---|---|
| `node_id` | string | From an `[LCM summary node <id>]` marker. |
| `message_id` | string | From a search hit or a page row. |
| `page` | integer | Lineage page index. Default 0. `node_id` only. |
| `page_size` | integer | Clamped to `max_page_size`. |
| `max_chars` | integer | Per-message body cap, clamped to the hard maximum. |

Exactly one of `node_id` / `message_id`. Node expansion returns
`{node{node_id,level,source_count,summary}, lineage[], returned, total, page,
page_size, has_more, next_page, body_chars, bounded}`. Each `lineage` entry is
either a resolved `{kind:"message", id, seq, role, body, truncated}` or a
`{kind:"node", id, level, note}` pointer telling you to descend further.

## `lcm_page`

Walk the session's raw messages ascending by sequence.

| Arg | Type | Notes |
|---|---|---|
| `cursor` | integer | `next_cursor` from the previous call. 0 starts at the beginning. |
| `page_size` | integer | Clamped to `max_page_size`. |

Returns `{items[], returned, total, page_size, has_more, next_cursor, bounded}`.

## `lcm_status`

No arguments. Returns the retained-message and summary-node counts, schema
version, FTS availability, effective bounds, and the recall policy. This is the
same view the `/lcm status` slash command prints.

## CLI

```
/lcm status
/lcm search <query>
/lcm expand <node_id|message_id> [page]
/lcm page [cursor]
```

## Reading a DAG

A node's lineage may contain `node` edges as well as `message` edges. That
happens when an already-summarized range is summarized again: the new node
points at the superseded *node*, not at the rendered marker message, so the DAG
stays a DAG rather than a chain of copies. Descend with `lcm_expand
node_id=<child>` until every edge is a `message`.
