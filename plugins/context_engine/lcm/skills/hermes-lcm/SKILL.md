---
name: hermes-lcm
description: Use when context was compacted by the LCM engine and a detail must be recovered. Bounded recall over the current session's retained raw messages and DAG summary nodes.
version: 0.1.0
author: NousResearch (Protean B04)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [LCM, context-engine, recall, compaction, DAG]
    related_skills: []
---

# LCM recall

The LCM context engine (`context.engine: lcm`) never deletes a message. When
the live context gets too long it replaces a range of messages with a *summary
node* that keeps a lineage edge to every message it absorbed. This skill is how
you get a compacted detail back, and how you avoid the two mistakes that make
recall expensive.

Read `references/recall-policy.md` before running a broad sweep, and
`references/recall-tools.md` for the exact tool arguments and response shapes.

## When to use this

Use it when the current session's context contains an LCM summary node (it
reads `[LCM summary node <node_id>]`) and you need something that was compacted
out: an earlier instruction, a value, a decision, the exact wording of a file
you already edited.

Do **not** use it to search across other sessions. That is `session_search`,
which covers the whole host history. LCM recall is scoped to *this* session and
its DAG, and the two surfaces are deliberately separate.

## The three moves

1. **Locate.** `lcm_search` with a distinctive term. It returns a few matches
   with a `message_id` each and never more than the configured ceiling.
2. **Expand.** `lcm_expand` with the `node_id` from the summary, or with a
   `message_id` from a search hit. A node expands into *pages* of its lineage.
   Each page carries `has_more` and `next_page`.
3. **Page on purpose.** If you need a contiguous stretch of the session, use
   `lcm_page` with a cursor and walk it. Never try to pull the session in one
   call. The tools will not do it.

## Rules

- **One page at a time.** Every LCM response is bounded by design. If you need
  more, ask for the next page. Do not ask for a bigger page than the ceiling.
- **Cite what you recovered.** When a recovered message drives a decision, name
  its `message_id` so the chain stays auditable.
- **Raw beats summary.** The summary node's digest is a lossy convenience. If a
  detail matters exactly (a path, a number, an identifier), expand the node and
  read the original message rather than trusting the bullet in the digest.
- **No unbounded transcript loads.** There is no parameter that returns the
  whole session, and asking for an enormous `page_size` is clamped, not honored.
- **Rollback is not your problem.** If the engine is disabled or fails to
  start, Hermes uses the built-in compressor and these tools do not exist.
  Nothing here needs repairing when that happens.
