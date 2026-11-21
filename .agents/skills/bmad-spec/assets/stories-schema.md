# Stories schema

`stories.yaml` — the optional output of Story Breakdown: a top-level YAML list, one entry per story, in execution order — stories run top to bottom. Sibling of `SPEC.md`, discovered by its fixed filename (same convention as `SPEC.md` and `.memlog.md`); never listed in `companions:`, never referenced from frontmatter.

## Fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Identity, unique within the file. Compared for equality and used as a filename prefix; carries no other meaning. Use unpadded integers (`"1"`), or composites (`"3-2"`) when the epic sits inside a larger project spec. No zero-padding — execution order is the list order, not filename sort. Pinned once the story's spec file exists (see Update semantics); until then it may be renumbered. |
| `title` | string, one line | yes | Display name; becomes the story's heading downstream. |
| `description` | string | yes | What this story covers, pointing into `SPEC.md` — not a story spec. Keep it to two sentences; a writing rule, not something tooling enforces. |
| `spec_checkpoint` | boolean | no (default `false`) | Set by the human at breakdown time; read only by the dispatching caller, never by the implementing dev skill. When true, a human reviews the story spec between planning and implementation. |
| `done_checkpoint` | boolean | no (default `false`) | Caller-only, like `spec_checkpoint`. When true, dispatch pauses after this story completes, before anything further runs. |
| `invoke_dev_with` | string | no (default `""`) | Free text appended verbatim to the prompt that dispatches this story; the implementing dev skill reads it as part of its prompt, and nothing else interprets it. If the text needs structure, put it inside the string. Which dev skill to invoke is the caller's configuration, never data in this file. |

## Validity rules

1. Every entry parses with all required fields; ids unique.
2. Ids are prefix-free under the `<id>-` filename-matching convention: no id may equal another id plus a dash-suffix (`"3"` and `"3-2"` cannot coexist).
3. No `status` field, ever.
4. Ids are YAML strings, always quoted, containing only letters, digits, and dashes. An unquoted `id: 1` parses as a number and breaks string comparison; characters like `/` or `*` break the filename match.

## Example

```yaml
- id: "1"
  title: Add rate limiting to the public API
  description: >-
    Introduce a token-bucket limiter in front of the public endpoints;
    return 429 with a Retry-After header on limit breach.
  spec_checkpoint: true
  invoke_dev_with: >-
    Rate limit state must be shared across instances; use the existing
    Redis client, not in-process memory.
- id: "2"
  title: Expose limiter metrics to the ops dashboard
  description: >-
    Emit per-route accept/reject counters the existing dashboard can
    scrape; no new dashboard panels in this story.
```

## Update semantics

Updates to `stories.yaml` go through Story Breakdown: append the change to `.memlog.md`, then re-derive. An id is pinned once its story spec file exists (any `stories/<id>-*.md` in the spec folder): a pinned story keeps its id through edits, its removal retires the id, and retired ids are never reassigned. Stories with no spec file yet may be renumbered, reordered, or removed freely on re-derive — typically so ids keep following list order. Never give a story an id that collides with a `stories/` file belonging to a different story.
