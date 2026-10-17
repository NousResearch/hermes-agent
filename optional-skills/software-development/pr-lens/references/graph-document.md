When to load: writing or debugging a graph.json document — full field-by-field format, enums, limits.

# Authoring a graph document

This page is the whole shape, and what a schema cannot tell you besides: which parts matter, and where documents actually go wrong. `references/example.graph.json` is one document that validates, if you would rather read than be told.

The validator enforces the same thing from a JSON Schema, published at `https://unpkg.com/@coldtea/pr-lens-schema/json-schema/graph-doc.schema.json` if you want it machine-readable.

Every schema here is **strict**: an unknown key is a rejection, not a warning. A field with a default may be left out.

## The document

```json
{
  "schemaVersion": "0.1.1",
  "kind": "graph",
  "title": "Batch broadcast sending through Postmark",
  "summary": "One paragraph answering: what does this change do?",
  "lenses": ["architecture", "data-flow"],
  "provenance": { "repo": { "owner": "…", "name": "…" }, "base": { "sha": "…" }, "head": { "sha": "…" } },
  "lanes": [],
  "nodes": [],
  "edges": [],
  "flows": [],
  "stats": {},
  "views": []
}
```

`lenses` declares what the document carries enough detail to draw: `architecture`, `data-flow`, or both. A document carrying flows must declare `data-flow`.

`provenance` is where the document came from: the repository, the base and head commit shas (lowercase hex, 7-40 characters), optionally the pull request and the generator. When you produce a document through the CLI these are filled in from the repository, so do not invent them.

## Ids

`^[A-Za-z0-9][A-Za-z0-9._:/-]*$`, at most 128 characters, unique within their own collection. Use readable kebab-case: `broadcast-sender`, not `n1`. An id ends up in an SVG id, a URL fragment and a comment anchor, so nothing else is allowed through.

## Deltas

Every node, edge, flow and flow step declares one: `added`, `modified`, `removed`, `unchanged`.

`unchanged` is not padding. It is the neighbouring code the change touches, and it is what turns a diagram into a blast radius. A document whose every element is `added` describes a change nobody can place.

## Lanes

1 to 16. Every node belongs to exactly one.

```json
{ "id": "functions", "label": "Cloud Functions", "subtitle": "Node 20", "order": 1 }
```

`order` (0-64) places lanes left to right; ties fall back to array order. Give a lane a `delta` only when the lane itself is new or gone.

## Nodes

1 to 256.

```json
{
  "id": "send-broadcast-bulk",
  "label": "sendBroadcastBulk",
  "kind": "function",
  "delta": "added",
  "lane": "functions",
  "group": "broadcast-lib",
  "subtitle": "(broadcastId) => Promise<void>",
  "summary": "Claims the broadcast, builds one bulk payload and posts it.",
  "files": [{ "path": "functions/src/broadcast/sendBroadcastBulk.ts", "startLine": 1, "endLine": 142 }],
  "badges": ["retry"]
}
```

`kind` is one of `service app module function route job queue datastore cache external ui config test package other`. It drives the card's icon and shape and nothing else; when in doubt, `other` still renders.

`group` clusters nodes inside a lane: a package, a folder that means something. `files` (up to 64) become diff permalinks. `badges` (up to 6) are extra chips; the delta badge is drawn for you, so do not restate it.

## Edges

Up to 512.

```json
{
  "id": "bulk-to-postmark",
  "from": "send-broadcast-bulk",
  "to": "postmark",
  "kind": "http",
  "delta": "added",
  "label": "POST /email/bulk",
  "emphasis": "hero",
  "animated": true
}
```

`kind` is one of `call http rpc event queue data dependency render other`. `emphasis` is `normal` (default), `hero` or `muted`. More than one or two heroes and the emphasis stops meaning anything. `from` and `to` must be node ids you declared. This is the single most common failure.

## Flows

Up to 16, for the data-flow lens.

```json
{
  "id": "send-pipeline",
  "title": "Sending a broadcast",
  "delta": "modified",
  "participants": [{ "node": "queue-route" }, { "node": "send-broadcast-bulk" }, { "node": "postmark" }],
  "messages": [
    { "id": "enqueue", "from": "queue-route", "to": "send-broadcast-bulk", "label": "enqueue job", "kind": "async", "delta": "modified" },
    { "id": "send", "from": "send-broadcast-bulk", "to": "postmark", "label": "POST /email/bulk", "kind": "sync", "delta": "added", "repeat": 4 },
    { "id": "accepted", "from": "postmark", "to": "send-broadcast-bulk", "label": "200 Accepted", "kind": "return", "delta": "added" }
  ]
}
```

- 2 to 12 participants, ordered by array position; each names a node id.
- 1 to 64 messages. **Step order is array order**: there is no step number field, so a document cannot disagree with its own animation.
- `kind` is `sync`, `async`, `return` or `self`. `self` requires `from === to`, and no other kind may have them equal.
- Both endpoints must be participants of that flow, not merely nodes of the document.
- `repeat` says a step happens more than once per run, e.g. 4 batched requests.

## Stats

```json
{ "filesChanged": 27, "additions": 1979, "deletions": 1370, "chips": [{ "label": "Postmark calls", "value": "500x fewer", "tone": "hero" }] }
```

Up to 8 chips, `tone` one of `neutral added modified removed hero`. Per-delta element counts are deliberately absent from the schema: they are derivable from the document, and a stored copy can only go stale.

## Views

The drill-down tree in the comment: up to 32 at the root, nesting up to 32 children each. A document with no views renders as one picture and nothing else.

```json
{
  "id": "the-new-path",
  "title": "The new batch path",
  "lens": "architecture",
  "summary": "What replaced the per-recipient loop.",
  "defaultOpen": false,
  "scope": { "kind": "selection", "nodes": ["send-broadcast-bulk", "postmark"] },
  "children": []
}
```

`scope` is either `{ "kind": "all" }` (the default) or a selection naming at least one lane, node, edge or flow. The two are distinct states on purpose: removing the last element a view pointed at can never quietly turn it into a view of everything. A view's `lens` must be one the document declares.

### Choosing architecture views

Treat the architecture tree as a set of decisions, not a quota:

1. Ask whether the change affects a user, an external system or a system boundary. If it does, start with a system-context view. If it does not, leave that level out.
2. Show affected applications, services, jobs, data stores and runtimes in a container view. Make it the root when there is no useful context view; otherwise make it a child of that context.
3. Add a component child only when the internals of an affected container matter to the change. Components may be modules, routes or functions, but the view should explain their responsibilities and relationships rather than mirror folders.
4. Stop at components unless someone explicitly asks for code-level detail.

One architecture view may be the right answer for a small change. Each child must move down exactly one level and cover a materially narrower scope. Skip a level when it would be empty, speculative or a repeat of its parent. Do not create two views with substantially the same nodes and edges, and do not infer a boundary from a folder name alone. Keep unchanged direct neighbours when they make the blast radius clear.

Set `defaultOpen: true` on the highest useful architecture view. Lower levels should normally stay collapsed. A data-flow view describes an ordered sequence, so keep it as a separate root instead of placing it inside the architecture hierarchy.

This compact fragment shows the shape. The selected ids refer to elements declared elsewhere in the document:

```json
{
  "views": [
    {
      "id": "checkout-context",
      "title": "Checkout in its environment",
      "lens": "architecture",
      "defaultOpen": true,
      "scope": {
        "kind": "selection",
        "nodes": ["shopper", "commerce-platform", "payment-provider", "fulfilment-system"],
        "edges": ["shopper-to-commerce", "commerce-to-payment", "commerce-to-fulfilment"]
      },
      "children": [
        {
          "id": "checkout-containers",
          "title": "Checkout containers",
          "lens": "architecture",
          "scope": {
            "kind": "selection",
            "nodes": ["storefront", "checkout-api", "orders-db", "payment-provider"],
            "edges": ["storefront-to-checkout", "checkout-to-orders", "checkout-to-payment"]
          },
          "children": [
            {
              "id": "checkout-components",
              "title": "Checkout API components",
              "lens": "architecture",
              "scope": {
                "kind": "selection",
                "nodes": ["checkout-route", "order-service", "payment-client"],
                "edges": ["route-to-orders", "orders-to-payment-client"]
              }
            }
          ]
        }
      ]
    },
    {
      "id": "place-order-flow",
      "title": "Placing an order",
      "lens": "data-flow",
      "scope": { "kind": "selection", "flows": ["place-order"] }
    }
  ]
}
```

## Walkthrough

Optional in the format, but write one for anything that is not trivial: more than one diagram, a diagram with several changed parts, or any flow. Skip it only when the document is one small diagram whose single step would just repeat the title. A canvas or a share page plays it.

```json
{
  "walkthrough": {
    "steps": [
      {
        "id": "four-batch-calls",
        "heading": "Postmark now gets 500 emails per call",
        "body": "One call per batch, and Postmark answers with a result for each message.",
        "stage": { "kind": "flow", "flow": "send-pipeline" },
        "focus": { "kind": "selection", "messages": ["batch-post", "batch-results"] }
      },
      {
        "id": "blast-radius",
        "heading": "4 parts added, 2 removed, across 3 lanes",
        "body": "A 2,000-person broadcast used to make 2,000 calls to Postmark. It now makes 4.",
        "stage": { "kind": "view", "view": "overview" },
        "focus": { "kind": "all" }
      }
    ]
  }
}
```

A walkthrough is a short guided tour of the diagrams. It has two to twelve steps. Each step shows one diagram, points at one part of it, and says a few words about it.

Every step is one change, never a description of the diagram: the heading names the thing and what happened to it, built from change words such as added, removed, replaced, now, moved and split, and the body is one line on what that means for behaviour, with the numbers when they matter. The headline change is step one. Write it all for a smart twelve-year-old, in short common words and active voice. The skill page has the rule in full, with examples of a step written well and the same step written badly.

Each step has:

- `heading`: the thing and what happened to it, up to 48 characters, in sentence case. For example "Postmark now gets 500 emails per call".
- `body`: one line under the heading, up to 140 characters, on what the change means for behaviour. For example "One call per batch instead of one call per person". Required: a heading with no body reads as unfinished.
- `stage`: which diagram to show. A document can have several diagrams: its views (the drill-down diagrams) and its flows (the sequence diagrams). `{ "kind": "view", "view": "overview" }` shows the view called `overview`. `{ "kind": "flow", "flow": "send-pipeline" }` shows the flow called `send-pipeline`. Leave `stage` out and the step uses the diagram the reader is already on.
- `focus`: what to zoom in on inside that diagram. `{ "kind": "all" }`, the default, means the whole diagram. A selection means "just these things": name any lanes, nodes, edges or flow steps (`messages`) by id, and the camera zooms to them while everything else dims. A selection must name at least one thing.

The validator checks:

- Every id you name exists in the document. A flow step you name must belong to the flow the stage shows, because flow step ids are only unique inside their own flow.
- `messages` needs a stage that shows a flow. Leave it out when the stage is an architecture view.
- Step ids are unique within the walkthrough. Two steps minimum, twelve maximum.
- A stored map never carries a walkthrough. A map describes the system; a walkthrough tells the story of one change.

## Layout

```json
{ "direction": "right", "laneOrder": ["api", "functions", "external"], "rank": { "send-broadcast-bulk": 2 } }
```

Hints, not instructions: the renderer owns final placement, so a diagram stays deterministic and a stale hint cannot break it. Absolute coordinates are not expressible. Omitting `layout` entirely is normal.

## File references

```json
{ "path": "functions/src/broadcast/sendBroadcastBulk.ts", "startLine": 1, "endLine": 142, "revision": "head" }
```

Repository-relative POSIX paths: no leading `/`, no drive letter, no backslash, no `..` segment. Lines are 1-based, `endLine` requires `startLine` and may not precede it. `revision` defaults to `head`; use `base` on elements the change removes.

## Length limits

Labels 120 characters, summaries 2000, chip values 32. They are display fields: a label that needs 120 characters is a label the diagram cannot draw.

## Then validate

```bash
npx @coldtea/pr-lens-cli@latest validate .pr-lens/graph.json
```

Every problem is reported at once, with a path into the document. Fix them all and run it again until it is clean.
