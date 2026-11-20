When to load: fixing or overriding a generated map via the .github/pr-lens.yml correction overlay.

# Correcting the map: `.github/pr-lens.yml`

The generated document is regenerated on every run, so editing it is pointless. Corrections live in `.github/pr-lens.yml`, an overlay applied over fresh inference every time. Inference never writes back into this file, which is why a correction keeps holding as the code moves.

```yaml
schemaVersion: 0.1.1          # required
lenses: [architecture, data-flow]
branding: true
map:
  rename:
    - match: functions/src/broadcast/sendBroadcastBulk.ts
      to: Broadcast sender
  exclude:
    - "**/*.test.ts"
    - scripts/**
  lane:
    - match: packages/broadcast-lib/**
      lane: functions
  group:
    - match: id:build-bulk-payload
      group: broadcast-lib
```

Every field except `schemaVersion` is optional, and the file itself is optional. For editor autocomplete, point at the published JSON Schema — no install needed:

```jsonc
{ "$ref": "https://unpkg.com/@coldtea/pr-lens-schema/json-schema/config.schema.json" }
```

## Selectors

A `match` beginning with `id:` addresses exactly one node, as in `id:build-bulk-payload`. Anything else is a repository-relative path glob matched against the node's file paths.

**Prefer the glob.** Ids come from inference and may change when the code does; a path correction survives that. Reach for `id:` only when no path distinguishes the node, or when the node has no files at all (an external service, a queue).

## The four corrections

| | What it does |
| --- | --- |
| `rename` | replaces the inferred label |
| `exclude` | drops matching nodes, and the edges and flow steps that hung from them |
| `lane` | moves matching nodes into a lane, **creating it** when the document declares no such id |
| `group` | clusters matching nodes under a sub-group inside their lane |

Up to 128 of each. They are about intent rather than structure: there is no way to add a node or draw an edge here, and the one thing a correction can bring into existence is a lane, a band a repository wants that inference did not find. It takes the id for its label, because the id is the only name this file carries, so write `lane: infrastructure` rather than `lane: l3`. If the map is wrong in a way corrections cannot express, the fix belongs in the analysis, not in this file.

## Recipes

**"Stop showing me the test files."**
```yaml
map:
  exclude: ["**/*.test.ts", "**/__tests__/**"]
```

**"That node is called the wrong thing."** Match the file it comes from, not its id:
```yaml
map:
  rename:
    - match: server/lib/broadcast/createBroadcastSendTask.ts
      to: Send task
```

**"These belong in a band of their own."** The lane need not exist yet:
```yaml
map:
  lane:
    - match: infra/**
      lane: infrastructure
```

**"Keep the shared library together."**
```yaml
map:
  group:
    - match: packages/broadcast-lib/**
      group: broadcast-lib
```

**"Only draw the architecture."**
```yaml
lenses: [architecture]
```

## Hosted GitHub App comments

The hosted App reads `github` settings from the PR's head commit. Other options apply to the CLI.

| Setting | Default | Effect |
| --- | --- | --- |
| `github.comment.collapsed` | `false` | Start diagrams and details closed. Drawing still runs automatically. |

## Check it

```bash
npx @coldtea/pr-lens-cli@latest validate .github/pr-lens.yml
```

`pr-lens render` reports any correction that changed nothing about the document it drew. That is a config that has drifted out of date, usually because the file a selector named has moved or gone. It is not an error and nothing stops, but it is worth fixing: a correction that matches nothing is a correction nobody is getting.
