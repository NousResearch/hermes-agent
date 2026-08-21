# ICM Card Schema

Canonical field grammar for ICM cards. All cards are Markdown files with a
YAML frontmatter block followed by optional Markdown body.

## Shared fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique identifier within the card's universe. |
| `kind` | string | yes | Card type: `object` or `process`. |
| `universe` | string | yes | Universe this card belongs to. See `CONTEXT.md`. |
| `name` | string | yes | Human-readable name. |
| `summary` | string | yes | One-line description. |
| `aliases` | list[string] | no | Alternate names for lookup. |
| `tags` | list[string] | no | Free-form categorization tags. |

## Object card

Represents a discrete thing: file, module, symbol, resource, or artifact.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `shape` | string | yes | Always `object`. |
| `path` | string | no | Filesystem path or URI, if applicable. |
| `interface` | list[string] | no | Public surface: symbols, methods, endpoints. |
| `depends_on` | list[string] | no | IDs or qualified refs this object requires. |

## Process card

Represents a flow, pipeline, lifecycle, or execution path.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `shape` | string | yes | Always `process`. |
| `steps` | list[object] | yes | Ordered list of `{id, summary}` steps. |
| `entrypoints` | list[string] | no | Entrypoint IDs or names. |
| `produces` | list[string] | no | IDs of objects this process creates or modifies. |
| `consumes` | list[string] | no | IDs of objects this process reads or destroys. |

## Validation

- `id` MUST match `^[a-z0-9_.-]+$`.
- `kind` MUST be one of `object` or `process`.
- `universe` MUST be a known universe from `CONTEXT.md`.
- `steps` MUST be ordered and non-empty for `process`.
- `depends_on`, `produces`, `consumes` MUST use qualified refs when crossing universes.
