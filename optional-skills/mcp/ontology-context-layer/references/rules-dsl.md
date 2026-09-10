# Rules DSL Reference

Rules are plain JSON, stored in the ontology store and upserted by `ontology_add_rule` (matched on `name`).

## Rule Shape

```json
{
  "name": "Large deal needs approval",
  "description": "Deals over $10k need manager sign-off.",
  "if": [
    {"property": "value", "op": "gt", "value": 10000}
  ],
  "then": "requires_manager_approval",
  "severity": "warning",
  "mode": "all"
}
```

| Field | Default | Meaning |
|---|---|---|
| `name` | — (required) | Unique; adding a rule with an existing name replaces it |
| `if` | — (required) | One condition dict, or a list of condition dicts |
| `then` | `"ok"` | Machine-readable action/outcome label returned on PASS |
| `severity` | `"info"` | `info` \| `warning` \| `error` — display hint |
| `mode` | `"all"` | `all` = every condition must pass; `any` = at least one |
| `description` | `""` | Human explanation surfaced in validation output |

## Property Conditions

Conditions check `properties.<key>` first, then top-level entity fields (`verified`, `confidence`, `source`, `id`, `type`, `name`).

| op | Behavior | Example |
|---|---|---|
| `eq` | equality | `{"property":"status","op":"eq","value":"Won"}` |
| `ne` | inequality | `{"property":"status","op":"ne","value":"Lost"}` |
| `gt` / `gte` | numeric greater (or equal) | `{"property":"value","op":"gt","value":10000}` |
| `lt` / `lte` | numeric less (or equal) | `{"property":"age","op":"lt","value":18}` |
| `in` | value in a list | `{"property":"status","op":"in","value":["Scheduled","In Progress"]}` |
| `contains` | substring in value | `{"property":"email","op":"contains","value":"@acme.test"}` |
| `exists` | field present and non-empty | `{"property":"email","op":"exists"}` |
| `not_exists` | field absent or empty | `{"property":"datePaidInFull","op":"not_exists"}` |

## Relationship Conditions

Count a specific relationship type from the entity, optionally filtered by the target's type:

```json
{"relationship": "works_at", "target_type": "Company", "count_op": "gte", "count": 1}
```

This passes when the entity has ≥1 outgoing `works_at` relation pointing at a `Company`.

## Worked Examples

**Approval gate — reject deals over $10k without manager sign-off:**

```json
{
  "name": "Approval gate",
  "if": [
    {"property": "value", "op": "gt", "value": 10000},
    {"property": "approved_by_manager", "op": "eq", "value": true}
  ],
  "then": "ok",
  "severity": "error"
}
```

**Follow-up required — unverified entity with no recent touch:**

```json
{
  "name": "Follow-up required",
  "if": [
    {"property": "verified", "op": "eq", "value": false},
    {"property": "lastTouchDate", "op": "not_exists"}
  ],
  "then": "call_today",
  "severity": "warning"
}
```

**Any-mode — blocked by price OR timing:**

```json
{
  "name": "Objection detected",
  "mode": "any",
  "if": [
    {"property": "objection", "op": "eq", "value": "Price"},
    {"property": "objection", "op": "eq", "value": "Timing"}
  ],
  "then": "handle_objection"
}
```

## Validation Output

`ontology_validate(entity_id)` returns per-rule results; each carries `passed`, `mode`, per-condition `reasons` (`PASS`/`FAIL` lines with the evaluated detail), plus `then`, `severity`, and `description`. `overall` is `PASS` only when every rule passes.
