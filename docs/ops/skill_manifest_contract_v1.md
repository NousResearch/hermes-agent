# OpenClaw skill manifest contract v1

Status: proposed adopt-now donor fold-in slice
Version string: `openclaw-skill-manifest.v1`

## Goal

Introduce a governed, machine-readable skill manifest / invocation contract without breaking existing skills.

This slice does **not** make malformed manifests fail to load at runtime yet. Instead, it adds a normalized contract summary plus structured findings so operators and future tooling can detect:

- malformed `metadata.openclaw` blocks
- unsafe `skillKey` values
- invocation flag mistakes (`user-invocable`, `disable-model-invocation`)
- unknown manifest keys that would otherwise be silently ignored
- malformed or duplicate install entries
- `primaryEnv` / `requires.env` drift

## Normalized contract object

The patch emits a `contract` object on skill status records:

```json
{
  "version": "openclaw-skill-manifest.v1",
  "mode": "governed",
  "valid": true,
  "errorCount": 0,
  "warningCount": 0,
  "invocation": {
    "userInvocable": true,
    "disableModelInvocation": false,
    "modelInvocable": true
  },
  "manifest": {
    "metadataDeclared": true,
    "skillKey": "coding-agent",
    "primaryEnv": null,
    "rawInstallCount": 2,
    "normalizedInstallCount": 2
  },
  "issues": []
}
```

## Modes

- `legacy` — no governed metadata block and no explicit invocation flags
- `invocation-only` — explicit invocation flags but no governed metadata block
- `governed` — `metadata.openclaw` block present and parseable enough to inspect

## Structured issue format

Each issue is emitted as:

```json
{
  "code": "INVALID_SKILL_KEY",
  "level": "error",
  "path": "metadata.openclaw.skillKey",
  "message": "skillKey \"bad.contract\" must match /^[A-Za-z0-9_-]+$/"
}
```

### Current issue codes covered by the slice

- `INVALID_METADATA_BLOCK`
- `INVALID_INVOCATION_BOOLEAN`
- `INVALID_SKILL_KEY`
- `INVALID_REQUIRES_BLOCK`
- `INVALID_INSTALL_BLOCK`
- `INVALID_INSTALL_ENTRY`
- `MISSING_INSTALL_KIND`
- `MISSING_INSTALL_FIELD`
- `UNSUPPORTED_INSTALL_KIND`
- `UNKNOWN_METADATA_KEY`
- `UNKNOWN_REQUIRES_KEY`
- `UNKNOWN_INSTALL_KEY`
- `PRIMARY_ENV_NOT_DECLARED`
- `DUPLICATE_INSTALL_ID`
- `INSTALL_ENTRY_DROPPED`
- `NO_INVOCATION_SURFACE`

## Backward compatibility

This slice is intentionally **audit-first**:

- existing skills continue to load
- malformed contract fields become structured findings instead of silent drops
- CLI JSON surfaces contract state for automation
- human-facing `skills info` / `skills check` gain contract findings sections

## Included content fix

The slice also normalizes the bundled `spotify-player` skill to remove a duplicated install id and replace the ignored `tap` field with the supported fully-qualified brew formula `steipete/tap/spogo`.

## Promotion path

1. Land the audit-first patch.
2. Observe contract findings on real skill inventories.
3. After drift is resolved, optionally tighten selected findings from warning-only to fail-closed enforcement.
