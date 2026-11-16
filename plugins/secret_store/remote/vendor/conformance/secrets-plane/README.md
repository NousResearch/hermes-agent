# Secrets plane conformance fixtures

The source of truth for the `/v1/secrets/*` behaviour both sides run: this
repo (`test/secretsPlane/conformance.test.ts`, against the real router) and
hermes-agent (vendored verbatim with a `SOURCE.json`, against its `remote`
secret store client). `conformance/secrets-plane-fixtures.sha256` pins every
file (regenerate with `npm run conformance:secrets:manifest`); the reserved
name list itself is `conformance/secrets-reserved-names.json`, generated from
hermes-agent by `scripts/secrets-reserved-names.py`.

## Files

- `resolution/*.json` — `{ agent: { owner }, levels, expected }`. `levels` holds
  `tenant`, `owners` (keyed by owner subject: `principal:<sub>`,
  `group:<id>`, `org:<id>`), `instance` and `profile`, each `{ env?: {NAME:
  value}, auth?: {provider: record} }`. `expected` is the agent's resolved view
  (`env`/`auth` of the `GET /v1/secrets/self` body without `version` fields).
- `reserved-names.json` — `{ kind, name, expect: "reserved" | "allowed" }`. A
  reserved name is refused `400 secret_name_reserved` on every write route.
- `cas/*.json`, `wire/*.json` — scenarios: `agents` to register (in tenant
  `org_a`, credential subject `agent1-sub`), then `steps` run in order against
  a fresh plane.

## Step actors (`as`)

| `as` | Credential |
|---|---|
| `agent` | the first agent's identity bearer + `X-Hermes-Instance-Id` |
| `nas:<instanceId>` | a `connector-secrets-system` token for that instance (`sub: admin`) |
| `person:<sub>` | an identity bearer for a tenant member |
| `admin` | an identity bearer for a tenant admin (`sub: admin`) |
| `none` | no credential |

## Placeholders in `expect`

| Value | Matches |
|---|---|
| `"<any>"` | any string |
| `"<etag>"` | any non-empty string (the opaque view version) |
| `"<time>"` | an ISO-8601 timestamp |
| header `"<body.version>"` | the header equals the body's `version` |
| step `ifNoneMatch: "<previous.etag>"` | send the previous step's `ETag` |

Every other value matches exactly (objects: same keys, recursively). An
`expect` without `body` checks the status only.
