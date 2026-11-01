# Privacy Model

## Sensitive by default

Keep these local:

- raw exports
- normalized rows and snapshots
- post IDs, text, and links
- exact account and post metrics
- source paths and SHA-256 hashes
- lane classifications and strategy findings
- revenue or monetization fields
- private configuration

## Public-safe output

The generated public-safe report is limited to methodology, schema status, date coverage, validation summary, reconciliation status, partial-day method, and privacy notes. It intentionally omits exact analytics and content.

Public-safe means suitable for review as a starting point. It does not grant permission to publish.

## Repository boundary

A public copy may contain:

- generic workflow documentation
- executable parser and analysis logic
- example configuration
- synthetic fixtures
- behavior and regression tests

A public copy must not contain:

- real or sanitized exports
- real source hashes
- account-specific lanes or thresholds
- local user paths
- credentials, cookies, tokens, or browser state
- private reports, snapshots, manifests, or strategy notes

## Side-effect boundary

The importer does not download data, upload data, call external services, write to Git, update notes, or publish results. Those actions require separate tools and separate authorization.
