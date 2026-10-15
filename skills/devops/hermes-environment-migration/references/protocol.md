# Protocol

### 1. Discover both environments

Inventory platform, Hermes home, profiles, versions, providers, databases, skills, cron, gateways, plugins, and external dependencies.

### 2. Classify data

Mark each item copy, compare, merge, rewrite, secret, machine-bound, cache, optional, or quarantine.

### 3. Back up target

Create a dated target backup and verify that it can be read before importing anything.

### 4. Create export

Build a non-secret staged archive with a complete file manifest, sizes, modes where relevant, and SHA-256 hashes.

### 5. Verify archive

Inspect the archive before extraction, extract only to staging, verify every hash, and reject unexpected content.

### 6. Plan import

Produce a file-by-file action plan covering conflicts, path rewrites, schema compatibility, and rollback.

### 7. Import in phases

Apply identity, skills, scripts, memory, config, cron, gateway, vault metadata, and development work with verification after each phase.

### 8. Transfer secrets separately

Use an approved secure channel and reauthenticate machine-bound credentials whenever possible.

### 9. Cut over and verify

Prevent duplicate gateways, run Hermes health checks, test critical workflows, and retain rollback until acceptance.

## Evidence discipline

Record the source, timestamp when relevant, scope, access limitations, and any contradiction for each load-bearing finding.
