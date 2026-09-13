# macOS bundle updates

Tagged macOS bundles use `electron-updater`. The bundled and Light stamps
name that owner. Stable and canary are separate application identities; each
updates within its baked channel. Development and bootstrap installs keep
checkout updates.

One-off builds carry `source: commit-build` and have no updater. Their package
identity includes the short commit SHA, so different commit builds can coexist.
Checks and apply requests explain that the developer must provide a new build.
This applies to the desktop and its bundled CLI, not to an unrelated remote
backend the desktop connects to.

## Feed contract

`apps/desktop/update-feed.json` defines each channel directory and filename.
The desktop CJS adapter and Python publisher read these same facts.
The builder writes that URL into `app-update.yml`. The client uses this file
unless `updates.desktop_feed_base_url` supplies an explicit bucket-base URL.

- Stable: `releases/darwin/stable/stable-mac.yml`
- Canary: `releases/darwin/canary/canary-mac.yml`
- Light: the same paths with `light/` between `darwin/` and the channel.
- Artifacts: `releases/tag/TAG/FILENAME`, shared by download links and feeds.

The current workflow builds the bundled variant, on ARM64 and Intel runners.
Light has separate client/feed routing but no release matrix leg in this change.

`python -m scripts.releases.r2 finalize` requires one metadata file for each architecture,
named `arm64-CHANNEL-mac.yml` and `x64-CHANNEL-mac.yml`. It rejects wrong
versions, variants, architectures, hashes and inconsistent legacy path fields.
Each referenced ZIP/DMG is streamed back and checked against its SHA-512 and
size. Publication checks the live version, conditionally replaces its ETag,
and reads back the resulting feed. Same-tag macOS artifacts cannot be overwritten
with different bytes. Mutable feeds use `Cache-Control: no-store`.
Canary retention protects the artifacts and blockmaps referenced by live feeds.
An unreadable feed prevents pruning.

## Client lifecycle

Checks never download automatically. Apply rechecks the release, downloads it,
and waits for Squirrel.Mac to accept the signed app. Only then does Hermes stop
its app-owned backends and request installation/relaunch. Unrelated quits do
not trigger installation. Downloads and native-verification failures leave
backends running. Concurrent checks cannot replace an apply operation's target.
The existing checkout updater never mutates the sealed app bundle.

## Release environment

The existing `release-signing` environment supplies:

- `CSC_LINK` and `CSC_KEY_PASSWORD`: Developer ID Application signing identity.
- `APPLE_API_KEY_P8`, `APPLE_API_KEY_ID`, `APPLE_API_ISSUER`: notarization.
- `CLOUDFLARE_R2_ACCOUNT_ID`, `CLOUDFLARE_R2_ACCESS_KEY_ID`,
  `CLOUDFLARE_R2_SECRET_ACCESS_KEY`: bucket access secrets.
- `CLOUDFLARE_R2_BUCKET`, `CLOUDFLARE_R2_PUBLIC_URL`: repository/environment vars.

Publishing requires the Apple credentials. The existing after-sign hook owns
notarization, so electron-builder's second notarization path is disabled.
The publish gate verifies the signature, stapled ticket and Gatekeeper assessment.
The Darwin publish job waits for both native builds and serializes channel writes.

## Verification limits

Helper tests exercise the strategy, native-event ordering, feed validation,
conditional publication, and retention. They are not proof of a signed install
or actual app replacement.

Native macOS packaged-update drivers are part of the existing
[install/update family](../tests/install/BUNDLED_UPDATES.md). The stable gate
requires signed-package transitions on both architectures. Each acceptance
claim needs a successful native run for the exact old/new package pair.
Workflow definitions and historical helper results do not establish acceptance
of the current head. See [PM audit status](pm-audit-status.md) for scoped receipts.
