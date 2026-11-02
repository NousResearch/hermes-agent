# Bundled application update acceptance

The `packaged-app -> open-app-update` arms are Stable Release's signed-package upgrade arms (`windows-packaged`, `macos-packaged-arm64`, `macos-packaged-x64`). They call the per-OS run workflows (`install-e2e-windows-run.yml`, `install-e2e-macos-run.yml`) directly with the transition manifest their `transitions-*` job pinned (`bundle-manifest-url`, `bundle-manifest-sha256`). They are not crossed with source-checkout update methods, and the Install & Update E2E matrix (`install-e2e.yml`) runs source legs only.

## Input contract

Each JSON manifest has `schema: 1`, `platform` (`windows` or `macos`), `arch` (`x64` or `arm64`), and `old`/`new` objects. Each object contains:

- `tag`: exact release tag.
- `version`: actual package version (Windows numeric quad; macOS app semver).
- `commit`: full commit SHA from the artifact's install stamp.
- `identity`: exact MSIX Identity Name or macOS CFBundleIdentifier.
- `artifact`: HTTPS `url` and lowercase `sha256` of the actual release package.
- Windows additionally requires `publisher` and `applicationId`.
- macOS additionally requires `teamId`, the signing TeamIdentifier.

Windows artifacts are signed universal `.msixbundle` files. macOS artifacts are signed application `.zip` files. Do not use bootstrap Setup.exe/DMG artifacts, development builds, repackaged placeholders, or unsigned stand-ins. Both artifacts must already exist. The candidate commit must equal the workflow checkout SHA. Identities and signing ownership must match; versions and commits must change. The macOS transition stays on one update channel.

`bundle-inputs.mjs` validates the manifest and streams both downloads with SHA-256 verification. Local paths are added only after verification; mismatched partial downloads are removed. The test's temporary feed serves these real bytes through the normal production update protocol. No public channel is modified.

## Proof boundaries

The native drivers run only on disposable GitHub Actions hosts. They verify old installation identity/provenance, click the actual in-app Update control, and observe native package replacement plus automatic relaunch. The driver must not launch the new app or manually start the relaunch waiter as the pass signal. New process identity, payload provenance, version/commit, backend health, and preserved user/plugin state must all agree. Recordings and logs use the existing per-leg artifact/report conventions.

Unit tests of feed/manifest helpers use transport fixtures only. Passing them does not prove native signing, deployment, update, or relaunch. A green merge/typecheck does not establish those properties either. Each release acceptance claim needs a real native run and its parsed receipts.
