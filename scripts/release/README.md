# Hermes Release Bundle Pipeline

Decisions pinned for the updater rework (phase 0, task 0.0).
See `docs/updater-world.md` §2.1, §2.6 and `docs/plans/updater-rework/01-phase0-bundles.md`.

## Decisions

| Decision | Value | Rationale |
|---|---|---|
| Signing scheme | Ed25519 via PyNaCl | Pure-Python signing — no external minisign CLI to install per-platform. Uses PyNaCl (libsodium) for Ed25519 sign/verify; the Rust updater verifies with ed25519-dalek. Signature is a JSON `.sig` file with base64-encoded signature + pubkey. |
| Channels | `nightly` (daily cron) + `stable` (manually promoted tag) | Daily nightly replaces per-commit tracking (~100 commits/day makes SHA-tracking useless for users). Stable is manually promoted from a passing nightly. |
| Platform matrix (v1) | `linux-x64`, `linux-arm64`, `darwin-arm64`, `win-x64` | `darwin-x64` deferred unless CI capacity allows. linux-arm64 via `ubuntu-24.04-arm` runner. |
| Bundle versioning | calver `YYYY.MM.DD[.N]` for nightlies, semver for stable | Nightlies get a date stamp; `.N` suffix if multiple builds land same day. Stable tags are `v<semver>`. |

## Bundle layout contract

The layout produced by `scripts/release/build-bundle.sh` (task 0.3):

```
hermes-<version>-<platform>.tar.zst
├── manifest.json          # integrity + compat metadata (task 0.4)
├── runtime/
│   ├── python/            # uv-managed CPython (relocatable)
│   ├── venv/              # fully resolved site-packages from uv.lock (non-editable)
│   ├── node/              # Node LTS runtime
│   └── tools/             # bundled native CLIs (ripgrep) — launcher prepends to PATH
├── app/                   # hermes-agent source (git archive, no .git), .pyc precompiled
├── ui/
│   ├── tui/dist/          # pre-built Ink bundle
│   └── web/dist/          # pre-built dashboard SPA
├── desktop/               # pre-built electron app (optional — manifest flags absence)
└── bin/hermes             # launcher shim (phase 0: placeholder; phase 1: native binary)
```

## Updater ↔ bundle contract

Paths frozen at the same tier as `manifest.json` and `min_updater_version`:

- `manifest.json` at the bundle root — must exist in every future bundle.
- `bin/hermes` (launcher + updater binary) — must exist at this path in every
  bundle an old staged updater could be asked to hop into. Moving either
  requires a `min_updater_version` bump plus a compat copy at the old path for
  one contract window.

See `docs/updater-world.md` §2.3.1 for the bootstrap-hop mechanism.
