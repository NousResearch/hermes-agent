# Building the Desktop Installers

This document tells you how the bundled desktop installers are built, and how
to build and test one on your machine. For the app architecture, read
`AGENTS.md` in this directory.

## What a bundle is

A bundled installer contains the full Hermes runtime. The user installs one
file and gets everything. Nothing downloads at first launch.

The installer contains:

- The Electron app (the chat surface).
- The agent source tree at the release tag, without `.git`.
- `uv` and a CPython interpreter for the target architecture.
- A ready `site-packages` tree, built from the lockfile.
- A Node runtime and the prebuilt JS surfaces (ui-tui, dashboard SPA).
- A build stamp (`.hermes_build_info.json`) that records the tag, the commit,
  and the distribution (`desktop-app`).

The app runs the backend directly from its own resources. This is the
"embedded" install axis. See the plan in
`.hermes/plans/202607_resources-resident-bundled-runtime.md`.

## The installer for each platform

| Platform | Artifact | Notes |
|---|---|---|
| Windows | NSIS `.exe` | One-click, per-user, no install screens. Signed with Azure Trusted Signing. |
| Windows | `.msi` | For fleet deployment through IT tools. |
| macOS | `.dmg` | Signed and notarized when the `APPLE_*` and `CSC_*` secrets are set. |
| Linux | unpacked / AppImage | Unsigned. |

Each artifact ships with a blockmap and a `latest*.yml` file. electron-updater
uses these files for differential updates.

## How the build works

One script drives the whole build:

```
node scripts/build-bundled-desktop.mjs --tag=vX.Y.Z
```

The script always runs every step:

1. **Gate the toolchain.** The host `node` and `npm` must satisfy
   `package.json` engines. `uv --version` must print a build triple. The
   payload embeds these exact host versions, so gate == embed.
2. **Build the JS surfaces.** ui-tui (with hermes-ink) and the dashboard SPA.
3. **Build the desktop app.** `npm run build` in `apps/desktop`: vite,
   electron-main bundle, native deps, then payload staging.
4. **Stage the agent payload** (`scripts/stage-agent-payloads.mjs`). This step
   snapshots the repo at the tag with `git archive`, copies the prebuilt JS
   surfaces in, installs CPython and `site-packages` with `uv`, downloads a
   Node dist, and writes `manifest.json` plus the build stamp. Each staged
   binary must prove the target architecture in its own version banner. A
   wrong-architecture binary fails the build.
5. **Package with electron-builder.** NSIS on Windows, DMG on macOS.

Payload staging stays dormant unless `HERMES_DESKTOP_BUNDLED=1` is set. The
build script sets it. A normal `npm run dev` or `npm run pack` without the
script does not stage payloads.

## Code signing (Windows)

Signing turns on when the `AZURE_SIGN_*` environment variables are set:

```
AZURE_SIGN_ENDPOINT     https://cus.codesigning.azure.net
AZURE_SIGN_ACCOUNT      codesign2
AZURE_SIGN_PROFILE      hermesagent
AZURE_SIGN_PUBLISHER    CN=Nous Research Inc., ...
AZURE_CLIENT_ID         (the OIDC app id)
```

`scripts/run-electron-builder.mjs` reads these variables and composes the
`win.azureSignOptions` configuration itself. Do not pass the values as `-c`
arguments: the publisher name contains spaces, and spaces do not survive the
cmd.exe hops between npm and the builder on Windows. Without the variables,
the build produces unsigned artifacts. Forks and local builds work unsigned.

Authentication uses the Azure credential chain: OIDC federated login in CI,
or an `az login` session on a dev machine. There is no signing secret.

## Where builds run

- **CI:** `.github/workflows/desktop-bundled-release.yml`. A push of a
  `vX.Y.Z` tag builds all targets on a per-OS runner matrix. The signing
  variables live in the `release-signing` environment. Its deployment policy
  admits only `main` and `v*` tags.
- **Local:** any machine with `git`, `npm`, `tar`, and an official `uv`
  0.12+. Wheels resolve natively per host, so build on the architecture you
  target.

## Build and test locally

To build a full bundle:

```
# Linux (from the repo root; this worktree needs the devshell)
nix develop -c node scripts/build-bundled-desktop.mjs --tag=v0.20.0

# macOS (do not use nix develop on a Mac — it compiles for hours)
nix shell nixpkgs#nodejs_22 nixpkgs#uv --command \
  node scripts/build-bundled-desktop.mjs --tag=v0.20.0
```

The tag must point at a commit in the local repo, because staging runs
`git archive` against it. After a force-push, run `git tag -f v0.20.0` first.

Artifacts land in `apps/desktop/release/`.

To check the payload of a built artifact:

```
RES=<unpacked-app>/resources/agent-payload
cat $RES/manifest.json          # schemaVersion, tag, commit
$RES/python/cpython-*/bin/python3 -c 'import hermes_cli'
```

For app development without payloads, use the normal fast paths:

```
npm run dev     # dev server + electron
npm run pack    # unpacked, unsigned build in release/<platform>-unpacked
npm run check   # lint + tests + pack
```

## Known machine setup (Windows)

A Windows build machine needs:

- Official `uv` 0.12+ and Node on `PATH` for the target architecture.
- A .NET SDK on `PATH`. The TrustedSigning PowerShell module installs its
  `sign` CLI with it. CAUTION: Do not set `DOTNET_ROOT` to an arm64 SDK. The
  Azure signing dlib runs inside x64 `signtool.exe`, and that combination
  fails with exit code 3.
- PowerShell execution policy `RemoteSigned` for the current user.
- For source-built wheels on arm64: MSVC arm64 build tools and a static
  OpenSSL (`OPENSSL_DIR`, `OPENSSL_STATIC=1`).
