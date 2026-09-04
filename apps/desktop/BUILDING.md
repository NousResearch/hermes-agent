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
- An install stamp (`install-stamp.json`) recording the tag, the commit, the
  distribution, the update mechanism, and where the managed runtime dir
  lives.

The app runs the backend directly from its own resources. This is the
"bundled" install axis, built by `pm bundle` against the `pm/lock.json` pin
table — every managed tool (uv, python, node, npm, ripgrep, git, chromium,
cua-driver) is digest-pinned and staged at build time.

## The installer for each platform

| Platform | Artifact | Notes |
|---|---|---|
| Windows | MSIX `.msix` / `.msixbundle` | The shipping artifact. Signed with Azure Trusted Signing. Out-of-store installs update via the OS App Installer (.appinstaller source; the app checks + prompts, the OS applies). Store-submission builds (HERMES_DESKTOP_VARIANT=store) use the Partner Center identity and update via the Store. |
| macOS | `.dmg` | Signed and notarized when the `APPLE_*` / `CSC_*` secrets are set. Updated via electron-updater against the `latest-mac.yml` feed. |
| Linux | unpacked / AppImage | Unsigned. |

NSIS is intentionally dead (D1 decision): Windows ships MSIX only.

## How the build works

One script drives the whole build:

```
node scripts/build-bundled-desktop.mjs --tag=vX.Y.Z
```

The script always runs every step:

1. **Gate the toolchain.** The host `node` and `npm` must satisfy
   `package.json` engines, and the toolchain pins resolve from
   `pm/lock.json` (the "Resolve toolchain pins" CI step). The payload embeds
   these exact pinned versions, so gate == embed.
2. **Build the JS surfaces.** ui-tui (with hermes-ink) and the dashboard SPA.
3. **Build the desktop app.** `npm run build` in `apps/desktop`: vite,
   electron-main bundle, native deps, then payload staging.
4. **Stage the agent payload** (`pm bundle`). This snapshots the repo at the
   tag with `git archive`, copies the prebuilt JS surfaces in, installs the
   pinned CPython and `site-packages`, stages the pinned managed tools, and
   writes `manifest.json` plus the install stamp. Each staged binary must
   prove the target architecture in its own version banner. A wrong-
   architecture binary fails the build.
5. **Package with electron-builder.** MSIX on Windows, DMG on macOS.

Payload staging stays dormant unless `HERMES_DESKTOP_VARIANT=bundled` is
set. The build script sets it. A normal `npm run dev` or `npm run pack`
without the script does not stage payloads.

## Code signing (Windows)

Signing turns on when the `AZURE_SIGN_*` environment variables are set:

```
AZURE_SIGN_ENDPOINT     https://cus.codesigning.azure.net
AZURE_SIGN_ACCOUNT      codesign2
AZURE_SIGN_PROFILE      hermesagent
AZURE_SIGN_PUBLISHER    CN=Nous Research Inc., ...
AZURE_CLIENT_ID         (the OIDC app id)
```

`electron-builder.config.cjs` reads these variables and composes the
`win.sign` configuration itself. Do not pass the values as `-c` arguments:
the publisher name contains spaces, and spaces do not survive the cmd.exe
argument hop. Signing runs through `scripts/sign-msix.mjs`, a plain hook
that signs ONLY the .msix package: Windows validates the package signature
(AppxSignature.p7x over AppxBlockMap.xml), and inner binaries are covered
by the block-map hashes — per-file Authenticode is neither required nor
validated.

On win32, `scripts/after-pack.mjs` runs `sanitize-pe-signatures.mjs` before
the MSIX pack: python-build-standalone's `llvm-strip` can leave dangling PE
certificate tables, and one dangling table fails the whole package with
0x800700C1. This is a hard-fail step, not best-effort.

## The MSIX manifest

`assets/msix-manifest.xml` is the stock app-builder-lib template plus the
`uap5`/`desktop4` namespaces needed by the CLI execution aliases, and the
`uap3` fragment (declared on its own root) that registers the app as a
Windows Copilot hardware key provider. The extensions file
(`build/msix-extensions.xml`) is generated at config-require time by
`writeMsixExtensions()` in `electron-builder.config.cjs`. Keep the manifest
in sync with app-builder-lib when electron-builder bumps.

## Code signing (macOS)

The macOS build signs and notarizes with electron-builder's builtin
notarization when the `APPLE_ID` / `APPLE_APP_SPECIFIC_PASSWORD` /
`APPLE_TEAM_ID` secrets are present. The `sign-nested-chromium` path signs
the payload's nested Chromium Mach-O binaries (see `sign-nested-chromium.mjs`,
wired from `after-pack.mjs`); the outer app bundle is signed by the
electron-builder signing pass.

## Local build

```
cd apps/desktop
npm install
npm run dist:win:msix   # or dist:mac / dist:linux
```

For a fast dev loop without payload staging:

```
npm run dev             # vite + electron against a dev backend
```
