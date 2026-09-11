# Building the Desktop Installers

Use the complete bundle builder for release artifacts. Ordinary `dist:*`
commands package the current desktop build; they do not stage a fresh runtime.

## Artifact and update ownership

| Target | Artifact | Update owner |
|---|---|---|
| Windows x64 / ARM64, sideload | Signed per-architecture MSIX packages, combined into a universal `.msixbundle`; `.appinstaller` descriptor | Windows App Installer |
| Windows x64 / ARM64, Store | Store-identity MSIX packages and a separate Store bundle | Microsoft Store |
| macOS ARM64 / x64 | Signed, notarized `Hermes.app` in DMG and ZIP artifacts | `electron-updater` with Squirrel.Mac |
| Linux x64 / ARM64 | Local builder can produce AppImage | External replacement; Linux desktop release legs are disabled |

The current MSIX manifest requires Windows 11 22H2 (`10.0.22621.0`).
The source-script Windows support range is separate from this package floor.
Windows desktop packaging uses MSIX, not NSIS or MSI.

`bundled` carries the local runtime. `store` carries the same runtime under
Partner Center's package identity. `light` is a remote-only client without
Python or a local agent. Light has build/feed support but no release matrix leg.

## Payload contents

The bundled payload contains:

- The release's source snapshot without `.git`.
- Pinned CPython and uv, plus a ready Python dependency tree.
- Node.js, npm, supported managed tools, and prebuilt TUI/dashboard assets.
- A manifest, tool facts, the installed feature list, and build provenance.
- Generated CLI launchers from the archived project's script declarations.

`pm/lock.json` owns managed-tool pins. `pyproject.toml` and `uv.lock` own Python
requirements. Native staging uses `--all-extras`, subject to platform markers.
This is broader than the source installer's extra named `all`.

The backend runs from app resources. Launchers execute the store interpreter
with the source and dependency paths; they do not boot through a relocated
venv executable. Startup can create user-state records and CLI links.
Provider calls, model downloads, and optional integration setup can still use
the network.

Git is a platform exception: Windows stages Git for Windows with Bash.
POSIX targets use system Git. A Mac without Command Line Tools can therefore
need them for Git-dependent operations.

User data remains outside the package. Optional additions use writable PM
storage and complete Python environment generations, not writes into the app.
See [Package management](../../website/docs/reference/package-management.md).

## Python runtime contract

The PM bundle builder records interpreter and dependency paths in the payload
manifest. `scripts/write-build-stamp.mjs` copies that runtime contract into the
desktop stamp. The desktop build embeds the stamp in Electron.
`electron/payload-backend.ts` reads `runtime.sitePackages` from this stamp.
It does not select a Python version or search for dependency directories.

Staging does not prove native startup or package replacement. Those checks
use the [bundled-update acceptance suite](../../tests/install/BUNDLED_UPDATES.md).
Acceptance requires a native run with the actual signed release artifacts.

Packaged tool facts describe the final bytes, not only the staged archives.
Windows refreshes the tool digests after sanitization and batch signing.
The macOS custom signer retains the builder's entitlements and file selection.
It refreshes digests after child signatures and before the outer app signature.
Unsigned macOS builds refresh them at the end of `afterPack`.
Do not refresh facts in `afterSign`: that changes resources covered by the signature.

## Complete native build

From a checkout whose `HEAD` equals the release tag, run:

```sh
uv run --no-project --python 3.14 python scripts/bundles/desktop.py --tag=vX.Y.Z
```

Replace `vX.Y.Z` with an actual immutable tag. Stable tags must match the
version in `pyproject.toml`. Canary tags use the release script's tag grammar.
The host must provide Git, native-architecture Node/npm, and official uv 0.12+
with a build triple in `uv --version`. Native dependency builds also need the
platform's compiler and libraries.

The builder:

1. Checks the tag, checkout, Node architecture, and workspace engine constraints.
2. Installs the locked root JS workspace when its install stamp differs.
3. Builds the TUI and dashboard for variants with a payload.
4. Stages the PM payload, places JS assets, relocates links, and generates launchers.
5. Builds Electron and packages MSIX, DMG/ZIP, or AppImage for the current OS.

Use `--variant store` on Windows, or `--variant light` for the remote client.
Arguments after `--` go to Electron Builder. `electron-builder.config.cjs`
is the sole packaging configuration. The wrapper disables automatic publishing;
the release workflow owns uploads and channel promotion.

`pm bundle --out DIR --ref REF` stages the native runtime only.
`scripts/bundles/stage.py` also generates its launchers. Neither command builds
the TUI/dashboard outputs or creates a signed installer by itself.
The launcher stage checks the payload and records its relative launch paths.
The Electron build bakes these paths into its stamp. Desktop startup does not
inspect, create, or repair a PM payload. Non-bundled builds carry no placeholder payload.
See [shared bundle builds](../../docs/shared-bundle-builds.md) for Termux reuse.

## Commit-only builds

To preview a build for a pushed revision, run:

```sh
python scripts/release.py --build-commit REV --remote origin
```

The command fetches the remote and resolves `REV` to a full commit SHA.
It prints the dispatch command without changing local branches, tags, or releases.
Add `--publish` to dispatch that build. This flag does not publish a release
in commit-build mode.

Use repeatable `--bundle-env NAME=VALUE` options to bake non-secret runtime
defaults into the desktop app, for example:

```sh
python scripts/release.py --build-commit REV --remote fork --publish \
  --bundle-env HERMES_GUEST_ONBOARDING=1 \
  --bundle-env HERMES_DATA_DIR_SUFFIX=magic-test \
  --bundle-unset HERMES_HOME
```

These defaults run before Electron initializes its paths and are inherited by
local backend processes. Explicit runtime environment values win, including
empty values. Do not pass secrets: the values are visible in the workflow inputs
and packaged JavaScript. This option affects desktop bundles, not Termux.
The suffix is appended literally; include a leading hyphen if desired.
`--bundle-unset NAME` explicitly clears an inherited value at app launch, even
if the caller supplied it. Internally it sets the value to an empty string, not
an absent key. Clearing `HERMES_HOME` also disables the Windows registry fallback:
older installers saved that variable permanently, which otherwise takes priority
over the test suffix. It does not edit the registry or the existing install.
For full data-path isolation, also clear `HERMES_DESKTOP_USER_DATA_DIR` if that
machine uses an explicit Electron directory. Ordinary defaults still preserve
runtime overrides; `--bundle-env HERMES_HOME=` is a default, not a forced clear.

For local commit builds, `HERMES_BUNDLE_ENV_JSON` accepts a JSON object whose
string values are defaults and whose `null` values are explicit clears. For example,
`{"HERMES_HOME":null,"HERMES_DATA_DIR_SUFFIX":"magic-test"}`. These settings are
not applied to the build runner itself.
Commit archive keys still use the SHA, so use a fresh commit for different
defaults: an existing artifact is never overwritten with different bytes.

The workflow must exist on the repository's default branch. Admission requires
a default-branch `workflow_dispatch` and repository write, maintain, or admin
permission for both the original actor and the actor who reruns it.
It rejects mixed tag, release-phase, channel-publication, and upgrade inputs.

Builder jobs check out the admitted SHA. Their artifacts and completion receipts
go to `releases/commit/FULL_SHA/`, separate from tag archives and update channels.
The run summary lists Windows packages and both universal bundles, macOS DMG/ZIP
files, and the Termux package. Linux release legs remain disabled and are listed
as not built. Only receipt-listed artifacts that exist in storage get download
links. Missing receipts show the failed or incomplete leg.

Each commit build also writes a downloads page to
`releases/commit/<FULL_SHA>/index.html` on the R2 public origin; a tag run
writes its channel page instead (`releases/stable/index.html` or
`releases/canary/index.html`). Pages list only objects the build actually
staged, and a re-run of an older tag never replaces a newer channel page.
`release.py --build-commit` prints the commit page URL before dispatching.

Commit builds require the signing credentials used by their release legs.
Store bundle envelopes remain unsigned for Partner Center, but these builds
never submit them. No GitHub release, updater feed, or APT channel is changed.
An identical upload retry can succeed. Different bytes at an existing commit
object key fail rather than replace that object.

For a local native build, check out the exact SHA and run:

```sh
python scripts/bundles/desktop.py --commit=FULL_SHA --variant=bundled
```

The builder uses that commit's project version. Sideload MSIX versions append
`.0`. Store package versions use the commit timestamp with the existing UTC
calendar policy, even when the app version starts with zero.
Commit-built stamps disable automatic release-channel updates. Local command and transport tests do not
replace signed-package installation and update acceptance on each native host.

## Windows signing and App Installer

The signing jobs provide `AZURE_SIGN_ENDPOINT`, `AZURE_SIGN_ACCOUNT`,
`AZURE_SIGN_PROFILE`, `AZURE_SIGN_PUBLISHER`, and `AZURE_CLIENT_ID`, plus
Azure authentication. Do not pass publisher names through shell-split `-c`
arguments. Local unsigned builds are not release-acceptance artifacts.

The packaging hooks sanitize invalid PE certificate tables, then sign and
timestamp payload EXEs/DLLs. The product EXE receives its signature after
resource edits. `scripts/sign-msix.mjs` signs the sideload package envelope.
Store package envelopes remain unsigned for Partner Center to sign.

Unchanged payload files reuse verified signatures from
`${ELECTRON_BUILDER_CACHE}-payload-signatures`. The key binds input bytes,
Azure policy, signing tools, and timestamp policy. Filenames and release
versions do not determine identity. A hit must match executable content and
pass Authenticode publisher/timestamp checks. Invalid entries become misses.
The product EXE and package envelopes still receive fresh signatures.

The `before-build.mjs` hook generates MSIX extensions from payload launcher
names. One execution-alias extension carries the CLI aliases. The manifest
also registers the Copilot hardware-key provider. Its minimum Windows version
is part of the package contract.

The release job combines only matching sideload packages from both architectures.
Store packages never enter the sideload bundle. It publishes bundle bytes before
the channel descriptor at `releases/win32/CHANNEL/CHANNEL.appinstaller`.
Light uses `releases/win32/light/CHANNEL/`.

Windows records the descriptor source at installation. The app checks that
registered source through WinRT and distinguishes an unknown result from no
update. Apply downloads the descriptor before teardown, then opens the local
`.appinstaller` file. It does not require the disabled `ms-appinstaller:` protocol.
The app registers a detached relaunch waiter before handoff.

The build stamp declares `updateMechanism`: `app-installer` for sideload bundles,
`microsoft-store` for Store builds, `electron-updater` for macOS packages, and
`self` for source-built apps. Settings shows Microsoft Store for a Store build.
The runtime does not infer Store ownership from Electron flags or carry a
second Store boolean. Windows Light declares `external` because it has no
bundled Python checker. Its OS-registered App Installer source still owns
automatic updates.

Store builds use `Windows.Services.Store.StoreContext` to check, download,
and request installation inside Hermes. The native consent UI attaches to the
current desktop window. Download finishes before backend shutdown; the existing
relaunch waiter is registered before the install request. Unknown checks,
cancellation and request failures do not count as successful updates. Native
acceptance requires a Store-acquired package or flight, not a sideloaded MSIX.

On Windows bundles, PM copies the verified pinned base Python to its writable
store for uv builds. The app and execution aliases retain their signed bundled
launchers and bundled Python, which load the selected dependency generation.
The new venv's generated console scripts must not replace those launchers:
out-of-package Python cannot launch the packaged tools.

Sideload stable versions are `X.Y.Z.0`. Canary revisions derive from elapsed
minutes after the stable baseline. The release script rejects ambiguous or
overflowing cuts. Store versions use `year.hour-of-year.second-of-hour.0` in UTC,
with the fourth component reserved for Microsoft. App semver and package
version are different facts. `scripts/msix-shared.mjs` owns these derivations.

## macOS signing and updates

`CSC_LINK` and `CSC_KEY_PASSWORD` supply the Developer ID identity.
The notarization hook accepts `APPLE_API_KEY`, `APPLE_API_KEY_ID`, and
`APPLE_API_ISSUER`, or a keychain profile. CI materializes the API key from
`APPLE_API_KEY_P8`.

The app and nested Mach-O binaries, including Chromium, must be signed before
notarization. The existing after-sign hook owns submission and stapling.
Publication checks signatures, the stapled ticket, and Gatekeeper assessment.
Unsigned local builds do not satisfy these gates.

Both DMG and ZIP artifacts are required by the release pipeline. The ZIP is
the update artifact, not an optional duplicate of the download DMG.
See [macOS bundle updates](../../docs/macos-bundle-updates.md) for feed validation
and native-event ordering.

### DMG detach diagnostics

The builder wrapper reports `[dmg-detach]` snapshots when dmgbuild cannot detach
its staging image. It runs `lsof` before the supplier retries or performs forced
cleanup. The snapshot identifies the mounted filesystem, backing image and
devices, then lists process names, PIDs, parent PIDs, users, descriptors and paths.

The queries use noninteractive `sudo` when available. Permission failures and
query timeouts are reported explicitly. Empty output does not prove that the
image has no holder. The shim does not stop processes or change detach results,
retry settings, signing or notarization. Explicit `CUSTOM_DMGBUILD_PATH`
overrides bypass the shim because their interpreter layout is not known.

## Development, assets, and verification

Prepare and activate the [PM developer environment](../../website/docs/reference/package-management.md#developer-workflow)
first. Use Bash `source ./activate` or PowerShell `. .\activate.ps1`, and keep a
separate development home. Activation supplies the toolchain, not `node_modules`.

From the repository root:

```sh
npm ci
npm run dev --workspace apps/desktop
```

For an ordinary package of that desktop build, use the workspace's
`dist:win`, `dist:mac`, `dist:linux`, or `pack` command. Those commands do not
replace the complete tagged build described above.

Icons are generated from `assets/nous-girl-*.svg` and `assets/backgrounds/`.
`node scripts/generate-icons.mjs` uses the locked, isolated `icon-build` group.
Generated PNG/ICO/ICNS files are not source assets. Build-only renderers do not
belong in production payload dependencies.

[Stable release admission](../../docs/stable-releases.md) requires the full
pipeline, not just successful packaging. Native signed-package update tests
live in the [existing install/update family](../../tests/install/BUNDLED_UPDATES.md).
A helper test or unpacked-app smoke is not proof of native install, update,
or automatic relaunch. Historical receipts and current unresolved gates are
separate in [PM audit status](../../docs/pm-audit-status.md).
