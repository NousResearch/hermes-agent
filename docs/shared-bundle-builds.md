# Shared bundle builds

The build-only Python modules live in `scripts/bundles/`. They use PM for
package installation and dependency resolution. They do not implement a
second package manager.

| Module | Responsibility | Consumers |
|---|---|---|
| `payload.py` | Git snapshots, manifest/facts, JS output placement, relocation and launcher generation | Native desktop and Termux |
| `native.py` | Native tool-store staging and dependency venv assembly | `hermes pm bundle`, desktop builder |
| `desktop.py` | Build the JS surfaces, assemble the payload, invoke Electron packaging | Native release workflow, local builds |
| `stage.py` | Stage a payload and its launchers without creating an Electron package | `npm run payload` |
| `mint_launchers.py` | Mint Windows launchers with the payload interpreter's distlib | Shared launcher assembly |
| `launcher_wrapper.py` | Runtime template for the minted executable | Shared launcher renderer |

Build a desktop bundle from a checkout at its release tag:

```sh
uv run --no-project --python 3.14 python scripts/bundles/desktop.py --tag=vX.Y.Z
```

The builder derives console entrypoints from the archived `pyproject.toml`.
It checks the interpreter, dependency tree, and generated launchers before
publishing their relative paths in the payload manifest. The desktop build
bakes this launch contract into its stamp. Electron uses the declared paths
without payload probes, adoption, or repair. Non-bundled builds carry no
placeholder payload. The stamp also declares the update mechanism. Store
packages use `external`, and sideload bundles use `app-installer`. The runtime
has no separate Store probe or compatibility fallback.
The MSIX hook reads the declared launcher names
rather than carrying a second entrypoint list. POSIX launchers follow
links to the installed payload and call each declared function directly.
Windows launchers are minted by the payload's own Python, preserving native
architecture and relocation behavior.

Termux uses the same snapshot, manifest/fact writer, JS asset placement and
POSIX launcher generator. Its package-manager hooks stay in `scripts/termux/`.
Its bionic wheel compilation and offline installation remain target-specific:
a glibc host cannot execute the shipped interpreter, and Termux has a fixed
installation prefix. Native desktop staging instead resolves on the target OS.
Neither path compiles dependencies on the user's machine.

Electron-specific work stays with Electron: renderer/main-process bundling,
Node native bindings, MSIX metadata, signing, notarization and app packaging.
The after-pack hook invokes the shared Python relocation command instead of
maintaining its own link rewrite algorithm.

Ordinary bundle staging does not scan user plugin trees. Runtime plugin
admission is a separate PM transaction. Build output cleanup is confined to
its payload store; it must not prune the machine-wide downloader partials.

## Target boundaries

Desktop stages on the target OS and architecture. Its dependency step uses
`uv sync --frozen --all-extras --active` on the staged interpreter.
The complete desktop builder, not `pm bundle` alone, adds prebuilt JS surfaces
and invokes Electron packaging.

Termux's host scripts run on Linux ARM64. `build_cpython.sh` and `build_node.sh`
stage pinned bionic packages; `stage_runtime_libs.py` supplies their native
library closure. `termux_build.sh` exports the tag's core plus `acp` requirements,
builds the required native wheels in the pinned bionic container, and records
wheelhouse inputs. `build_deb.sh` assembles the environment offline, creates the
APT package, and validates it in a fresh container without network access.
Those container checks do not substitute for Android device acceptance.

The installed Termux root is `$PREFIX/lib/hermes-agent/`. It contains `app`,
`tools`, `runtime-libs`, `venv`, `bin`, and the PM manifest/facts. Maintainer
hooks manage only the package's CLI symlinks under `$PREFIX/bin` and refuse
foreign launcher conflicts. They do not compile dependencies on install.

`stage_apt_repo.py` owns repository metadata and signatures. Stable and canary
suites are `hermes-stable` and `hermes-canary`; package files publish before
signed metadata. Runtime and repository pins are distinct from the native
build-toolchain packages used only inside CI.

Docker and Nix consume PM pins but do not run this desktop payload assembly.
Docker has its own curated extras and image lifecycle. Nix uses uv2nix and
separate derivations. [Stable release admission](stable-releases.md) coordinates
their acceptance and publication with desktop and Termux packages.

PM and Docker ship full Chromium, without a separate headless shell. The same
executable serves headed and headless sessions. Browser launchers use the
selected PM executable; direct Playwright callers select the `chromium` channel.
Native staging removes retired package facts and directories from its build
cache. It does not remove browser files from the user's machine-wide store.

PM retains completed download archives until the package is verified and
published. Normal installs commit package facts before deleting their archives.
Cross-target staging verifies the published entry before deleting its archives.
Failed or paused installs keep downloads for retry. Cleanup leaves unrelated
archives and resumable partials alone. A later repair may download again.

## Verification boundary

Tests execute shared snapshot/manifest helpers on real git fixtures, stage
real subprocesses and venvs at controlled boundaries, and mint/run Windows
launchers after relocation. POSIX launcher and symlink tests run on POSIX.
A local helper test is not proof of a signed installer, bionic wheel build,
or stable-release upgrade. The release workflows own those native receipts.
