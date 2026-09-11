# Shared bundle builds

Hermes separates dependency preparation, product builds, and distribution
packaging. The compiler and agent-assembly interfaces live in
[`scripts/build/README.md`](../scripts/build/README.md). These are current
interfaces, not proof that every distribution passed native acceptance.

## Providers, products, and distributions

| Layer | Responsibility | Implementation |
|---|---|---|
| Dependency providers | Prepare tools, Python environments, JavaScript dependencies, and native bindings | PM/uv, `scripts/build/node-deps.mjs`, `scripts/build/python_env.py`, Nix, and Termux |
| Product builders | Compile icons/TUI/web/desktop UI or assemble a runnable agent from prepared inputs | `scripts/generate_icons.py` and `scripts/build/` |
| Distribution adapters | Select products and package them for their target | `scripts/bundles/`, `Dockerfile`, `nix/`, and `scripts/termux/` |

Providers retain their package managers and locks. Product builders do not
install missing dependencies or download inputs. The Node dependency provider
uses one locked workspace union. Nix supplies its own npm and Python inputs.
Termux supplies bionic tools and wheels. Shared recipes do not imply identical
bytes, dependency selections, or Python environments across targets.

| Product | Implementation | Inputs |
|---|---|---|
| Icons | `scripts/generate_icons.py` | Artwork and prepared `icon-build` environment |
| TUI | `scripts/build/tui.mjs` | Prepared TUI workspace |
| Dashboard | `scripts/build/web.mjs` | Prepared web workspace and generated icons |
| Desktop UI | `scripts/build/desktop.mjs` | Prepared desktop workspace, icons, stamp, and native bindings |
| Runnable agent | `scripts/build/agent.py` | Code, interpreter, dependencies, independent PM runtime, resources, and selected frontends |

The desktop UI compiler does not depend on the dashboard. A bundled desktop
selects an agent with TUI and dashboard products. A light desktop does not
select an agent payload.

## Build and packaging entrypoints

Build a desktop distribution from a checkout at its release tag:

```sh
uv run --no-project --python 3.14 python scripts/bundles/desktop.py --tag=vX.Y.Z
```

`desktop.py` requires exactly one of `--tag` or `--commit`. Commit builds
require the full commit SHA and matching checkout HEAD. `--variant` accepts
`bundled`, `store`, or `light`, with `bundled` as the default. `--repo` selects
the checkout. Arguments after `--` go to Electron Builder.

The driver prepares Node dependencies, builds the selected products, and runs
Electron packaging. The desktop npm build prepares its stamp and native tree
before the shared desktop compiler. Electron-specific dependency compilation,
MSIX metadata, signing, notarization, and package formats remain adapter work.

Stage a native agent with both frontend products, without Electron packaging:

```sh
python -m scripts.bundles.stage --out /work/agent-payload --ref HEAD
```

This command prepares and builds TUI/web products if none are supplied. It
also accepts `--tui` and `--web` together as prepared product roots. It rejects
a selection with only one of those arguments. The standalone products have
explicit input/output contracts in the builder README.

`--ref` selects the agent source snapshot. Automatic frontend compilation uses
a temporary snapshot of that same revision. Explicit frontend products must
come from the selected revision.

`hermes pm bundle --out /work/agent-payload --ref HEAD` stages native tools,
the application environment, and the agent. That PM command does not build
frontends. `npm run payload --workspace apps/desktop` uses the stage driver
that includes them. Neither command creates an Electron installer.

For Termux, tools and the wheelhouse are prerequisites:

```sh
python scripts/termux/build.py --repo /work/source --payload /work/termux-payload \
  --out /work/packages --tag vX.Y.Z
```

The Termux driver accepts exactly one of `--tag` or `--commit`. It prepares
the TUI workspace, calls the shared TUI builder, and passes that product to
`build_deb.sh`. It does not add the dashboard or replace bionic preparation.

The CLI contracts are in `scripts/bundles/desktop.py:129–143`,
`scripts/bundles/stage.py:17–46`, `pm/cli.py:442–444,488–491`, and
`scripts/termux/build.py:21–62`.

## Shared agent and launcher contract

`AgentInputs` supplies explicit paths and a placement mode:

- `contained` keeps runtime inputs inside the native payload.
- `fixed` retains the Docker or Termux installation prefix.
- `references` keeps installed code and dependencies in separate Nix store paths.

The assembler derives console commands from `[project.scripts]`. It copies
source-layout resources and selected frontends, or links explicit reference
inputs. It writes the completion manifest after structural assembly. Nix also
receives a command/environment map for its native wrappers.

The launcher implementation lives in `scripts/build/launchers.py`.
`launcher_wrapper.py` and `mint_launchers.py` live in that same directory.
`scripts/bundles/payload.py` owns git snapshots, PM tool facts, PM runtime
sealing, and portable link handling. It no longer owns frontend placement or
launcher generation.

The desktop stamp carries the declared launch paths. Electron consumes that
contract without payload adoption or repair. Non-bundled builds carry no
placeholder agent payload. Store packages declare `microsoft-store` as their
update mechanism. Sideload bundles declare `app-installer`.

Windows launcher minting runs the payload interpreter. POSIX launchers use
explicit interpreter, source, and dependency paths. Structural assembly does
not replace target-native launch tests. A manifest alone does not prove that
a relocated or signed artifact runs.

## Independent PM runtime

PM dependencies come from `pm/pyproject.toml` and `pm/uv.lock`, not the
application dependency graph. The application environment and `pm-runtime`
are separate inputs to agent assembly.

Native and Docker preparation use `pm.runtime_stage.stage_runtime`. Termux
uses that function with its offline wheelhouse. Nix uses the independent
`nix/pm-runtime.nix` derivation. Native sealing records the payload interpreter
and PM-only site directory in `pm-runtime.json`.

Resident PM workers run the declared interpreter with `-I -S -B`. They add
only the declared PM dependency directory before the worker script. They do
not borrow application dependencies or add PM packages to the caller's
interpreter. Docker and Nix stamps point to their declared PM runtime.
These contracts live in `pm/runtime.py:55–88,171–182` and
`scripts/bundles/payload.py:58–90`.

Ordinary native staging does not scan user plugin trees. It runs provisioning
in a child with temporary home and PM roots plus an explicit persistent build
cache. Windows ARM64 prerequisite preparation runs before HOME isolation and
uses the same provider as source setup and CI. Live PM retains plugin
admission, generation selection, and transaction state. Both paths use
`pm.environment.PythonEnvironment` for explicit uv environment construction.

## Distribution boundaries and output paths

| Distribution | Selected products | Current output layout |
|---|---|---|
| Desktop bundled/Store | Icons, TUI, web, agent, desktop UI | Products: `apps/desktop/build/products/`. Agent: `apps/desktop/build/agent-payload/`. UI: `apps/desktop/dist/`. Packages: `apps/desktop/release/` |
| Desktop light | Icons and desktop UI | `apps/desktop/dist/` and `apps/desktop/release/`, without an agent payload |
| Docker | Icons, TUI, web, agent | Agent at `/opt/hermes`, dependencies at `.venv`, PM at `pm-runtime`, generated commands at `libexec` |
| Nix TUI | TUI | `$out/lib/hermes-tui/{dist/entry.js,package.json}` |
| Nix web | Web | `$out/index.html` and assets |
| Nix agent | Agent with TUI/web references | `$out/bin`, `$out/share/hermes-agent`, `$out/ui-tui`, `manifest.json`, and `command-map.json` |
| Nix desktop | Icons and desktop UI, with the Nix agent | `$out/share/hermes-desktop` and `$out/bin/hermes-desktop` |
| Termux | TUI and agent | `OUT/hermes-agent_<version>_aarch64.deb`, installed at `$PREFIX/lib/hermes-agent/` |

Native payloads contain `hermes-agent`, `tools`, `venv`, `pm-runtime`, `bin`,
`uv-cache`, and their manifests/facts. Copied frontend assets live at
`hermes-agent/hermes_cli/tui_dist/` and `hermes-agent/hermes_cli/web_dist/`.
The TUI asset directory contains `entry.js` and module-mode package metadata.

Docker keeps its existing TUI product at `/opt/hermes/ui-tui` and web output
at `/opt/hermes/hermes_cli/web_dist`. The assembler also plants its shared TUI
asset layout beneath `hermes_cli/tui_dist`. Venv command symlinks preserve the
paths used by s6 and the privilege-drop shim.

The Docker frontend stage owns npm dependencies and compilation. The runtime
stage copies only the frontend products and the locked TypeScript package for
runtime linting. Node/npm and Photon's separately installed sidecar dependencies
remain runtime inputs. Python dependency preparation precedes the application
source copy. Actual cache-hit and layer-size claims require build evidence.

Nix retains `importNpmLock`, uv2nix, platform overrides, and store references.
Its builders run during derivation builds, not evaluation. Per-product source
filters keep frontend, Python, and resource inputs separate. Nix wrappers
consume the assembler's command map and retain their PATH and extra-Python
collision policies.

**Nix wheel policy:** `nix/python.nix:128–135` sets `HERMES_NIX_BUILD=1` only
for the Hermes derivation. `setup.py:34–72` rejects general Hermes wheel/sdist
builds. The shared assembler uses the installed Nix code without another copy.
Other providers use source-layout code and metadata, not a public Hermes wheel.

Termux retains bionic wheel compilation, offline installation, native library
paths, and its fixed prefix. Its installed root contains `app`, `tools`,
`runtime-libs`, `venv`, `pm-runtime`, `bin`, and manifests/facts. APT maintainer
hooks manage the declared CLI symlinks under `$PREFIX/bin` and refuse foreign
conflicts. They do not compile dependencies during installation.

`stage_apt_repo.py` owns repository metadata and signatures. Stable and canary
suites are `hermes-stable` and `hermes-canary`. Package files publish before
signed metadata. [Stable release admission](stable-releases.md) coordinates
acceptance and publication across distributions.

## Cache ownership

PM binary download archives and the native uv wheel cache have different
consumers. They are not interchangeable cleanup targets.

- PM retains completed downloads until package verification and publication.
  Successful installs remove their exact fetch archives. Failures retain
  downloads for retry. Cleanup leaves unrelated partials alone.
- Native staging prunes obsolete entries only in its build-owned tool store.
  It does not prune the user's machine-wide store.
- Native staging retains `uv-cache/` for offline mutable-environment rebuilds.
  It keeps extracted wheel entries but removes wheel ZIPs beneath `sdists-v*`
  so native signing can reach the extracted binaries.
- PM-runtime and application builds share the provider's persistent uv cache.
  CI restores/saves that cache, not the temporary build HOME. Failed builds
  retain completed wheels. Only v2 keys are restored; there is no legacy fallback.
  Plain `uv cache prune` removes dangling entries without discarding offline
  wheel inputs. It does not remove all historical versions or enforce a size cap.
- Icon preparation uses its own `SOURCE/.cache/icon-build`. Its build-only
  dependencies do not belong in the native application cache.
- Frontend `node_modules` is a provider input, not a frontend product.
  Docker's runtime TypeScript and Photon selections are separate exceptions.

The native cache behavior is in `scripts/bundles/native.py:55–65,202–216`.
Removing all caches from a payload can break offline environment reconstruction.

## Pinned binary inputs

`python -m scripts.ci.archive_inputs` preserves every HTTPS artifact in
`pm/lock.json`, across all targets, plus the Termux runtime-library and license
pins. `pm/artifact-mirror.json` owns the public mirror location. Object keys
are `upstream/sha256/HASH`, independent of filenames and release tags.

CI requests R2 first. Only 404 permits an upstream download. It verifies the
existing SHA256 before an immutable `If-None-Match: *` upload, then downloads
and verifies the stored object. Corrupt bytes, denied access, and failed
uploads stop the build. Concurrent writers may reuse identical bytes but
cannot replace an existing object.

The all-target workflow runs on pin changes on main, manual dispatch, and as
an admitted release prerequisite. It uses runner Python before the pinned
toolchain is available. Protected build jobs opt into the same R2-first step
through `setup-pm`'s `archive-inputs` input. Target payload steps seed the
actual PM store's disposable fetch entries with `--target` and `--store`;
Termux also passes `--payload` for runtime libraries. Cache hits do not skip
preservation. Untrusted PR jobs receive no publication credentials.

Installed PM clients, bootstrap installers, and Nix pin consumers use the
primary URL followed by the public mirror if the download is unavailable.
The pinned hash remains binding; no credentials or uploads are needed by
clients. PM keeps per-source resume state and reports attempted URLs.
For Termux files already removed upstream, the CI publisher can recover the
exact bytes from the community Internet Archive after an upstream 404/410.
It never repins to the latest package.

The archive must have no expiration lifecycle rule. Release pruning does
not cover its prefix. This covers PM binary pins and Termux runtime inputs,
not unpinned apt packages, OCI images, language-package registries, or native
Electron/SDK archives independently downloaded by their build tools.

## Verification boundary

The checks below distinguish product tests from distribution acceptance.
Signed installers and Android device execution require their native runners.

Focused helper tests cannot establish all of these requirements:

- Offline frontend compilation from prepared immutable inputs.
- Standalone TUI interaction and dashboard backend/assets behavior.
- Native Electron bindings and final signed launchers on each target.
- PM isolation after native payload relocation.
- Retained-cache offline reconstruction of a mutable application environment.
- Docker non-root CLI/TUI/web/browser behavior and every image layer's contents.
- Actual Nix package execution through reference-based wrappers.
- Fresh network-disabled Termux package installation and Android device behavior.

Use the existing acceptance checks in `tests/install/BUNDLED_UPDATES.md`,
`tests/docker/`, Nix checks, and `scripts/termux/check_deb.sh` plus
`validate_installed.py`. Container acceptance does not substitute for Android
device acceptance. No build-pass claim follows from this document.
