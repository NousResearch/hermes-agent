# Shared product builders

These modules compile frontends and assemble a runnable agent from prepared
inputs. They do not replace npm, uv, PM, or Nix.
[Shared bundle builds](../../docs/shared-bundle-builds.md) describes the
providers and distribution adapters.

This reference describes the current interfaces, not completed artifact
acceptance. A successful parser or helper test does not prove a distribution
build or a target-native runtime.

## Ownership

| Layer | Owns | Examples |
|---|---|---|
| Dependency provider | Tools, locked dependencies, native libraries and build environments | `node-deps.mjs`, PM Python operations, Nix `importNpmLock`/uv2nix, Termux wheelhouse |
| Product builder | Compilation or application assembly from prepared inputs | `tui.mjs`, `web.mjs`, `desktop.mjs`, `agent.py`, `../generate_icons.py` |
| Distribution adapter | Product selection, target preparation, package layout, signing and publication | `../bundles/`, `../../Dockerfile`, `../../nix/`, `../termux/` |

`node-deps.mjs` and `pm.build_environment` prepare dependencies. Unlike the product
builders, they can access package registries. There is no universal installer,
all-products dispatcher, or cross-platform Python environment.

## Complete desktop preparation

`../bundles/desktop.py` composes these providers for local and CI packaging:

```sh
python scripts/bundles/desktop.py --tag vX.Y.Z --variant bundled --prepare-only \
  --work "$PWD/.build/desktop-job" --cache "$PWD/.cache/desktop-inputs"
python scripts/bundles/desktop.py --prepared "$PWD/.build/desktop-job/prepared.json"
```

Start from a clean checkout at that tag (or use `--commit FULL_SHA`). A host
Python and Git bootstrap preparation; PM selects the pinned tools. Native
compiler/SDK prerequisites are still platform-specific. Omitting `--prepare-only`
runs both phases. Defaults use the same `.build/desktop-job` and
`.cache/desktop-inputs` roots. Do not pre-create the work directory: preparation
claims it and publishes its job-local path selection only after success.

Preparation covers the exact workspace union, icon Python environment, runtime
and independent PM dependencies, Electron-native bindings, Electron archive and
selected packaging utilities. Light omits the runtime and TUI/web union.
Compilation and packaging consume those inputs, refusing missing/stale inputs
rather than acquiring replacements. The result is tied to this source revision,
target and absolute paths; it must not be restored as an authoritative CI cache.
Stable bundled/Store variants can share it. Product compilation still reruns.

The desktop cache action derives reusable paths from provider declarations and
saves them after preparation, before compilation/signing. It excludes job-local
environments and products. Native wheel reuse additionally depends on measured
compiler/SDK inputs. Signing-result caches remain separate. Strict dependency
consumption is not offline signing: timestamps, notarization and publication can
still require network access. Native unsigned network-denied packaging and final
signed launch acceptance are distinct verification gates.

## Prepared JavaScript workspace

Run commands from the repository root. Replace the absolute example paths with
build-owned paths.

```sh
node scripts/build/node-deps.mjs --source /work/source \
  --workspace ui-tui --workspace web
```

`--source` and at least one `--workspace` are required. A workspace can be its
locked path or package name. The provider deduplicates the selection and runs
one root `npm ci` operation. It includes root dependencies, development
dependencies, and optional dependencies. It checks Node/npm against the root
`engines` declarations. npm lifecycle scripts remain enabled.

The prepared source needs these inputs:

- Root `package.json` and `package-lock.json`.
- Selected workspace manifests and their required `file:` dependency sources.
- Source files and configuration for the selected product.
- Resolved dependencies in workspace-local or root `node_modules`.

Request the complete workspace union once. A later, narrower `npm ci` can
remove dependencies that another product needs. This provider modifies the
prepared workspace. It does not build a frontend.
Nix supplies dependencies through `importNpmLock` instead of this command.

`--reuse` opts into reusing a completed dependency install. Desktop bundles use
this with CI's cached `node_modules` tree. The receipt lives inside that tree
and matches the lockfile, local package manifests, project npm configuration,
Node/npm versions, OS/architecture, and exact workspace union. It also checks
npm's installed-tree lock and the presence of its recorded package directories.
A missing or mismatched receipt runs a clean `npm ci`; failed installs cannot
leave a reusable receipt. Omit `--reuse` to force a clean dependency install.
Source launchers add `--no-install` when PM's lazy-install policy is disabled.
That mode still reuses a matching completed receipt, but rejects stale or missing
dependencies before mutating the tree. Explicit build/update operations may install.
The receipt does not validate arbitrary edits inside installed packages and
never skips product compilation. CI saves the prepared tree before packaging
can mutate it, and before unrelated build/signing failures can discard it.

## Frontend products

```sh
node scripts/build/tui.mjs --source /work/source --out /work/products/tui
node scripts/build/web.mjs --source /work/source \
  --icons /work/products/icons --out /work/products/web
node scripts/build/desktop.mjs --source /work/source \
  --icons /work/products/icons --stamp /work/install-stamp.json \
  --native-deps /work/native-deps --out /work/products/desktop
```

| Builder | Required CLI arguments | Additional CLI arguments | Product contents |
|---|---|---|---|
| `tui.mjs` | `--source`, `--out` | None | `dist/entry.js` and `package.json` with `type: module` |
| `web.mjs` | `--source`, `--out`, `--icons` | None | `index.html`, Vite assets, and public assets |
| `desktop.mjs` | `--source`, `--out`, `--icons`, `--stamp`, `--native-deps` | `--typecheck`, `--platform` | Renderer assets, `electron-main.mjs`, `electron-preload.js`, and native `node_modules` |

Each output is the product directory itself. The desktop output is a `dist`
directory, not an application package. The exported functions are `buildTui`,
`buildWeb`, and `buildDesktop`. They return output paths and publish the build-input
receipt described below.

Each compiler publishes `hermes-build.json` inside its output (inside `dist/`
for TUI). `freshness.mjs` owns this receipt and all source input selection.
TUI inputs are its source tree, the Ink source alias, shared sources, their
manifests and TypeScript configuration, dependency locks, and its compiler and
shared compiler helpers. Tests, workspace documentation, dependency-provider
recipes, and other products' compiler recipes do not invalidate the TUI.
It records product/host identity, content hashes of workspace/shared sources and
build inputs, and the exact supplied icon directory, desktop install stamp, and
native-dependency tree. Inputs are checked again before publication: a concurrent
input change fails the build and preserves the previous output. Output validation
checks renderer/main/preload/public bytes and the native file inventory; native
bytes may change through signing after compilation. Native ABI verification remains
with the native provider and desktop compiler.

Source launchers query this owner without provisioning tools:

```sh
node scripts/build/freshness.mjs --source /work/source --product web --out /work/products/web
node scripts/build/freshness.mjs --source /work/source --product tui --out /work/products/tui/dist
```

The result is a JSON boolean. Missing receipts, changed inputs (including supplied
inputs outside source), missing prepared trees, or damaged outputs are stale.
Receipts describe a source build, not a portable dependency cache; immutable
distributions use their existing prebuilt launch path instead. They replace the
old Python per-profile hashes and TUI mtime lists, not PM's dependency receipts.

The compilers resolve modules from the supplied workspace. They do not run
npm, uv, PM installation, or icon preparation. TypeScript/Vite scratch files
stay outside source inputs. Compilation uses a private output directory next
to the destination. A successful compile replaces the destination. A failed
compile leaves the previous product in place and reports failure. Its presence
alone does not prove that the latest build succeeded.

Existing arbitrary output directories require the builder's `.hermes-product`
marker. Files, symlinks, and source directories are rejected. The exact npm
destinations (`ui-tui/dist`, `hermes_cli/web_dist`, `apps/desktop/dist`, and
`apps/desktop/build/native-deps`) remain rebuildable without a prior marker.
Other in-tree products live beneath `.build/` or `apps/desktop/build/products/`.
`frontend-common.mjs` classifies these destinations independently of which source
children already exist, so a warm desktop rebuild uses the same rule as a fresh
build. Explicit stamp, icon, native-tree and dependency inputs remain protected,
even when they live beneath a generated destination.

### Icons and native inputs

`--icons` names the generator's output root, not a directory of loose icons.
The web builder reads `web/public/` beneath it and requires `favicon.ico`.
The desktop builder reads `apps/desktop/public/` beneath it and requires
`apple-touch-icon.png`.

Run the generator with a Hermes runtime Python (Pillow and resvg-py are core
dependencies):

```sh
python scripts/generate_icons.py --source /work/source --out /work/products/icons
python scripts/generate_icons.py --source /work/source --out /work/products/icons --check
```

The generator reads artwork from `SOURCE/assets` and writes its declared paths
beneath `OUT`. These paths also include desktop packaging, website, and
bootstrap-installer assets. `--check` regenerates targets in memory and checks
output image properties. It does not compare output bytes with regenerated bytes.
The generator writes targets directly, not through the frontend
publication helper.

The convenience wrapper runs the same generator on `HERMES_PYTHON` (else
`python` on PATH) and never installs anything:

```sh
node scripts/generate-icons.mjs --source /work/source --out /work/products/icons
```

Both icon commands accept `--check`. Without explicit paths, they use the
source checkout as the output root. Builders without a runtime environment
prepare one with `scripts/build/icon_environment.py`.

The desktop native tree contains prepared packages, including `node-pty` with
its compiled binding. macOS also requires `get-windows/main`. The provider owns
the architecture and Electron ABI match. `--platform` defaults to the running
Node platform and controls native-file checks. It does not cross-compile a
binding. `--typecheck` enables the desktop renderer TypeScript check and defaults
to false. The web builder always runs its TypeScript project check.

The supplied install stamp controls the desktop main/preload build identity.
The compiler does not create an install stamp, build the dashboard, package
Electron, or sign native files.

### Source-development entrypoints

Existing npm commands use these recipes:

```sh
npm run build --workspace ui-tui
npm run build --workspace web
npm run build --workspace apps/desktop
```

| Command | Output | Preparation outside the product compiler |
|---|---|---|
| TUI build | `ui-tui/dist/entry.js` | Existing installed workspace dependencies |
| Web build | `hermes_cli/web_dist/` | npm `prebuild` prepares icons |
| Desktop build | `apps/desktop/dist/` | Icons, root-install assertion, install stamp, and native-dependency staging |

Compositions prepare icons once and pass `npm run build -- --icons /prepared/root`
to the desktop's source-development driver. It copies prepared packaging artwork
and passes the same root to the compiler. A standalone `npm run build` still
prepares its own icons. Source desktop launch runs the already-prepared Electron
binary directly; `--skip-build` does not provision Node, npm, or Electron.

TUI and web scripts support a no-argument development mode. Explicit product
mode requires the arguments in the earlier table. The desktop product script
has no no-argument mode. The Node product parsers expose no `--help` flag.

## Python dependency provider

```sh
python -m pm.build_env --source /work/source \
  --python /work/tools/python --out /work/venv --sealed \
  --extra all --extra messaging
```

| Argument | Contract |
|---|---|
| `--source`, `--out` | Required prepared source and fresh environment destination |
| `--python` | Optional build interpreter; otherwise PM selects its pinned Python |
| `--cache` | Optional build cache directory; otherwise PM selects its cache |
| `--group` | Repeatable build/test dependency-group selection |
| `--sealed` | Prune build-time editable and virtualenv marker `.pth` files |
| `--extra` | Repeatable extra selection |
| `--all-extras` | Select all extras instead of `--extra` |
| `--no-install-project` | Exclude the root application install, but retain workspace-member installation |
| `--offline` | Prohibit uv network access. Required artifacts must already be available |

`OUT` must not exist. Failure removes this invocation's environment, not a
pre-existing environment. Success prints its Python executable. The CLI is an
explicit build request. The Python
function `pm.build_environment` accepts the same semantic inputs and an optional
explicit build `env` mapping. It returns the same executable as a `Path`.

PM owns pinned installer acquisition, environment creation, frozen workspace
sync, dependency checks, and failure cleanup. Callers never resolve or pass a uv
executable. It preserves project policy, uses the supplied interpreter, and
disables interpreter downloads. It does not discover user plugins or publish a
live PM selection. Nix retains its declarative dependency provider. Termux builds
native wheels separately, then uses PM's requirements-environment operation with
an explicit bionic interpreter and an offline wheelhouse. The build cache remains
available after this call.

Other build adapters use the same command with `--requirements FILE` (or repeated
`--requirement SPEC`) for a caller-owned dependency list, `--manager-runtime` for
the independent PM graph, `--check-lock` for non-mutating CI lock validation, and
`--export-requirements FILE` for marker-preserving frozen export. Cache teardown
uses `python -m pm.build_env --prune-cache --cache PATH`; add `--ci` only when the
cache will not be packaged for offline installation.

Native bundle staging keeps its HOME and PM state temporary, but not its uv
cache. `scripts.bundles.stage --cache PATH` (or `hermes pm bundle --cache PATH`)
selects the persistent cache explicitly. Direct staging also accepts the
provider's `UV_CACHE_DIR`; otherwise it uses the output parent's `.uv-cache`.
The PM runtime and application dependency builds receive this same cache.
General PM staging uses `setup-pm` and `save-pm-cache` to restore and save it
after the build, including failures, under its v2 namespace. Desktop composition
instead prepares the full dependency set before its single dependency snapshot
save. The packaged `uv-cache/` is a copy, not the writable build cache.

### Windows ARM64 build prerequisites

`scripts/windows-build-deps.ps1` owns Visual Studio ARM64, Clang, Rust, and
static OpenSSL preparation. PM calls it through `pm/native_build.py` before any
dependency build from a checkout, so every source install path gets it. Native
build adapters use the same `scripts/build/windows-deps.ps1` entrypoint through
`pm.native_build`, before
isolating HOME or compiling Node/Python dependencies. CI uses the same script
through `setup-windows-build-deps`, with an OpenSSL cache outside the product.
The product compilers and assembler do not install these prerequisites.

The PowerShell entrypoint accepts `-StateRoot` for persistent build-tool state
and `-EnvironmentFile` for its prepared environment. The Python adapter passes
that environment only to build children. Rust's original toolchain homes stay
explicit, so temporary HOME isolation cannot hide an initialized toolchain.
General CI setup exports that compiler environment to later steps. Desktop
preparation keeps it child-scoped and records native cache identity there.
Warm OpenSSL reuse validates both static libraries and its development header.

## Runnable agent assembly

```sh
python -m scripts.build.agent --inputs /work/agent-inputs.json --out /work/agent
```

The only builder arguments are `--inputs` and `--out`. Python entrypoints also
accept argparse's `-h`/`--help`. The library interface is
`assemble(AgentInputs(...), out)`. The input JSON rejects unknown fields.

### Input fields

All supplied filesystem input paths must be absolute and exist. `repo` and
`bin_dir` are output-relative names, not filesystem inputs.

| Field | Required | Meaning |
|---|---|---|
| `project` | Yes | `pyproject.toml` with static project metadata and `[project.scripts]` |
| `code` | Yes | Prepared source tree, or installed code root for reference placement |
| `repo` | Yes | Code/resource location beneath `OUT`, such as `hermes-agent`, `.`, or `share/hermes-agent` |
| `placement` | Yes | `contained`, `fixed`, or `references` |
| `target` | Yes | `linux-x64`, `linux-arm64`, `darwin-x64`, `darwin-arm64`, `win32-x64`, `win32-arm64`, or `linux-arm64-bionic` |
| `python` | Yes | Prepared target interpreter file |
| `site_packages` | Yes | Prepared application dependency directory |
| `environment` | Yes | Prepared application environment root |
| `pm_runtime` | Yes | Independent PM runtime directory with `pm-runtime.json` |
| `bin_dir` | No | One output-relative directory name. Default: `bin` |
| `tools` | No | Prepared runtime tool directory. The manifest defaults to `tools` if omitted |
| `command_dir` | For `references` | Directory of prepared commands declared by the project |
| `resources` | No | Resource-name to directory mapping |
| `frontends` | No | `tui` and/or `web` product paths |
| `ref` | No | Source identity copied to the completion manifest |
| `stamp` | No | Prepared install stamp copied to `OUT/repo/install-stamp.json` |
| `features` | No | Prepared feature inventory copied to `OUT/enabled-features.json` |
| `env` | No | Additional environment values in the reference-placement command map |

Recognized resource names are `skills`, `optional-skills`, `plugins`, `locales`,
and `optional-mcps`. Distribution adapters supply the required resources.
The generic assembler does not infer a missing resource mapping. TUI inputs
require `dist/entry.js` and `package.json`. Web inputs require `index.html`.

The PM marker supplies `python` and `sitePackages` paths relative to its runtime
directory, or absolute store references. These paths must resolve to real
inputs. If a supplied stamp declares `nix` or `docker`, its `pmRuntime` must
match the supplied PM runtime. The assembler does not create that environment.

### Placement and output

- **`contained`:** The provider prepares the interpreter, dependencies, PM
  runtime, and tools inside `OUT`. Assembly copies source/resources, plants
  frontends, and calls the portable link helper.
- **`fixed`:** The provider owns final-prefix preparation. Assembly copies or
  reuses source/resources and generates launchers without portable relocation.
  Docker and Termux use this placement.
- **`references`:** Assembly references installed code and commands without
  copying Python code or replacing wheel metadata. It links explicit resources
  and frontends, then emits `command-map.json`. Nix creates its native wrappers
  from this map.

Source-layout placement writes project distribution metadata without building
a Hermes wheel. It also writes `site_packages/hermes-agent.pth` with a relative
code path. Source-layout placement therefore requires an output-owned dependency
directory, including in `fixed` mode. Reference placement does not write this file.

Commands derive from `[project.scripts]`, not a second command
list. POSIX launchers use the supplied runtime paths. Windows launcher minting
runs the target interpreter and therefore needs a runnable native environment.
A target label alone does not prove ABI compatibility.

For copied frontends, the assembler places TUI files at
`OUT/repo/hermes_cli/tui_dist/` and web files at
`OUT/repo/hermes_cli/web_dist/`. Reference placement links TUI at `OUT/ui-tui`
and web at `OUT/repo/web_dist` and records their environment bindings.

After structural assembly, `manifest.json` records `schema`, `target`, `repo`,
`venv`, `store`, `launchers`, and `runtime`, plus `ref` when supplied. Its
`runtime` contains `repoDir`, `toolsDir`, `storePython`, `sitePackages`, and
`commands`. Reference placement also emits command sources, destinations,
entrypoints, and environment values in `command-map.json`.

Agent assembly modifies its output in place. It removes old completion records
before work and writes the manifest last. This is not the frontend builder's
atomic-directory publication contract. A failed assembly can leave partial
files. The manifest is not evidence of a target-native launch or signed-package
acceptance.

## Implementation references

These locations define the interfaces described here:

| Contract | Source |
|---|---|
| Frontend arguments and publication | `frontend-common.mjs:31–79` |
| TUI product and development output | `tui.mjs:25–103` |
| Web inputs and TypeScript check | `web.mjs:8–71` |
| Desktop inputs and native checks | `desktop.mjs:11–69` |
| Locked workspace union | `node-deps.mjs:31–68` |
| Python provider | `../../pm/operations.py`, `../../pm/environment.py` |
| Agent input fields and checks | `inputs.py:30–115` |
| Agent assembly and outputs | `agent.py:76–168` |
| Launcher implementation | `launchers.py`, `launcher_wrapper.py`, `mint_launchers.py` |

## Verification still required

Parser checks and source inspection establish the documented call shapes.
They do not establish offline compilation, cache reuse, or runtime success.
Artifact acceptance still needs these checks:

- Real frontend builds from immutable prepared inputs without network access.
- Standalone TUI interaction and dashboard assets/backend behavior.
- Native Electron bindings under the packaged Electron version.
- Agent CLI, ACP, plugins, and catalogs from an unrelated working directory.
- Native payload relocation and offline mutable-environment reconstruction.
- Docker runtime probes as its non-root user and layer-content inspection.
- Actual Nix builds and commands through store-reference wrappers.
- Fresh network-disabled bionic installation and Android device acceptance.

Docker/Nix build execution belongs to the distribution verification work, not
this documentation pass. No build-pass claim follows from this reference.
