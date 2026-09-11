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
| Dependency provider | Tools, locked dependencies, native libraries and build environments | `node-deps.mjs`, `python_env.py`, PM, Nix `importNpmLock`/uv2nix, Termux wheelhouse |
| Product builder | Compilation or application assembly from prepared inputs | `tui.mjs`, `web.mjs`, `desktop.mjs`, `agent.py`, `../generate_icons.py` |
| Distribution adapter | Product selection, target preparation, package layout, signing and publication | `../bundles/`, `../../Dockerfile`, `../../nix/`, `../termux/` |

`node-deps.mjs` and `python_env.py` prepare dependencies. Unlike the product
builders, they can access package registries. There is no universal installer,
all-products dispatcher, or cross-platform Python environment.

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
prepared workspace. It does not build a frontend or publish a completion stamp.
Nix supplies dependencies through `importNpmLock` instead of this command.

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
`buildWeb`, and `buildDesktop`. They return output paths, not a new provenance
manifest.

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

### Icons and native inputs

`--icons` names the generator's output root, not a directory of loose icons.
The web builder reads `web/public/` beneath it and requires `favicon.ico`.
The desktop builder reads `apps/desktop/public/` beneath it and requires
`apple-touch-icon.png`.

Run the generator with a prepared `icon-build` Python environment:

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

The convenience wrapper prepares the locked environment and can access the
network:

```sh
node scripts/generate-icons.mjs --source /work/source --out /work/products/icons
```

It uses the isolated `icon-build` dependency group and
`SOURCE/.cache/icon-build`. Both icon commands accept `--check`. Without
explicit paths, they use the source checkout as the output root.

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

TUI and web scripts support a no-argument development mode. Explicit product
mode requires the arguments in the earlier table. The desktop product script
has no no-argument mode. The Node product parsers expose no `--help` flag.

## Python dependency provider

```sh
python -m scripts.build.python_env --source /work/source \
  --python /work/tools/python --uv /work/tools/uv --out /work/venv \
  --extra all --extra messaging
```

| Argument | Contract |
|---|---|
| `--source`, `--python`, `--uv`, `--out` | Required prepared source, interpreter, uv executable, and fresh environment destination |
| `--cache` | Optional cache directory. Default: `OUT.parent/.uv-cache` |
| `--extra` | Repeatable extra selection |
| `--all-extras` | Select all extras instead of `--extra` |
| `--no-install-project` | Exclude the root application install, but retain workspace-member installation |
| `--offline` | Prohibit uv network access. Required artifacts must already be available |

`OUT` must not exist. Failure removes this invocation's environment, not a
pre-existing environment. Success prints its Python executable. The Python
function `build_python_environment` additionally requires an explicit `env`
mapping. It returns the same executable as a `Path`.

The provider uses `pm.environment.PythonEnvironment` for environment creation,
frozen workspace sync, and dependency checks. It preserves project uv policy,
uses the supplied interpreter, and disables interpreter downloads. It does not
discover user plugins or publish a live PM selection. Nix and Termux retain
separate dependency providers. The uv cache remains available after this call.

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
| Python provider | `python_env.py:21–76`, `../../pm/environment.py:50–133` |
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
