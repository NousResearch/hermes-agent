# Official Rive CLI and RML

The official `rive` CLI builds local RML projects without the desktop Editor.
It is separate from the editor MCP and this skill's Python MCP adapter. Use the
CLI for text-authored scenes, schema discovery, inspection, builds and captures;
use editor MCP for an authorized open editor document.

## Install without touching a shared environment

Discover the host and existing executable with `terminal` first. The official
1.2.0 release supplies Apple Silicon macOS, Linux x86_64 and Windows x64 builds.
Only macOS arm64 was exercised for this recipe. Do not install into a shared
location or change shell startup files without authorization.

Use `terminal` for the following POSIX-shell setup. Set `TASK_DIR` to a new,
absolute task-owned scratch directory; do not substitute a shared install.

```sh
export RIVE_HOME="$TASK_DIR/rive-home"
export RIVE_INSTALL_DIR="$RIVE_HOME/bin"
export XDG_CONFIG_HOME="$TASK_DIR/config"
export TMPDIR="$TASK_DIR"
export RIVE_ANALYTICS=off
export RIVE_NO_TUI=1
export RIVE_BASE_URL=https://releases.rive.app/cli
export RIVE_STREAM=stable
export RIVE_VERSION=1.2.0
curl -fL https://releases.rive.app/cli/install.sh -o "$TASK_DIR/install.sh"
```

**Stop and inspect the downloaded file with `read_file` before executing it.**
The reviewed installer honors `RIVE_HOME`/`RIVE_INSTALL_DIR`, verifies the release
archive against the official manifest's checksum, and installs bundled docs and
samples next to the versioned binary. Preserve these vendor checks and notices;
do not copy the CLI or vendor documentation into the skill. The script uses Bash
syntax, so invoke it with `bash`, not an arbitrary POSIX `sh`.

After review, use `terminal` in the same scoped environment:

```sh
bash "$TASK_DIR/install.sh"
RIVE_CLI="$RIVE_HOME/bin/rive"
"$RIVE_CLI" --version
"$RIVE_CLI" doctor --format=json
```

Keep these environment overrides on **every** invocation. `RIVE_HOME` isolates
install/state, but credentials on macOS/Linux instead honor `XDG_CONFIG_HOME`;
setting only `RIVE_HOME` is insufficient. `doctor` contacts the network and checks
version, update, auth, live-link and project. A signed-out warning with exit 0 is
expected, not a reason to log in.

Windows uses the official `https://releases.rive.app/cli/install.ps1`: download
and inspect it before running, and check its current isolation parameters.
Windows credentials use Credential Manager, not `XDG_CONFIG_HOME`; do not assume
POSIX credential isolation applies there. This recipe did not exercise Windows
or Linux installation. Homebrew is another documented option, but a shared
Homebrew install is not this isolated workflow; manage it with Homebrew rather
than the CLI's installer-only `update`/`switch`/`uninstall` commands.

`RIVE_ANALYTICS=off` forces telemetry consent off for the process. `rive analytics
on|off` writes the saved preference; plain `rive analytics` reports that saved
choice and can say “not set” even with the environment override. Do not enable
analytics implicitly. None of these controls proves absence of all network
traffic: install/update/doctor and authenticated commands have network functions.

## Discover the authoring API, then write

Use `terminal`, substituting the discovered executable for `rive`:

```sh
rive docs
rive docs format
rive docs gotchas
rive docs --search gradient
rive schema Rectangle --json
rive schema Shape --animatable --json
rive schema Shape --bindable --json
rive schema --search gradient --json
```

Use the installed schema's names, enum values and integer property keys rather
than guessing or copying editor MCP payloads into RML. `rive docs --path` gives
the installed reference directory for `read_file`/`search_files`.

- A handwritten scene starts with `<Rive version="1" kind="fragment">`.
  A fragment does not declare `Backboard`; put defaults in `rive.yaml`.
- Set `main` to the exact artboard name. IDs such as `0:2` are unique across all
  `.rml` files in the project. Every `.rml` compiles into the same document.
- **RML rotation is radians; editor MCP transform rotation uses degrees.**
  RML scale/opacity values are not the editor's displayed percentages. Read the
  relevant schema before converting units.
- **RML colors are `AARRGGBB` without `#`**, e.g. `FFBA7A2E`. They are not CSS
  `RRGGBBAA`. By contrast, YAML `artboard.background` requires a quoted `#RRGGBB`
  or `#AARRGGBB` value and only configures generated artboards without RML.
- Animation frames use the timeline's `fps`; transitions use milliseconds.
  Earlier drawable siblings render in front of later ones. Inspect a screenshot
  rather than assuming SVG-style paint order.
- Add a linked `LayoutComponentStyle` to the artboard and non-overlapping editor
  graph coordinates for states. `inspect` can report warnings that a successful
  `--verify` report does not show.
- Keep the project scan narrow. Every discovered `.luau` and `.wgsl` compiles even
  if unreferenced; exclude unwanted files. Unknown YAML keys are silently ignored.

## Build, inspect and render the original template

Copy `templates/minimal-rml/` into a fresh task-owned directory using file tools
or Python's `shutil.copytree` through `terminal`. Do not build in the installed
skill or add unrelated assets. Set `PROJECT` to that copy and use `terminal`:

```sh
"$RIVE_CLI" "$PROJECT" --verify --format=json
"$RIVE_CLI" inspect "$PROJECT" --json
"$RIVE_CLI" "$PROJECT" --once --format=json
"$RIVE_CLI" "$PROJECT" --screenshot="$PROJECT/build/start.png" \
  --artboard=Smoke --viewport=320x200 --fit=contain
"$RIVE_CLI" "$PROJECT" --screenshot="$PROJECT/build/mid.png" \
  --artboard=Smoke --viewport=320x200 --fit=contain --advance=30
"$RIVE_CLI" "$PROJECT" --screenshot="$PROJECT/build/end.png" \
  --artboard=Smoke --viewport=320x200 --fit=contain --advance=60
```

Check exit codes **and** JSON `success`, `errors`, `warnings`, `data.problems`
for build reports; check `problems`, the actual artboard list and the resolved
graph for `inspect`. `--verify` writes no `.riv`; `--once` does. Capture is its
own build mode, not a modifier of `--once`. Avoid `--quiet` while diagnosing:
it also hides compiler errors. Do not combine `--bench` with `--screenshot`;
that combination benchmarks without writing a PNG.

Load captures with `vision_analyze`. The template has an off-white 320×200
artboard, a horizontal rail, dark endpoints and an ochre diamond that travels
left → right → left. It has no scripts, View Models, text, fonts, image assets or
external content. See its README for the named animation/state-machine contract.

### Guard against successful commands that did the wrong thing

- An unknown render `--artboard` silently falls back to the first artboard.
  `inspect --artboard=UNKNOWN` instead returns an empty `artboards` list with
  exit 0. Assert the exact name exists **before** rendering and verify the
  runtime's selected artboard afterward; neither exit code proves selection.
- With a bound View Model, docs warn that a nonexistent `--data=path=value`
  property can be dropped while the build succeeds; `--quiet` hides the warning.
  Resolve the actual typed property first, use `--data-dump=<output.json>` to
  assert the readback, and check the visible effect. The included template has
  **no** bound View Model: on CLI 1.2.0, `--data=missing=1` fails with exit 1 and
  “no view model is bound”; it is not a binding demo.
- Interaction order matters. `--advance=30` means 30 frames at 60 Hz, not 30
  timeline frames at any arbitrary authored FPS. Entry-state settling can shift
  the first capture: do not demand pixel-identical timing between different
  render harnesses unless their initialization/advance contracts match.

## Independent runtime reload

A CLI screenshot does not prove the exported file loads in the target runtime.
Reload `build/minimal-rml.riv` with a separately installed, version-matched JS/WASM
pair, not the CLI preview. Keep WASM and the `.riv` local, disable the runtime's
WASM CDN fallback and asset-CDN loading, and fail unexpected external requests.
Verify `Smoke`, `Slide` and `Travel` exist; then render the initial, interior,
rightmost and return poses and assert visible marker movement, not just callbacks.
Dispose the artboard/state-machine/file/renderer and stop any test server.

This fixture was independently decoded and rendered with `@rive-app/canvas`
2.42.0 (its matching `rive.js` and `rive.wasm`), Playwright 1.63.0 and Chromium
151.0.7922.34. The harness used `RuntimeLoader.awaitInstance()`, `load(bytes,
undefined, false)`, `artboardByName('Smoke')`, `stateMachineByName('Slide')`, and
`StateMachineInstance.advanceAndApply()` with explicit frame steps. It checked
pixels and four poses with no external browser requests or page errors. This
is a web Canvas smoke test, not WebGL2/native compatibility or production UI QA.

## Account and entitlement boundaries

Local create, verify, unsigned build and screenshot do not require login. The
script-free template was built and reloaded unsigned. **Web files containing
scripts require `--publish`; unsigned scripts are rejected by web runtimes/CDN.**
Do not bypass this by patching exports or substituting a tool after an entitlement
refusal. `--publish` and `--rev` require a live authenticated session; neither was
exercised here, and no editor-source reopening is claimed.

Publishing may add a watermark, even without scripts. Current docs require a
project bound to a Rive account file in a Cadet-or-higher workspace for a clean
published file. Check current terms; local runtime licensing is not permission
to export someone else's artwork or bypass plan restrictions.

`push` uploads and replaces a bound remote file, including content someone has
open in the Editor. `pull` overwrites local scene/scripts/assets; `--yes` removes
its confirmation. Do not run login, publish, `.rev` export, remote import, push
or pull without explicit authorization for the account, exact file, rights and
overwrite scope. Version-control/recover the affected sources first.

## Verification scope and sources

Verified official CLI **1.2.0**, installed from the official stable manifest with
its bundled installer checks, on macOS arm64. The template passed verify and
inspect with no problems, built a **436-byte** unsigned `.riv`, reproduced the
same bytes, and rendered CLI captures plus the independent web-runtime poses.
Doctor passed with only the expected signed-out warning. The negative artboard
and unbound-data cases above were exercised. Account operations, scripts,
shaders, remote sync, `.rev` reopening and other platforms remain untested.
Only original source is shipped; binaries, captures and machine-specific harnesses
are disposable test evidence, not dependencies of this skill.

Official sources (refresh with `web_extract` when behavior changes):
- https://rive.app/docs/cli/overview.md
- https://rive.app/docs/cli/getting-started.md
- https://rive.app/docs/cli/agents.md
- https://rive.app/docs/cli/reference/commands.md
- https://rive.app/docs/cli/reference/project-config.md
- https://rive.app/docs/runtimes/advanced-topic/rml.md
- https://releases.rive.app/cli/install.sh
- https://releases.rive.app/cli/v1.2.0/manifest.json
