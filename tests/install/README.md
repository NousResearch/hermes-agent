# Install and Update E2E Tests

These tests answer one question: can a user on a released version get to this commit?

Each leg installs a released version and updates it through the real user
surface. Source legs target the selected checkout revision; packaged legs use
explicit old/new signed artifacts. The harness can use a mock LLM provider for
onboarding and interaction. It does not substitute a mock installer, updater,
or headless proxy for a native GUI flow.

## The layers

The test family has four layers. Each layer has one job.

1. `scripts/sandbox/generate-e2e-matrix.mjs` declares the support matrix. It lists every {os, install-method, update-method} pair. It expands the pairs against the sampled release tags. It knows nothing about which pairs CI can run.
2. `.github/workflows/install-e2e.yml` is the primary workflow. It picks the release tags, runs the generator, and fans out one matrix job per OS. It also writes the plan chart and the result chart on the run summary.
3. The run workflows own the capability knowledge. `install-e2e-run.yml` serves linux. `install-e2e-windows-run.yml` serves windows. `install-e2e-macos-run.yml` selects either the shared script driver or the macOS GUI driver. Job-level `if:` gates select the supported pairs. All other pairs skip natively and show as grey.
4. The drivers do the work. `tests/install/installer-script-e2e.sh` handles POSIX script installs, `tests/install/macos-desktop-e2e.sh` handles macOS dmg installs, and `tests/install/windows-e2e.ps1` handles Windows installs. Install and update methods are separate axes, subject to each workflow's capability gates.

To declare a new method, edit the generator. To implement a method, flip the gate in the run workflow and extend a driver.

## The isolation trick

Source drivers redirect canonical Hermes Git URLs to a local bare clone at
`serve.git`, using a driver-owned `GIT_CONFIG_GLOBAL` rewrite. This controls
the source install/update boundary, not all network access: tool/dependency
downloads and published bootstrap artifacts can still use the network.
Packaged-update legs instead use verified signed downloads and a temporary feed.

The driver parks the `main` branch of `serve.git` at the old release. The installer runs and lands on the old release. Then the driver moves `main` to HEAD. An update becomes available in the same way that it does for a real user.

For script-install legs, the installer comes from the old Git ref. A
script-reinstall update uses the target revision's script. A `hermes-update`
leg starts the old release's updater, and app-update legs start its app flow.
These paths are intentionally different.

## What one leg does

Each leg with the script drivers has these phases:

1. Stage: make the bare clone, park `main` at the old release.
2. Install: run the old release's own installer script. Make sure that the checkout is at the old commit and that `hermes --version` works.
3. Update: move `main` to HEAD. Apply one update method. An app update must produce a new successful receipt or handoff result; a changed checkout alone is not completion.
4. Verify the installed command and products before running `hermes --version`. Select the command under the installation's `.hermes/bin`; use the old venv only for a source tree without PM. Check PM currency and compiler receipts where supported. Preserve the no-desktop scenario for a plain install. Do not rebuild, remove dependencies, or force-stop an updater during verification. Historical installs without these receipts get artifact-presence checks, not a freshness claim.

The cheap fixture checks are `tests/scripts/test_source_driver.py` and `tests-js/source-update-observer.test.mjs`. They do not run installers or prove native GUI relaunch. The source app helper does not relaunch the updated app for it; the native packaged-update drivers own automatic-relaunch acceptance. The observer does not change source files, products, dependency selections, or facts.

The Windows `desktop-installer@latest` install downloads the published
`Hermes-Setup.exe` and drives its GUI with AutoHotkey. The selected update
method is a separate axis. App-update methods click the running app's Update
control; script and CLI methods use their corresponding entry points.

## Old versions

A leg can install a release from months back. The driver must not assume that the old version has today's CLI surface. The rule: probe, do not assume.

- For the installer, read the flag from the old ref's own script text.
- For the installed CLI, ask the binary with `--help`.
- If a flag is not found, omit the flag. This is not an error.

## The install methods

- `packaged-app`: a signed Windows MSIX bundle or macOS application ZIP.
  This pairs only with `open-app-update`, using separately pinned package
  inputs rather than the source-tag cross product. See [bundled update
  acceptance](BUNDLED_UPDATES.md) for the manifest and proof contracts.
- `installer-script`: the platform's one-liner (`curl | bash` on linux and macos, `irm | iex` on windows).
- `installer-script+desktop`: the same one-liner with its desktop stage opted in (`--include-desktop` / `-IncludeDesktop`). The stage builds the desktop app during the install. On windows it also registers Start Menu and Desktop shortcuts. On linux and macos it builds the app inside the checkout and registers no OS entry point.
- `desktop-installer@latest`: the published GUI installer (`Hermes-Setup.exe` on windows, `Hermes-Setup.dmg` on macos), driven through the real user flow.

## The two app-update variants

The desktop app has two launch paths, so the matrix has two app-update methods. Both click "Update now" in the running app. They differ in how the app starts:

- `open-app-update`: the app starts from the installed app entry point. On Windows, both the desktop installer and `installer-script+desktop` create shortcuts, so both support this route. On Linux and macOS, the script's opt-in desktop stage builds inside the checkout without registering an OS entry point. The macOS route therefore requires a desktop-installer install; Linux has no open-app-update leg.
- `hermes-desktop-app-update`: the app starts with the `hermes desktop` command. Every install method provides this command, on each OS that ships the desktop app. On linux this is the only app surface: no desktop installer and no packaged desktop artifact exist for linux. The driver captures the product's own launch call (argv, cwd, environment) with `e2e-assets/launch-capture/sitecustomize.py` and re-executes it under Playwright, which owns the app and clicks the update flow.

## Skips

A grey leg is normal. There are two causes:

- The method pair is declared but cannot run: either no OS entry point exists for it (open-app-update after a plain script install registers nothing to open), or no driver arm exists yet. The gate in the run workflow lists the pairs that run.
- The starting release predates the surface under test. Example: a release without `apps/desktop` has no window to launch. The tag annotation `tag_has_desktop` from the primary workflow marks these releases.

The result chart on the run summary shows each leg as passed, failed, or skipped. [Confirmed historical upgrade limitations](KNOWN_FAILURES.md) records failures that cannot be fixed in the update target, with exact release commits and CI evidence. These are not blanket skips: the original paths still run. Exact signature matches are non-red, counted separately as known failures, and linked to footnotes at the bottom of the result chart. An unrelated error on the same tag still fails.

## Triggers and cost

The matrix does not run on pull requests. One leg installs real toolchains and takes more than 10 minutes. The triggers are:

- A schedule, every 12 hours. This finds upstream drift.
- A matching release tag push.
- A reusable workflow call from the stable release gate.
- Manual dispatch. You can select the route and the tag count:

```
gh workflow run install-e2e.yml --ref <branch> -f route=all -f tag-count=2
```

The generator's output is the leg-count authority. Read the workflow's plan
chart before dispatching a large run. Scheduled/tag runs default to two sampled
tags; manual dispatch defaults to three. `tag-count` accepts 1–10.

`all` selects every source OS. `both`, `update`, and `installer` select Linux
source legs, not Windows plus macOS. `windows-desktop` and `macos-desktop`
select those source/GUI routes. `windows-bundled`, `macos-bundled`, and `bundled`
require their package manifests; they do not expand the source-tag cross product.
`install-ref` selects one exact source baseline. Stable release calls also
exclude the candidate tag so it cannot serve as its own old version.

Per-leg timeouts and GitHub's matrix limits remain workflow constraints, not
proof that every declared combination ran. Native package acceptance is separate
from the deferred desktop Playwright application suite.

Running the drivers locally: don't, except in a disposable VM. The windows driver kills every process named Hermes during teardown and the macos driver operates on `/Applications/Hermes.app`; on a machine with a real Hermes install they will interfere with it.

## Plugin upgrade preservation

Every upgrade leg also carries a plugin-survival contract: a tagged upgrade
must not delete or modify anything under the active home's `plugins/**` tree
or any profile's `profiles/<name>/plugins/**` tree. Destructive flows
(explicit uninstall, plugin removal, profile or user deletion) are out of
contract and not exercised.

- `e2e-assets/verify-plugin-preservation.py` is the shared, stdlib-only,
  read-only verifier. `snapshot` records every entry (kind, byte size +
  sha256, link targets, and a recursive fingerprint of a symlink's external
  target) across all plugin roots, including empty directories and the roots
  themselves; `verify` diffs the live tree against that snapshot and fails
  on any deletion or modification. Unreadable paths are hard errors; an
  empty snapshot is refused as inconclusive rather than claimed as a pass.
- `e2e-assets/preserve-plugins.sh` is the POSIX/macOS hook pair: after the
  install phase it seeds controlled, non-dependency directory fixtures (a
  `mnemosyne-wrapper` plugin with its marker, a symlinked runtime, a second
  profile plugin tree, and the externally-owned sidecar witness outside the
  home — no pyproject anywhere in the scanned root, nothing downloaded) and
  snapshots; after the update lands it verifies and fails the leg on any
  violation. Both shell and Windows hooks call the same Python `seed`
  command. Existing fixtures or snapshots abort rather than masking damage
  by reseeding. Windows uses a junction without requiring symlink privilege.
- Unit tests live at `tests/scripts/test_verify_plugin_preservation.py` and
  exercise the verifier against a real temp filesystem (real files, real
  symlinks; junction fallback on Windows).
- Stable-to-stable: the drivers accept `--update-ref REF` (Windows:
  `-UpdateRef`), defaulting to HEAD. Pass the next release tag to target a
  stable→stable upgrade through the same serve.git staging; only label a leg
  stable-to-stable when BOTH the install ref and the target ref are release
  tags. The workflow matrix itself is unchanged.

## Artifacts

Each leg uploads its logs as an artifact. Every leg also records the screen for its whole run: the composite action `.github/actions/e2e-screen-record` installs ffmpeg, records with the OS's capture backend (x11grab on linux, gdigrab on windows, avfoundation on macos), and fails the leg if the recording is missing or has zero frames. Linux runners have no display, so the action starts `Xvfb :99` first and exports `DISPLAY` for every later step — the app under test and the recorder share that display. The windows GUI leg also uploads screenshots and the update result file. Get them with `gh run download <run-id>`.
