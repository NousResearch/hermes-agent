# Install and Update E2E Tests

These tests answer one question: can a user on a released version get to this commit?

Each test leg installs an old released version, then updates it to HEAD. The install and the update run the real user surfaces. The legs do not use mocks and do not use headless proxies of GUI flows.

## The layers

The test family has four layers. Each layer has one job.

1. `scripts/sandbox/generate-e2e-matrix.mjs` declares the support matrix. It lists every {os, install-method, update-method} pair. It expands the pairs against the sampled release tags. It knows nothing about which pairs CI can run.
2. `.github/workflows/install-e2e.yml` is the primary workflow. It picks the release tags, runs the generator, and fans out one matrix job per OS. It also writes the plan chart and the result chart on the run summary.
3. The run workflows own the capability knowledge. `install-e2e-run.yml` serves linux. `install-e2e-windows-run.yml` serves windows. `install-e2e-macos-run.yml` selects either the shared script driver or the macOS GUI driver. Job-level `if:` gates select the supported pairs. All other pairs skip natively and show as grey.
4. The drivers do the work. `tests/install/installer-script-e2e.sh` handles POSIX script installs, `tests/install/macos-desktop-e2e.sh` handles macOS dmg installs, and `tests/install/windows-e2e.ps1` handles Windows installs. Install and update methods are separate axes, subject to each workflow's capability gates.

To declare a new method, edit the generator. To implement a method, flip the gate in the run workflow and extend a driver.

## The isolation trick

The drivers do not touch the network for git operations. Each driver makes a bare clone of the checkout at `serve.git`. Then it points every git process at this clone. The mechanism is a driver-owned `GIT_CONFIG_GLOBAL` file with `url.<file://serve.git>.insteadOf` rewrites for both canonical repository URLs.

The driver parks the `main` branch of `serve.git` at the old release. The installer runs and lands on the old release. Then the driver moves `main` to HEAD. An update becomes available in the same way that it does for a real user.

The installer script is not downloaded. The install leg runs the copy from the old git ref. This is the copy that a user of that version executed. The update leg runs the copy from HEAD.

## What one leg does

Each leg with the script drivers has these phases:

1. Stage: make the bare clone, park `main` at the old release.
2. Install: run the old release's own installer script. Make sure that the checkout is at the old commit and that `hermes --version` works.
3. Desktop smoke: run `hermes desktop --build-only` from the installed CLI. This proves that the installed version can build the desktop app. If the installed version does not have this flag, the phase reports a skip and continues.
4. Update: move `main` to HEAD. Apply one update method. Make sure that the checkout is at HEAD and that `hermes --version` works.
5. Desktop smoke again, at HEAD.

The windows GUI driver replaces phases 2 and 4 when the install method is `desktop-installer@latest`. It downloads the published `Hermes-Setup.exe`, clicks through the installer window with AutoHotkey, and clicks "Update now" in the running app with Playwright.

## Old versions

A leg can install a release from months back. The driver must not assume that the old version has today's CLI surface. The rule: probe, do not assume.

- For the installer, read the flag from the old ref's own script text.
- For the installed CLI, ask the binary with `--help`.
- If a flag is not found, omit the flag. This is not an error.

## The install methods

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
- A release tag push. This is the moment the set of start versions changes.
- Manual dispatch. You can select the route and the tag count:

```
gh workflow run install-e2e.yml --ref <branch> -f route=both -f tag-count=2
```

Cost per run, so nobody is surprised: 41 legs per sampled tag (windows 18, macos 15, linux 8), so scheduled and release-tag runs sample 2 tags for up to 82 legs. Manual dispatch defaults to 3 tags for up to 123 legs. A typical green leg finishes in 7-15 minutes; every leg is capped at 60. Route slices for cheaper reads: `update` (linux only, 8/tag), `windows-desktop` (18/tag), `macos-desktop` (15/tag). `tag-count` is validated to 1-10. GitHub's 256-job cap applies to each OS matrix separately, not to the combined leg count; at 10 tags the matrices hold 180 windows, 150 macos, and 80 linux entries. Windows would first exceed the cap at 15 tags (270).

Running the drivers locally: don't, except in a disposable VM. The windows driver kills every process named Hermes during teardown and the macos driver operates on `/Applications/Hermes.app`; on a machine with a real Hermes install they will interfere with it.

## Artifacts

Each leg uploads its logs as an artifact. Every leg also records the screen for its whole run: the composite action `.github/actions/e2e-screen-record` installs ffmpeg, records with the OS's capture backend (x11grab on linux, gdigrab on windows, avfoundation on macos), and fails the leg if the recording is missing or has zero frames. Linux runners have no display, so the action starts `Xvfb :99` first and exports `DISPLAY` for every later step — the app under test and the recorder share that display. The windows GUI leg also uploads screenshots and the update result file. Get them with `gh run download <run-id>`.
