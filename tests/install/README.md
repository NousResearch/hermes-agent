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

The cheap fixture checks are `tests/scripts/test_source_driver.py` and `tests-js/source-update-observer.test.mjs`. They do not run installers or prove native GUI relaunch. The native packaged-update drivers own automatic-relaunch acceptance. A later driver-owned launch checks chat only after the original update/relaunch assertions pass; it cannot repair a failed handoff. The observer does not change source files, products, dependency selections, or facts.

The Windows `desktop-installer@latest` install downloads the published
`Hermes-Setup.exe` and drives its GUI with AutoHotkey. The selected update
method is a separate axis. App-update methods click the running app's Update
control; script and CLI methods use their corresponding entry points.

## Desktop chat at install and update checkpoints

Desktop-bearing routes run the same message check used by post-build bundle
smoke and the desktop chat spec: `tests-js/scripts/desktop-chat-smoke.ts`.
It types a unique prompt through the real composer, requires a new request at
the existing loopback mock provider, and waits for a new completed assistant
reply in the active transcript. A reply preserved from OLD cannot satisfy NEW.
Only inference is mocked; Electron, the installed backend, and rendering run
for real. These checks require no external model credentials.

- A desktop-bearing OLD installation must chat before its update. App-driven
  updates also check the actual OLD window before clicking Update now.
- After the existing completion and product checks, NEW must chat if the route
  expects desktop. A route that adds desktop only at update retains the OLD
  no-desktop assertion.
- Plain source install/update routes that never request desktop record
  `not-applicable`; the chat driver does not build desktop for them.
- Native package updates must prove automatic relaunch first. The driver then
  closes the verified app normally and launches the same installed binary for
  `post-update-launch` chat. That second launch is not automatic-relaunch proof.

`tests/install/e2e-assets/desktop-smoke.ts` runs an exact installed executable
with an isolated home/userData and checks the actual backend listener's origin.
Source checkpoints use that leg's installed tree; bundle checkpoints use its
embedded payload. Missing products fail rather than trigger a build or repair.
Driver Node/Playwright come from the current checkout, not OLD's dependencies.

Each checkpoint writes a `desktop-chat-old.json`, `desktop-chat-new.json`, or
`desktop-chat-installed.json` result plus a screenshot. Failed chat fails the
leg even when version, health, and update receipts passed. Existing plugin and
user-state preservation checks remain required. The mock configuration writer
preserves unrelated settings, including the controlled update feed.

## Post-build artifact smoke

`desktop-bundled-release.yml` calls `desktop-bundle-smoke.yml` on fresh native
Windows and macOS runners. It downloads the admitted commit's receipt-bound
bytes through the public archive and installs the actual artifact before chat:

| Native target | Independently installed formats |
|---|---|
| macOS arm64 and x64 | DMG, ZIP |
| Windows arm64 and x64 | MSIX, universal MSIXBUNDLE |

The universal envelope runs on both Windows architectures, verifying native
slice selection. Windows assembly/staging is separate from feed publication;
both Windows smoke matrices must pass before publishing its canary feed.
macOS publication and stable candidate acceptance also require their smoke
results. Failed checks retain the downloadable build and diagnostic artifacts.

Stable candidate schema 2 binds these successful smoke groups into the pinned
candidate manifest. Promotion renders that evidence, not the skipped smoke
jobs from its own phase. Schema-1 candidates lack this admission and cannot be
promoted by the new workflow; create a new candidate tag. They remain valid as
previous-version upgrade baselines.

This covers signed downloadable commit builds, canaries with upload enabled,
and stable candidates. No-upload tag builds have no cross-job download handoff.
Linux bundle jobs remain disabled, but Linux source-install desktop checks run.
Unsigned Store submissions are not sideloaded or re-signed for this smoke;
their Store acceptance remains separate. Passing helper tests does not establish
native installation or chat acceptance: those require the corresponding runner.

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

## Manual update rehearsals

There is no `manual/` directory in this repo. The rehearsal scripts — `pre`
(back up the whole `HERMES_HOME`, the desktop app's Electron userData, the
`hermes` shims on PATH and the global git config, then point the install's
update source at a custom repo + ref) and `post` (undo all of it and report how
exact the restore was) — are a hand-off kit for someone with a *real* install,
kept outside this checkout on purpose: they are not part of the repo and not
wired into any lane here.

What *is* in the repo is the assertion layer they exist to demonstrate. The
user-state preservation checks below run on every upgrade leg; the manual kit
only reproduces the same upgrade on a machine you care about, where a backup
and a restore are the only honest way to do it.

## User-state preservation

Alongside the plugin-tree contract below, every upgrade leg also carries a
**user-state** contract: an upgrade may add state, may rewrite `config.yaml`
(additive config migration) and the bundled `skills/` tree (the product
re-syncs it), but it may not delete or modify the user's own durable state —
`.env`, `auth.json`, `state.db`, `gateway_state.json`, `memories/`, `cron/`,
`sessions/`, `profiles/`, `photon/`, `desktop-plugins/`, `tui-widgets/`,
`skins/`, `pets/`, `skills/.archive/` — and `state.db` may not lose rows.

- `e2e-assets/verify-user-state.py` is the shared, stdlib-only, read-only
  verifier. It records row counts for `state.db` rather than bytes (a live
  SQLite file changes for benign reasons), and it has **no** `seed` mode on
  purpose: the state it defends must be produced by the product through the
  ordinary user path, never hand-written by the harness.
- `e2e-assets/user-state-actions.sh` produces that state with real commands —
  `hermes chat -q` (a real turn → `sessions/` + `state.db` rows),
  `hermes auth add` (→ `auth.json`), `hermes profile create`
  (→ `profiles/<name>/`) — after probing `--help` for each flag, per the
  harness's "probe, do not assume" rule. Each action asserts it actually landed,
  so a leg can never "pass" while testing nothing.
- `e2e-assets/preserve-user-state.sh` is the POSIX/macOS hook pair
  (snapshot before the upgrade, verify after). Windows calls the same Python
  `snapshot`/`verify` from `windows-e2e.ps1`'s phases, alongside its existing
  plugin hooks.
- Two more invariants ride along: the redirect must stay transport-level
  (`config --get remote.origin.url` stays official while `remote get-url` is
  rewritten), and the user-visible launcher must still exist and run after the
  upgrade.
- Unit tests live at `tests/scripts/test_verify_user_state.py` and exercise the
  verifier against a real temp filesystem with real sqlite databases.

## Artifacts

Each leg uploads its logs as an artifact. Every leg also records the screen for its whole run: the composite action `.github/actions/e2e-screen-record` records with the OS's capture backend (x11grab on linux, gdigrab on windows, avfoundation on macos), and fails the leg if the recording is missing or has zero frames. ffmpeg itself comes from the PM toolchain — jobs pass `packages: ffmpeg` to `actions/setup-pm`, which installs the locked, sha256-pinned build (native per-OS, including win32-arm64) and puts it on `PATH`; the action only verifies it is there. Linux runners have no display, so the action starts `Xvfb :99` first and exports `DISPLAY` for every later step — the app under test and the recorder share that display. The windows GUI leg also uploads screenshots and the update result file. Get them with `gh run download <run-id>`.
