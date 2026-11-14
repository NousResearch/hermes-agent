# GitHub Actions workflows

This directory has 57 workflow files. Most of them are **reusable workflows**
(`on: workflow_call`). They never start on their own; an entry-point workflow
calls them. This README covers what starts each workflow, which jobs it skips
and why, how to run it by hand, and how a regular CI run differs from a
release run.

The YAML is the source of truth. If this file and a workflow disagree, the
workflow wins. Fix this file in the same PR.

- [Map](#map)
- [Things to know first](#things-to-know-first)
- [Regular CI: `ci.yaml`](#regular-ci-ciyaml)
- [Standalone PR / main checks](#standalone-pr--main-checks)
- [PR helpers](#pr-helpers)
- [Post-merge automation on `main`](#post-merge-automation-on-main)
- [Scheduled monitors](#scheduled-monitors)
- [Releases](#releases)
- [Manual-only and iteration-branch workflows](#manual-only-and-iteration-branch-workflows)
- [Full index](#full-index)
- [Known dormant or inert paths](#known-dormant-or-inert-paths)

---

## Map

```
pull_request / push main
 ├─ ci.yaml  (CI)  ── detect ──┬─ tests.yml ─────────────── (e2e jobs: release/dispatch only)
 │                             ├─ tests-os.yml ──────────── windows-install-update-e2e.yml
 │                             ├─ lint.yml, js-tests.yml, rust-tests.yml, catalog-tests.yml
 │                             ├─ remote-backends.yml, bootstrap-installer.yml, docs-site-checks.yml
 │                             ├─ e2e-desktop-core.yml, e2e-desktop-update.yml (release/dispatch only)
 │                             ├─ e2e-desktop.yml (disabled: `if: false`)
 │                             ├─ repo-guards.yml (one job: every cheap guard as a lane-gated step)
 │                             └─ all-checks-pass  ← the ONLY required branch-protection check
 │                                                   (+ the CI timing report steps)
 ├─ docker.yml   (own run; build/test, push :main on main)
 ├─ nix.yml      (own run; nix flake check)
 └─ path-filtered: pm-bundle, termux-verify, plugin-catalog-ci, sandbox-image, windows-bundle-sdk

workflow_run CI started  → ci-review-comment.yml (live PR comment)
PR labeled run-e2e       → label-rerun.yml (reruns CI, Docker, Nix)

schedule daily 06:41 UTC → canary-release.yml → (release.py dispatches) desktop-bundled-release.yml
                                                                         └─ desktop-bundle-smoke.yml

release.py release       → stable-release.yml ─┬─ ci.yaml (release: true, strict gate)
 (pushes rc.N-vX.Y.Z,                          ├─ docker.yml (release-phase test → publish)
  dispatches on main)                          ├─ nix.yml, pm-bundle.yml, termux-verify.yml
                                               ├─ windows-venv-e2e.yml, install-e2e.yml
                                               ├─ desktop-bundled-release.yml (candidate / publish)
                                               └─ install-e2e-{windows,macos}-run.yml (signed upgrade)
Stable Release completed (failed) → stable-release-publication.yml (rerun failed jobs / publish)
```

---

## Things to know first

### Entry points vs reusable workflows

| Kind | What starts it | Files |
|---|---|---|
| Entry point | `pull_request`, `push`, `schedule`, `workflow_run`, `workflow_dispatch`, `release` | `ci.yaml`, `docker.yml`, `nix.yml`, `canary-release.yml`, `stable-release.yml`, and the others under the headings below |
| Reusable only | Another workflow's `uses: ./.github/workflows/<file>` | Most `ci.yaml` children (`tests.yml`, `lint.yml`, …), `desktop-bundle-smoke.yml` (also dispatchable), `install-e2e-run.yml`, `install-e2e-{windows,macos}-run.yml` |
| Both | Its own triggers **and** a caller | `nix.yml`, `docker.yml`, `pm-bundle.yml`, `termux-verify.yml`, `windows-venv-e2e.yml`, `windows-install-update-e2e.yml`, `install-e2e.yml`, `desktop-bundled-release.yml`, `ci.yaml`, `archive-inputs.yml` |

A reusable workflow does not show up as its own run in the Actions tab. Its
jobs appear inside the caller's run, under the caller's job name (for example
`Python tests / Run tests (1/2)`).

### The `commit` input

Almost every reusable workflow takes `commit` ("Commit to test and build.
Empty keeps the triggering commit."). Regular CI leaves it empty, so it tests
`github.sha`. A stable release dispatches on `main`, so `github.sha` is
`main`'s tip there, not the commit under test. `stable-release.yml` therefore
passes the admitted claim commit down through `commit` (or `ref` for
`pm-bundle.yml`). `tests/ci/test_stable_release_commit_threading.py` enforces
this.

### Concurrency (what cancels what)

The pattern repeats across files:

- **Pull requests:** one run per PR per workflow. A new push cancels the older
  run (`cancel-in-progress: true` for `pull_request`).
- **Dispatches on a side branch** (`refs/heads/*` other than `main`): keyed on
  `github.ref`; in `ci.yaml` a newer dispatch on the same branch cancels the
  older run, so a re-dispatch never queues behind a stale one.
- **Pushes to `main`, dispatches on `main` or a tag:** keyed on `github.ref`.
  `ci.yaml` does not set `cancel-in-progress` for them (GitHub may still
  replace a *pending* run in the group). Some children (`tests.yml`,
  `lint.yml`, …) still use `cancel-in-progress` for non-tag refs.
- **Release / `commit`-pinned calls:** keyed on `github.run_id`, never
  cancelled. A parent rerun cannot kill a child that is still running.
- **Release controllers:** `canary-release`, `stable-release` (shared with
  `stable-release-publication`), `pages` (deploy), `docs-search-monitor`,
  `install-e2e-red-tracker`. All of them queue (`cancel-in-progress: false`).
  GitHub keeps only one *pending* run per group, so a newer queued run
  replaces an older queued one.

### "Skipped" is not "failed"

A job whose `if:` is false reports **skipped**, and no runner starts. In
regular CI, `all-checks-pass` treats a skipped job as fine when `detect`
did not select it. In release mode a skipped required job **fails** the gate
(see [Release mode](#release-mode-inputsrelease--true)).

### Forks and secrets

- `ci.yaml` runs PR-controlled code. It never gets `secrets: inherit` or
  GitHub App credentials. Do not add them.
- Jobs that need secrets either run only on `main` pushes or dispatches
  (`docker.yml` publish, `archive-inputs.yml`, `deploy-site.yml`) or check the
  repo (`github.repository == 'NousResearch/hermes-agent'`). On a fork they
  skip.
- Jobs that need R2 read the secret into job `env` and skip their steps when
  it is empty (a job-level `if:` cannot read `secrets`). This applies to
  `archive-inputs`, the `canary-release` R2 prune, and
  `termux-verify` → `r2-provisioned`. An unprovisioned fork or environment
  stays green and does not go red.
- `ci-review-comment.yml` skips PRs from forks (it needs a write token).

### Events created by `GITHUB_TOKEN` do not start workflows

GitHub does not start workflows for `push` (including tags) or `release`
events made with the default `GITHUB_TOKEN`. `workflow_dispatch` and
`workflow_run` are the exceptions. This is why:

- Release controllers start their follow-up builds with explicit
  `gh workflow run` dispatches (`release.py`, `skills-index.yml` → `deploy-site.yml`).
- Tags and releases that automation creates with `GITHUB_TOKEN` (canary
  tags, final `vX.Y.Z` receipts, GitHub release publication by the sequencer)
  do **not** start the `push: tags` / `release: published` triggers below.
  Those triggers fire only when a human or an App token creates the tag or
  release.
- `js-autofix.yml` pushes with an App token, so its pushes **do** start CI.
  Its concurrency group prevents a loop.

### Runners

`blacksmith-*` labels are Blacksmith runners (`4vcpu` for small jobs,
`32vcpu` for test suites and builds, `-arm` for native arm64). `macos-15`,
`windows-latest-32-core` and `windows-latest-32-arm-core` are GitHub-hosted.
No workflow runs on Intel macOS. Some publish jobs run on `ubuntu-latest` on purpose:
`docker buildx imagetools` cannot commit manifest lists to Docker Hub from
Blacksmith.

---

## Regular CI: `ci.yaml`

**Name:** `CI`. **Required check:** `All required checks pass (v3)`.

### Triggers

| Event | Behaviour |
|---|---|
| `pull_request` (any branch, any path) | The classifier selects lanes from the PR diff. E2E lanes are forced **off**. |
| `push` to `main` | Every lane is on (a push has no PR diff, so the classifier fails open). E2E lanes are forced **off**. PR-only jobs skip. |
| `workflow_dispatch` | Every lane is on, **E2E included**. Inputs: `release` (strict gate), `strict_acceptance`. |
| `workflow_call` from `stable-release.yml` | `release: true` and `commit: <claim commit>`. Every lane is forced on, E2E included, with the strict gate. |

There are no `paths:` filters: CI starts on every PR. The `detect` job
decides what runs.

### The `detect` job and lanes

`detect` runs `.github/actions/detect-changes`, which feeds the changed files
to `scripts/ci/classify_changes.py`. Each lane comes out as `true` or `false`.
The classifier fails **open**:

- An empty diff (push, dispatch, or a compare-API failure) turns every lane on.
- Any change under `.github/` turns every lane on.
- `python` is a denylist. It is off only when every changed file is provably
  prose, frontend-only, Docker meta, Nix-only, or catalog entry data.

Then a gate step in `ci.yaml` rewrites the result:

- The `e2e*` lanes are forced **false** unless the run is a release or a
  `workflow_dispatch`. A diff or a label cannot turn them on for a PR or a push.
- With `release: true`, every lane is forced **true**.

| Lane | Selected on a PR when the diff touches… (summary; see `classify_changes.py`) |
|---|---|
| `python` | anything not provably irrelevant to Python (includes `skills/`, `website/docs/`, `website/scripts/`, `website/static/oauth/`) |
| `python_prod` | as `python`, excluding `tests/` |
| `frontend` | `ui-tui/`, `web/`, `apps/`, `tests-js/`, root pnpm manifests (`package.json`, `pnpm-lock.yaml`, `pnpm-workspace.yaml`), the WhatsApp bridge, JS under `scripts/build/` |
| `rust` | `.rs`, Cargo files, bootstrap-installer crate paths |
| `catalog` | `plugin-catalog/*.yaml`, `contributors/emails/` (a diff with only these runs catalog tests **instead of** the Python suite) |
| `site` | the Docusaurus site |
| `bootstrap` | `install.sh` and its pin fragments and stamp |
| `desktop_updater` | `scripts/desktop-update/` and its Windows hand-off tests |
| `remote_backends` | the SSH/Docker terminal backends, the terminal tool, their real-backend tests |
| `uv_lock` | `pyproject.toml` or `uv.lock` |
| `deps` | `pyproject.toml` |
| `scan` | Python files, `.pth`, setup hooks |
| `npm_lock` | `pnpm-lock.yaml` or any `package-lock.json` |
| `docker_meta` | Dockerfiles and Docker scripts |
| `docker`, `nix` | slow lanes, read by `docker.yml` and `nix.yml` (not by `ci.yaml`) |
| `e2e`, `e2e_upgrade`, `e2e_desktop_core`, `e2e_desktop_update` | computed, then forced off except on release or dispatch |

A cross-language fixture (`_SHARED_FIXTURE_CONSUMERS`) also turns on the lanes
of every suite that reads it.

### Jobs and when each runs

| Job (check name) | Workflow | Runs when | Skipped when |
|---|---|---|---|
| Detect affected areas | inline | always | never |
| Python tests | `tests.yml` | `python` or `e2e` or `e2e_upgrade` | none of those lanes |
| ↳ `Run tests (1/2, 2/2)` | | always inside `tests.yml` | |
| ↳ `e2e` | | `e2e` lane (release or dispatch) | every PR and every push to `main` |
| ↳ `e2e-upgrade-plan`, `e2e-upgrade (shard)` | | `e2e_upgrade` lane (release or dispatch) | every PR and every push |
| OS-specific tests | `tests-os.yml` | `python` or `desktop_updater` or `e2e` or `e2e_upgrade` | none of those |
| ↳ macOS (2 slices), Windows x64, Windows arm64 (2 slices) | | always inside `tests-os.yml`; the real Windows hand-off tests only with `desktop_updater` | |
| ↳ Windows install + update E2E | `windows-install-update-e2e.yml` | `e2e_upgrade` (release or dispatch) | PRs and pushes |
| ↳ Windows E2E (real processes) | | `e2e` (release or dispatch) | PRs and pushes |
| Python lints | `lint.yml` (one job) | `python` or `frontend`; the job itself skips a frontend-only push | neither |
| ↳ Windows footguns + sibling guards, `ruff check .` (steps) | | `python` | frontend-only |
| ↳ code health ratchet (step) | | PR (Python and TS) | push / dispatch / release |
| ↳ ruff + ty diff (advisory steps) | | PR and `python` | push, or frontend-only |
| JS & TS checks | `js-tests.yml` | `frontend` | no frontend change |
| Rust tests | `rust-tests.yml` | `rust` | no Rust change |
| Catalog tests | `catalog-tests.yml` | `catalog` | no catalog data change |
| Remote terminal backends | `remote-backends.yml` | `remote_backends` | no backend change |
| Bootstrap installer | `bootstrap-installer.yml` | `bootstrap` | no installer change |
| Desktop E2E | `e2e-desktop.yml` | **never** (`if: false`, deferred as flaky) | always |
| Desktop core E2E | `e2e-desktop-core.yml` | `e2e_desktop_core` (release or dispatch) | PRs and pushes |
| Desktop update E2E | `e2e-desktop-update.yml` | `e2e_desktop_update` (release or dispatch) | PRs and pushes |
| Docs Site | `docs-site-checks.yml` | `site` | no site change |
| Repository guards | `repo-guards.yml` (one job) | **always** | never |
| ↳ No committed infographics / profile archives / case collisions / large blobs / lazy_deps imports / stale icons (steps) | | always | never |
| ↳ Deny unrelated histories (step) | | `pull_request` only | push, dispatch, release |
| ↳ Check contributors (step) | | `python` or `catalog` | neither |
| ↳ Check uv.lock (step) | | `uv_lock` | neither lockfile nor pyproject changed |
| ↳ JS lockfile diff, `pnpm-lock.yaml` and `package-lock.json` (advisory steps) | | PR and `npm_lock` | push, or no lockfile change |
| ↳ Prompt surface diff (advisory steps) | | PR and `python` | push, or no Python change |
| ↳ Lint Docker scripts (steps) | | `docker_meta` | no Docker meta change |
| ↳ Supply-chain scan / PyPI upper bounds (steps) | | PR and `scan` / `deps` | push, or neither lane |
| All required checks pass (v3) | inline | `always()` | never |
| ↳ CI timing report (advisory steps of the gate job) | | `always()`. Saves a baseline on `main`; makes a diff report on PRs | never |

Steps inside one job keep what told the old jobs apart: the lane gate is the step's
`if:`, each guard runs whether or not an earlier one failed (`!cancelled()`), any red
step fails the job, and the `review-status-*` artifacts the live PR comment reads keep
their names. The jobs were split only by history: each paid its own checkout and a
full billed minute for a few seconds of work.

Not part of `ci.yaml`: the Docker image build, `nix flake check`, and the OSV
scan. Docker and Nix run in their own workflows. A reusable-workflow call
keeps the caller run "in progress", and GitHub refuses `gh run rerun` on a run
that is in progress, so one slow job would block reruns of the fast required
ones. OSV runs on a weekly schedule.

### The gate: `all-checks-pass`

`scripts/ci/required_results.py` reads `toJSON(needs)`:

- **Regular mode (PR / push / dispatch without `release`):** any non-success
  that is not an allowed skip fails the gate. Skips are allowed, except for
  the lane-bound consumers `tests`, `tests-os`, `e2e-desktop-update`,
  `remote-backends` and `catalog-tests`: when `detect` selected their lane,
  they must succeed.
- **Cancelled duplicate (PR runs, `--allow-superseded`):** GitHub can deliver
  one push as two `pull_request` runs (seen when the push also dismisses a
  stale review). The per-PR concurrency group cancels the older copy, and
  branch protection requires the gate of every check suite on the SHA. A gate
  whose only non-successes are `cancelled` jobs passes when the Actions API
  shows a later-started (`run_started_at`, then higher run id) live or
  successful run of the same workflow, event, head SHA, head branch and head
  repository; that run's gate is the real verdict. (Fork runs carry no
  `pull_requests`, so the PR itself can't be matched; the same branch opened
  against two bases is not told apart.) Delegation only points forward, so the
  last run to start never passes by delegation, and a rerun never delegates
  (attempt 2 of a fork run approved after `action_required` counts as its
  first). A run cancelled while still waiting in the concurrency group never
  runs its gate at all, so it leaves no gate behind. Otherwise it fails as
  before.
- **Release mode:** every required job must be `success`. Only the deferred
  `e2e-desktop` may skip. The PR-only checks are steps of `repo-guards`, which
  runs on every event, so they never reach the gate as a skip.

Branch protection requires only this check. The `(v3)` suffix is deliberate.
When a new job joins the required set, bump the version here and in the
`protect-main` ruleset together, so old greens on open PRs stop counting.

### Release mode (`inputs.release == true`)

| | Regular PR | Push to `main` | Manual dispatch | Release (`stable-release.yml`) |
|---|---|---|---|---|
| Lanes | diff-selected | all except E2E | all incl. E2E | all incl. E2E (forced) |
| E2E suites | off | off | on | on |
| PR-only jobs | on | skip | skip | skip |
| Gate | lenient | lenient | lenient (strict with `-f release=true`) | strict |
| Cancelled by a newer push | yes | orchestrator no; `ref`-keyed children (`tests.yml`, `lint.yml`, …) can be | orchestrator no; unpinned children can be on the same branch, even with `release=true` | no (`run_id` group) |
| Commit tested | PR merge ref | `main` head | dispatched ref | claim commit via `commit` |

### Triggering CI by hand

```sh
# Full CI on a branch, E2E included, lenient gate
gh workflow run ci.yaml --ref my-branch

# Same, with the strict release gate
gh workflow run ci.yaml --ref my-branch -f release=true

# Acceptance: let tracked known_failure gaps FAIL instead of xfail
gh workflow run ci.yaml --ref my-branch -f strict_acceptance=upd-txn
```

To get the Docker and Nix slow lanes on a PR without a dispatch, add the
`run-e2e` label (see [`label-rerun.yml`](#pr-helpers)). The label does **not**
turn on the E2E suites. Only a release or a dispatch runs them.

---

## Standalone PR / main checks

Each of these has its own run and its own check. They are not required by
branch protection.

| Workflow | Triggers | Runs / skips | Manual trigger |
|---|---|---|---|
| `docker.yml`, **Docker Build, Test, and Publish** | `pull_request`, `push: main`, `workflow_call` (stable) | **PR:** `mode` (phase + classification, one job) → `build` only when the `docker` slow lane is on (Docker meta, dependency manifests, `pm/`, `tests/docker/`, webapp build recipe, or the `run-e2e` label). Builds slim+desktop × amd64+arm64 and runs the docker integration tests. Never pushes. **Push to `main` on the upstream repo:** always builds, then `publish` (env `container-publish`) and `merge` push `:main` / `:main-desktop` only (never `:latest`). **Stable `release-phase: test`:** builds the claim commit, tests it (unless the claim has no `docker-tests` lane), and archives the exact image. **`release-phase: publish`:** pushes the archived bytes under the attempt ref and returns the manifest digest. No `release: published` trigger, on purpose. | none (`gh run rerun`) |
| `nix.yml`, **Nix flake check** | `pull_request`, `push: main`, `workflow_call` | `flake-check` runs when the `nix` lane is on (`nix/`, `flake.*`, dependency/npm manifests, `.github/`, `run-e2e` label). Every push to `main` turns it on. `release: true` forces it. | none |
| `pm-bundle.yml`, **PM Bundle** | `pull_request` on `pm/**` (not `*.md`), `scripts/bundles/**`, the Windows build deps, `uv.lock`, `pyproject.toml`, itself; `workflow_dispatch`; `workflow_call` (stable) | Stages and smoke-tests `hermes pm bundle` on darwin-arm64, win32-x64, win32-arm64. Linux targets are commented out; there is no Intel macOS leg. | `gh workflow run pm-bundle.yml [-f ref=<sha>]` |
| `termux-verify.yml`, **Termux verification** | `push: main` / `pull_request` on `scripts/termux/**`, `pm/**` (not `*.md`), `tests/scripts/test_termux*`, itself; `workflow_dispatch`; `workflow_call` (stable) | `contracts` and `tui` always run. `r2-provisioned` is skipped on PRs. `native-runtime` (arm64) runs only when R2 secrets exist or `release: true`. A release run fails loudly without R2. | `gh workflow run termux-verify.yml` |
| `plugin-catalog-ci.yml`, **Plugin Catalog CI** | `pull_request` on `plugin-catalog/**` | `structural` (schema) and `pinned-source-validate` (the pinned SHA is reachable and the plugin passes `hermes plugins validate`; inner steps skip when no entry file changed). | none |
| `sandbox-image.yml`, **Sandbox desktop image** | `pull_request` on its Dockerfile and smoke script; `workflow_run` of a successful Stable Release; weekly `schedule`; `workflow_dispatch` (`publish`) | `build`+smoke on the upstream repo only. `publish`/`merge` (push `nousresearch/hermes-sandbox:desktop`, plus `:v<version>-desktop` after a stable release) on those three, a dispatch only with `publish=true`; publishing builds skip the layer cache to pick up base and apt fixes. The weekly run rolls `:desktop` from main's Dockerfile, so a Dockerfile change reaches users within a week. Never on `main` pushes. | `gh workflow run sandbox-image.yml -f publish=true` |
| `windows-bundle-sdk.yml`, **Windows bundle SDK** | `pull_request` on the Windows bundle and signing scripts plus their tests; `workflow_dispatch` | Runs the real Windows SDK downloader on cold Windows runners. | `gh workflow run windows-bundle-sdk.yml --ref <branch>` |

---

## PR helpers

| Workflow | Trigger | What it does / skip conditions |
|---|---|---|
| `ci-review-comment.yml`, **CI review comment** | `workflow_run` of `CI`, `types: [in_progress]` | Polls the CI run and the sibling Docker run for up to ~50 min. Keeps one live review comment on the PR up to date. Always runs scripts from the default branch. **Skips** non-PR runs and PRs from forks. A first-time contributor's run starts it only after a maintainer approves it, because `in_progress` fires then. One poller per head repo+branch; a newer CI run cancels the old poller. |
| `label-rerun.yml`, **Label rerun** | `pull_request: labeled` | Only for the `run-e2e` label. Cancels the head commit's `ci.yaml`, `docker.yml` and `nix.yml` runs if they are still going, then reruns them. Their `detect` reads the live labels, so Docker and Nix build. Exits if the PR head moved. The E2E suites stay off. |

---

## Post-merge automation on `main`

| Workflow | Trigger | What it does / skip conditions |
|---|---|---|
| `js-autofix.yml`, **auto-fix lint issues & formatting** | `push: main` touching `*.js/cjs/mjs/ts/tsx` or npm manifests; `workflow_dispatch` | `generate-patch` (no write access) runs `eslint --fix`. `apply-patch` (write access, never runs repo code) runs only when there are fixes. It pushes `bot/js-autofix` and opens an auto-merge PR. |
| `archive-inputs.yml`, **Archive pinned PM inputs** | `push: main` on `pm/lock.json`, `pm/artifact-mirror.json`, `scripts/termux/runtime_libs.json`, its script, itself; `workflow_dispatch`; `workflow_call(sha)` | Verifies every pinned HTTP input and archives it to R2 (env `release-signing`). The archive step is skipped with a notice when R2 secrets are missing. |
| `deploy-site.yml`, **Deploy Site** | `push: main` on `website/**`, `skills/**`, `optional-skills/**`, `plugin-catalog/**`, itself; `release: published`; `workflow_dispatch` (`skills_index_run_id`, `rebuild_skills_index`) | `deploy-docs` (GitHub Pages, upstream repo only) runs on every trigger. `deploy-vercel` (Vercel hook) runs only on `release` or dispatch, **not** on path pushes. Shared `pages` group, queued. |
| `skills-index.yml`, **Build Skills Index** | `schedule` 06:00 and 18:00 UTC; `push: main` on its script or itself; `workflow_dispatch` | `build-index` runs on the upstream repo only. `trigger-deploy` dispatches `deploy-site.yml` with the new index, and only on schedule or dispatch (not on the path push). |

---

## Scheduled monitors

| Workflow | Schedule (UTC) | Other triggers | What it does |
|---|---|---|---|
| `canary-release.yml` | daily 06:41 | dispatch | See [Canary](#canary-releases) |
| `skills-index.yml` | 06:00, 18:00 | push, dispatch | See above |
| `skills-index-freshness.yml` | every 4 h | dispatch | Upstream only. Opens an issue if `/docs/api/skills-index.json` is more than 26 h stale, missing, or a source collapsed. Also checks CORS on `/docs/api/plugins.json`. |
| `docs-search-monitor.yml` | every 6 h at :17 | `workflow_run` after every **Deploy Site**; dispatch | Upstream only. Live Algolia and headless-browser probe of docs search. A red run is the alert. |
| `osv-scanner.yml` | Mondays 09:00 | dispatch | OSV scan of `uv.lock`, `pnpm-lock.yaml` and the nested `package-lock.json` files. Results go to the Security tab. Advisory (does not fail on vulnerabilities). Intentionally not per-PR. |
| `sandbox-image.yml` | Mondays 04:23 | PR (paths), `workflow_run` after Stable Release, dispatch | Upstream only. Rebuilds `nousresearch/hermes-sandbox:desktop` without layer cache so base-image and apt security fixes land between releases. See [standalone](#standalone-pr--main-checks). |

---

## Releases

There are three release paths, and none of them starts from a tag push:

| | Canary | Stable | One-off / channel build |
|---|---|---|---|
| Entry | `canary-release.yml` (daily cron or dispatch) | `python scripts/release.py release …` → dispatches `stable-release.yml` | `python scripts/release.py --build-commit REV [--channel NAME] --publish` → dispatches `desktop-bundled-release.yml` |
| Identity | `v<stable>+canary.<UTC>` tag, separate `hermes-canary` app | `rc.<N>-vX.Y.Z` attempt ref, then a `vX.Y.Z` receipt at publish | commit: `hermes-<sha7>` app; channel: `Hermes NAME` or stable branding |
| Tests | native smokes of each bundle | full CI (strict) + Docker + Nix + PM bundle + install/update E2E + signed upgrade acceptance, per the claim's lanes | native smokes |
| Publishes | canary feeds, canary GitHub prerelease | Docker versioned tag → `stable`/`latest`, R2 stable feeds, App Installer, Store, APT, GitHub release | commit/channel pages under R2; never a feed or Store |

Long-form reference: `website/docs/developer-guide/stable-releases.md`.

### Canary releases

`canary-release.yml`, **Canary Release**

- **Runs:** daily at 06:41 UTC, or on `workflow_dispatch` from the default branch.
- **Skipped:** the `tag` job skips when dispatched from any branch other than
  the default branch (it would tag unmerged code). `prune` skips when `tag`
  was skipped or cancelled.
- **No new commits:** `release.py --canary` resumes an unpublished canary
  until its protected head is live, or does nothing.
- **Flow:** `tag` runs `release.py --canary --publish`, which pushes the tag,
  drafts the prerelease, and **dispatches `desktop-bundled-release.yml`**
  (a separate run). `prune` deletes canaries older than 14 days (GitHub, plus
  R2 when secrets are present).
- **Manual:** `gh workflow run canary-release.yml` (on `main`).
- A queued dispatch that a newer one superseded ends as `cancelled`. Only
  `failure` signals a canary failure.

### Stable releases

`stable-release.yml`, **Stable Release** (run name `Stable release <tag>`)

- **Runs:** `workflow_dispatch` only, from the default branch, with
  `tag=<rc.N-vX.Y.Z>` (the annotated claim ref) and an optional
  `baseline-manifest`. Use `scripts/release.py` instead of dispatching by
  hand. It derives the version, pushes the claim atomically, creates the
  draft, and dispatches:

  ```sh
  python scripts/release.py release --commit "$(git rev-parse origin/main)" --bump patch --remote origin
  python scripts/release.py release … --only windows,docker      # lane subset
  python scripts/release.py release … --with termux --without install-e2e
  python scripts/release.py publish --version X.Y.Z --remote origin
  python scripts/release.py abandon --version X.Y.Z --remote origin
  ```

  Never dispatch it with a final `vX.Y.Z` tag. Never publish the draft from
  the GitHub UI.

- **What runs is decided by the claim's lanes, not by inputs or paths.**
  `admit` checks the tag object, the commit, and ancestry on `origin/main`,
  then reads the lane set from the claim message (`scripts/releases/lanes.py`).
  It outputs `lanes` and `jobs`. Every lane-controlled job has
  `if: fromJSON(needs.admit.outputs.jobs)['<job>']`. A job whose lane is off
  reports **skipped**, and the gate **requires** that skip. A job that ran
  although its lane was off also fails the release.

  | Lane (default) | Jobs it controls |
  |---|---|
  | `ci` (on) | `ci` → `ci.yaml` with `release: true` |
  | `docker` (on), `docker-tests` (on) | `docker` (test phase), `publish-docker` |
  | `nix` (on) | `nix` → `nix.yml` with `release: true` |
  | `pm-bundle` (on) | `pm-bundle` |
  | `termux` (off), `termux-checks` (on, only with `termux`) | `candidates-termux`, `termux-checks` |
  | `windows-live` (on) | `windows-live` → `windows-venv-e2e.yml` |
  | `install-e2e` (on) | `install-e2e` → `install-e2e.yml` (route `all`) |
  | `windows` (on), `windows-store` (on) | `candidates-win32-{x64,arm64,bundle}`, Store submission |
  | `macos-arm64` (on) | `candidates-darwin-arm64` |
  | `smoke` (on) | native smokes inside the candidate calls |
  | `upgrade` (on) | `transitions-*` and `windows-packaged` / `macos-packaged-*` (signed OLD→NEW). Also skipped when no published baseline exists. |
  | `bootstrap-version` (on) | `bootstrap-version` |

- **Order:** every gate and candidate starts right after `admit`. CI does not
  block the signed builds. `acceptance` is the single join that blocks
  publication. Then `publish-docker` and `publish-bundles` (desktop workflow,
  `release-phase: publish`), `publication`, and finally `complete`
  ("Stable release is green").
- **Concurrency:** group `stable-release`, queued, shared with publication.
- **Secrets:** called workflows get `secrets: inherit`, **except `ci.yaml`**.

`stable-release-publication.yml`, **Stable Release Publication**

- **Runs:** on `workflow_run` when **Stable Release** completes, or on a manual
  dispatch (`version`).
- **Skipped:** on `workflow_run`, the `reconcile` job runs only when the
  stable run **did not succeed**, was itself a `workflow_dispatch`, and came
  from this repo. A green run publishes from its own final job. A failed run
  gets its failed jobs rerun at once, at most twice (attempts 2 and 3). After
  that the claim is burned.
- **Manual:** `gh workflow run stable-release-publication.yml -f version=X.Y.Z`
  (this is what `release.py publish` does). It is idempotent, so you can run
  it again after a partial failure.

### Desktop bundles: `desktop-bundled-release.yml`

**Desktop Bundled Release.** This is the native build, sign, stage and publish
pipeline for darwin-arm64, win32-x64/arm64 (+ universal bundle and Store
bundle), and Termux. There is no Intel macOS (darwin-x64) build. The linux-x64/arm64 groups are **disabled** (dummy jobs).

| Mode | How it starts | Key inputs | Publishes |
|---|---|---|---|
| Canary tag | dispatched by `release.py --canary` from `canary-release.yml` | `tag=v…+canary.…`, `upload_release=true` | canary Windows/macOS feeds, Termux .deb, builds table, canary release |
| Stable candidate | `workflow_call` from `stable-release.yml` | `release-phase=candidate`, `jobs=<one group>`, `skip-tests` / `skip-signed` / `skip-store` / `windows-smokes-with-bundle` from lanes | stages to `releases/tag/<attempt>/` only, plus receipts |
| Stable publish | `workflow_call` (`publish-bundles`) | `release-phase=publish`, `manifest-sha256` | verifies candidate files; `stable-store` submits the Store bundle |
| Commit build | `release.py --build-commit SHA --publish` (default branch, maintainer) | `build_commit=<40-hex>`, `bundle_env` | `releases/commit/<sha>/` page; no feeds |
| Preview channel | `release.py --channel NAME --build-commit … --publish` | `channel`, `branding` | channel head CAS after every smoke passes |
| Disposable namespace (tests) | manual dispatch | `disposable_channel`, `build_commit`, `disposable_receivers` | allocation only, no native build |

Inside one run:

- Per-arch `*-release` vs `*-commit` build jobs are chosen by `validate`'s
  outputs. The matching `… result` job collapses them.
- `smoke-*` call **`desktop-bundle-smoke.yml`**. It needs a staged SHA and
  skips after a failed `validate`.
- `publish-win32-updater` / `publish-darwin-updater` require
  `upload_release=true` and no `build_commit`/`channel`.
- `stage-receipt-*` run only for `release-phase=candidate`. `stable-*` jobs
  run only for `release-phase=publish`.
- `builds-table` runs only for a plain uploading tag build (canary).
  `commit-builds-summary` runs only for commit builds.

Re-smoke an already staged artifact without rebuilding:

```sh
gh workflow run desktop-bundle-smoke.yml --ref <test-branch> \
  -f platform=win32 -f arch=x64 -f format=msix -f sha=<built commit> \
  -f commit-build=true -f public-base=https://hermes-assets.nousresearch.com/
```

### Install & update E2E family

| Workflow | Triggers | Notes |
|---|---|---|
| `install-e2e.yml`, **Install & Update E2E** | `workflow_dispatch`; `workflow_call` from stable (`route=all`) | "Can a user on a released version get to this commit?" `pick-releases` samples release tags (newest, oldest, spread; `tag-count` or `install-ref`). `generate-matrix` expands `scripts/sandbox/generate-e2e-matrix.mjs` into linux/windows/macos legs. The bundled legs need `windows-bundle-manifest` / `macos-bundle-manifest` and a matching `route`. **Never runs on PRs, pushes, or a schedule.** |
| `install-e2e-run.yml` | `workflow_call` | Linux/OS-agnostic leg: real `curl \| bash` install, then update. Unsupported method pairs skip (grey). |
| `install-e2e-windows-run.yml` | `workflow_call` | Windows leg (Hermes-Setup.exe via AutoHotkey, `install.ps1`, app "Update now" under Playwright). Packaged-bundle arm only with `install-method=packaged-app`. Also called by stable for signed acceptance. |
| `install-e2e-macos-run.yml` | `workflow_call` | macOS leg (script installs, Hermes-Setup.dmg, packaged-bundle arm). Also called by stable. |
| `install-e2e-red.yml`, **Install & Update E2E red tracker** | `workflow_run` of Install & Update E2E; `workflow_dispatch` (`run-id`, `dry-run` default true) | Keeps one `install-e2e-red` issue in sync. The automatic path runs only when the watched run was a `schedule` event (see [dormant paths](#known-dormant-or-inert-paths)). |

```sh
gh workflow run install-e2e.yml --ref main -f route=all
gh workflow run install-e2e.yml --ref main -f route=windows-desktop -f install-ref=v0.21.5
gh workflow run install-e2e-red.yml -f run-id=<id> -f dry-run=false
```

### Other release-adjacent workflows

| Workflow | Triggers | Notes |
|---|---|---|
| `bootstrap-installer-build.yml`, **Bootstrap installer build** | `workflow_dispatch` (`ref`, `pin_branch`, `pin_commit`) | Signed Hermes-Setup.exe (Azure Trusted Signing) and notarized Hermes-Setup.dmg. Uploaded as run artifacts only, never published. No caches. |
| `live-providers.yml`, **Live provider canaries** | `push: tags v*`; `workflow_dispatch` (`filter`) | Live calls to real provider APIs, upstream only. Each case skips when its `LIVE_*_API_KEY` secret is missing. Tags pushed by `GITHUB_TOKEN` do not start it, and stable attempt refs (`rc.*`) do not match `v*`. In practice it runs on dispatch or on a `v*` tag a person pushes. `gh workflow run live-providers.yml -f filter='openrouter-anthropic'` |
| `sandbox-image.yml` | see [standalone](#standalone-pr--main-checks) | Publishes `:v<version>-desktop` and `:desktop` when Stable Release succeeds (`workflow_run`), `:desktop` weekly or on dispatch. |
| `deploy-site.yml` | see [post-merge](#post-merge-automation-on-main) | The Vercel deploy runs on `release: published` or dispatch. |

---

## Manual-only and iteration-branch workflows

These let you iterate on Windows behaviour without the rest of CI. Push to a
matching branch, or dispatch:

| Workflow | Branch pattern | Dispatch | Other callers |
|---|---|---|---|
| `windows-venv-e2e.yml`, venv-holder live E2E | `wine2e/**` | no | stable (`windows-live` lane) |
| `windows-install-update-e2e.yml`, install.ps1 + `hermes update` | `wine2e-install/**` | yes (`strict_acceptance`) | `tests-os.yml` (`e2e_upgrade` lane) |
| `windows-desktop-attach-e2e.yml`, Desktop attach to a running gateway | `wine2e-desktop/**` | yes | none |

```sh
gh workflow run windows-install-update-e2e.yml --ref <branch> -f strict_acceptance=upd-txn
```

---

## Full index

`R` = reusable only. `PR`/`main` = `pull_request` / `push: main`. `paths` = path-filtered.

| File | Name | Triggers | Called by |
|---|---|---|---|
| `archive-inputs.yml` | Archive pinned PM inputs | main (paths), dispatch, call | none |
| `bootstrap-installer-build.yml` | Bootstrap installer build | dispatch | none |
| `bootstrap-installer.yml` | Bootstrap installer | R | ci |
| `canary-release.yml` | Canary Release | cron daily 06:41, dispatch | none |
| `catalog-tests.yml` | Catalog tests | R | ci |
| `ci-review-comment.yml` | CI review comment | workflow_run(CI in_progress) | none |
| `ci.yaml` | CI | PR, main, dispatch, call | stable |
| `deploy-site.yml` | Deploy Site | main (paths), release published, dispatch | skills-index (dispatch) |
| `desktop-bundled-release.yml` | Desktop Bundled Release | dispatch, call | stable; canary/release.py (dispatch) |
| `desktop-bundle-smoke.yml` | Desktop bundle smoke | dispatch, call | desktop-bundled-release |
| `docker.yml` | Docker Build, Test, and Publish | PR, main, call | stable |
| `docs-search-monitor.yml` | Docs Search Monitor | cron 6 h, workflow_run(Deploy Site), dispatch | none |
| `docs-site-checks.yml` | Docs Site Checks | R | ci |
| `e2e-desktop-core.yml` | E2E Desktop core | R | ci |
| `e2e-desktop-update.yml` | E2E Desktop update | R | ci |
| `e2e-desktop.yml` | E2E Desktop | R | ci (disabled) |
| `install-e2e-macos-run.yml` | install-e2e macos leg | R | install-e2e, stable |
| `install-e2e-red.yml` | Install & Update E2E red tracker | workflow_run(Install & Update E2E), dispatch | none |
| `install-e2e-run.yml` | Install & Update E2E (reusable) | R | install-e2e, macos leg |
| `install-e2e-windows-run.yml` | install-e2e windows leg | R | install-e2e, stable |
| `install-e2e.yml` | Install & Update E2E | dispatch, call | stable |
| `js-autofix.yml` | auto-fix lint issues & formatting | main (paths), dispatch | none |
| `js-tests.yml` | JS Tests | R | ci |
| `label-rerun.yml` | Label rerun | PR labeled | none |
| `lint.yml` | Lint (ruff + ty) | R | ci |
| `live-providers.yml` | Live provider canaries | tags `v*`, dispatch | none |
| `nix.yml` | Nix flake check | PR, main, call | stable |
| `osv-scanner.yml` | OSV-Scanner | cron Mon 09:00, dispatch | none |
| `plugin-catalog-ci.yml` | Plugin Catalog CI | PR (paths) | none |
| `pm-bundle.yml` | PM Bundle | PR (paths), dispatch, call | stable |
| `remote-backends.yml` | Remote terminal backends | R | ci |
| `repo-guards.yml` | Repository guards | R | ci |
| `rust-tests.yml` | Rust tests | R | ci |
| `sandbox-image.yml` | Sandbox desktop image | PR (paths), workflow_run(Stable Release completed), cron weekly, dispatch | none |
| `skills-index-freshness.yml` | Skills Index Freshness Check | cron 4 h, dispatch | none |
| `skills-index.yml` | Build Skills Index | cron 06:00/18:00, main (paths), dispatch | none |
| `stable-release-publication.yml` | Stable Release Publication | workflow_run(Stable Release completed), dispatch | none |
| `stable-release.yml` | Stable Release | dispatch | release.py |
| `termux-verify.yml` | Termux verification | PR/main (paths), dispatch, call | stable |
| `tests-os.yml` | OS-specific tests | R | ci |
| `tests.yml` | Tests | R | ci |
| `windows-bundle-sdk.yml` | Windows bundle SDK | PR (paths), dispatch | none |
| `windows-desktop-attach-e2e.yml` | Windows Desktop attach E2E | push `wine2e-desktop/**`, dispatch | none |
| `windows-install-update-e2e.yml` | Windows install + update E2E | push `wine2e-install/**`, dispatch, call | tests-os |
| `windows-venv-e2e.yml` | Windows venv-holder live E2E | push `wine2e/**`, call | stable |

---

## Known dormant or inert paths

These match the YAML today and are listed so nobody mistakes them for live
coverage:

- **`e2e-desktop.yml`** is wired into `ci.yaml` with a bare `if: false`
  (deferred as flaky). Keep it a bare `false`. An expression form once made
  GitHub's parser reject the whole of `ci.yaml`.
- **Linux desktop bundles** (`build-linux-x64/arm64` in
  `desktop-bundled-release.yml`) and the Linux targets of `pm-bundle.yml` are
  disabled.
- **`install-e2e-red.yml`** updates its issue automatically only when the
  watched run came from a `schedule` event, and `install-e2e.yml` has no
  schedule. Today only a manual dispatch does anything.
- **`archive-inputs.yml`** declares `workflow_call`, but nothing in this
  directory calls it.
- Some file header comments are older than the "E2E only on release or
  dispatch" rule. For example, `windows-install-update-e2e.yml` says it runs
  on every push to `main`. The `detect` gate in `ci.yaml` is authoritative:
  the E2E lanes are off for every PR and every push.
- The `push: tags` / `release: published` triggers (`live-providers`,
  `deploy-site` Vercel job) do not fire for tags and
  releases that automation creates with `GITHUB_TOKEN`.
