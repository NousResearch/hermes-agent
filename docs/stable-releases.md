# Stable release admission and promotion

`Stable Release` is the release gate. A successful desktop builder alone is
not a successful stable release. Canary builds retain their separate workflow.

## Order

1. Admit an exact `vMAJOR.MINOR.PATCH` tag and non-prerelease draft.
2. Run the whole `ci.yaml` pipeline in release mode.
3. Build and test the same Docker images as the Docker workflow.
4. Run Nix, native PM bundles, install/update E2E, Termux and Windows live
   process tests. Build, sign and notarize the candidate packages.
5. Test upgrades to the actual signed Windows/macOS packages on both
   architectures. The baseline must be a published stable release.
6. Publish the tested Docker and bundle artifacts. Do not rebuild for publication.
7. After every required check and artifact publication succeeds, advance the
   Docker, App Installer, macOS and APT stable channels.
8. Verify promotion, record the accepted package manifest, then publish the
   GitHub release. Only this final job reports the stable release green.

Versioned candidate uploads are staging, not stable promotion. Failed,
cancelled, missing and unexpectedly skipped requirements block the gate.
Public channel updates never run merely because one architecture built.

Docker Hub, R2, APT and the Store do not support one cross-service transaction.
All prerequisites finish before promotion starts, but a failure during final
promotion can leave some services advanced and others unchanged. Such a run
stays red. Inspect its per-service results before retrying; do not rebuild or
replace the tested candidate to recover a pointer update.

Promotion also replaces the stable downloads page at `releases/stable/index.html`
on the R2 public origin, so the latest stable builds stay readable without
GitHub. Canary tag builds own `releases/canary/index.html`, and commit builds
own `releases/commit/<sha>/index.html`. Pages list only objects the build
actually staged; a re-run of an older tag never regresses a channel page.

## Run a release

Use `scripts/release.py --bump patch --publish --remote <remote>` to create
its version commit, stable tag and draft. Stable dispatch selects
`stable-release.yml` at that tag. The workflow rejects a different ref/SHA,
a moved tag, a tag outside repository main, or a project-version mismatch.
Canary dispatch still uses the default branch for its workflow/cache scope.

To start the gate for an existing draft, dispatch `Stable Release` with the
exact tag as both the workflow ref and the `tag` input. Keep tags immutable.
The local workflow calls and their checkouts use the tag's commit, not moving
main.

For a failed publication or promotion, retry the failed jobs in the original
run. Desktop and Termux handoffs live in the immutable R2 tag archive, not
expiring GitHub job artifacts. Each producer uploads its files, then writes a
`handoff-<target>.json` receipt containing the tag, commit, paths, sizes and
SHA256 digests. Consumers validate that identity and verify streamed downloads
before using them. Candidate assembly publishes only `release-candidates.json`;
it does not upload the packages again.

Windows, macOS and Termux candidate jobs stage packages, native metadata and
APT inputs under `releases/tag/<tag>/` before acceptance. No stable feed is
written at this stage. Publication and promotion read the candidate manifest
from R2 and require the digest passed by the accepted candidate job. The
existing acceptance and publication gates still control every stable channel.

Immutable uploads accept an existing object only after verifying that its bytes
match; different bytes fail. Do not delete archive objects or rebuild signed
candidates under the same tag to bypass a conflict. A fresh dispatch starts the
build phases again and is not a promotion-only retry. If the original objects
are unavailable, stop recovery rather than replace the accepted candidate.

The desktop workflow's optional Termux upgrade input is
`termux_upgrade_from_tag`: an exact published release tag with a Termux R2
handoff. It no longer selects packages by a GitHub workflow run. Explicit
non-publishing desktop builds retain no downloadable job artifacts. Unrelated
CI diagnostics and tested Docker image handoffs keep their existing storage.

## Canary and one-off desktop identities

`release.py --canary` builds the separate canary application. Its package
identity and CLI command (`hermes-canary`) differ from stable; the existing
canary feed updates that application only. One-off builds use
`release.py --build-commit REV --remote REMOTE` (add `--publish` to dispatch).
Their application identity and CLI command (`hermes-<7-character-sha>`) include
the pinned commit. Two different commit builds do not replace each other.

Branding is selected from those build inputs, not from runtime settings:
canary uses yellow/dark-yellow icons; one-off builds use red icons bearing
the short SHA. All desktop icon formats derive from the same artwork.

One-off stamps use `source: commit-build`. No app update feed or App Installer
subscription is published for them, and both the GUI and bundled CLI refuse
update requests. They direct the recipient to ask the developer for a new
build. Source checkout channels are separate: `hermes update --set-channel`
remains available there and selects the published release's source commit.

Store submission retains its fixed official stable identity. Nonstable
packages must not be submitted under that identity.

## Signed-package baseline

The last successful stable release records
`releases/stable/release-candidates.json` on the configured R2 public origin.
It identifies actual Windows universal MSIX bundles, macOS ZIPs and package
provenance. The next run combines those records with its candidate manifest
and uses the existing native bundled-update drivers.

For an existing stable release that predates this metadata, supply
`baseline-manifest` as an HTTPS URL on the configured R2 public origin to an
equivalent manifest of its actual published packages. Manifest redirects must
stay on the same origin. The baseline tag must be a published stable release,
package identities must agree, and versions must increase. Stable sideload
Windows versions must equal the tag's three components plus `.0`; Store
packages retain their separate version policy. Missing baseline
artifacts are a blocker, not permission to fabricate or skip acceptance.
See [the bundled update contract](../tests/install/BUNDLED_UPDATES.md).

## Explicit exclusions and policy

- Desktop Playwright E2E (`e2e-desktop.yml`) is deferred at the owner's request
  because it is flaky. It is reported as deferred, not passed. Stabilize it
  and prove repeatable CI runs before adding it to this gate.
- Install/update E2E and native signed-package acceptance are **not** deferred.
- PR-only history, label and diff review checks do not apply to a stable tag.
  All applicable source CI jobs still run, regardless of changed paths.
- OSV vulnerability findings retain their existing advisory policy. Required
  scanner execution failures are failures, not advisory findings.
- Disabled Linux desktop packaging is not claimed as shipped. Native Linux
  PM bundles, Docker, Nix and install/update checks remain required.
- Housekeeping, autofix, comment and skills-index/deploy workflows are not
  release acceptance suites.

## Implementation ownership

Actions owns job ordering, runner selection, permissions and environments.
Python owns shared release admission, manifests, artifact hashes, publication
and channel promotion under `scripts/releases/` and `scripts/bundles/`.
Electron-builder configuration/hooks and native Windows/macOS adapters remain
in JavaScript or PowerShell. These adapters consume release facts rather than
reimplementing the release gate. Gate jobs use only Python's standard library;
they do not install the application or the JS workspace to report a verdict.

Signing and publication credentials stay in their protected job environments.
The source CI call does not inherit deployment secrets. Configure the existing
release-signing and container-publish environments before running this pipeline.
