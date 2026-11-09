# Repair stale source version stamps after a release tag arrives

Status: complete

## Acceptance criteria

- A mutable `source="git"` install stamp is rejected when its commit still matches `HEAD` but a
  different stable release tag now points directly at that commit.
- The existing live-Git fallback then reports the newly available release version.
- A matching source stamp remains authoritative, and packaged/non-Git stamps are unchanged.

## Data flow

`get_version_info()` reads the install stamp first. While validating a source stamp, it already
checks live `HEAD`; after that check it will inspect only stable tags pointing at `HEAD`. A version
mismatch makes the stamp invalid, so the existing `_git_version_info()` path computes the full live
identity and caches it for the process.

## Boundaries

The change stays inside `hermes_cli/version_info.py` and its behavior tests. Stamp writing, update
channel resolution, packaged artifact identity, and installer behavior remain unchanged.

## Security and blast radius

The additional probe is read-only Git metadata against the executing checkout. It applies only to
stamps explicitly marked `source="git"`; Docker, Nix, desktop, and other packaged stamps retain
their existing authority.

## Verification

- Run the focused version/source-stamp tests through `scripts/run_tests.sh`.
- Run the CLI startup fast-path tests through `scripts/run_tests.sh`.
- Exercise `--version` from a temporary real Git checkout whose matching source stamp predates a
  stable tag at `HEAD`.
- Run the repository's blocking checks appropriate to the changed Python and documentation files.

## Outcome

- Full version resolution now rejects a mutable Git stamp when a different stable tag points at
  the same `HEAD`, then derives the current identity through the existing live-Git path.
- SHA-only startup/control probes skip release-tag validation and retain their one-`rev-parse`
  contract; this refinement came from independent correctness review.
- Focused suite: 33 passed, 0 failed, 3 host-specific skips.
- Disposable real checkout: a `0.23.0` source stamp followed by a `v0.23.1` tag at the same `HEAD`
  produced `Hermes Agent v0.23.1` through the actual `--version` fast path.
- `python scripts/check`: 15 checks passed.

No open questions remain. The initial environment activation provisioned the pinned test runtime
but failed its final shell composition; verification used that provisioned interpreter explicitly
through the required `scripts/run_tests.sh` runner.
