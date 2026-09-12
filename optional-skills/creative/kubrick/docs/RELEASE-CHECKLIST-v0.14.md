# Kubrick v0.14 Release Checklist

## Status

| Gate family | Status |
|---|---|
| Deterministic contract implementation | Complete |
| Canonical and exported Hermes contract tests | Complete |
| Cross-platform stdlib and validation CI | Complete on release candidate |
| `v0.14.0` git tag / GitHub Release | Pending green release commit |

## Required gates

- [x] Manifest validates and router surfaces derive from it.
- [x] `SKILL.md` validates with zero warnings.
- [x] Stdlib profile passes without site packages.
- [x] Validation profile exercises YAML/schema-dependent behavior.
- [x] Structured failure matrix verifies exits 2, 3, and 4.
- [x] Compile receipts contain deterministic release, corpus, schema, adapter,
      command, and input identities.
- [x] Canonical normalized artifacts are byte-identical across repeated runs.
- [x] Provider reports preserve all critical semantic invariants.
- [x] Injected provider losses fail closed.
- [x] Installer validates before activation and preserves the active install on failure.
- [x] Installer dry-run, fresh install, upgrade, receipt, rollback, and version paths pass.
- [x] 22/22 Hermes evaluations pass.
- [x] Outcome governance, Wave 2/3, and design-specification regressions pass.
- [x] Official Hermes export contains all storyboard recipe fixtures.
- [x] Repository-level Hermes contract tests pass in canonical and exported layouts.
- [x] Release versions and current-document references align.
- [x] GitHub Actions is green on the release commit.
- [ ] Create and push the `v0.14.0` tag from that green commit.
- [ ] Publish GitHub release notes from `docs/RELEASE-NOTES-v0.14.md`.

## Release procedure

1. Run `python3 scripts/audit_release_version.py --strict`.
2. Run the stdlib profile with `python3 -S`.
3. Run manifest, diagnostics, adapters, installer, identity, Hermes evals, governance,
   Wave 2/3, design, repeatability, and repository contract tests.
4. Export into a clean Hermes Agent tree and run `tests/skills/test_kubrick_skill.py`.
5. Confirm no generated output, corpus report, cache, or usage receipt is committed.
6. Commit and push the release surfaces.
7. Wait for every GitHub Actions job to pass.
8. Tag the verified commit as `v0.14.0` and push the tag.
9. Publish `docs/RELEASE-NOTES-v0.14.md` as the GitHub release body.
10. Synchronize the active official Hermes PR and re-run its repository contract test.

## Non-goals

- Windows-native PowerShell installation parity is post-v0.14 work.
- OpenClaw host alignment is tracked separately in issue #32.
- Typed domain-layer refactoring, claim-level provenance, and workflow compression are
  post-release architectural work.
