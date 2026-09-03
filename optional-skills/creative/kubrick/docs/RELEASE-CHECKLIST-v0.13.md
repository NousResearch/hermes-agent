# Kubrick v0.13 Release Checklist

## Status

| Gate family | Status |
|---|---|
| Wave 2 + Wave 3 implementation | Merged to `main` ([PR #24](https://github.com/scrimshawlife-ctrl/Kubrick/pull/24)) |
| Issues #3 / #4 | Closed |
| Hermes Skill Evals on PR | Passed at merge |
| `v0.13.0` git tag / GitHub Release | **Published** — https://github.com/scrimshawlife-ctrl/Kubrick/releases/tag/v0.13.0 |

## Required gates

- [x] `python scripts/kubrick.py validate-skill`
- [x] `python scripts/kubrick.py validate-corpus`
- [x] `python scripts/kubrick.py coverage` (warnings for historical sidecars outside registry are accepted)
- [x] `python scripts/kubrick.py eval`
- [x] `python scripts/test_outcome_governance.py`
- [x] `python scripts/test_wave2_wave3.py`
- [x] `python scripts/test_design_specification.py`
- [x] `python scripts/kubrick.py repeatability --output out/kubrick/repeatability-report.json`
- [x] Canonical authority-transfer storyboard compiles with `status: COMPILED`
- [x] Multi-provider adapters (flux/sd3/midjourney) preserve `source_graph_id`
- [x] Multi-signal evolution receipts validate and never set `automatic_application_allowed: true`
- [x] Forge signal bundles remain `OBSERVATION` with `forge_canonical: true`
- [x] Closed-loop QA reports geometry/state/residue/convergence separately
- [x] Surface-occult audit fails closed on named esoterica
- [x] No audience-facing packet exposes private pattern or lexicon links
- [x] README and QUICKSTART document Wave 2/3 + design-build commands
- [x] Create and publish the `v0.13.0` tag from verified `main`
- [x] Publish GitHub Release notes from `docs/RELEASE-NOTES-v0.13.md` / changelog

## Release procedure

1. Confirm the default branch is green.
2. Review `CHANGELOG.md` and version declarations (`VERSION`, `SKILL.md`, `README.md`).
3. Confirm no generated output or usage receipt is unintentionally committed.
4. Confirm docs index (`docs/README.md`) and QUICKSTART command lists match `scripts/kubrick.py`.
5. Create the `v0.13.0` tag from the verified main commit.
6. Publish release notes from the `0.13.0` changelog section / `docs/RELEASE-NOTES-v0.13.md`.
7. Reinstall into a clean Hermes skill directory and run `validate-skill`.

## Non-goals

- External model APIs are not required.
- Continuity Forge and MCP remain optional.
- Evolution proposals do not modify corpus authority automatically.
- MCP is never authoritative over local CLI or Forge.
