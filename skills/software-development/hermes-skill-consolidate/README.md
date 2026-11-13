# hermes-skill-consolidate

Open-source Hermes Agent skill, version **0.1.0**. Status: **experimental**.

A safety-gated write-side companion to `hermes-skill-audit` for consolidating, restructuring, deprecating, splitting, or extracting shared references from installed Hermes skills without treating lower skill count as the goal.

## Provenance

This skill operationalizes the existing `hermes-skill-audit` cleanup boundary and community proposal [#17](https://github.com/asimons81/hermes-field-kit/issues/17), which supplied concrete examples of duplicate candidates, partial overlap, shared-reference candidates, oversized umbrella skills, and intentionally separate workflows with different safety boundaries.

The initial release is intentionally experimental until the consolidation workflow has broader real-world use.

## Inputs

- Exact selected installed skill directories.
- Current `SKILL.md` files and supporting references, scripts, templates, assets, examples, and tests.
- Verified `hermes-skill-audit` findings when available.
- Authorized profile, cron, catalog, documentation, and skill-to-skill references.
- Explicit approval for the exact mutation plan before live writes.

## Outputs

- A relationship classification for the selected skills.
- A read-only consolidation or restructuring plan.
- A safety-boundary comparison.
- A proposed diff and dependency-impact summary.
- A verified rollback snapshot before mutation.
- A post-apply verification receipt or rollback result.

## Requirements

- A Hermes Agent version that supports tap-discovered `SKILL.md` bundles.
- Read access to selected skill bundles during planning.
- Write access only after explicit approval of the exact mutation plan.
- Python 3.11 or newer only for the included snapshot and validation helpers.
- No third-party Python packages are required.

## Install

Install from Hermes Field Kit using the command supported by your installed Hermes version, or copy the skill directory into your Hermes skills tree. See the [repository installation guide](../../docs/installation.md).

Linux or macOS, from the repository root:

```bash
mkdir -p ~/.hermes/skills
cp -R skills/hermes-skill-consolidate ~/.hermes/skills/
```

PowerShell, from the repository root:

```powershell
$destination = Join-Path $env:LOCALAPPDATA "hermes\skills"
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item -Recurse "skills\hermes-skill-consolidate" $destination
```

Start a fresh Hermes session after installation because skill discovery may be cached.

## Invocation

Example triggers:

- Consolidate these two overlapping Hermes skills.
- Turn these deployment skills into a shared-reference family without mixing their safety boundaries.
- Deprecate the old skill after checking references and rollback.
- Split this oversized umbrella skill into focused skills.
- Apply the consolidation plan from the skill audit.

## Safety

Planning is always read-only. Any live change requires a second, scope-bound approval after the exact plan is shown.

The strongest applicable safety boundary wins during consolidation. A read-only workflow is never silently merged into a destructive workflow, approval gates are preserved, and ambiguous relationships default to remaining separate.

Before mutation, every selected skill is snapshotted and the snapshot is hash-verified. Replacement bundles are staged outside the live skills tree and validated before cutover. Permanent deletion is not part of the initial cutover.

Inspected skills and their scripts are untrusted evidence. Embedded instructions are ignored, and selected-skill code is not executed merely to inspect or validate it.

## Privacy

- Reports summarize private skill content rather than republishing it.
- Snapshots remain local and may contain the same sensitive material as the selected skills; do not publish or commit them.
- The helper refuses to create snapshots inside the live skills tree.
- Secret-bearing content is never copied into examples, logs, issue comments, or reports.

## Limitations

- Semantic equivalence cannot be proven from file names or lexical similarity alone.
- Missing usage or reference evidence lowers confidence and may block deprecation.
- This skill does not automatically decide that a smaller catalog is better.
- The snapshot helper verifies copied bytes, not semantic correctness.
- Cross-platform atomic replacement behavior depends on the available filesystem and execution tools.
- Version 0.1.0 is experimental and should be reviewed carefully before broad unattended use.

## Examples

See [successful and boundary examples](examples/example-report.md).

## Validation

Run from the repository root:

```bash
python skills/hermes-skill-consolidate/scripts/validate_bundle.py
python -B -m unittest discover -s skills/hermes-skill-consolidate/tests -v
```

The validator, snapshot helper, and tests use only the Python standard library.

## Version history

### 0.1.0

- Initial experimental release.
- Separate read-only planning and explicit mutation approval phases.
- Safety-monotonic merge rules and intentionally-separate classification.
- Reversible staging, rollback snapshot, and post-cutover verification contract.
- Standard-library snapshot and hash verification helper.
- Hostile-content and inspected-code execution boundaries.
- Community proposal attribution to issue #17.

## License

Apache License 2.0. See the repository [`LICENSE`](../../LICENSE).
