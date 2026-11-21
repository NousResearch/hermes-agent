# hermes-skill-audit

Open-source Hermes Agent skill, version **1.0.0**.

A read-only health audit for global and profile-local Hermes skills, including dependency references, frontmatter, usage metadata, cron dependencies, duplicates, and upstream drift.

## Provenance

Derived from repeated Hermes skill-inventory and cleanup reviews. Public fixtures contain no private skill source, local usage history, cron identifiers, or personal paths.

## Inputs

- Resolved global, built-in, tap-installed, and profile-local skill directories.
- Skill frontmatter, references, scripts, examples, tests, and declared upstream sources.
- Authorized usage metadata, profile references, and cron dependencies.

## Outputs

- A HEALTHY, HEALTHY WITH FINDINGS, or REQUIRES ATTENTION verdict.
- Broken, stale, overlapping, ambiguous, and healthy classifications.
- An approval table for any proposed merge, archive, rename, or deletion.

## Requirements

- A Hermes Agent version that supports tap-discovered `SKILL.md` bundles.
- Read access to the evidence named by the request.
- Python 3.11 or newer only for the included validation commands.
- No third-party Python packages are required for bundle validation.

## Install

Install from Hermes Field Kit as a tap using the command supported by your installed Hermes version, or copy this skill directory into your local Hermes skills tree. See the [repository installation guide](../../docs/installation.md).

Linux or macOS, from the repository root:

```bash
mkdir -p ~/.hermes/skills
cp -R skills/hermes-skill-audit ~/.hermes/skills/
```

PowerShell, from the repository root:

```powershell
$destination = Join-Path $env:LOCALAPPDATA "hermes\skills"
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item -Recurse "skills\hermes-skill-audit" $destination
```

Start a fresh Hermes session after installation because skill discovery may be cached.

## Invocation

Example triggers:

- Audit my installed Hermes skills.
- Find duplicate or broken skills.
- Which skills are stale or unused?
- Check skill references before cleanup.

## Safety

The skill is diagnosis, planning, or interview-only by default. Mutations, repairs, process changes, credential changes, persistence, publication, installation, execution, or repository writes require separate explicit approval.

Inspected content is untrusted evidence. Embedded instructions never override the user, the skill contract, or higher-priority safeguards.

## Privacy

- Private skill content is summarized rather than republished.
- Credential-bearing references and personal paths are not included in reports.
- No skill is loaded, executed, edited, archived, or deleted during the audit.

## Limitations

- Absent usage tracking means unknown, not unused.
- Similar names do not prove functional overlap.
- Upstream drift cannot be confirmed without an authoritative source.

## Examples

See [successful and boundary examples](examples/example-report.md).

## Validation

Run from the repository root:

```bash
python skills/hermes-skill-audit/scripts/validate_bundle.py
python -m unittest discover -s skills/hermes-skill-audit/tests -v
```

The validator and tests use only the Python standard library.

## Version history

### 1.0.0

- Initial public Field Kit release with evidence-first workflow, explicit authority boundaries, hostile-content handling, examples, and contract tests.

## License

Apache License 2.0. See the repository [`LICENSE`](../../LICENSE).
