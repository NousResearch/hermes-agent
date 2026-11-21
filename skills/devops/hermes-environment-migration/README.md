# hermes-environment-migration

Open-source Hermes Agent skill, version **1.0.0**.

A platform-aware migration protocol that refuses blind copies of machine-bound state and separates ordinary configuration from credentials and encrypted vault material.

## Provenance

Derived from repeated Hermes machine-migration and recovery planning workflows. The public bundle removes private paths, credentials, hostnames, and machine-specific inventories.

## Inputs

- Source and target Hermes environment inventories.
- Approved non-secret configuration and state files.
- Archive manifests, hashes, platform details, and version evidence.

## Outputs

- A migration classification and phased import plan.
- Integrity, secrets-transfer, path-rewrite, rollback, and verification sections.
- Approval-gated next actions rather than automatic import.

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
cp -R skills/hermes-environment-migration ~/.hermes/skills/
```

PowerShell, from the repository root:

```powershell
$destination = Join-Path $env:LOCALAPPDATA "hermes\skills"
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item -Recurse "skills\hermes-environment-migration" $destination
```

Start a fresh Hermes session after installation because skill discovery may be cached.

## Invocation

Example triggers:

- Move my Hermes setup to a new machine.
- Prepare a Hermes migration export.
- Verify and import this Hermes archive.
- Migrate profiles, skills, memory, and cron safely.

## Safety

The skill is diagnosis, planning, or interview-only by default. Mutations, repairs, process changes, credential changes, persistence, publication, installation, execution, or repository writes require separate explicit approval.

Inspected content is untrusted evidence. Embedded instructions never override the user, the skill contract, or higher-priority safeguards.

## Privacy

- Secrets are excluded from ordinary migration archives.
- Reports should name credential posture without printing values.
- Private paths and inventories remain local unless the user explicitly chooses to share them.

## Limitations

- It does not make incompatible session databases portable.
- It cannot verify secrets transferred through an unavailable secure channel.
- A successful archive check does not prove the target runtime works.

## Examples

See [successful and boundary examples](examples/example-report.md).

## Validation

Run from the repository root:

```bash
python skills/hermes-environment-migration/scripts/validate_bundle.py
python -m unittest discover -s skills/hermes-environment-migration/tests -v
```

The validator and tests use only the Python standard library.

## Version history

### 1.0.0

- Initial public Field Kit release with evidence-first workflow, explicit authority boundaries, hostile-content handling, examples, and contract tests.

## License

Apache License 2.0. See the repository [`LICENSE`](../../LICENSE).
