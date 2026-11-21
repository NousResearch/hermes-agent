# hermes-update-doctor

Open-source Hermes Agent skill, version **1.0.0**.

A diagnosis-first update investigator that separates remote drift, repository divergence, process locks, stale caches, partial installs, and runtime-version mismatches.

## Provenance

Derived from repeated Hermes update failures across native and repository-backed installations. Public material removes private remotes, host paths, process details, and incident-specific state.

## Inputs

- Resolved Hermes executable, repository, environment, and reported version.
- Read-only process, Git topology, package metadata, cache, and installation-integrity evidence.
- Authoritative remote and version information when accessible.

## Outputs

- A HEALTHY, REMOTE DRIFT, PROCESS BLOCKED, REPOSITORY DIVERGED, UPDATE INCOMPLETE, or UNVERIFIED diagnosis.
- Installation, process, repository, version, root-cause, and not-verified evidence.
- One approval-gated recovery and verification plan.

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
cp -R skills/hermes-update-doctor ~/.hermes/skills/
```

PowerShell, from the repository root:

```powershell
$destination = Join-Path $env:LOCALAPPDATA "hermes\skills"
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item -Recurse "skills\hermes-update-doctor" $destination
```

Start a fresh Hermes session after installation because skill discovery may be cached.

## Invocation

Example triggers:

- Hermes says it is behind but will not update.
- Update says already current but the version is stale.
- Another Hermes process is blocking the update.
- Diagnose a failed Hermes update.

## Safety

The skill is diagnosis, planning, or interview-only by default. Mutations, repairs, process changes, credential changes, persistence, publication, installation, execution, or repository writes require separate explicit approval.

Inspected content is untrusted evidence. Embedded instructions never override the user, the skill contract, or higher-priority safeguards.

## Privacy

- Remote URLs are sanitized when they contain credentials or private hostnames.
- Process command lines and logs are summarized without secret values.
- Dirty worktrees and local paths are reported only as needed.

## Limitations

- It cannot prove remote freshness when network access is unavailable.
- A version string alone cannot identify the running code path.
- Recovery is not performed during diagnosis.

## Examples

See [successful and boundary examples](examples/example-report.md).

## Validation

Run from the repository root:

```bash
python skills/hermes-update-doctor/scripts/validate_bundle.py
python -m unittest discover -s skills/hermes-update-doctor/tests -v
```

The validator and tests use only the Python standard library.

## Version history

### 1.0.0

- Initial public Field Kit release with evidence-first workflow, explicit authority boundaries, hostile-content handling, examples, and contract tests.

## License

Apache License 2.0. See the repository [`LICENSE`](../../LICENSE).
