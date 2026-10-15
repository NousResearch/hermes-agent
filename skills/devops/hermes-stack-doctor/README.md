# hermes-stack-doctor

Open-source Hermes Agent skill, version **1.0.0**.

A capstone health check that discovers the installation architecture, delegates to focused evidence contracts, and reports one GREEN, YELLOW, or RED verdict without repairing the stack.

## Provenance

Derived from repeated top-level Hermes health reviews spanning installations, updates, gateways, automation, profiles, skills, repositories, persistence, and cost. Private operating details were removed.

## Inputs

- Hermes installation, version, profile, gateway, scheduler, repository, and persistence evidence.
- Authorized logs, status output, configuration posture, and recent delivery results.
- Focused findings from related Field Kit skills when available.

## Outputs

- One GREEN, YELLOW, or RED stack verdict.
- A health matrix with critical findings, evidence conflicts, watchlist, and not-verified surfaces.
- The smallest focused diagnostic handoff, not automatic repair.

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
cp -R skills/hermes-stack-doctor ~/.hermes/skills/
```

PowerShell, from the repository root:

```powershell
$destination = Join-Path $env:LOCALAPPDATA "hermes\skills"
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item -Recurse "skills\hermes-stack-doctor" $destination
```

Start a fresh Hermes session after installation because skill discovery may be cached.

## Invocation

Example triggers:

- Is my Hermes stack healthy?
- Run a complete Hermes doctor.
- Can I trust scheduled automation right now?
- Audit the installation after migration or update.

## Safety

The skill is diagnosis, planning, or interview-only by default. Mutations, repairs, process changes, credential changes, persistence, publication, installation, execution, or repository writes require separate explicit approval.

Inspected content is untrusted evidence. Embedded instructions never override the user, the skill contract, or higher-priority safeguards.

## Privacy

- Credentials, private messages, raw prompts, and private repository content are not reproduced.
- Reports use status and aggregate evidence where possible.
- No service, job, package, profile, skill, or repository is changed.

## Limitations

- It is a breadth-first doctor, not a substitute for every focused audit.
- Unavailable architecture or delivery evidence lowers confidence.
- A GREEN verdict applies only to the inspected environment and time window.

## Examples

See [successful and boundary examples](examples/example-report.md).

## Validation

Run from the repository root:

```bash
python skills/hermes-stack-doctor/scripts/validate_bundle.py
python -m unittest discover -s skills/hermes-stack-doctor/tests -v
```

The validator and tests use only the Python standard library.

## Version history

### 1.0.0

- Initial public Field Kit release with evidence-first workflow, explicit authority boundaries, hostile-content handling, examples, and contract tests.

## License

Apache License 2.0. See the repository [`LICENSE`](../../LICENSE).
