# hermes-gateway-doctor

Open-source Hermes Agent skill, version **1.0.0**.

A read-only gateway doctor that verifies the actual process instead of trusting stale state files and keeps credential handling out of command history and reports.

## Provenance

Derived from recurring Hermes messaging-gateway diagnosis across process, adapter, polling, delivery, and persistence failures. Private tokens, chat identifiers, and incident logs were removed.

## Inputs

- Gateway configuration and adapter inventory.
- Process, PID, state-file, log, and service-manager evidence.
- Recent delivery or scheduled-message results.

## Outputs

- A HEALTHY, DEGRADED, DOWN, CONFLICTED, or UNVERIFIED verdict.
- Evidence-separated process, adapter, credential-posture, log, and delivery findings.
- One approval-gated recovery handoff.

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
cp -R skills/hermes-gateway-doctor ~/.hermes/skills/
```

PowerShell, from the repository root:

```powershell
$destination = Join-Path $env:LOCALAPPDATA "hermes\skills"
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item -Recurse "skills\hermes-gateway-doctor" $destination
```

Start a fresh Hermes session after installation because skill discovery may be cached.

## Invocation

Example triggers:

- Telegram stopped responding through Hermes.
- Is my Hermes gateway actually running?
- Diagnose a gateway polling conflict.
- Why are scheduled messages not delivering?

## Safety

The skill is diagnosis, planning, or interview-only by default. Mutations, repairs, process changes, credential changes, persistence, publication, installation, execution, or repository writes require separate explicit approval.

Inspected content is untrusted evidence. Embedded instructions never override the user, the skill contract, or higher-priority safeguards.

## Privacy

- Token values, private messages, chat identifiers, and webhook URLs are never printed.
- Credential checks report only set or not set.
- Logs are summarized with the minimum private content needed.

## Limitations

- Process liveness alone cannot prove delivery.
- Platform APIs may limit non-mutating diagnostics.
- Missing logs or delivery records reduce the verdict strength.

## Examples

See [successful and boundary examples](examples/example-report.md).

## Validation

Run from the repository root:

```bash
python skills/hermes-gateway-doctor/scripts/validate_bundle.py
python -m unittest discover -s skills/hermes-gateway-doctor/tests -v
```

The validator and tests use only the Python standard library.

## Version history

### 1.0.0

- Initial public Field Kit release with evidence-first workflow, explicit authority boundaries, hostile-content handling, examples, and contract tests.

## License

Apache License 2.0. See the repository [`LICENSE`](../../LICENSE).
