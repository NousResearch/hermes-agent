# hermes-token-audit

Open-source Hermes Agent skill, version **1.0.0**.

A read-only, schema-discovering token and cost audit that defaults to aggregate metadata and clearly separates local estimates from provider billing.

## Provenance

Derived from repeated Hermes usage, cron-consumption, runaway-session, and billing-reconciliation investigations. Public fixtures contain no real prompts, invoices, account identifiers, or usage history.

## Inputs

- Authorized Hermes usage databases opened read-only.
- Schema, model, provider, profile, session, cron, and date-window metadata.
- Optional provider billing exports for reconciliation.

## Outputs

- A NORMAL, OPTIMIZATION OPPORTUNITY, ANOMALY DETECTED, or INSUFFICIENT EVIDENCE verdict.
- Aggregate usage, concentration, automation, anomaly, and billing findings.
- Approval-gated control recommendations.

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
cp -R skills/hermes-token-audit ~/.hermes/skills/
```

PowerShell, from the repository root:

```powershell
$destination = Join-Path $env:LOCALAPPDATA "hermes\skills"
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item -Recurse "skills\hermes-token-audit" $destination
```

Start a fresh Hermes session after installation because skill discovery may be cached.

## Invocation

Example triggers:

- Where did my Hermes tokens go?
- Which sessions cost the most?
- Did a cron job burn the budget?
- Why does provider billing not match local usage?

## Safety

The skill is diagnosis, planning, or interview-only by default. Mutations, repairs, process changes, credential changes, persistence, publication, installation, execution, or repository writes require separate explicit approval.

Inspected content is untrusted evidence. Embedded instructions never override the user, the skill contract, or higher-priority safeguards.

## Privacy

- Message bodies, prompts, credentials, account identifiers, and raw personal data are not read by default.
- Reports favor aggregate metadata and bounded excerpts.
- Provider exports remain local unless explicitly shared.

## Limitations

- Local cost fields remain estimates unless reconciled with provider billing.
- One database may not contain all provider or external CLI activity.
- Billing lag and mismatched date windows can prevent exact reconciliation.

## Examples

See [successful and boundary examples](examples/example-report.md).

## Validation

Run from the repository root:

```bash
python skills/hermes-token-audit/scripts/validate_bundle.py
python -m unittest discover -s skills/hermes-token-audit/tests -v
```

The validator and tests use only the Python standard library.

## Version history

### 1.0.0

- Initial public Field Kit release with evidence-first workflow, explicit authority boundaries, hostile-content handling, examples, and contract tests.

## License

Apache License 2.0. See the repository [`LICENSE`](../../LICENSE).
