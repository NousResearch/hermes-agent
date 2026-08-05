# Prompt audit — operator and contributor guide

`scripts/prompt_audit.py` is a read-only, deterministic AISPA-based audit for the default Hermes profile and every named profile beneath `<root>/profiles/`. This guide is for operators running scans, reviewers interpreting findings, and contributors changing prompts or configs.

It resolves each profile's effective identity prompt from `SOUL.md` plus `agent.system_prompt`, records explicit fallback and config errors, and emits concise redacted findings. The bundled `scripts/prompt_audit_rubric.json` is the stable 17-rule contract; the MVP implements deterministic checks for required sections, contradiction candidates, stale dated claims, baseline drift, unsafe privilege/tool/privacy/injection patterns, silent identity fallback, and duplicate cross-profile prompts. Semantic-only rubric checks remain outside the MVP and are not represented as passes.

## Installation

No install step is required beyond the KenseiAgent repo runtime that already ships PyYAML. Invoke the script directly from the repo root:

```bash
python3 scripts/prompt_audit.py --root ~/.hermes
```

Run `python3 scripts/prompt_audit.py --help` to confirm the flag set. Any environment missing PyYAML is treated as a config error in the audit output rather than a hard crash.

## Invocation

### Fleet-wide scan

```bash
python3 scripts/prompt_audit.py --root ~/.hermes
```

This audits the default profile and every non-hidden directory under `~/.hermes/profiles/`. Discovery order is alphabetical and deterministic.

### Subset scan

```bash
python3 scripts/prompt_audit.py --root ~/.hermes --profile gojo --profile remii
python3 scripts/prompt_audit.py --root ~/.hermes --severity high --severity critical
```

Repeat `--profile` to restrict discovery. Repeat `--severity` to filter emitted findings without changing scoring.

### Machine-readable output

```bash
python3 scripts/prompt_audit.py --root ~/.hermes --format json --output /tmp/prompt-audit.json
```

## Output fields

Top-level keys:

- `schema_version` — report schema major version.
- `rubric.id` / `rubric.version` — stable contract identifiers.
- `root` — resolved audit root.
- `profiles` — array of audited profiles.
  - `profile` — profile name.
  - `home` — inspected directory.
  - `config_path` — path to `config.yaml`.
  - `identity_status` — `applied` or `fallback`.
  - `effective_prompt_sha256` — SHA-256 of the canonical effective prompt.
  - `sources` — prompt sources with `kind`, `location`, `status`, `sha256`, `chars`, and optional `reason`.
  - `config_errors` — parser/config error strings.
- `checks_run` / `checks_not_run` — dependency completeness record.
- `findings` — sorted finding array.
  - `profile` — affected profile.
  - `rule_id` — rubric rule ID, e.g. `KPA-011`.
  - `severity` — `low`, `medium`, `high`, or `critical`.
  - `evidence` — redacted one-line evidence.
  - `source_location` — file path or file path with line references.
  - `remediation` — operator-readable fix.
- `score` — weighted sum of confirmed findings, one entry per profile+rule.
- `verdict` — `pass`, `review`, or `fail`.

## Severity and verdict interpretation

Severity weights:

- `critical` — 25
- `high` — 10
- `medium` — 4
- `low` — 1

Verdict rules:

- `fail` if any critical finding is confirmed, score >= 10, `KPA-011` is `not_run` where specialist routing is expected, or more than two required AISPA dimensions are uncovered.
- `review` if score is 4-9 with no critical finding, any unresolved gray-area candidate remains, or a baseline/cross-profile dependency is `not_run` without an explicit fail condition.
- `pass` if score is 0-3, no critical finding, every mandatory check ran, and AISPA coverage meets the profile risk matrix.

Only `pass` is releasable. `review` requires named human adjudication. `fail` blocks activation or baseline promotion.

## Baseline workflow

### Create a baseline

```bash
python3 scripts/prompt_audit.py \
  --root ~/.hermes \
  --create-baseline governance/prompt-baseline.json \
  --approved-by sahil
```

Requirements and behavior:

- `--approved-by` is mandatory.
- The command refuses to overwrite an existing baseline file.
- Baseline files contain hashes and source provenance, not full prompt text or secrets.

Review the created JSON explicitly before committing it. Baselines are governed artifacts.

### Review baseline diff

```bash
python3 scripts/prompt_audit.py \
  --root ~/.hermes \
  --baseline governance/prompt-baseline.json \
  --format json --output /tmp/prompt-audit.json
```

A `KPA-004` finding means the profile is absent from the baseline or its effective prompt hash differs from the approved baseline. Treat new profiles as requiring explicit baseline capture.

### Replace a baseline

Create a new path or deliberately remove/rename the old file through governed review. The audit command never mutates baselines.

## Suppress a finding with auditable justification

The MVP does not implement inline suppression annotations. To suppress a finding:

1. Record an auditable override in governance with the exact profile, rule ID, finding evidence, justification, approver, and expiry or review date.
2. Re-run the scan with `--severity` filters to confirm the suppressed class is still visible in raw output if required.
3. Do not modify the rubric rule IDs or weights to hide findings.

## CI enforcement

```bash
python3 scripts/prompt_audit.py \
  --root ~/.hermes \
  --profile octacon \
  --baseline governance/prompt-baseline.json \
  --fail-on high
```

Threshold guidance:

- `--fail-on off` — never fails; useful for visibility-only jobs.
- `--fail-on low` — fails on any finding; suitable for release gates.
- `--fail-on medium` — fails on medium and above.
- `--fail-on high` — fails on high and above; the default.
- `--fail-on critical` — fails only on critical findings.

Use report-only mode in CI until the first baseline is approved and drift is understood.

## Example: detecting a silent default-profile fallback

A missing or empty `SOUL.md` produces an explicit `KPA-011` critical finding. Example evidence:

```
[CRITICAL] octacon KPA-011 /home/kensei/.hermes/profiles/octacon/SOUL.md
  Missing or empty SOUL.md caused explicit default identity fallback.
  Remediation: Add a profile-specific SOUL.md or explicitly approve generic Hermes fallback.
```

Config parse errors also surface as `KPA-011`. Run with `--severity critical` to isolate fallback and config failures quickly.

## Example: reviewing prompt drift

When a profile drifts from an approved baseline, the report emits `KPA-004` high findings with baseline and current SHA-256 values. Example evidence:

```
[HIGH] gojo KPA-004 governance/prompt-baseline.json
  baseline=abc123... current=def456...
  Remediation: Restore approved text or review the diff and explicitly capture a new baseline.
```

Review the unified diff between the baseline and current effective prompt before approving a new baseline.

## Low-risk rollout for 50+ profiles

Recommended sequence:

1. Report-only scan across the full fleet without a baseline. Establish finding density, severity distribution, and baseline candidates.
2. Baseline review. Capture an approved baseline for every active profile in one governed change.
3. Remediation window. Fix fallbacks, missing sections, stale claims, and duplicate identities before enabling enforcement.
4. CI enforcement. Start with `--fail-on high`; raise to `medium` or `low` only after drift is stable.

Avoid enabling `--fail-on critical` on a fleet that still has unresolved fallback or config errors; that will fail every affected profile until the root cause is fixed.
