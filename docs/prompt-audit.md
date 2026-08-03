# Prompt audit CLI

`scripts/prompt_audit.py` is a read-only, deterministic AISPA-based audit for the default Hermes profile and every named profile beneath `<root>/profiles/`.

It resolves each profile's effective identity prompt from `SOUL.md` plus `agent.system_prompt`, records explicit fallback and config errors, and emits concise redacted findings. The bundled `scripts/prompt_audit_rubric.json` is the stable 17-rule contract; the MVP implements deterministic checks for required sections, contradiction candidates, stale dated claims, baseline drift, unsafe privilege/tool/privacy/injection patterns, silent identity fallback, and duplicate cross-profile prompts. Semantic-only rubric checks remain outside the MVP and are not represented as passes.

## Audit

```bash
python3 scripts/prompt_audit.py --root ~/.hermes
python3 scripts/prompt_audit.py --root ~/.hermes --format json --output /tmp/prompt-audit.json
python3 scripts/prompt_audit.py --root ~/.hermes --profile octacon --severity high --severity critical
```

`--fail-on low|medium|high|critical` sets the CI threshold. The default is `high`; `--fail-on off` always exits zero. Findings at or above the threshold exit `1`. Invalid usage or an attempted baseline overwrite exits `2`.

JSON is sorted and contains:

- profile inventory and effective-prompt hashes;
- rule ID, severity, concise evidence, source location, and remediation;
- checks run/not run, weighted score, and verdict;
- no complete prompt content.

## Explicit baselines

Baseline creation is a separate command and requires an approver:

```bash
python3 scripts/prompt_audit.py \
  --root ~/.hermes \
  --create-baseline governance/prompt-baseline.json \
  --approved-by sahil
```

The command refuses to overwrite an existing file. Review and commit the new JSON explicitly. To replace a baseline, create a new path or deliberately remove/rename the old file through the governed review process; the audit command never updates snapshots.

Compare without mutation:

```bash
python3 scripts/prompt_audit.py \
  --root ~/.hermes \
  --baseline governance/prompt-baseline.json \
  --format json
```

Baselines contain hashes and source provenance, not full prompt text or secrets.
