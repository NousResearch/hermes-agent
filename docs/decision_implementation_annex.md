# Decision Implementation Annex

## Purpose

The **implementation annex** (or execution annex) bridges the gap between the
decision pipeline's analytical output and real-world execution by a human
stakeholder (e.g., a parent, teacher, or clinician).  Where the final decision
report uses internal evidence-tier labels, structured uncertainty, and
contract-level language, the annex translates those findings into actionable
guidance while preserving the original meaning and caveats.

## Phase 1 Scope

Phase 1 supports a single domain: `child_adhd_education` (ADHD-I family
execution guidance).

| Feature | Phase 1 | Future |
|---------|---------|--------|
| Domain profiles | `child_adhd_education` only | Extensible YAML profiles |
| Generation approach | Template-assisted, deterministic | Adaptive / LLM-assisted |
| Provenance | Block-level | Line-level trace |
| Quality gate | Independent CLI/tool | Integrated pipeline step |
| Automatic pipeline integration | No | Optional |

## Architecture

```
final_report.md ─┐
external_calibration.md ─┤
contract.md ─────────────┤
domain profile ──────────┤
                         v
              implementation_annex_generator.py
                         │
                         ├── execution_annex.md
                         └── manifest.json
                              │
                              v
              implementation_annex_quality_gate.py
                         │
                         └── structured verdict
```

### Components

1. **`tools/implementation_annex_generator.py`**
   - `generate_annex()` — main entry point
   - Reads inputs, applies domain profile, produces annex + manifest
   - Optionally runs quality gate after generation

2. **`tools/implementation_annex_quality_gate.py`**
   - `run_quality_gate()` — standalone quality checker
   - CLI: `python -m tools.implementation_annex_quality_gate --annex ...`
   - Checks: no U+FFFD, no internal terms, no drug names, no treatment
     instructions, Grade 3 forward-only, overcommit guards, required sections

3. **`config/domain_safety_profiles.yaml`**
   - Per-domain profiles with forbidden terms, overcommit guards, required
     sections, evidence-tier mappings

## Annex Structure

Every execution annex contains:

1. **目的与依据** — Purpose, basis, evidence tier legend
2. **核心建议** — Core recommendations with evidence levels
3. **分角色行动项** — Action items by stakeholder group
4. **时间线与里程碑** — Timeline and milestones
5. **注意事项与边界** — Caveats and boundaries (original meaning preserved)
6. **监测指标** — Monitoring indicators and warning signs

## Quality Gate Checks

| Check | What it verifies |
|-------|------------------|
| `no_ufffd` | No U+FFFD replacement characters |
| `no_internal_terms` | No leaked `evidence_supported`, `plausible_inference`, etc. |
| `no_drug_names` | No medication names or dosages |
| `no_treatment_instructions` | No medical treatment directives |
| `grade3_forward_only` | Grade 3 content remains forward-looking hypothesis |
| `overcommit_guards` | No over-promises on BPT, CLAS, DRC, school support, exercise |
| `required_sections` | All required sections present |

## Development

### Adding a domain profile

1. Add a profile block to `config/domain_safety_profiles.yaml`
2. Create fixture files under `tests/fixtures/`
3. Add domain-specific checks to the quality gate (if needed)
4. Add tests

### Running tests

```bash
cd /path/to/hermes-agent-research-decision
python3 -m pytest tests/test_implementation_annex_generator.py -v
```

### Running the generator manually

```bash
python3 -m tools.implementation_annex_generator \
    --final-report tests/fixtures/adhd_i_decision_report.md \
    --external-calibration tests/fixtures/adhd_i_external_calibration.md \
    --contract tests/fixtures/adhd_i_decision_report.md \
    --output-dir /tmp/annex-out \
    --domain child_adhd_education
```
