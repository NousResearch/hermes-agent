# Skill Packaging Standard Contract v1 (`SYS-04` bounded foundation)

Date: 2026-04-02  
Status: active (bounded foundation)  
Owner: Architect  
Scope: governed skill packaging only (no marketplace/runtime auto-loading in v1)

---

## 1) Purpose

Define a fail-closed package contract for OpenClaw skills so reusable capabilities are:

1. **discoverable** (standard metadata),
2. **composable** (consistent package structure),
3. **governable** (risk/tool/approval metadata), and
4. **portable** (explicit interoperability/versioning metadata).

This v1 slice is intentionally narrow: schema/template/validator/scaffolder + one migrated skill package.

---

## 2) Donor patterns folded in

- `claude-skills` / `everything-claude-code`: `SKILL.md` + `scripts/` + `references/` as standard package envelope.
- `PaddleOCR`: explicit runtime preflight requirements + versioned package metadata.
- OpenClaw doctrine baseline: advisory-first risk/governance envelope and explicit operator approval semantics.

Authority sources:
- `reports/openclaw_external_findings_foldin_program_2026-04-02.md`
- `reports/deep_dive_claudecode_skills_deepagents_openclaw_foldin_2026-04-02.md`
- `reports/deep_dive_paddleocr_openclaw_foldin_2026-04-02.md`

---

## 3) Required package structure (v1)

Each governed skill package must include:

- `SKILL.md` (with YAML frontmatter)
- `skill.package.json` (manifest)
- `scripts/` (tooling/runtime helpers)
- `references/` (docs, benchmark plans, citations)

Frontmatter and manifest must agree on at least:

- `skill_id`
- `version`
- `display_name`
- `risk_class`
- `execution_mode`

---

## 4) Required manifest schema

Normative schema:
- `docs/ops/schemas/skill_package_manifest.schema.json`

Starter template:
- `docs/ops/templates/skill_package_manifest.template.json`

Pack manifest:
- `ops/openclaw/architecture/skill_packaging_schema_pack.v1.json`

Key required manifest groups:

1. **package** (paths + assets)
2. **activation** (intents/keywords/priority)
3. **runtime** (execution mode, offload policy, preflight)
4. **interoperability** (import/export formats)
5. **quality** (benchmark + contract tests)
6. **governance** (risk class, allowed tools, approval policy)
7. **provenance** (source refs + confidence)

Fail-closed rule: missing/unknown required fields = invalid package.

---

## 5) Validation entrypoints

- `pytest -q tests/test_sys_04_skill_packaging_schema_pack.py`
- `bash ops/openclaw/architecture/validate_skill_packaging_schema_pack.sh --json`
- `python scripts/skill_package_scaffold.py --help`

---

## 6) Boundaries / non-goals (v1)

Not included in this slice:

- automatic runtime skill loading
- public marketplace/distribution flow
- cross-harness conversion runtime
- broad migration of all existing skill-like directories

This slice only establishes the governed packaging substrate for future expansion.

---

## 7) Immediate next-step hooks

After v1 foundation is stable:

1. add registry snapshot surface (`skill discovery index`) to continuity latest artifacts,
2. gate worker dispatch by `runtime.preflight` + `governance.approval_policy`,
3. migrate additional high-value skills to the standard package shape.
