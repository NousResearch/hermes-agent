# Skill Intake Pipeline Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a governed intake pipeline for third-party and legacy skills with explicit stages for audit, convert, re-audit, and admit.

**Architecture:** Implement a single shared engine with separate operational stages. Keep compatibility, security, semantic portability, and admission policy as distinct concerns so conversion never implies trust. Start with V1 operational skill + V2 audit-first tool, then layer convert, re-audit, and admit.

**Tech Stack:** Python 3, Hermes tool registry, pytest, JSON schema validation, Markdown/YAML parsing.

---

## Product decision

Recommended product name: `skill-intake`

Top-level commands:
- `skill-intake audit SOURCE`
- `skill-intake convert SOURCE --out DIR`
- `skill-intake re-audit DIR`
- `skill-intake admit DIR --policy conservative`

Core rule:
- audit -> convert -> re-audit -> admit
- conversion never implies approval
- admission requires explicit policy gate

## Scope

Supported inputs:
- local file
- local directory
- raw URL
- GitHub file URL
- GitHub tree URL
- GitHub repo URL
- native Hermes skill
- Agent Skills-compatible package
- `CLAUDE.md`
- OpenClaw-like skill package
- procedural markdown / README-style workflow docs

Explicit non-goals for first shipping slice:
- auto-install after conversion
- auto-publish to registries
- executing third-party scripts during routine audit
- trust-by-origin shortcuts

## Deliverables

1. V1 operational skill installed in local Hermes profile
2. V2 technical spec and JSON schemas
3. Policy matrix and governance docs
4. Future code skeleton for Hermes integration

---

## Target file layout in Hermes repo

```text
hermes-agent/
├── .plans/
│   ├── skill-intake-pipeline.md
│   └── skill-intake/
│       ├── output-schema.json
│       ├── policy-matrix.md
│       ├── sample-audit-result.json
│       ├── sample-convert-result.json
│       ├── sample-reaudit-result.json
│       └── sample-admit-result.json
├── tools/
│   └── skill_intake_tool.py              # future V2 tool implementation
├── tests/
│   └── tools/
│       └── test_skill_intake_tool.py     # future tests
```

---

## Data model

### Source model
- `source`
- `source_type`
- `resolved_at`
- `provenance`
  - `git_owner`
  - `git_repo`
  - `git_ref`
  - `git_commit`
  - `content_hashes`

### Discovery model
- `entrypoints_found`
- `skill_candidates`
- `manifest`

### Compatibility report
- `score` 0-100
- `classification` A/B/C/D
- `checks[]`

### Security report
- `risk_score` 0-100
- `risk_level` info/low/medium/high/critical
- `findings[]`

### Semantic report
- `score` 0-100
- `findings[]`

### Quality report
- `score` 0-100
- `eval_plan` optional

### Decision model
- `recommendation.action` install/adapt/quarantine/reject
- `admission.status` approved/approved_with_warning/review_required/rejected

---

## Rule packs

### Compatibility
Pass checks:
- `SKILL.md` exists
- frontmatter YAML exists
- `name` present and valid
- `description` present
- directory structure reasonable

Warn checks:
- metadata missing
- compatibility field missing where needed
- monolithic docs requiring restructuring

Fail checks:
- no procedural entrypoint
- invalid or absent frontmatter
- invalid skill slug
- not skill-like

### Semantic portability
Checks:
- clear usage trigger
- repeatable procedure
- verification steps present
- pitfalls present or inferable
- temporary context absent
- repo coupling not excessive
- reusable imperative language present

### Security / supply-chain
Checks:
- `curl|bash` or equivalent
- remote binary execution
- unpinned dependencies
- no checksum on downloads
- privilege escalation (`sudo`)
- credential store access
- keychain / `.env` / SSH / cloud credential reads
- hidden telemetry or webhooks
- prompt injection / safety bypass language
- obfuscated shell or base64 payloads
- broad recursive delete or traversal patterns

---

## Policy profiles

### conservative
- critical => rejected
- high => rejected
- medium => review_required
- low => approved_with_warning
- info => approved

### balanced
- critical => rejected
- high => review_required
- medium => approved_with_warning
- low => approved

### permissive
- critical => review_required
- high => approved_with_warning
- medium => approved
- low => approved

Recommendation: default to `conservative`.

---

## Implementation phases

### Phase 0: V1 operational skill
**Objective:** Standardize manual/semiautomatic intake before building the native tool.

**Files:**
- Create: `~/.hermes/skills/autonomous-ai-agents/hermes-skill-intake-audit/SKILL.md`
- Optional later: references for policy matrix/schema

**Acceptance criteria:**
- Skill exists and loads via `skills_list`/`skill_view`
- Instructions explicitly separate audit, convert, re-audit, admit
- Output labels fact/inference/risk/recommendation are defined

### Phase 1: Spec artifacts
**Objective:** Persist design documents in repo for future implementation.

**Files:**
- Create: `~/.hermes/hermes-agent/.plans/skill-intake/output-schema.json`
- Create: `~/.hermes/hermes-agent/.plans/skill-intake/policy-matrix.md`
- Create: sample result JSON files under the same directory

**Acceptance criteria:**
- Files exist and are internally consistent
- Sample outputs validate conceptually against schema

### Phase 2: V2 audit tool
**Objective:** Implement read-only intake analysis as the first native tool.

**Files:**
- Create: `tools/skill_intake_tool.py`
- Modify: `toolsets.py`
- Create: `tests/tools/test_skill_intake_tool.py`

**Acceptance criteria:**
- Supports `mode=audit`
- Handles local file, local dir, and GitHub URL
- Returns structured JSON with compatibility, security, semantic sections

### Phase 3: Conversion
**Objective:** Convert CLAUDE.md / generic markdown / OpenClaw-like packages to Hermes draft format.

**Files:**
- Modify: `tools/skill_intake_tool.py`
- Add helper modules if code size warrants refactor
- Extend tests with conversion fixtures

**Acceptance criteria:**
- Supports `mode=convert`
- Produces `SKILL.md` draft plus report
- Preserves provenance and discarded-content log

### Phase 4: Re-audit and admit
**Objective:** Close the governance loop.

**Acceptance criteria:**
- Re-audit consumes converted directory
- Admit applies policy profile
- Installation remains an explicit later step

### Phase 5: Eval planning / optional sandbox
**Objective:** Measure whether admitted skills improve real execution safely.

**Acceptance criteria:**
- Generates eval skeletons
- Can compare with-skill vs without-skill later

---

## Test strategy

Fixtures to include:
- native Hermes skill
- valid Agent Skills skill
- `CLAUDE.md` workflow
- OpenClaw-like package
- docs-only candidate
- malicious candidate with `curl|bash`
- candidate with secret-exfil instructions

Core tests:
- audit local file
- audit local directory
- audit GitHub repo URL
- convert CLAUDE.md
- convert docs-only candidate rejected or downgraded appropriately
- re-audit converted output
- admit decision per policy profile

---

## Open product decisions

1. Should converted drafts ever be auto-installed? Recommendation: no.
2. Should trust policies consider repo owner/reputation? Not in V2 baseline.
3. Should sandboxed script execution be part of audit? Not by default.
4. Should quality score be heuristic-only in V2? Yes.

---

## Recommended next implementation step

After this plan is saved, implement Phase 0 and Phase 1 first. Do not start with conversion code before the audit model and policy artifacts are stable.
