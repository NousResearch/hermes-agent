# PDF → Knowledge → Skill Update Pipeline Runbook

## Purpose

Turn incoming reference PDFs into:
1. extracted/indexed knowledge artifacts,
2. distilled modular playbooks,
3. diff-ready skill update proposals,
4. gated/manual apply + post-change evaluation.

Pipeline entrypoint:
- `scripts/pdf_skill_pipeline.py`

Outputs:
- `memory/pdf_skill_pipeline/runs/<run_id>/...`
- `memory/pdf_skill_pipeline/playbooks/<run_id>/...`
- `memory/pdf_skill_pipeline/proposals/<run_id>/...`

---

## Stage Commands (one command per stage)

> Recommended Python (has `pypdf`): `./.venv_tools/bin/python`

Set run id once per batch:

```bash
RUN_ID="twitter_pack_2026-02-12_seed"
```

### 1) ingest (detect + sha256 dedupe)

```bash
./.venv_tools/bin/python scripts/pdf_skill_pipeline.py ingest \
  --run-id "$RUN_ID" \
  --manifest-json memory/inbound_pdfs/twitter_digest_research_pack_2026-02-12_MANIFEST.json
```

Alternative (directory scan):

```bash
./.venv_tools/bin/python scripts/pdf_skill_pipeline.py ingest \
  --run-id "$RUN_ID" \
  --inbound-dir /home/yeqiuqiu/.openclaw/media/inbound
```

### 2) extract (text)

```bash
./.venv_tools/bin/python scripts/pdf_skill_pipeline.py extract --run-id "$RUN_ID"
```

### 3) index/manifest

```bash
./.venv_tools/bin/python scripts/pdf_skill_pipeline.py index --run-id "$RUN_ID"
```

### 4) distill (modular playbooks)

```bash
./.venv_tools/bin/python scripts/pdf_skill_pipeline.py distill --run-id "$RUN_ID"
```

### 5) propose updates (diff-ready patches)

```bash
./.venv_tools/bin/python scripts/pdf_skill_pipeline.py propose --run-id "$RUN_ID"
```

### 6) approve/apply (manual gate)

Create approval template:

```bash
./.venv_tools/bin/python scripts/pdf_skill_pipeline.py approve --run-id "$RUN_ID"
```

Edit:
- `memory/pdf_skill_pipeline/runs/$RUN_ID/approve/APPROVAL.md`
- set `Approved: YES` only after review.

Apply after approval:

```bash
./.venv_tools/bin/python scripts/pdf_skill_pipeline.py approve \
  --run-id "$RUN_ID" \
  --apply \
  --decision-file memory/pdf_skill_pipeline/runs/$RUN_ID/approve/APPROVAL.md
```

### 7) evaluate (post-change quality checks)

```bash
./.venv_tools/bin/python scripts/pdf_skill_pipeline.py evaluate --run-id "$RUN_ID"
```

---

## Daily Operator Flow

1. **Ingest** latest inbound batch (`ingest`).
2. **Check dedupe report** in `runs/<run_id>/ingest/ingest_manifest.md`.
3. **Extract + index** (`extract`, `index`).
4. **Distill** new modular playbooks (`distill`).
5. **Review proposal plan + patches** (`propose`):
   - `memory/pdf_skill_pipeline/proposals/<run_id>/skill_update_plan.md`
   - `memory/pdf_skill_pipeline/proposals/<run_id>/patches/*.patch`
6. **Manual approval gate**:
   - fill `APPROVAL.md`,
   - apply only if approved.
7. **Evaluate** quality gates (`evaluate`), confirm:
   - citation sections exist,
   - contradiction flag list exists,
   - sha256 dedupe passes,
   - required six playbooks generated.

---

## Safety/Quality Gates Checklist

- [x] sha256 dedupe at ingest/index
- [x] source citation retention in manifests/playbooks
- [x] contradiction flag list generated (`CONTRADICTION_FLAGS.md`)
- [x] rollback notes generated before/after apply

Rollback locations:
- Proposal notes: `memory/pdf_skill_pipeline/proposals/<run_id>/rollback_notes.md`
- Apply backups: `memory/pdf_skill_pipeline/runs/<run_id>/approve/rollback/`
- Apply rollback commands: `memory/pdf_skill_pipeline/runs/<run_id>/approve/rollback_instructions.md`
