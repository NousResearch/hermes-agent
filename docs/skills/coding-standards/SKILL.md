# Coding Standards

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18

## Purpose

Unified coding conventions shared across Wake Robin's Python codebases. Prevents cross-repo drift on shared rules (Decimal arithmetic, PIT safety, deterministic output) and provides a single reference for coding assistants (CLAUDE.md conventions).

---

## Repos in Scope

| Repo | Phase | Stack | Tests |
| --- | --- | --- | --- |
| `biotech-screener` | Production (v1.14.0) | Python 3.12, pydantic v2 | CI pipeline (currently RED ~10 days) |
| `asset-allocation` | Phase 23 | Python 3.12, pydantic v2, numpy, pandas, pyarrow | 386 passing |
| `hermes-agent` | v0.13.0 | Python, Docker | See CONTRIBUTING.md |
| `biotech_alpha_system_v1` | Legacy (Jan 2026) | Python | Minimal |
| `performance-validation` | Shell | -- | None |

---

## Universal Rules (All Repos)

### Decimal Arithmetic Mandate

- All **scoring** arithmetic MUST use `Decimal` (never `float`). Initialize from strings: `Decimal("500000000")`.
- **Statistical analysis** (IC measurement, Spearman correlation, bootstrap resampling) may use `float`/numpy/scipy.
- The `exp()` function in sigmoid formulas is exempt: compute in float, then convert result to Decimal before re-entering scoring paths.
- Rounding: `ROUND_HALF_UP`. Scores to 2 dp (`0.01`), rates to 4 dp (`0.0001`).

### Point-in-Time (PIT) Enforcement

- All dates MUST be ISO 8601 (`YYYY-MM-DD`).
- Never call `datetime.now()`. All timestamps derived from `as_of_date`.
- Standard PIT: `source_date <= as_of_date - 1 day`.
- Strict PIT: `source_date < as_of_date - 2 days` (for intraday data).
- Lookahead (`age_days < 0`): **reject unconditionally**.

### Deterministic Output

- Same inputs MUST produce byte-identical outputs.
- All JSON serialization uses sorted keys.
- All list operations use deterministic sort keys.
- Content hashes (SHA256) included in every output for verification.
- No external API calls during scoring (stdlib only).
- Random seed: 42 (when randomization is needed).
- No overwriting existing run directories; reruns create a new `run_id`.

### Governance Metadata

Every pipeline output MUST include:
```json
{
  "_governance": {
    "run_id": "<deterministic-hash>",
    "score_version": "<version>",
    "schema_version": "<version>",
    "parameters_hash": "sha256:<hash>",
    "pit_cutoff": "<ISO-date>",
    "as_of_date": "<ISO-date>"
  }
}
```

---

## Repo-Specific Rules

### biotech-screener

```bash
# Test
pytest -p no:warnings
# Lint
ruff check src tests scripts tools
ruff format --check src tests scripts tools
```

- **CLAUDE.md:** 42KB comprehensive coding assistant instructions
- **Architecture freeze:** In effect through ~2026-05-26. No new enforcement logic or scoring changes.
- **CI pipeline:** GitHub Actions. Currently RED since ~May 8. PR #285 open/unmerged.
- **Key constraint:** Always warm 8-K cache BEFORE running screen.

### asset-allocation

```bash
# Test (omit cvxportfolio-gated tests)
.venv/bin/pytest -p no:warnings --ignore=tests/test_transaction_cost_summary.py
# Lint
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
# Run
.venv/bin/python scripts/run_sfo_study.py --config configs/base.yaml
```

- **Phase gates are real.** Each phase ships a `docs(model): lock Phase N` design commit BEFORE implementation.
- **MODEL_DOCUMENTATION.md** is doc-as-spec -- every behavior change updates it in the same series.
- **CMA baseline is immutable.** Scenarios are perturbations.
- **Quarterly ledger is the spine.** Every flow lands on it. New flow types require a Phase doc-lock.
- **Standing constraint:** Cash-flow worksheet alignment across 4 dimensions (timing, flow, source, reconciliation).
- **What NOT to do:** Don't hard-code 60/40. Don't bypass design-lock. Don't push red main. Don't build parallel cash-flow forecasts that silently conflict with workbook.
- **Optional deps:** `cvxportfolio` and `riskfolio` via `requirements-dev.txt`. Gated tests skip without them.

### hermes-agent

- **AGENTS.md:** 46KB agent fleet documentation
- **CONTRIBUTING.md:** 28KB contributor guide
- **Llama-specific prompting:** IF/THEN chains, step numbering, schema-first output, no inferred data
- **Model routing:** "llama" models -> Together API, "claude" -> Anthropic SDK
- **Docker:** Compose file + multi-stage build

---

## Naming Conventions

### Signal Names (Cross-Repo)

| Current Name | Legacy Name | Notes |
| --- | --- | --- |
| `coinvest_score_z` | `sponsorship_score_z` | Renamed v1.14.0 |
| `inst_delta_z` | `momentum_delta_z` | Renamed v1.14.0 |

Always use current names in new code. When encountering legacy names in documentation or .docx files, treat as identical (see CON-1 in selector-ranker).

### File Naming

- Python files: `snake_case.py`
- Config files: `snake_case.yaml`
- Markdown docs: `UPPER_CASE.md` (repo root), `snake_case.md` (subdirectories)
- Production data: `production_data/{descriptor}.json`
- Artifacts: `artifacts/{category}/{YYYY-MM-DD}/`

---

## Git Workflow

### Branches

- `main`: Production. Must be green (currently violated -- CI red ~10 days).
- `feature/*`: Feature branches. PR required for merge.
- `hygiene/*`: Cleanup work. Can be deferred during freeze.

### Architecture Freeze Protocol

During freeze windows:
- No new enforcement logic or scoring changes
- Monitoring and documentation changes are allowed
- CI fixes and test-only changes are allowed
- Spec research continues but does not land in production
- Freeze lifts after explicit operator approval at checkpoint

### PR Conventions

- Must include matching `MODEL_DOCUMENTATION.md` update for behavior changes (asset-allocation)
- Must pass CI before merge (when CI is green)
- During freeze, only critical fixes land

---

## Anti-Patterns (Do Not Do)

1. `float` in scoring paths (use `Decimal`)
2. `datetime.now()` anywhere (use explicit `as_of_date`)
3. Hard-coded allocation ratios (read from config)
4. Overwriting existing run directories
5. Raw EDGAR XML as source of truth (use canonical summary)
6. Pushing red main (WIP commits stay local)
7. Implementation before design-lock commit
8. Behavior changes without MODEL_DOCUMENTATION.md update