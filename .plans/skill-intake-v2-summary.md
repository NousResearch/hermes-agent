# Skill Intake V2 Summary

## Executive conclusion

`skill_intake` reached a solid V2 baseline in the Hermes repo.

The pipeline now supports:
- `audit`
- `convert`
- `re-audit`
- `admit`

This is no longer a prototype. It is a functional, test-covered intake pipeline for candidate skills.

## What V2 delivers

### 1. Audit
- local file and directory support
- basic remote GitHub/raw URL support
- compatibility scoring/classification
- security heuristic scanning
- semantic portability scoring
- recommendation output
- provenance for local and remote sources

### 2. Convert
- generic markdown / workflow conversion
- project-instructions conversion (`CLAUDE.md`-style)
- OpenClaw-like package preservation for `scripts/` and `references/`
- `discarded` tracking for temporary context
- `mapping` from source to generated artifacts
- optional persistence via `out_dir`

### 3. Re-audit
- revalidation of generated drafts
- readiness states: `ready`, `review`, `blocked`

### 4. Admit
- policy-driven admission gate
- profiles: `conservative`, `balanced`, `permissive`
- decision output with required actions

## Quality improvements completed

### Generic converter
The generic converter now extracts or infers:
- `When to use`
- `Prerequisites`
- `Steps`
- `Commands`
- `Examples`
- `Pitfalls`
- `Verification`

### OpenClaw-like support
Current heuristic:
- directory with `AGENT.md`
- plus `scripts/` or `references/`

Preserved in output:
- scripts
- references
- mapping entries for preserved assets

## Provenance now available

### Local
- `local_path`
- `local_kind`
- `asset_counts` for scripts/references on directories

### Remote
- `source_url`
- `source_type`
- `remote_fetch.fetched`
- `remote_fetch.attempts`
- `remote_fetch.fetched_count`

## Validation status

Current validation baseline:
- `tests/tools/test_skill_intake_tool.py`
- `tests/test_toolsets.py`

Latest checked result during build-out:
- targeted tool tests passing
- toolset regression passing

## What remains after V2

This is post-V2 work, not required to call V2 complete:
- richer remote provenance (repo/ref/commit where resolvable)
- broader OpenClaw-specific fixture coverage
- eval planning / with-skill vs without-skill comparison
- sandbox / CI hardening
- deeper semantic classification of examples vs references

## Recommended next phase

Call the current state `V2 complete`.

Recommended next phase order:
1. integrated tests with more realistic fixtures
2. provenance refinement
3. eval planning
4. sandbox / CI hardening
