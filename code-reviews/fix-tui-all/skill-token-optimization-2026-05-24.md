# Skill Compression & Token Optimization — Code Review Panel Verdict

**Date:** 2026-05-24  
**Branch:** `fix/tui-all` vs `main`  
**Reviewers:** Adversarial (DeepSeek V4 Flash), Neutral (DeepSeek V4 Flash)  
**Score:** 6/10 (after CRITICAL fixes; was 4/10 pre-fix)

---

## CRITICAL — Must fix before deploy

### C1 ✅ FIXED — Double `bump_use` on stow
**File:** `tools/skill_manager_tool.py:766-770`  
**Root cause:** `stow` handler had inline `bump_use()` call AND telemetry block also bumps `bump_use()`. Every other action only bumps in the telemetry block.  
**Impact:** Usage counters inflated 2×, corrupting curator auto-archive decisions.  
**Fix:** Removed inline bump. Telemetry block handles it once.

### C2 ✅ FIXED — Non-additive include/categories_include filters
**File:** `agent/prompt_builder.py:1077-1096`  
**Root cause:** `include_set` and `cats_include_set` were sequential AND gates. A skill matching `cats_include_set` but NOT `include_set` silently failed.  
**Impact:** Users setting both directives would see fewer skills than expected.  
**Fix:** Rewrote filter with OR semantics — positive filters match if name/category hits EITHER set. Negative filters remain subtractive.

---

## HIGH — Should fix soon

### H1 — Brittle context file parsing
**File:** `agent/prompt_builder.py:1421-1447`  
**Issue:** Regex only matches `skills.include: [a, b]` bracket syntax. Multi-line YAML, single-item, and template formats silently fail with no warning.  
**Fix:** Add a proper YAML frontmatter parser for the skill config block, or at minimum log a WARNING on unrecognized patterns.

### H2 — config.yaml fallback skipped on partial context match  
**File:** `agent/prompt_builder.py:1452`  
**Issue:** `if not result:` gate prevents config.yaml merge when context file has ANY directive. A `.hermes.md` with `skills.include: [github]` would lose `skills.project.exclude` from config.yaml.  
**Fix:** Always load config.yaml first, then overlay context file values.

### H3 — stow doesn't validate skill exists
**File:** `tools/skill_manager_tool.py:762-781`  
**Issue:** All other actions call `_find_skill(name)` first. `stow` calls `bump_use()` directly with no existence check. Returns `success: True` for non-existent skills.  
**Fix:** Add `if not _find_skill(name): return tool_error(...)`.

### H4 — Case-sensitive external skill dedup
**File:** `agent/prompt_builder.py:1178,1191,1203`  
**Issue:** `seen_skill_names` uses original-case values but filter matching uses `.lower()`. Local skill "Git" and external "git" could both appear.  
**Fix:** Normalize `seen_skill_names` to lowercase.

### H5 — Missing import guard for gateway.session_context  
**File:** `agent/prompt_builder.py:1038`  
**Issue:** Unprotected `from gateway.session_context import get_session_env` can crash on minimal CLI installs without gateway package.  
**Fix:** Wrap in `try/except ImportError`.

---

## MEDIUM — Fix before charging users / next iteration

| # | Issue | File | Fix |
|---|-------|------|-----|
| M1 | Regex matches inside code blocks (doc self-reference) | `prompt_builder.py:1421` | Add code-fence awareness |
| M2 | Last-match-wins overwrites same directive (no merge) | `prompt_builder.py:1421` | Append to list instead of overwrite |
| M3 | Snapshot not versioned by filter params (cache miss) | `prompt_builder.py:1069` | Include filter hash in manifest |
| M4 | Empty name in stow returns success | `skill_manager_tool.py:762` | Guard against empty name |
| M5 | Compact format uses colons, not pipes (doc mismatch) | `prompt_builder.py:1239` | Update docstring or change delimiter |

---

## LOW — Nice to have

| # | Issue | File |
|---|-------|------|
| L1 | Unnecessary cache clear on stow (no metadata change) | `skill_manager_tool.py:786` |
| L2 | No cwd propagation to parse_project_skill_config | `system_prompt.py:180` |
| L3 | External dirs cache key uses non-canonical paths | `prompt_builder.py:1051` |
| L4 | Unbounded regex capture group size | `prompt_builder.py:1421` |
| L5 | Snapshot mtime+size vs content hash | `prompt_builder.py:889` |

---

## What's GOOD — Strengths

1. **10× token reduction observed** — 112 skills / ~3,000 tokens → 5 skills / ~260 tokens with proxxied config
2. **Fully backward compatible** — all new params default to None/"full", zero behavior change for unconfigured projects
3. **Hierarchical category matching** — prefix-level `"/"` splitting correctly handles nested categories
4. **Two-layer cache preserved** — in-process LRU + disk snapshot, filter params in cache key, invalidate-on-mutate
5. **Clean separation of concerns** — config parsing → filtering → rendering, all independent functions
6. **Defensive throughout** — all I/O and imports wrapped in try/except, graceful degradation
7. **165 existing tests pass** — no regression in skill prompt builder or skill usage tracking

---

## Rollout Readiness Assessment

| Gate | Status | Notes |
|------|--------|-------|
| Migration | ✅ | No schema changes, no data migration needed |
| Endpoints | ✅ | No new API endpoints |
| Client | ✅ | Context file directives are opt-in, no client changes |
| Tests | ⚠️ | 0 tests for new code paths (filters, compact format, parse_project_skill_config) |
| Stow mechanism | ⚠️ | `stow` bumps usage counter but actual compression depends on context compressor |

---

## Fix Priority Order

1. **H3** — stow existence check (prevents misleading success responses)
2. **H5** — gateway import guard (prevents crash on minimal installs)
3. **H2** — config.yaml merge (prevents silently lost settings)
4. **H1** — bracket-only parser (biggest UX papercut for users adopting)
5. **H4** — case-sensitive dedup (rare in practice, easy fix)
6. **L2** — cwd propagation (gateway mode correctness)
