# Phase 1 definitive assessment — delegation tiers

Generated: 2026-04-14
Sources: self-assessment, Claude Code review, Blackbox review, real runs, web benchmarks

---

## 1. Question asked

"Is the tier implementation 100% reliable and robust? Are we at the ceiling?"

---

## 2. Short answer

**No, not 100%. It is good beta quality. Not upstream-ready without fixes.**

Both independent reviewers (Claude Code and Blackbox) rated it **7/10**.
Self-assessment identified the same gaps.
We are NOT at the ceiling — there are concrete, fixable issues.

---

## 3. Scores converged from 3 sources

| Source | Rating | Key finding |
|---|---|---|
| Self-assessment | — | 5 long functions, 9 hardcoded tier strings, 6 global mutation refs, 2 pre-existing test failures |
| Claude Code (gpt-5.4-mini) | 7/10 | silent fallback too forgiving, policy mixed into routing, global state risk |
| Blackbox (gpt-5.4-mini) | 7/10 | unknown tier brittle, max_iterations falsy, pool malformed-data edge, prompt injection surface |

Agreement:
- Architecture is sound
- Tests cover happy path well
- Edge-case coverage insufficient
- Silent fallback behavior is risky
- Not upstream-ready without hardening

---

## 4. Concrete issues found (cross-referenced)

### Issue A — Unknown tier silently falls through to flat config
Found by: all 3 sources

When a caller passes `tier="nonexistent"` and the config has tiers, the code returns
the original flat config unchanged — no warning, no error.

Problem: hides misconfiguration, makes debugging harder.

Fix options:
1. Log a warning and fall back to default_tier if available
2. Fail fast with a clear error
3. At minimum, test and document the behavior explicitly

### Issue B — _validate_pool_model first-entry fallback
Found by: all 3 sources

If the requested model is not in the pool, it falls back to `pool[0].get("model")`.
If `pool[0]` is malformed or missing `model`, it returns `None` even when later entries are valid.

Problem: can silently route to None or wrong model.

Fix: scan for first valid model entry, not just index 0.

### Issue C — max_iterations falsy check
Found by: Blackbox, self-assessment

`max_iterations or default_max_iter` means `max_iterations=0` is impossible to express.
Any falsy value (0, "", None) is treated as "unset".

Problem: 0 is technically invalid for delegation, but this should be explicit validation, not implicit Python falsy semantics.

### Issue D — Global mutable state (_last_resolved_tool_names)
Found by: all 3 sources

`delegate_task` mutates `model_tools._last_resolved_tool_names` during child construction
and restores it in a try/finally. If delegate_task is called concurrently from multiple
parent agents, the global can race.

Problem: thread safety concern in concurrent delegation.

This is a pre-existing issue from the upstream code, not introduced by tiers, but tiers make the surface area larger because each child construction now involves tier resolution.

### Issue E — Edge cases not tested
Found by: Claude Code, Blackbox

Missing tests for:
- `tiers` config is not a dict
- `default_tier` is not a string
- tier entry is not a dict
- pool entries missing `model` field
- `max_iterations=0`, negative, or non-int
- concurrent delegate_task() calls from different parents
- blocked tools can't leak through mixed toolset combinations
- task.toolsets with invalid names silently stripped

### Issue F — Security surface
Found by: Blackbox

Pool metadata (strengths, provider) is injected into tool schema descriptions.
If config is user-controlled, this is an instruction-injection surface.

Not a code execution issue, but worth noting for upstream.

### Issue G — Config read overhead
Found by: Blackbox

`_get_max_concurrent_children()` calls `_load_config()` on every delegate_task invocation.
Minor overhead, but avoidable with caching.

---

## 5. What is strong (cross-referenced, no disagreement)

All 3 sources agree on these strengths:
- `resolve_tier_config()` is clean and well-isolated
- Reasoning floor guardrails are a smart safety feature
- Per-task tier precedence is well-defined
- Schema changes are additive and backward-compatible
- Real validation report is strong evidence
- Tests cover the happy path well
- 3-child concurrency was correctly traced to credential-pool issue

---

## 6. Real performance data

| Tier | Model | Duration | Tokens In | Tokens Out | Est. Cost | Quality |
|---|---|---|---|---|---|---|
| light | gpt-5.4-mini | ~6-10s | ~7k-22k | ~90-190 | ~$0.01-0.02 | good |
| heavy | gpt-5.4 | ~70-188s | ~246k-1.2M | ~1.5k-3.6k | ~$0.67-3.11 | good |
| review | gpt-5.4 | ~195s | ~597k | ~4k | ~$1.55 | good |
| planning | mimo-v2-pro | ~37-80s | ~67k-176k | ~1.3k-1.3k | ~$0.17-0.44 | good |
| research | gpt-5.4 | ~79s | ~95k | ~1.8k | ~$0.27 | good |

All 5 tiers produce usable output.
`light` is the most cost-efficient.
`review` is the most expensive and hits iteration caps easily.

Batch with 2 children: ~6-78s (works reliably).
Batch with 3 children: ~12s (works when credential pool is clean).

---

## 7. Gap analysis vs "100% reliable"

| Dimension | Status | Gap |
|---|---|---|
| Core logic | ✅ correct | none |
| Happy path tests | ✅ good | none |
| Edge case tests | ⚠️ insufficient | 8+ missing cases |
| Backward compat | ⚠️ mostly safe | resolve_tier_config sometimes returns original dict with nested tiers still in it |
| Silent fallback | ❌ risky | pool model fallback to index 0 without validation |
| Unknown tier | ❌ silent | no warning or error on invalid tier name |
| Global state | ⚠️ pre-existing | _last_resolved_tool_names mutation not safe for concurrent parents |
| Security | ⚠️ minor | pool metadata injection into tool schema descriptions |
| Performance | ⚠️ minor | config re-read on every call, no caching |
| max_iterations | ⚠️ falsy | implicit falsy check instead of explicit validation |

---

## 8. What "upstream-ready" would require

Based on all 3 sources converging:

**Must fix:**
1. Warn or error on unknown tier names
2. Harden pool fallback to scan for first valid entry
3. Add edge-case tests for malformed configs
4. Explicit max_iterations validation instead of falsy check
5. Ensure resolve_tier_config always strips nested keys

**Should fix:**
6. Cache config reads in _get_max_concurrent_children()
7. Add concurrency safety test for global state
8. Add test proving blocked tools can't leak through mixed toolsets
9. Document pool fallback behavior in schema description

**Nice to have:**
10. Strict mode option for pool validation (fail instead of fallback)
11. Centralized tier definitions with human-readable descriptions
12. Config read caching for performance

---

## 9. Honest assessment

The tier system is a good idea with solid implementation that needs hardening.

It is NOT 100% production-ready.
It IS the best version of delegation model routing that currently exists for Hermes.
It is NOT at the ceiling — there are specific, fixable gaps.

The fixable issues are well-characterized and not architectural.
No reviewer found fundamental design flaws.

Consensus: **7/10 with a clear path to 9/10 through the listed fixes.**
