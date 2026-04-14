# Twitter Digest Reverse-Engineering Brief
**Date:** 2026-02-13
**Owner:** Yeqiuqiu
**Prepared by:** Architect

## 1) Executive Summary
The Twitter 4h digest pipeline is operationally stable, but output quality is still below target because low-quality topic/entity labels leak through (especially from keyword-rescue paths). The format has already been migrated to friend-style Top 40 layout; the remaining gap is selection/ranking/entity quality.

---

## 2) Current State (What’s Good vs Not Good)

### ✅ Good
- Pipeline reliability is stable (capture → ingest → normalize → detect → brief).
- 4h digest format now mirrors requested friend-style layout (Top 40 sections + numbering flow).
- Significant reduction in earlier noise compared to older baseline outputs.

### ❌ Not Good Yet
- Rescue-label artifacts still appear (examples observed):
  - `agent can`
  - `fully functional`
  - `#ai` (too generic in-context)
- Some NEW/EMERGING rows are still weak semantically.
- Entity canonicalization still allows linguistically valid but low decision-value fragments.
- Novel section can inherit weak labels, making narrative feel noisy despite cleaner formatting.

---

## 3) Root-Cause Hypothesis (Current Best Read)
The primary issue is **not ingestion freshness**; it is **labeling/gating logic** in topic extraction + rescue admission + ranking.

We validated this by:
1. Running a fresh browser capture cycle,
2. Immediately rerunning 4h digest,
3. Observing the same low-quality rescue/entity artifacts.

Conclusion: stale/old data is not the main culprit; deterministic quality gates need tightening.

---

## 4) Recent Relevant Changes Already Landed

### Revival-only watcher scope
- Breakout setup disabled by env gate; revival remains active.
- Commit: `fc3aab4`

### Friend-style 4h digest layout
- Added `friend_v1` layout mode and defaulted 4h runs to it when mode unset.
- Commit: `63e095c`

### Earlier quality polish
- Blueprint + deterministic gate improvements + compact narrative polish landed before this report.

---

## 5) Ask for GPT-5.2 Pro (Primary Reverse-Engineer Request)

## Copy/paste prompt begins
You are reverse-engineering and fixing a deterministic Twitter trend digest pipeline.

### Context
I have a working 4h Twitter digest pipeline, but output quality is still not acceptable.
Infra is stable; the issue is **topic/entity quality**, especially in keyword-rescue labels.

Current bad examples that still leak:
- `agent can`
- `fully functional`
- `#ai` (too generic in this context)
- occasional weak NEW/EMERGING rows with poor semantic value

Layout is already fixed to friend format 1:1 (Top 40 style).
**Do not redesign layout.** Fix selection/ranking/entity quality only.

### Hard constraints
- Deterministic-first (no ML classifier required for gating).
- Keep existing pipeline architecture and schedule compatibility.
- Do not touch unrelated NGMI watchers.
- Keep concise Telegram-friendly output.
- Prefer precision over recall.

### Repo paths to inspect first
- `src/ngmi_terminal/twitter/topics.py`
- `src/ngmi_terminal/twitter/brief.py`
- `src/ngmi_terminal/twitter/keyword_rescue.py` (if needed)
- `scripts/run_twitter_digest_once.sh`
- tests under `tests/test_twitter_lane_*.py`

### Required work
1. **Root-cause analysis**
   - Identify exact failure paths where low-quality labels bypass gates.
   - Explain candidate generation → normalization → admission → ranking → rendering failure chain.

2. **Deterministic fix design**
   - Propose concrete rule updates for:
     - entity quality gate
     - rescue lane hardening
     - NEW/EMERGING admission tightening
     - ranking demotion/boost logic
   - Explicitly handle stopword-led fragments and generic hashtag leakage.

3. **Patch implementation**
   - Produce minimal, surgical code edits in the files above.

4. **Tests**
   - Add/update tests for:
     - rescue junk suppression
     - stopword-fragment blocking
     - generic hashtag suppression (without anchor)
     - NEW/EMERGING quality floor
     - no regression for valid emerging named entities

5. **Verification run**
   - Run fresh 4h check and provide before/after snippet.
   - Explain why each removed item was removed.
   - List remaining imperfections (if any).

### Output format (strict)
Return exactly in this structure:
1) Diagnosis
2) Rule changes
3) Files changed
4) Tests added/updated
5) Test results
6) Before/after sample
7) Residual risks + next knob suggestions

Be strict and practical. I want production-ready deterministic gating, not vague ideas.
## Copy/paste prompt ends

---

## 6) Companion Ask for Gemini 3.1 Pro (Second Opinion)
Use this as a parallel quality reviewer (same problem, independent lens):

## Copy/paste prompt begins
Please review this deterministic Twitter 4h digest pipeline for output-quality failure modes.

Goal: identify why low-value labels still leak (e.g., `agent can`, `fully functional`, generic `#ai`) even after formatting cleanup, and propose deterministic, production-safe fixes.

Constraints:
- Keep current friend-style Top 40 layout unchanged.
- Keep deterministic-first behavior.
- Prioritize precision and decision usefulness.

Focus files:
- `src/ngmi_terminal/twitter/topics.py`
- `src/ngmi_terminal/twitter/brief.py`
- `src/ngmi_terminal/twitter/keyword_rescue.py`

Deliver:
1) independent diagnosis,
2) ranked patch recommendations,
3) exact deterministic rule suggestions,
4) test cases to prove improvement,
5) calibration knobs with trade-offs.

Please optimize for concise, operator-grade output quality in Telegram.
## Copy/paste prompt ends

---

## 7) Acceptance Criteria for Final “Fully Good” State
- No stopword-fragment labels in NEW/EMERGING or NOVEL.
- Generic hashtags only appear when strongly anchored to a concrete named event/entity.
- NEW/EMERGING list remains informative and decision-useful.
- Novel section reads cleanly with meaningful entities only.
- No regressions in scheduling/pipeline reliability.

---

## 8) Artifacts to Share Alongside This Brief
When sending to GPT/Gemini, attach or reference:
- Latest 4h output (`twitter_brief_latest.md`)
- A before/after comparison artifact (if available)
- Current trend facts JSON (`twitter_trends_latest.json`)
- This brief

---

## 9) Notes
This brief intentionally separates **format** (already solved) from **signal quality** (remaining core issue).
