# x-Algorithm Content Adjustments — Change Log

## Date: 2026-05-16
## Trigger: xAI open-sourced x-algorithm (May 15, 2026)
## Repo: https://github.com/xai-org/x-algorithm

## Findings from x-algorithm analysis

1. **BangerInitialScreen classifier** runs Grok-based eval on EVERY post before ranking
   - quality_score (threshold 0.4 for out-of-network distribution)
   - slop_score (explicit AI-generation detection)
   - taxonomy_category (content type classification)
   
2. **Phoenix weighted scorer** uses 18 engagement signals
   - reply-with-author = 150x weight (highest)
   - bookmark = 10x weight
   - retweet = 20x weight
   - like = 0.5x (nearly worthless)
   - Negative signals: not_interested, block, mute, report

3. **Grox content-understanding pipeline** uses multimodal embedders
   - Text AND images embedded together
   - Template cards score lower than screenshots
   - Stock images + generic text = weak embedding signal

4. **Query hydration** considers: mutual follows, followed topics, subscription status,
   impression bloom filters, demographics, IP

## Changes Made

### 1. Benchmark (done)
- Evaluated all 8 unique sahil_twitter templates against banger criteria
- All pass quality gate (avg 0.66/1.0) but slop scores averaged 4.25/10
- Issue: "Shipping apps. Breaking things. Fixing them. Repeat." mantra in 6/8 templates

### 2. Fixed `pillar_templates[0]` bug (MANDATORY FIX)
- File: `llm_drafts.py` line 546
- Before: `body = pillar_templates[0]` — always picks first template
- After: `body = random.choice(pillar_templates)` — random rotation
- Impact: This was the root cause of 17x and 7x template reuse

### 3. Added slop_score audit (HIGH PRIORITY)
- File: `llm_drafts.py` new function `_audit_slop()`
- Detects: boilerplate mantras, template-itis, generic filler, over-polished structure
- Slop filter tries up to 5 random templates before falling through
- Score + issues stored in DB (`slop_score`, `slop_issues` columns)
- Surfaces in CLI output during generation with ✅/⚠️ markers

### 4. Added taxonomy diversity (HIGH PRIORITY)
- 3 new template groups added to `llm_drafts.py`:
  - `SAHIL_TWITTER_TUTORIAL` — 4 tutorial/how-to templates
  - `SAHIL_TWITTER_DATA` — 4 data-driven templates  
  - `SAHIL_TWITTER_PROMOTION` — 4 promotion templates
- Updated `topics.py` with 6 new sahil_twitter topics covering tutorial, data, promotion
- Taxonomy variety: build_update, observation, opinion, story, tutorial, data_driven, promotion

### 5. Reference-content slots (MEDIUM PRIORITY)
- Covered by tutorial and data pillars (~15% of rotation)
- These are naturally bookmark-optimised (lists, frameworks, decision guides)

### 6. Reply suggestion pipeline (LOW — WIRED BUT DORMANT)
- File: `reply_suggester.py` — new module
- 20 target accounts configured (indie dev, Claude Code, football, startup)
- 16 reply patterns with exemplar templates
- Post categorisation via keyword heuristics (no LLM cost per reply)
- xurl wiring stubbed — ready for activation when xurl goes live

### 7. Visual card audit (MEDIUM PRIORITY)
- Removed "KENSEI" watermark from all card generators (biggest slop signal)
- Added content_type-aware visual layout variation in `make_card()`:
  - Data pillars: larger fonts, stat-first layout
  - Tutorial pillars: code-style border, numbered-list spacing
  - Promotion pillars: centered, problem-solution framing
  - Default: original layout (build updates, wry, football)

## Files modified
- llm_drafts.py: random.choice fix, _audit_slop(), 3 new template groups
- database.py: insert_draft now accepts slop_score, slop_issues
- content_engine.py: slop display in generate output, pillar passed to make_card
- topics.py: 6 new sahil_twitter topic entries
- visuals.py: KENSEI watermark removed, content_type-aware layouts
- reply_suggester.py: NEW — reply pipeline (dormant, needs xurl)

## Still pending (future)
- Activate reply pipeline when xurl is enabled (wire xurl into reply_suggester.py)
- Add more template variety to PLENISHD, COACHOS, MATCHDAYMAESTRO pillars (same pattern)
- Benchmark post-implementation: run generate_all on cron and compare slop scores
