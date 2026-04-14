# Twitter 4h Digest — Current Logic Snapshot (for Architecture Audit)
Date: 2026-02-14
Owner: Yeqiuqiu
Scope: Explain why NEW/EMERGING + ACCELERATING can look sparse while NOVEL still has content, and why some output still looks gibberish.

---

## TL;DR diagnosis
1. **ACCELERATING can appear blank by design** because friend_v1 de-duplicates topics across sections and prints NEW first. If candidate overlap is high, acceleration gets emptied.
2. **NOVEL can still show items** because it has a separate source path (topic clusters + keyword rescue bundles), so it can produce narrative bullets even when acceleration rows are empty.
3. **“Gibberish” survives mainly from broad hashtag rescue + weak phrase extraction remnants**, especially when low-volume windows pass corroboration thresholds.

---

## 1) Current pipeline (runtime)
Script path: `scripts/run_twitter_digest_once.sh`

4h flow (current default behavior):
1. `twitter dedupe --since 4h`
2. `twitter detect-trends --short-window 4h --long-window 24h`
3. `twitter emit` → `data/outputs/twitter_trends_latest.json`
4. `twitter brief` (4h defaults to `digest_mode=friend_v1`)
5. rank follow candidates + outcome updates

So the presentation is friend-style, but underlying selection logic is still multi-lane (strict + shadow + rescue + novel clustering).

---

## 2) How NEW/EMERGING is currently admitted
In `src/ngmi_terminal/twitter/brief.py`, `_topic_passes_new_emerging_admission(...)` applies hard filters for 4h:

- reject `shadow_rescued`
- require **velocity >= 1.0** unless explicit override
- for small counts: require author/feed corroboration
- for low counts (`<=3`): require corroborated evidence
- reject broad generic AI tags unless concrete anchor / override
- reject broad umbrella topics unless concrete anchor / override

Effect: this improves precision, but in small windows it can leave only 1-3 valid NEW rows.

---

## 3) Why ACCELERATING can show “No clear acceleration signal”
In friend renderer (`_render_friend_v1_compact_markdown`):

1. Build `new_rows` from primary-ranked topics first.
2. Build `accelerating_rows` from acceleration-ranked topics.
3. **Deduplicate by topic key across both using `seen_keys`**.
4. If all acceleration candidates were already consumed by NEW (or filtered), acceleration list becomes empty and prints the fallback line.

So blank acceleration is often a **section-allocation artifact**, not necessarily “no momentum exists.”

---

## 4) Why NOVEL can still have content when ACCELERATING is empty
NOVEL is rendered via `_render_novel_context_panel(...)` and uses:
- selected novel topics,
- rescue bundles (`keyword_traction_rescue`),
- cluster merge pass (`_cluster_novel_narratives` + rescue attachment),
- substantive text/schema gates.

This is a separate path from acceleration rows, so it can still output bullets even when acceleration section is empty.

---

## 5) Where gibberish still comes from
Even after recent hardening, artifacts can still leak from:

1. **Rescue hashtags with weak semantic specificity**
   - e.g., broad tags like `#ai` / `#llm` can pass if corroboration and velocity are sufficient.

2. **Shadow lane still contains many low-info phrase entities**
   - trends artifact currently shows lots of `phrase:*` junk in shadow/rescue pools.
   - even if not directly shown in Top rows, they influence fallback/novel candidate pressure.

3. **Low-volume windows amplify noisy wins**
   - with small counts (2-3 mentions), a few correlated posts can satisfy deterministic thresholds.

---

## 6) Evidence from latest artifacts
From latest `twitter_trends_latest.json` snapshot:
- `topics_scored=8`, `topics_active=8` (tight window)
- `shadow_topics=187`, `rescued_topics=182` (very large rescue/shadow universe)
- rescue bundles still include broad tags (`#ai`, `#llm`) plus phrase rescue candidates

This confirms a key mismatch: **strict top set is small, but rescue/shadow candidate space remains huge/noisy**.

---

## 7) Current logic map (code hotspots)
Primary files:
- `src/ngmi_terminal/twitter/brief.py`
  - friend layout renderer
  - section allocation / dedupe
  - NEW admission gate
  - NOVEL clustering and rendering
- `src/ngmi_terminal/twitter/topics.py`
  - entity extraction + quality gating modes (strict/shadow/rescue)
- `src/ngmi_terminal/twitter/keyword_rescue.py`
  - rescue scoring, velocity/baseline logic, bundle generation

Critical functions to audit:
- `_topic_passes_new_emerging_admission`
- `_select_accelerating_topics`
- `_render_friend_v1_compact_markdown`
- `_render_novel_context_panel`
- `_cluster_novel_narratives`
- rescue candidate admission/scoring in `keyword_rescue.py`

---

## 8) Concrete audit questions for GPT Pro / Gemini Pro
1. Should acceleration section be computed from a disjoint pool, or should overlap with NEW be allowed in display?
2. Should broad hashtags (`#ai`, `#llm`) require stronger anchor criteria (named entity + trigger + multi-source) before Novel inclusion?
3. Should rescue output be capped by a stricter semantic quality score before clustering?
4. Should shadow lane phrase entities be aggressively pruned earlier to reduce downstream noise pressure?
5. Should low-volume (2-3 mentions) entries require stronger author diversity for 4h mode?

---

## 9) Immediate practical fix options (if approved)
A. **Display policy fix**: allow ACCELERATING to show overlaps with NEW (or reserve slots) to avoid “blank acceleration” perception.
B. **Rescue semantic hard floor**: block broad tags unless strong named-event anchor.
C. **Shadow phrase purge**: raise minimum quality in shadow lane so rescue pool is cleaner.
D. **4h strictness bump**: increase low-volume corroboration thresholds (authors/feeds) for Novel eligibility.

---

## 10) Bottom line
The current behavior is internally consistent with the code: sparse NEW/ACCEL + non-empty NOVEL is expected under strict 4h admission + section dedupe + independent novel clustering. The remaining quality issue is not infra; it is the **rescue/shadow semantic boundary** and **display allocation policy**.
