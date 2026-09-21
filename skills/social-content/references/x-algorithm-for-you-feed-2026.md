# X (Twitter) For You Feed Algorithm — Analysis Summary

**Source:** `xai-org/x-algorithm` (open-sourced May 15, 2026)
**License:** Apache 2.0 | **Stars:** 20k+
**Languages:** Rust (57%), Python (43%)

## Key Components

### 1. Grox Content-Understanding Pipeline (`grox/`)
Runs before ranking. Python-based classifiers using Grok/VLM:
- Banger Initial Screen (quality_score, slop_score, taxonomy_category)
- Post Safety Screen (deluxe)
- Spam detection (low-follower accounts)
- Reply ranking
- PTOS policy enforcement
- Multimodal post embedders (v2, v5) — text + images analyzed together

### 2. Home Mixer Orchestration (`home-mixer/`)
Rust service that assembles the For You feed. Pipeline:
1. **Query hydration** — user action seq, followed topics, starter packs, impression bloom filter, IP, mutual follows, served history, demographics
2. **Candidate sources** — in-network (Thunder), out-of-network (Phoenix retrieval), ads, who-to-follow, MoE, topics, prompts
3. **Candidate hydration** — engagement counts, brand safety, language, media detection, quotes, mutual follow Jaccard, subscription status
4. **Filters** — age, visibility, author social graph, dedup, muted keywords, previously seen/served, retweet dedup, self-tweet, topic IDs, video
5. **Scoring** — Phoenix (Grok-based transformer) + weighted scorer + author diversity + OON
6. **Selection** — final picks for the timeline

### 3. Phoenix Ranking (`phoenix/`)
Two-stage Grok-based transformer:
- **Retrieval:** Two-tower model (user tower + candidate tower) → ANN search → top candidates
- **Ranking:** Transformer with candidate isolation (candidates can't attend to each other)
- Output: per-candidate logits for each engagement type
- Mini model included: 128-dim, 4-layer transformer, trained on real-time data
- Sports corpus included: ~537K sports post IDs from a 6-hour window

### 4. Thunder In-Network Pipeline (`thunder/`)
Kafka-based real-time ingestion for posts from followed accounts.

## Weighted Scoring (Explicit Weights)

From `home-mixer/scorers/weighted_scorer.rs`:
```
combined =
  favorite_score × FAVORITE_WEIGHT +
  reply_score × REPLY_WEIGHT +
  retweet_score × RETWEET_WEIGHT +
  share_score × SHARE_WEIGHT +
  share_via_dm_score × SHARE_VIA_DM_WEIGHT +
  share_via_copy_link_score × SHARE_VIA_COPY_LINK_WEIGHT +
  dwell_score × DWELL_WEIGHT +
  dwell_time × CONT_DWELL_TIME_WEIGHT +
  profile_click_score × PROFILE_CLICK_WEIGHT +
  click_score × CLICK_WEIGHT +
  photo_expand_score × PHOTO_EXPAND_WEIGHT +
  quote_score × QUOTE_WEIGHT +
  quoted_click_score × QUOTED_CLICK_WEIGHT +
  follow_author_score × FOLLOW_AUTHOR_WEIGHT +
  video VQV_score × VQV_WEIGHT (if video_duration > threshold) +
  NOT_INTERESTED_SCORE × NOT_INTERESTED_WEIGHT (negative) +
  BLOCK_AUTHOR_SCORE × BLOCK_AUTHOR_WEIGHT (negative) +
  MUTE_AUTHOR_SCORE × MUTE_AUTHOR_WEIGHT (negative) +
  REPORT_SCORE × REPORT_WEIGHT (negative)
```

Exact weight values are not public, but relative ordering is clear from the x-algorithm documentation in home-mixer/params.

## Content Strategy Implications

1. **Banger gate is first:** quality_score < 0.4 = no out-of-network distribution at all. The quality gate is binary: pass/fail. Running appears before ranking.

2. **Slop_score matters:** Grok explicitly tags AI-generated content. Repetitive templates, boilerplate mantras, and over-polished structure are detectable.

3. **Reply chains are king:** The 150x weight for reply-with-author means reply strategy is the single highest-leverage activity on X.

4. **Taxonomy diversity matters:** If all posts land in one bucket, the algorithm narrows your out-of-network reach. Mix build updates, tutorials, data posts, observations, opinion, and promotion.

5. **Mutual follow graph boosts replies:** The `following_replied_users_hydrator` and `mutual_follow_jaccard_hydrator` mean replies to accounts your followers also follow get boosted.

6. **Images are embedded with text:** The multimodal embedder reads card text AND body text together. Template cards with watermarks score weak. Real screenshots score higher.

## File Reference

Full analysis with recommendations applied: `~/repos/KenseiAgent/content_engine/docs/x-algorithm-changes-2026-05-16.md`
Content pipeline skill: `devops/content-pipeline/SKILL.md`
