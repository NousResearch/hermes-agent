# Content Engine 2.0 — Rebuild Plan

## Current State (broken)
- Template-based drafts with random placeholders (no real context)
- No content_type field — everything defaults to "twitter"
- 150 stale drafts stuck in DB since May 11
- Telegram bot dead (stale lock file)
- Postiz bridge empty (no integrations)
- Stage 2 blocked (cannot generate AI visuals)
- Delivery is a mass dump of all pending drafts with no approval workflow

## Target State (Content Engine 2.0)
- LLM-generated drafts per brand voice (no templates)
- Content type: Text, Text+Image, Voice, Video, Text+Video
- Stage 1: LLM generates draft + visual description
- Stage 2: FAL.ai/HyperFrames generate actual media for approved drafts
- Telegram delivery: per-draft card with brand, platform, content type, text, media preview
- Approve → queue → Stage 2 media → publish
- Reject → discard
- Regenerate option

## Architecture

### New Draft Model
```json
{
  "id": "draft_uuid",
  "brand": "matchdaymaestro",
  "platform": "twitter|linkedin|x|instagram|tiktok",
  "content_type": "text|text+image|voice|video|text+video",
  "pillar": "live_predictions|build_in_public|...",
  "topic": "real research topic",
  "title": "post title",
  "body_text": "actual post content",
  "visual_description": "what the image/video should look like",
  "static_visual_path": "Pillow card or static image from Stage 1",
  "ai_image_path": "FAL.ai generated (Stage 2, post-approval)",
  "ai_video_path": "HyperFrames/Ffmpeg (Stage 2, post-approval)",
  "status": "draft|approved|rejected|enriched|published",
  "created_at": "timestamp",
  "approved_at": null,
  "published_at": null
}
```

### Stage 1: Draft Generation (free)
1. **Topic Sourcing:**
   - LLM researches trending topics per brand (web search, brand context)
   - Or manual topic list seeded by user
2. **Draft Generation:**
   - LLM generates draft per (brand, platform, topic) using brand voice
   - Assigns content type based on pillar + platform fit
   - Generates visual description for Stage 2
3. **Static Preview:**
   - Pillow card with brand colors + text preview
   - Or placeholder image from visual description
4. **Delivery to Telegram:**
   - Per-draft card: Brand emoji, platform badge, content type badge
   - Post text (truncated to 280 chars for preview)
   - Image attached if static visual exists
   - Inline buttons: Approve, Reject, Regenerate

### Stage 2: AI Media Generation (paid)
1. Triggered only for approved drafts
2. FAL.ai generates image from visual_description + brand style
3. HyperFrames/Ffmpeg generates video for video pillars
4. Media attached to draft → ready to queue for publishing

### Publishing
1. Postiz bridge (if integrations working)
2. Direct Telegram/LinkedIn/Twitter API (if Postiz not ready)
3. Or manual delivery: user copies content + media

## Phases

### Phase 1 (immediate - fix broken things)
- Kill stale drafts (reset DB or truncate)
- Fix Telegram bot (remove stale lock, restart)
- Clean up old template system

### Phase 2 (LLM draft generation)
- Replace templates.py with LLM draft generation
- Brand voice prompts per brand (use existing voice skills)
- Content type assignment logic
- Visual description generation

### Phase 3 (Telegram delivery overhaul)
- Per-draft Telegram message with image attachment
- Content type badge
- Inline action buttons (approve/reject/regenerate)
- Bot callback handlers

### Phase 4 (Stage 2 media)
- FAL.ai image generation for approved drafts
- HyperFrames video for video pillars
- Media attachment workflow

### Phase 5 (publishing)
- Postiz bridge fix or alternative
- Direct platform posting
- Scheduling support

## Files to create/modify
- `content_engine.py` — new orchestration
- `llm_drafts.py` — LLM-based draft generation (new)
- `telegram_digest_v2.py` — per-draft delivery with media (new)

- `telegram_digest.py` — rewrite for v2 format
- `database.py` — add content_type column, visual_description
- `postiz_bridge.py` — fix or replace
- `config.py` — update brand/platform definitions

## Brand Voice Prompts (LLM)
Each brand gets a system prompt based on existing voice skills:
- matchdaymaestro — matchdaymaestro-voice skill
- plenishd — plenishd-voice skill
- coachos — coachos-voice skill
- sahil_twitter — sahil-twitter-voice skill
- sahil_linkedin — sahil-linkedin-voice skill

Prompts specify: tone, style, CTA format, length limits, content types.

## Content Type Logic
Based on pillar + platform fit:
- Twitter: text, text+image, voice
- LinkedIn: text, text+image, video
- Instagram: text+image, video
- TikTok: video, text+video
- YouTube Shorts: video

Default mapping per pillar:
- live_predictions → text+image (quick stat reveals)
- build_in_public → text (opinion, insight)
- product → text+image (feature showcase)
- launch → video (announcement)
- ai_tools → text+video (demo walkthrough)
