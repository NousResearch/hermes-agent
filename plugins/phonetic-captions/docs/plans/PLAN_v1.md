# Hermes Bilingual Caption Agent — Hackathon Plan

**Hackathon**: Hermes Agent Creative Hackathon  
**Prize pool**: $25k (Main $15k + Kimi Track $5k)  
**Deadline**: EOD Sunday May 3rd, 2026  
**Judged on**: Creativity, usefulness, presentation  
**Kimi Track**: Must prove use of Kimi models in submission video

---

## Problem Statement

Content creators making bilingual (English + Vietnamese) videos spend 30–45 minutes per video manually:
1. Choosing font, size, color, and position in their editor every time
2. Typing or re-generating Vietnamese captions with no consistency
3. Fixing translation mistakes with no memory of prior corrections

There is no tool that learns a creator's personal caption style and applies it automatically across every future video while improving from corrections.

---

## Solution

**A self-improving Hermes-powered bilingual caption agent.**

One-line pitch: *"Hermes learns how you subtitle once, then applies that caption style and bilingual workflow to every future video."*

### Pipeline

```
User sends video
   → faster-whisper (local, free) transcribes English audio
   → Kimi K2.5 via NVIDIA NIM translates EN → VI + cleans up
   → ASS subtitle file built: EN on top, VI below (stacked, same moment)
   → FFmpeg burns styled captions into video
   → User receives captioned video
   → User corrects mistakes in chat → Hermes re-burns
   → After approval → corrections saved to Hermes memory
   → Next video: agent applies saved corrections automatically
```

### The Memory Loop (Hermes-native differentiator)

Three layers compound over time:

| Memory Layer | Stores | Example |
|---|---|---|
| Style profile | Font, size, color, outline, position | "Montserrat Bold 52pt, white, black stroke, bottom-center" |
| Correction memory | Recurring translation preferences | "always use 'các bạn', not 'mọi người' for 'guys'" |
| Glossary | Names, brands, channel terms | "channel name is 'Cooking with Linh'" |

**Demo story**: Video 1 needs 6 corrections. Video 2 needs 2. Video 3 needs 0. The agent got smarter.

---

## Technical Approach

### Stack (Total hackathon cost: ~$0)

| Component | Tool | Cost |
|---|---|---|
| Transcription | faster-whisper `medium` model (local) | $0 |
| Translation EN→VI | Kimi K2.5 via NVIDIA NIM | $0 |
| Subtitle styling + burn | FFmpeg + ASS subtitle format | $0 |
| Orchestration + memory | Hermes Agent | $0 |
| Primary interface | Telegram gateway (already in Hermes) | $0 |
| Secondary interface | Hermes CLI | $0 |

### Why faster-whisper + Kimi, not CapCut or Argos

- **CapCut**: No public API. Cannot automate programmatically.
- **Argos Translate**: Vietnamese support is weak (primarily European language pairs). Poor demo quality.
- **faster-whisper**: Local, free, good Vietnamese accuracy on `medium`/`large-v3` models.
- **Kimi K2.5**: Production-quality Vietnamese translation + qualifies for the $5k Kimi track.

### NVIDIA NIM Details

Kimi K2.5 is available free via NVIDIA NIM:
- Base URL: `https://integrate.api.nvidia.com/v1`
- Model ID: `moonshotai/kimi-k2.5`
- OpenAI-compatible API — works with Hermes's custom provider config
- Env var: `NVIDIA_API_KEY` in `~/.hermes/.env`
- **Known quirk**: Output sometimes lands in `reasoning_content` not `content` — the tool will check both

### Subtitle Format: ASS

ASS (Advanced SubStation Alpha) is preferred over SRT because:
- Full styling control: font, size, color, outline width, shadow, position
- Per-style definitions: can encode EN and VI styles separately
- FFmpeg supports burning ASS natively (`-vf "ass=file.ass"`)
- SRT is also written alongside burned video for archival/editing

---

## Surfaces

### Primary: Telegram

**Why**: Natural for creators — open Telegram, drop video, get it back. No new app. The chat-based correction loop is native to messaging.

```
You: [sends video]
Hermes: Processing... ✓ Here's your draft! MEDIA:/path/output.mp4
        [sends captioned video]
        Lines numbered 1-12 for corrections.

You: fix line 4 Vietnamese to "thêm muối vừa phải"
Hermes: Updated! MEDIA:/path/output_v2.mp4 [sends re-burned video]

You: looks great!
Hermes: Saved your correction to memory. I'll apply it automatically next time.
```

**Gateway fix required**: `_preprocess_inbound_text` in `gateway/run.py` does not currently inject video file paths into agent messages (documents get this, video type does not). Need to add a ~5-line handler mirroring the document injection pattern.

### Secondary: CLI

**Works automatically** — no extra code. Because the tool is registered in Hermes's tool registry, any Hermes surface can use it. CLI users pass the file path directly in conversation:

```
> caption /path/to/video.mp4
> caption this video: /tmp/cooking_ep5.mp4 and fix the Vietnamese for "add salt gradually"
```

CLI is useful for:
- Batch captioning multiple videos
- Local dev/testing without Telegram setup
- Users who prefer terminal workflows

---

## Files to Create / Modify

| File | Action | Purpose |
|---|---|---|
| `gateway/run.py` | Modify | Fix `_preprocess_inbound_text` to inject video file paths (~5 lines) |
| `tools/video_caption.py` | Create | Core tool: transcribe, translate, build_ass, burn, reburn operations |
| `toolsets.py` | Modify | Register `video-caption` toolset |
| `hermes_cli/config.py` | Modify | Add `caption_style` key to `DEFAULT_CONFIG` |
| `skills/video/bilingual_captions/SKILL.md` | Create | Skill wrapping the full conversational workflow |

---

## Build Phases

### Phase 1 — Core pipeline (Day 1: April 30)

**Goal**: Send video via Telegram → get back burned video with captions.

1. Fix `gateway/run.py` video path injection (critical path — everything depends on this)
2. Create `tools/video_caption.py`:
   - `transcribe(video_path, language="en")` → list of `{start, end, text}` segments
   - `translate_to_vietnamese(segments, nvidia_api_key)` → list of `{start, end, en, vi}` segments
   - `build_ass(segments, style_config, output_path)` → writes `.ass` file with stacked EN/VI
   - `burn(video_path, ass_path, output_path)` → FFmpeg subprocess call
   - `reburn(video_path, corrected_segments, style_config, output_path)` → apply edits + re-burn
3. Register in `toolsets.py`

**Verification**: Send 15s test video via Telegram → receive captioned video back.

### Phase 2 — Style + Kimi + memory (Day 2: May 1)

**Goal**: Persistent style, real Kimi translation, correction memory.

4. Ship default `caption_style` config in `hermes_cli/config.py`:
   ```python
   "caption_style": {
       "font": "Montserrat Bold",
       "font_size": 52,
       "primary_color": "&H00FFFFFF",   # white (ASS AABBGGRR format)
       "outline_color": "&H00000000",   # black
       "outline_width": 3,
       "alignment": 2,                  # bottom-center
       "margin_bottom": 80,
       "max_line_length": 42,
   }
   ```
5. Wire Kimi translation via NVIDIA NIM (replace Whisper-only placeholder from Phase 1)
   - Add `reasoning_content` fallback for NVIDIA NIM's Kimi quirk
6. Post-correction memory hook: on user approval → agent writes to `MEMORY.md`:
   - Translation preferences (e.g., "guys" → "các bạn")
   - Glossary terms
7. Create `skills/video/bilingual_captions/SKILL.md`

**Verification**: Two-pass demo — Video 1 with corrections, Video 2 shows corrections auto-applied.

### Phase 3 — Demo video + submission (Day 3: May 2–3)

**Goal**: Compelling submission video + tweet + Discord post.

8. Record demo script:
   - Show the problem (CapCut manual caption workflow, ~45 min per video)
   - Show the solution (Telegram: send video → 30s → get captioned video back)
   - Show Kimi translation quality vs raw Whisper output (proves Kimi usage)
   - Show memory improvement: Video 1 (corrections), Video 2 (auto-applied)
9. Write submission tweet tagging @NousResearch with demo video
10. Post link to `#creative-hackathon-submissions` on Nous Research Discord

---

## Sample User Journey

> **Meet Linh — she creates bilingual cooking content for Vietnamese and English audiences.**

**Before**: Every video takes 45 extra minutes in CapCut re-doing captions — choosing font, sizing it, typing the Vietnamese, positioning both lines. Same thing every single video.

**With Hermes Caption Agent:**

**Step 1** — Linh opens Telegram and sends her video:
> *[sends video file]*  
> "caption this please"

**Step 2** — ~30 seconds later, Hermes replies:
> "Done! Here's your draft. English on top, Vietnamese below — here are the lines:
> 1. EN: Today we're making pho | VI: Hôm nay chúng ta làm phở
> 2. EN: Add salt gradually | VI: Thêm muối từ từ
> ...
> Let me know what to fix!"  
> *[sends back captioned video]*

**Step 3** — Linh watches it and catches two issues:
> "Line 2 Vietnamese should say 'thêm muối vừa phải', not 'thêm muối từ từ'"

**Step 4** — Hermes re-burns:
> "Fixed! Here's the updated video." *[sends new video within 15s]*

**Step 5** — Linh approves:
> "Perfect!"

Hermes saves the correction silently.

**Next video (one week later)**: Linh sends a new video. Hermes already knows her style, her font, and that "add salt" → "thêm muối vừa phải". Draft comes back already correct. Zero corrections needed.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| NVIDIA NIM slow during peak hours (5–30s) | Acceptable for translation of ~50-150 words; show async "processing..." message |
| faster-whisper `medium` misses some Vietnamese words | Expected; user corrects and Hermes learns. This is the demo's value prop, not a bug. |
| FFmpeg not installed on user's machine | Add `check_requirements()` in tool; print clear install message |
| Telegram 50MB video send limit | 10-20s YouTube Shorts = 5-25MB; well within limits |
| NVIDIA NIM `reasoning_content` quirk | Add 3-line fallback in translation call |
| Kimi ToS (no commercial use via NIM) | Hackathon demo only — compliant |

---

## Kimi Track Strategy

The submission video will explicitly show:
1. The `NVIDIA_API_KEY` / `moonshotai/kimi-k2.5` config in `~/.hermes/.env`
2. Side-by-side: raw Whisper output vs Kimi-cleaned Vietnamese translation
3. Hermes memory storing Kimi-refined corrections for future runs

This makes the Kimi model usage unambiguous and demonstrates meaningful quality improvement — not just a wrapper call.
