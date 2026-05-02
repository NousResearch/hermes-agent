

## Hackathon Fit Analysis

Details: https://www.reddit.com/r/hermesagent/comments/1socbwa/hermes_agent_creative_hackathon/

**Strong fit.** The hackathon explicitly covers "video, audio, creative software" domains. Your problem is a real, painful creative workflow that millions of bilingual content creators share — and Hermes Agent can automate nearly all of it.

**Key constraint**: Deadline is **EOD Sunday May 3rd — 3 days from now.** Tight but doable.

---

## Plan: Hermes Bilingual Auto-Caption Tool

**TL;DR**: Build a Hermes skill + tool that takes a video, transcribes it with Whisper, translates with Kimi (qualifying for the $5k Kimi track bonus), and generates a styled subtitle file matching your existing CapCut style — all from a single conversation.

---

### How It Works (the pipeline)

```
Video file
   → faster-whisper (local, free, best Vietnamese accuracy)
   → Kimi model (translation EN↔VI + caption cleanup)
   → ASS/SRT file with your saved style profile
   → (optional) FFmpeg burns captions into final video
```

**No CapCut API is needed.** CapCut supports importing `.srt`/`.ass` subtitle files, so you import the finished file and the captions appear with all your styling pre-applied.

---

### The "Style Profile" Feature (the smart part)

You describe your style once in plain language to Hermes:
> *"I use Arial Bold, size 48, white text, black outline, centered at the bottom third, with Vietnamese below English"*

Hermes stores this as a style profile in your Hermes config. Every future run uses it automatically — no re-specifying.

Even better: you could show Hermes a screenshot of an existing captioned video, and it uses vision to **extract** the font/color/size/position automatically.

---

### What Hermes Agent Adds (vs. just running ffmpeg yourself)

| Without Hermes | With Hermes |
|---|---|
| Run separate tools manually | Single conversation: "caption my video" |
| Re-specify style each time | Style profile saved, reused forever |
| No translation | Kimi translates EN↔VI, cleans up errors |
| Raw Whisper output | LLM post-processes timing gaps, speaker breaks |
| CLI expertise required | Natural language interface |

---

### Build Scope (3-day sprint)

**Phase 1 — Core pipeline (Day 1):**
1. `tools/video_caption.py` — Hermes tool wrapping faster-whisper + FFmpeg
2. Accepts: video path, source language (en/vi/auto), output format (srt/ass/burned)
3. Generates `.ass` subtitle file with default style

**Phase 2 — Style profiles + Kimi integration (Day 2):**
4. Style profile storage via Hermes config (`skills.config.caption_style`)
5. Kimi API integration for translation and Vietnamese text cleanup
6. Vision-based style extraction from reference screenshot (LLM call)

**Phase 3 — Skill + Demo (Day 3):**
7. `skills/video/video_caption/SKILL.md` wrapping the tool into a conversational workflow
8. Demo video showing full workflow: drop video → chat → get captioned output
9. Tweet + Discord submission

**Relevant files to modify/create:**
- [tools/video_caption.py](tools/video_caption.py) — new tool (create)
- [toolsets.py](toolsets.py) — register new `video_caption` toolset
- `skills/video/video_caption/SKILL.md` — new skill (create)
- [hermes_cli/config.py](hermes_cli/config.py) — add `caption_style` config defaults

---

### Kimi Track Strategy

Use Kimi as the **translation + refinement model** and explicitly show it in the demo:
- Whisper transcribes the audio (open source)
- Kimi translates and cleans up the Vietnamese text
- The demo video shows Kimi's output side-by-side with raw Whisper output to highlight the improvement

This proves Kimi model usage and gives you a shot at both prize pools (+$5k).

---

### Vietnamese Quality Reality Check

- **faster-whisper large-v3** handles Vietnamese reasonably well — you'll get ~80-90% accuracy on clear speech
- Your existing plan of "I'll fix the mistakes" is exactly right and is the practical approach
- Kimi's translation/cleanup pass will help normalize common Whisper errors (e.g., wrong tones, missing diacritics)

---

**Further Considerations**

1. **Output format preference**: Do you want a file to import into CapCut (`.srt`/`.ass`), OR a video with captions already burned in? Both are easy — just need to know which to demo first.
2. **Caption layout**: Do you want EN and VI as separate subtitle tracks (two lines, stacked) or separate passes? CapCut handles both — stacked is simpler to implement.
3. **Dependencies**: faster-whisper requires ~6GB download for the large-v3 model first run. A smaller model (`medium`) could be used for the demo to keep setup fast — accuracy is still solid for English, slightly lower for Vietnamese.
