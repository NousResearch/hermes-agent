# Hackathon Submission — SpeakAlong for Hermes

**Track**: Kimi Track (primary) + General Track  
**Submitted by**: Allard Quek  
**Repo**: hermes-agent / feat/hackathon-creative-captions branch  
**Plugin path**: `plugins/phonetic-captions/`

---

## One-line pitch

> **SpeakAlong adds AI-generated phonetic captions to language teaching videos — watch, read pronunciation, speak along. Any language. Seconds, not minutes. Powered by Kimi K2.6.**

---

## Problem Statement

Language teachers creating teaching shorts face a painful captioning gap:

- Auto-generated captions (YouTube, CapCut) transcribe the audio but **never add pronunciation guides** — the one thing learners need to actually *say* the words.
- Manually adding phonetic captions to a 60-second video takes 20–40 minutes per language.
- Existing tools (Descript, Captions.ai) are designed for monolingual content and don't handle **code-switching** — the natural mix of instruction language + target language that defines most teaching videos.
- For tonal languages (Vietnamese, Mandarin, Thai) or diacritical systems (Korean, Arabic), the problem is even harder: transcription tools mangle tones/diacritics, and rebuilding them manually + adding pronunciation guides is prohibitively time-consuming.
- The result: teachers either skip captions or publish inaccessible content.

**Case study: Vietnamese Shorts** — but the same problem exists for Mandarin, Korean, Thai, and any language teachers want to make accessible.

---

## Solution

**SpeakAlong** is a self-contained Hermes plugin that handles the full pipeline for any language:

```
Teaching video (Telegram / upload)
  → faster-whisper  ─── local transcription, auto language detect
  → Kimi K2.6 (NVIDIA NIM)  ─── language classification + diacritic restoration + phonetic generation
  → FFmpeg  ─── ASS subtitle burn-in
  → captioned_<id>.mp4  +  dashboard editor link
```

Works with any language; demonstrated on Vietnamese.

### What makes it different

| Feature | Why it matters |
|---|---|
| **Auto language detection** | Handles code-switched audio without any configuration |
| **Diacritic correction** | Whisper mangles Vietnamese tones (e.g. `khong biet` → `không biết`). Kimi fixes them. |
| **Language-agnostic phonetics** | Generates readable phonetic guides in the student's native language (e.g. Vietnamese: `không biết` → `[humm biet]`) |
| **Teaching layout** | Target-language segments: bold main text + italic phonetic below. English segments: clean text only. |
| **Visual editor** | Dashboard editor with per-segment language toggles, word-split, NL edits, QA review |
| **Telegram-native** | Full pipeline from a single chat message — no desktop app needed (works for any language teaching channel) |
| **Hermes plugin** | Drop-in install, no core file changes, uses Hermes model as fallback if no NVIDIA key |

---

## Why Kimi K2.6

Phonetic generation is the hardest part of this pipeline. It requires:

1. **Language boundary detection** at segment level (one sentence may start EN, end in the target language)
2. **Diacritic and script restoration** from Whisper's transcription artifacts — Whisper mangles tones (`khong biet` → `không biết`), romanises scripts, and drops combining marks
3. **Phonetic approximation** that sounds right to English ears — not IPA, just `[humm biet]`
4. **Consistent JSON output** across hundreds of segments, reliably honouring the "no IPA" constraint

Kimi K2.6's extended reasoning handles all four simultaneously in a single pass, across every supported language. The model's instruction-following and structured output reliability made it the clear choice — smaller models frequently hallucinate IPA symbols, skip diacritics, or misclassify code-switched segments.

**Proof of Kimi usage**: The health banner in the dashboard shows `"Phonetics engine: NVIDIA Kimi K2.6"` when `NVIDIA_API_KEY` is set. The API call is visible in `~/.hermes/logs/agent.log` with `model: meta/llama-...` / `base_url: https://integrate.api.nvidia.com`. (See demo video.)

---

## Branding & Messaging

**Name**: SpeakAlong  
**Tagline**: *"Watch it. Read it. Speak it."*  
**Alt tagline**: *"AI phonetics for any language teaching video. Seconds, not minutes."*  
**Audience**: Language teachers (any language), heritage educators, polyglot content creators, particularly those teaching tonal or diacritical languages  
**Tone**: Warm, learner-focused, demo-forward, language-agnostic

### Key messages

- "Send a video. Learners speak along." — the core loop
- "Works for any language." — the generality angle
- "Kimi handles code-switching and tones automatically." — the AI angle
- "Edit what needs fixing, burn the rest." — the editor angle
- "Built as a Hermes plugin." — ecosystem angle

---

## Demo Video Outline

### Target length: 90 seconds (aiming for Shorts-compatible, ≤ 60s is ideal)

---

### Opening (0–10s) — The Problem

**Shot**: A language teaching Short (Vietnamese example) playing — no captions, mixed instruction language + target language audio.  
**Voiceover / title card**:
> "Teaching videos mix instruction and target language. Auto-captions can't add pronunciation guides. Teachers spend 30 minutes doing this manually — per video. For tonal languages, it's even harder."

---

### Act 1 — Telegram flow (10–30s)

**Shot 1**: Telegram chat. Send a raw teaching video to the Hermes bot (Vietnamese example).  
**Shot 2**: Bot replies: *"Transcribing…"* → typing indicator.  
**Shot 3**: Bot sends back the captioned `.mp4` — captions visible in the preview (Vietnamese + phonetics).  
**Title card overlay**: *"Auto language detect → Diacritic fix → Phonetic generation → FFmpeg burn (via Kimi K2.6)"*  
**Shot 4**: Bot message includes the dashboard link: *"Edit at http://localhost:9119/captions/abc123"*

---

### Act 2 — Dashboard editor (30–55s)

**Shot 1**: Click the link → dashboard opens on the editor.  
**Show**: Video player on left with burned captions. Segment list on right — target-language segments with phonetic field, English segments clean.  
**Shot 2**: Fix a segment — tap the phonetic field, type a correction (show Vietnamese example).  
**Shot 3**: NL edit panel — type *"merge segments 4 and 5"* → AI proposes patch → accept → segments merged.  
**Shot 4**: QA review — click "Review" → amber flags on two segments, green ticks on good ones.  
**Shot 5**: Click "Re-burn" → spinner → video reloads with updated captions.

---

### Act 3 — Kimi proof + style (55–75s)

**Shot 1**: Health banner at top of job list — **"Phonetics engine: NVIDIA Kimi K2.6"** clearly visible.  
**Shot 2** (optional): `~/.hermes/logs/agent.log` tail showing the NVIDIA NIM API call.  
**Shot 3**: Style presets — click "Style with Hermes", type *"bold yellow Impact, TikTok style"* → preview appears → apply → re-burn → new captions appear.

---

### Closing (75–90s)

**Shot**: Side-by-side: raw video (no captions) vs. captioned output (Vietnamese + phonetics).  
**Title card**:
> *"SpeakAlong — a Hermes plugin.*  
> *Watch it. Read it. Speak it."*

**Show**: `hermes plugins enable phonetic-captions` in terminal.  
**CTA**: *"Built for the Nous Research × Kimi hackathon."*

---

## Post Copy

### X / Twitter

> Standard accounts get **280 characters** (not 140 — doubled in 2017). Two options below.

**Full version (~270 chars, fits in 280 with link):**
```
🎬 Built SpeakAlong for @NousResearch × Kimi

Vietnamese Shorts mix EN + VI. Auto-captions miss pronunciation guides.
Kimi K2.6 classifies, fixes diacritics, adds [phonetic guides].
FFmpeg burns them in.

Send a clip. Learners speak along.

[link] #NousHackathon #KimiTrack
```

**Ultra-short (~120 chars, add link + hashtags after):**
```
Language teachers spend 30 min adding pronunciation captions per video.
Kimi K2.6 does it in seconds. Built SpeakAlong to prove it.
```

### Discord (shorter, for `#creative-hackathon-submissions`)

```
**SpeakAlong** — AI phonetic captions for language teaching videos (any language)

Built on Hermes + Kimi K2.6. Send a video via Telegram → Kimi auto-detects language,
fixes transcription artifacts (tones, diacritics), generates pronunciation guides
→ FFmpeg burns them in → you get back a Short learners can actually speak along with.

Works for any language; demonstrated on Vietnamese. Includes dashboard editor:
segment editing, NL instructions, QA review, style presets. Fully self-contained
Hermes plugin — no core file changes.

[demo video link]
```

---

## Technical Proof Points for Judges

| Claim | Where to verify |
|---|---|
| Kimi K2.6 via NVIDIA NIM | `plugins/phonetic-captions/pipeline.py` → `generate_phonetics()` — `base_url = "https://integrate.api.nvidia.com/v1"`, `model = "moonshotai/kimi-k2.6"` |
| Hermes model fallback | Same function — falls back to `AIAgent` if `NVIDIA_API_KEY` absent |
| Diacritic correction | `_phonetics_prompt()` in `pipeline.py` — explicit instruction to fix Whisper tone artifacts, parameterised per target language |
| Phonetic-only output (no IPA) | `_phonetics_prompt()` + `_nl_system_prompt()` — "NEVER use IPA symbols" rule in both |
| Any-language support | `LANG_NAMES` dict (17 languages), `_detect_target_lang()` auto-detects from Whisper segment tags, `target_lang` parameter flows through the full pipeline |
| Dashboard is a Hermes plugin | `plugins/phonetic-captions/plugin.yaml`, `__init__.py` `register(ctx)`, `dashboard/manifest.json` |
| No core file changes | Only `web/src/App.tsx` gets a 1-line catch-all guard; all pipeline + API + UI lives in `plugins/phonetic-captions/` |
| SSE streaming prevents timeouts | `plugin_api.py` `_agent_sse()` generator, used by `/nl-edit` and `/qa` endpoints |

---

## Future Direction

SpeakAlong is scoped to bilingual teaching videos for the hackathon, but the pipeline is already general. The natural growth path:

**v2 — Monolingual captions (trivial)**  
Whisper already transcribes any single-language video. Adding a "captions only, no phonetics" mode is a prompt removal and an additional ASS style — no architecture change. This makes the tool useful for any content creator, not just language educators.

**v3 — Translation subtitles**  
Replace the phonetics-generation prompt with a translation prompt. A monolingual Korean video could get English subtitles burned in; a Spanish teaching video could get French captions. Mechanically identical to the current pipeline — the LLM call is already there, just with a different instruction. This is deliberately deferred: it enters competitive territory with CapCut auto-translate, Descript, and YouTube auto-captions, and the positioning argument for SpeakAlong is strongest in the niche where those tools fail.

**v4 — Phonetic direction reversal**  
Currently the phonetic guide is always in English (the instructor's language). The inverse — generating phonetic guides in the *target* language for English words — is equally useful for language learners going the other direction (e.g., a Vietnamese learner watching an English teaching channel). Same pipeline, different prompt.

**The core bet**: the bilingual teaching use case is underserved, growing, and technically hard in exactly the ways this stack handles well. Winning this niche first is a stronger foundation than competing on general captioning from day one.

---

## Submission Checklist

- [ ] Demo video recorded (see outline above)
- [ ] Tweet drafted and ready to post (see copy above)
- [ ] Tweet posted tagging @NousResearch
- [ ] Discord link dropped in `#creative-hackathon-submissions`
- [ ] Verify health banner shows "NVIDIA Kimi K2.6" in demo video
- [ ] Verify API call to NVIDIA NIM visible in logs or terminal during demo recording
