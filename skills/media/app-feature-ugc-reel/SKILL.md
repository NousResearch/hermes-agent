---
name: app-feature-ugc-reel
description: |
  Story-first UGC Reel promoting ONE specific feature of a mobile/web app
  (not the whole app). Creator is hero of the frame; app screen appears
  as a product-substitute insert. Designed to activate mirror-neuron
  empathy: vulnerable storytelling first, feature reveal in the middle,
  soft personal CTA at the end. Vertical 9:16, typically 30–60s.
  Triggers when: (a) subject is a software/SaaS app, (b) brief targets
  ONE feature, (c) tone is organic/intimate/wellness/vulnerable, (d) output
  is a Reel/Short. NOT for physical-product UGC, hard-sell ads, tutorials,
  unboxings, or multi-feature brand reels.
---
# App-Feature UGC Reel Workflow

A self-contained pipeline that wraps ugc-flow with the substitutions needed
when the "product" is a software feature, not a physical object. Builds an
emotional, storytelling-first Reel where the creator confides about a personal
moment, then reveals how a specific app feature met that moment.

## Architecture (one-line summary)
Brief (app URL + feature + tone) →
  clarify gaps (duration, language) →
  N boards from duration table →
  generate character (Soul 2.0) + feature screen mockup (GPT Image 2.0) in parallel →
  generate N boards sequentially (chained for continuity) →
  generate N Seedance clips in parallel →
  ffmpeg hard-cut concat →
  final.mp4

## Duration → N boards (inherited from ugc-flow)

| Total D | N | Per-clip durations | Arc |
|---|---|---|---|
| 4–15  | 1 | D                  | FULL_ARC                                  |
| 16–19 | 2 | balance ≥4s each   | HOOK+SETUP / APPLY+CLOSER                 |
| 20–30 | 2 | 15, D−15           | HOOK+SETUP / APPLY+CLOSER                 |
| 31–45 | 3 | 15, 15, D−30       | HOOK / MAIN / CLOSER                      |
| 46–60 | 4 | 15, 15, 15, D−45   | HOOK / REVEAL / APPLY / CLOSER            |

Default if duration not specified: 45s (3 × 15s, the most natural narrative shape for this format).

## Tone lock (mandatory)

This flow is always Pattern A (classic UGC) calm-intimate unless the brief
explicitly requests high-energy. Override the default Pattern B (sustained hyped)
in ugc-clip-prompt.md. Specifically:

- Word density: ~25–35 words per 15s clip (lower than hyped UGC).
- NO [*explosive gasp*] / [*hyped yelp*] / [*excited shriek*] bracketed sounds.
- NO performed by an INSANELY hyped creator with explosive screaming energy throughout calibration phrase in the Narrative Summary.
- Audio cadence: warm, slightly hushed, like confiding to a friend.
- Micro-behaviors picked from the calm half of the menu: weight shifts, slow blinks, soft inhales, hand to collarbone, half-smiles, glance breaks, mid-thought stumbles.
- Mandatory: at least ONE unguarded mid-thought micro-stumble or post-laugh settle per clip (real-creator imperfection).

## Pipeline

### Step 1 — Parse the brief

Extract:
- App URL → run web_extract(urls=[url], extract_images=10) to capture brand voice, tone constraints, claim limitations, color palette signals, target audience.
- Specific feature the brief is targeting (e.g. "dream interpretation", "natal chart", "mood tracking", "habit streaks").
- Disclaimers / restrictions from the brief or site — what the feature is NOT (e.g. "not medical advice", "not a substitute for therapy", "for reflection and entertainment only"). These bound the script.
- Duration (default 45s) and language (default match the site's primary language).

If duration or language is missing AND ambiguous, ask in ONE bundled
ask_user_question call. Do NOT re-ask anything the brief already pins.

### Step 2 — Workflow check
Confirm the creator-as-hero / app-as-insert framing matches the brief. If
the brief actually wants the product-as-hero (rare for SaaS), redirect to
ugc-product-flow. If it wants step-by-step in-app demonstration, redirect
to ugc-tutorial-flow.

### Step 3 — Character generation (Soul 2.0)

Follow skill_view("image-generation", "references/soul-v2-ugc-character.md")
in full, with these defaults:

- Variety roll: run the mandatory 8-integer roll via
  python3 -c "import secrets; print(' '.join(str(secrets.randbelow(100)) for _ in range(8)))"
  and map per the table in soul-v2-ugc-character.md.
- Tier: premium by default (most app brands sit here). If the site signals
  luxury (high-end wellness, spiritual luxury) → luxury tier; if mass-market /
  free freemium → drugstore tier.
- Location: derive from the feature context, NOT the product category.
  Dream interpretation → cozy bedroom morning. Mood tracking → kitchen or
  bedroom. Habit tracker → home gym or desk. Astrology → bedroom evening.
  Meditation → living room with plants. Language learning → desk / café.
  Fitness → home gym. Sleep → bedroom morning.
- Outfit: cozy / lounge / soft — never editorial. The viewer must feel
  they're catching a moment in the creator's real life, not a content-day shoot.
- Lighting: COOL diffused window daylight ONLY. Never golden hour, never
  warm sunset, never orange cast.
- Submit with enhance_prompt: false and quality: "1080p", aspect 3:4.

Capture character_media_id after polling + re-upload.

### Step 4 — Feature-screen mockup (GPT Image 2.0)

This is the product-substitute. It replaces the physical product in the
referenced medias arrays of every downstream call.

- Model: imagegen_2_0 (only model that renders precise on-screen text
  reliably). Aspect 3:4, resolution: "2k".
- Subject: A photorealistic smartphone (iPhone-15-style, matte titanium
  frame) held perfectly straight-on, centered, on a neutral cream / off-white
  seamless background with soft contact shadow.
- On-screen content (mandatory elements):
  1. Status bar at top (e.g. 9:47, signal/wifi/battery).
  2. App wordmark in upper-left in the brand's typography.
  3. Large serif/display heading naming the feature exactly as the brand
     names it (verbatim from the site if extracted).
  4. Subhead in a muted secondary color (e.g. Last night's reflection,
     Today's check-in, `This week's pattern`).
  5. Central content card with rounded corners, soft gradient, containing
     a short, evocative piece of output that the feature would actually
     produce — this becomes the emotional anchor of the video. Quote-style
     italic + body text.
  6. 2–5 small symbol chips or icon glyphs labeled in small caps (e.g.
     MOON · KEY · THRESHOLD for dreams, FIRE · WATER · EARTH for
     astrology, SLEEP · STRESS · ENERGY for mood, etc.).
  7. Primary CTA button (rounded pill) with brand-appropriate copy (`Save to Soul Journal`, Track this day, Read more, etc.).
  8. Bottom tab bar with 4–5 minimal icons + small-caps labels matching the app's actual sections.
- Color palette: extract from the site. Default if absent: warm cream
  (#f8f1e4), soft accent (lavender / sage / blush), deep charcoal text,
  muted gold accent.
- Typography: soft serif for headings + display content, clean
  sans-serif for UI labels and small-caps rows.

Capture screen_media_id after polling + re-upload.

### Step 5 — Monologue draft

Voice-over-style on-camera monologue, English by default (match site's
primary language otherwise). Total length tuned to calm Pattern A density:

| Total D | Total words | Per board |
|---|---|---|
| 30s | ~60   | 30 / 30      |
| 45s | ~90   | 30 / 30 / 30 |
| 60s | ~120  | 30 / 30 / 30 / 30 |

Structure across N boards:

- Board 1 (HOOK) — vulnerable / human / story setup. ALL of:
  - First 1–2 sentences describe a personal moment, recurring experience,
    something the viewer can recognize ("I had this dream three nights in
    a row", "I keep waking up tired", "I've been avoiding journaling for months").
- 1 sentence about turning to the app — naturally, low-friction
    (`I opened [App]`, `I finally typed it in`).
  - 1 sentence teasing the reveal (`What came back wasn't what I expected`,
    `It said something I wasn't ready for`).
  - NO greetings, NO "hi guys", NO "today I'm showing you".

- Board K (MAIN — for N≥3) — the feature meets the moment. Quote or
  paraphrase a piece of the actual on-screen output the mockup shows. End
  with the emotional landing beat (`I just sat there`, `That was the
  thing I'd been avoiding`, `I had to put the phone down`).
  - Audio for K>1 NEVER opens with greetings, "hi", "as I was saying",
    or "so this is the app".
  - Opens mid-thought.

- Board N (CLOSER) — soft personal endorsement + CTA. ALL of:
  - 1 sentence of explicit honesty about what the app is NOT
    (`I'm not saying an app reads your soul`, It's not therapy, `It
    won't fix everything`) — this is what makes it feel non-salesy.
  - 1 sentence of what it IS for the creator (`But it's become my mirror`,
    `It's the one habit that stuck`).
  - 1 sentence of CTA naming the feature and platform (`It's free on iOS
    and Android. Try the [feature name]. See what comes up.`). NEVER
    you have to try this, NEVER you NEED this, NEVER obsessed.

Forbidden AI-tell phrases (NEVER use, inherits from `ugc-clip-prompt.md`):
obsessed, literally obsessed, game changer, 10/10, 100%,
you have to try this, you NEED this, mind-blowing, unreal.

Forbidden first words (positional): OK, Okay, Alright, So,
Yeah so, Right so, Um, Well, Like, Wait, Hold on. Rewrite the
opener if the draft starts with any of these.

### Step 6 — Parallel: character + screen mockup

Submit ONE higgsfield_generate(requests=[...]) call with both items:
- request 1: text2image_soul_v2 (character prompt from Step 3)
- request 2: imagegen_2_0 (screen mockup prompt from Step 4)

Poll both, download to assets/character.png and assets/app_screen.png,
re-upload both via higgsfield_upload(files=[...]), capture two media IDs.

### Step 7 — Boards (sequential, mandatory)

Boards must be generated sequentially — each board after Board 1 takes the
previous board as a 3rd reference for continuity. Order: medias = [screen, character, previous_board].

Board 1 prompt rules (build per ugc-boards.md, with these
substitutions):
- Treat the screen mockup like a product image with strict Angle Lock:
  `the phone shows only the visible front-facing side from @Image1 with
  the exact same app screen content — same exact UI, same exact text,
  same exact symbols, same exact colors`.
- Phone visibility cadence across 3 slots:
  - Slot 1: phone NOT visible (creator just talking, the story moment).
  - Slot 2: phone visible in hand, screen toward camera, feature screen readable.
  - Slot 3: phone NOT visible (creator's emotional reaction, hand on collarbone or similar grounding gesture).
- Distance band rule: TIGHT → MEDIUM → WAIST-UP/WIDE (must span tight, mid, wide).
- POV cadence: SELFIE → TRIPOD → SELFIE.

Boards K>1: keep wardrobe / room / lighting / palette IDENTICAL to
previous board, story continues mid-thought. Distance and POV cadence per
the arc role tables in ugc-boards.md.

Per board: imagegen_2_0, aspect "16:9", resolution: "2k", nested
medias data wrapper. Poll → download → re-upload → capture board_K_media_id.

### Step 8 — Seedance prompts (parallel submission)

For each board, write a Seedance prompt per ugc-clip-prompt.md with the
Pattern A calm overrides in this skill. Submit all N clips in a SINGLE
higgsfield_generate(requests=[r1, r2, ..., rN]) call:

**Operator rule (mandatory monitoring):** once video generation starts, do not leave it as a blind wait. Actively monitor each running stage and report progress checkpoints (`clip_1 done`, `clip_2 running`, `clip_3 queued/running`) until final stitch is complete. Use both local process status and Higgs job status so the user can see forward motion, not just static images.
{
  "type": "generation",
  "model": "seedance_2_0",
  "media_type": "video",
  "params": {
    "prompt": "<board K Seedance prompt>",
    "duration": <per-clip>,
    "aspect_ratio": "9:16",
    "resolution": "1080p",
    "generate_audio": true,
    "medias": [
      {"role": "image", "data": {"id": "<BOARD_K_MEDIA_ID>",   "type": "media_input"}},
      {"role": "image", "data": {"id": "<CHARACTER_MEDIA_ID>", "type": "media_input"}},
      {"role": "image", "data": {"id": "<SCREEN_MEDIA_ID>",    "type": "media_input"}}
    ]
  }
}

Respect the workspace concurrent-jobs cap (see system prompt `limits`) —
loop in chunks if N > cap. Default cap is high enough for 3–4 boards at once.

### Step 9 — Poll, download, stitch

Poll batched with higgsfield_job_status(job_ids=[...], poll=true). Download
each clip to output/clip_<K>.mp4 (suffix must equal board index).

Stitch via the montage skill:
- Build output/filelist.txt with absolute paths (each line: `file '/abs/path/to/clip_K.mp4'`).
  Relative paths fail when ffmpeg is invoked from a different working directory.
- ffmpeg -y -f concat -safe 0 -i output/filelist.txt -c copy output/final.mp4
- Verify with ffprobe (width / height / duration / fps).

### Step 10 — Deliver

higgsfield_upload(files=["output/final.mp4"]) then embed in the reply
as MEDIA:/abs/path/to/final.mp4. Save an artifact line with model, prompt
gist, job ID, and result URL.

In the reply, include:
1. The MEDIA path.
2. A scene-by-scene narrative breakdown (3–4 lines per beat).
3. Why-it-works bullet tied to the brief's emotional ask.
4. Caveats (anything that didn't render perfectly — pose oddities, hand
   issues, identity drift if any).
5. Offer next-step variations: other languages, shorter cuts (15s Story),
   different creator look.

## ask_user_question — what to ask, what to never ask

### NEVER ask (workflow defaults are pinned)

- Aspect ratio (locked to 9:16).
- Model selection (Soul 2.0 for character, GPT Image 2.0 for boards + screen, Seedance 2.0 for video).
- Resolution / quality (per-step pinned).
- Audio (always generate_audio: true, off-camera/on-camera VO built in).
- N boards or per-clip duration (derived from total duration table).
- Stitch method (hard cut via `montage`).
- Tone register (locked to Pattern A calm-intimate unless brief explicitly contradicts).

### DO ask (only when missing AND ambiguous)

| Question | Ask when |
|---|---|
| Duration | Brief did not specify a length. Offer 15s / 30s / 45s / 60s. Default 45s. |
| Language | Brief did not specify, AND the app's primary site language is unclear. |
| Feature focus | Brief mentioned the app but not which feature. Offer the 3–5 most prominent features from the site extraction. |

Bundle into ONE call if more than one is missing.

## Tone calibration — DO and DO NOT

| Quality | Do | Do NOT |
|---|---|---|
| Opening | Personal moment ("I had this dream three nights in a row") | Greetings ("Hey guys, today...") |
| Energy | Calm, intimate, confiding | Hyped, screaming, explosive |
| Reveal beat | Quote the app's actual output as the emotional anchor | Generic praise ("it's amazing", "obsessed") |
| Endorsement | Soft + honest about what the app is NOT | "Game changer", "you NEED this" |
| CTA | "Try the [feature]. See what comes up." | "Download now!", "Click the link!" |
| Body language | Hand to collarbone, slow blink, weight shift | Open-mouth scream, head jerk back |
| Lighting | Cool diffused window daylight | Golden hour, warm sunset, amber cast |

## Voice consistency across clips (read before generating N≥2)

Seedance 2.0 generates audio natively in the same pass as the video, but
**does NOT accept a voice_id, voice reference audio, or any cross-job
voice token**. Each clip is generated independently. Seedance picks vocal
parameters (timbre, perceived age, accent, pitch, cadence) internally from
the textual description in the Audio: block. Without active prompt
anchoring, the voice can drift between clips and a careful listener will
notice — different timbre, slightly different pitch, slightly different
accent register.

This skill achieves ~70–85% perceptual consistency via prompt anchoring.
That is enough for most viewers, but not a mathematical guarantee. If
the brief explicitly requires identical voice across clips (broadcast ad,
high-budget campaign, audio-led brand), use the fallback path below.

### Mandatory voice-anchor block — IDENTICAL across every clip

Every clip's Audio: line MUST start with the EXACT same opening
descriptor — copy-pasted verbatim, no rewording, no synonym swaps, no
variations. Template:
Audio: She speaks to camera, iPhone microphone audio with natural room
tone, warm intimate cadence, slightly hushed like she's confiding,
mid-thirties [or matched-age] warm voice, no accent shift, no pitch shift,
same speaker throughout: "<monologue segment>"

Lock these tokens identically across all N clips:

| Anchor | Required form |
|---|---|
| Pronoun + verb | She speaks to camera / He speaks to camera — never Her voice, She says, She narrates |
| Mic descriptor | iPhone microphone audio with natural room tone |
| Cadence | one fixed descriptor (e.g. `warm intimate cadence, slightly hushed like she's confiding`) — same exact wording in every clip |
| Age anchor | mid-thirties warm voice (or whatever matches the character) — same exact wording |
| Stability tokens | no accent shift, no pitch shift, same speaker throughout — appended verbatim |
| Language | same language flag (English / Spanish / etc.) in every clip; never mix |

For K>1 clips the rest of the rules from ugc-clip-prompt.md still apply:
no greetings, no [*explosive gasp*] bracketed sounds, open mid-thought.

### Voice gender lock

Set voice_gender once at the start of the pipeline based on the
character. Default female for wellness / spiritual / lifestyle app
briefs. Lock it across:
- the character prompt (`young woman` / young man in Soul 2.0 prompt)
- every board prompt's character reference language
- every Seedance clip's Audio: line (`She speaks` / `He speaks`)

Never mix genders inside one pipeline run.

### Fallback — when mathematical voice identity is required

If the brief demands strict voice identity (broadcast ad, audio-first brand
campaign, the user explicitly says "same voice across all clips"), switch
to this path:

1. Generate every Seedance clip with generate_audio: false and
   Audio: block replaced with `Audio: silent, ambient room tone only,
   natural room reverb, no dialogue` (Seedance still needs an Audio:
   block, just non-speech).
2. Generate the full monologue (all N segments concatenated) as a single
   TTS file using a provider with stable voice_id — ElevenLabs,
   OpenAI TTS, or Play.ht. Use the same voice_id and same parameters
   across the whole monologue.
3. Time-align each TTS segment to the corresponding clip duration. Pad
   with silence at segment boundaries if needed.
4. Mix in ffmpeg:
     ffmpeg -i output/final.mp4 -i tts_full.mp3 \
     -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest \
     output/final_with_tts.mp4
   
Trade-off: you lose Seedance's automatic lip-sync. The clip's mouth
movements will follow the original silent generation's micro-expressions,
not the TTS audio. This is usually acceptable for off-camera narration
framing but visible for tight close-ups of the mouth.

Alternative fallback: kling3_0 — supports voice extraction from a
reference video. Trade-off: visual quality below Seedance for this
use case, and aspect-ratio support is narrower.

### Don't do this

- Don't change the cadence descriptor between clips ("warm intimate
  cadence" in clip 1, "soft confiding tone" in clip 2 — Seedance reads
  these as different speaker briefs).
- Don't add bracketed non-verbal sounds (`[*sigh*]`, `[*small laugh*]`)
  to only some clips — they shift the model's voice-selection prior.
  This skill says: no bracketed sounds at all (Pattern A calm lock).
- Don't omit the age anchor — without it, Seedance can drift the
  perceived age between clips by 5–10 years.
- Don't write She speaks softly in one clip and `She speaks with
  warmth` in another. Pick one phrasing, repeat verbatim.

## Critical pitfalls (already debugged)

- **Do not start Seedance from text-only prompts when continuity matters.** For this workflow, visual coherence is mandatory: generate `character` + `app_screen` first, then sequential boards (B1→B2→B3/4), and only then generate video clips from those boards. Skipping this order causes identity/wardrobe/lighting drift and inconsistent product UI.
- If the user reports quality failure (deformed format, weak hook, broken close), **stop the current batch immediately**, report status clearly, and restart from the beginning of the flow (new assets + new boards), instead of patching only the final concat step.
- If a video stage appears "stuck" (no fresh stdout for 2–3 min), do not assume idle/failure: check both the local runner process and `higgsfield generate list --video` for newly completed jobs, then continue the pipeline and post a concise status update.

- ffmpeg concat fails with relative paths. When ffmpeg -f concat is
  run, paths inside the filelist are resolved relative to the filelist's
  own directory, NOT the current working directory. Always write absolute
  paths in filelist.txt.
- Board chaining is sequential. Board K depends on Board K−1 as a
  reference for identity / room / lighting continuity. Do NOT submit boards
in parallel — they will drift.
- Re-upload after every image step. Generated job IDs are
  image_job type; the boards / Seedance accept media_input. After polling,
  download and higgsfield_upload to convert. (Alternatively pass
  type: "image_job" — both work, but media_input via re-upload is the
  pattern this skill uses end-to-end.)
- Wardrobe modesty triplet. Already in soul-v2-ugc-character.md —
  do not skip it. Seedance will reject NSFW-adjacent compositions
  silently otherwise.
- The screen mockup IS the product. Treat it with strict Angle Lock
  in every board prompt — same exact text, same symbols, same layout
  across every appearance. The model will otherwise invent new UI per slot
  and the brand reads as inconsistent.
- Voice gender lock. Match the brief. Don't mix genders mid-pipeline.
  Default female for wellness / spiritual / lifestyle apps unless the
  brief specifies otherwise.

## BeSoul-specific messaging add-on

For BeSoul's AI natal-chart feature (45s EN storytelling), use:
- `references/besoul-natal-chart-storytelling.md`

This reference enforces a strong hook, non-generic copy, zodiac-vs-natal-chart contrast, and an empathy-first closer.

For BeSoul viral/save-worthy 60s storytelling, also use:
- `references/besoul-viral-curiosity-60s-template.md`
- `references/besoul-no-talking-heads-storytelling.md` (when user rejects ad look / says no women talking)

Reference add-on: `references/visual-coherence-execution-order.md` (enforces image-first execution order before Seedance).

### User-preference guardrail (critical)
When the user flags prior output as "not in English", "not viral enough", or "too horoscope-centric", switch immediately to this lock:
- Language: English-first copy and VO by default for BeSoul.
- Objective: maximize save/share/rewatch mechanics (hook + emotional turn + soft CTA), not generic feature listing.
- Topic constraint: avoid horoscope/zodiac framing unless the user explicitly requests horoscope content.
- Calibration step before full production: deliver **2 short pilot concepts** (hook + arc + CTA) and wait for user pick/feedback.
- After user supplies reference reels, extract common mechanics (hook pattern, emotional beat, cadence, CTA style) and mirror those mechanics in the pilots before generating final assets.

### BeSoul language/topic guardrail (user-correction, mandatory)

When the user flags quality with feedback like "no está en inglés", "no sigue regla viral", or rejects horoscope framing:
- Default output language to **English-first** (Spanish optional only if requested).
- Optimize script for **viral mechanics** (strong 1–2s hook, retention turns every 4–7s, explicit save/share CTA in final beat).
- **Do not center horoscope/zodiac tropes** unless the user explicitly asks for horoscope content; prefer broader emotional/self-discovery angles.
- If a non-compliant piece was already published, stop batch iteration and switch to review mode (show template used + list recent post links by day) before generating the next asset.

## Reference resolution table

| Step | Skill | File path | Purpose |
|---|---|---|---|
| Character generation | image-generation | references/soul-v2-ugc-character.md | Soul 2.0 creator prompt rules + variety roll |
| Storyboard generation | image-generation | references/ugc-boards.md | 16:9 3-slot UGC board structure |
| Seedance prompt | video-generation | references/ugc-clip-prompt.md | 3-cut clip prompt structure |
| Concat | montage | (SKILL.md) | ffmpeg hard-cut stitching |
| Brief extraction (URL) | web_extract (tool, not skill) | — | Site analysis for brand voice |

## BeSoul viral curiosity variant (60s, EN-only default when requested)

Use this branch when the user explicitly asks for **viral/save-worthy** content ("datos curiosos", short stories, curiosity hooks) instead of direct feature promo.

### Format lock
- Language: English only if user says "solo inglés".
- Target duration: 60s (acceptable 55–60s with natural pacing).
- Voiceover-first structure: single continuous VO track, then scene assembly.
- Background music: soft ambient at ~30% under VO (voice remains dominant).
- Scenes: 10–12 short shots that map to VO beats, then stitch with soft transitions.
- No subtitles unless explicitly requested.
- If user rejects "ad-like" output or says "No mujeres hablando", switch to **no talking-head mode**: abstract/atmospheric/app-UI visuals only, no woman/man speaking directly to camera.
- In no talking-head mode, avoid hard-sell composition patterns (presenter framing, promo overlays, salesy transitions); keep intimate, reflective pacing.

### BeSoul anti-ad / no-talking-head override (user-correction driven)
When the user asks for "storytelling curiosity" and rejects ad-like output (e.g. "no mujeres hablando"), apply this override:
- Hard forbid: female talking heads, direct-to-camera presenter shots, UGC selfie confession frames.
- Visual strategy: abstract/mystic b-roll, symbolic closeups, app UI fragments, hands-only or silhouette-only shots.
- Keep viral structure in audio/script (hook, promise, micro-turns, save CTA) while removing human presenter visuals.
- If existing render contains presenter footage, do a full visual replacement pass (not minor trims) before delivery.
- Validate with frame-sampling before delivery (early/mid/late timestamps) to confirm no presenter shots remain.
- **Delivery gate (mandatory):** before sharing final MEDIA, run a fast compliance checklist in this exact order: (1) no talking heads, (2) non-ad tone, (3) hook visible in first 1–2s, (4) save-trigger CTA present in final beat. If any check fails, regenerate/re-edit before delivery.
- **Language trigger mapping:** treat Spanish feedback like "parece ad", "se ve anuncio", "no mujeres hablando", "no talking head", "más orgánico" as the same override signal and switch modes immediately without asking.

### Script pattern (viral/save-worthy)
1. Hook in first 1–2s (pattern interrupt / curiosity gap).
2. Personal friction line ("this felt generic", "I kept repeating this").
3. Reveal insight mechanism (why deeper than horoscope-level generalization).
4. 1–2 concrete relatable realizations.
5. Honesty line (not magic, not therapy).
6. Save/share CTA ("save this and test tonight").

### Timing & assembly gates
- Generate VO first and **measure real duration** with ffprobe.
- Build shot list against measured duration windows (not estimated script length).
- Keep cut rhythm around 3–5s per shot for 60s storytelling reels.
- If VO lands at ~55s, do not pad with filler copy; keep concise and finish strong.

### Audio mix guardrail
- Preserve voice intelligibility: VO at 100%, BGM around 30% equivalent (or ~-10 dB to -12 dB under VO depending track loudness).
- Avoid busy percussion BGM that competes with confessional tone.

## NOT for this workflow

- Physical product UGC (creator holds a real bottle, tube, device) → ugc-flow or ugc-product-flow.
- Step-by-step in-app screen demonstration → ugc-tutorial-flow (creator narrates each tap).
- Hard-sell ad creative with maximalist energy → tv-ad or motion-design-flow.
- Multi-feature app showcase (no single feature focus) → standard cinematic-flow brand story.
- High-energy comedy / skit register → use base ugc-flow with Pattern B defaults.