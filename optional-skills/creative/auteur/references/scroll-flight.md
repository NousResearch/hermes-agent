> **Hermes adaptation note:** upstream auteur generated assets through local agent CLIs (`agy`, `codex`, `grok`). In Hermes, read every such invocation as a call to the built-in `image_generate` tool with the same prompt (then move the returned file into the project's `assets/gen/` path), use the `terminal` tool for `ffmpeg`/`node`/`npx`, and `browser_exec` or Playwright-via-terminal for screenshot loops. The per-CLI routing/strength tables below are upstream reference material — the taste guidance transfers, the CLI names do not.

# scroll-flight — photoreal scroll-scrubbed video ("fly through the world")

auteur's **video-scrub tier**. The hero is a pre-rendered camera flight whose
`currentTime` is driven by scroll — the viewer *pilots* a photoreal world.
Complementary to the real-time WebGL recipes in `scroll-cinema.md`, not a
replacement.

Engine: `templates/scroll-flight-engine.js` — a zero-dependency, framework-
agnostic, drop-in scrubber. Vendored from **scroll-world**
(github.com/cth9191/scroll-world, MIT © cyw); it solves ~18 shipped-in-anger
edge cases you do NOT want to re-derive. Read its header for the full config API.

## When to reach for this (vs WebGL scroll-cinema)

| Use scroll-flight (video) when… | Use WebGL scroll-cinema when… |
|---|---|
| the world must look **photoreal** — a real place, product, interior, landscape | the look is generative/abstract — fluid, particles, shaders, type |
| the motion is a **camera flight** through a fixed scene | the motion is procedural and reacts to cursor/audio/data live |
| you can generate/shoot **video clips** of it | you can express it as math in one WebGL context |

They compose: a WebGL hero can hand off into a scrubbed-video mid-section.

## ⚠️ AI clips barely move the camera — the scroll-dolly does the travelling

The single most important thing to know here. **`grok image_to_video` (and
most image→video models) animate a still *ambiently* — light shimmers, water
drifts, particles float — but they do NOT fly the camera through the scene.**
Verified on disk: a clip's first and last frame are near-identical. So a scene
built only from a raw AI clip reads as *a slightly-moving photo*, not a journey.
This is the #1 reason a scroll-flight looks like a slideshow.

The fix lives in the engine, not the prompt: `scroll-flight-engine.js` applies a
**scroll-driven camera dolly** — as you scroll through a scene, it pushes the
whole scene IN (scale) and drifts it down, inside the scale overscan so edges
never reveal. *That* is what manufactures forward/descent travel; the clip's
`currentTime` scrub only adds the ambient life on top. Consequences:

- **Don't over-invest in per-clip motion.** A near-static clip + the dolly looks
  the same as an aggressively-prompted clip + the dolly. Prompt for *mood and
  ambient life* (drifting particles, light, creatures), not "fast camera dash"
  you won't get.
- **Stills and clips are interchangeable.** A scene with only a `still` gets the
  same dolly, so it reads identically to a video scene. Mix freely — ship the
  clips you reliably get, fill the rest with stills, the dolly unifies them.
  (ABYSS showcase: 2 video scenes + 3 stills, indistinguishable.)
- **Pick worlds whose medium drifts** — underwater, clouds, space, dust, smoke.
  Ambient drift hides the lack of camera-baked motion; a dry static landscape
  exposes it.
- Real camera travel baked into the footage needs a true flythrough model
  (Higgsfield/Runway/Veo) or the 2.5D depth-parallax route (still → depth map →
  WebGL camera through the layers). The dolly is the pragmatic default that
  needs neither.

## The pipeline (auteur's toolchain)

1. **Scene stills** — one anchor still first, get art-direction approval, then
   batch the rest **style-locked to the anchor** (pass the approved still as the
   style reference). A style miss caught on the anchor costs 1 gen, not N.
   Sources: `codex`/`agy` (Gemini)/`grok` image gen — see `assets.md`.
2. **Dive clips** — animate each still into a short camera push-in (grok
   `image_to_video`, or any image→video model). One clip per scene.
3. **Seams** — two ways, pick by what your video model can do:
   - **Crossfade seams (default, grok/start-image-only models).** Leave
     `connectors` empty/`null`; the engine crossfades directly between adjacent
     dives. Ship-safe, always works, reads clean. This is auteur's baseline.
   - **Seamless flight (only with an end-image model — Higgsfield seedance /
     kling).** Generate connector clips whose **start = prev dive's last frame,
     end = next dive's first frame** (extract the actual rendered frames, never
     the stills). Verify every seam with the SSIM gate below before eyeballing.
4. **Encode for scrubbing** (§ below) — the single most important step.
5. **Posters** — extract each encoded clip's first frame (§ below).
6. **Wire** the engine config; run the motion + slopscan gates.

> auteur has no Higgsfield account by default; grok `image_to_video` is the
> baseline. So the honest default is **crossfade-seam** photoreal scrub — still
> Apple-tier. Fully-seamless chaining is an upgrade you unlock only with an
> end-image-capable video model.

## Encode for scrubbing — the `-g 8` recipe (critical)

Scrubbing sets `currentTime` every frame; a decoder's **seek cost scales with how
many frames it must decode from the nearest keyframe**. A tiny GOP (keyframe
every 8 frames) is what makes frame-accurate seeking cheap. Native res, crf 20,
no audio, faststart, light sharpen:

```bash
enc() { ffmpeg -v error -y -i "$1" -an -vf "unsharp=5:5:0.8:5:5:0.0" \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p \
  -g 8 -keyint_min 8 -sc_threshold 0 -movflags +faststart "$2"; }
```

- **Never upscale** — encode what `ffprobe` reports (some models return 720p).
- **Mobile sibling** (`-m.mp4`): 720p, `-g 4` (twice the keyframes = ~half the
  seek-decode work), crf 23. Wire as `clipMobile`/`connectorsMobile`. Still
  choppy on a low-end phone → `-g 2`, or `-g 1` (all-intra = instant seeks,
  bigger files).
- The engine loads each clip as a **Blob** (always seekable) and scrubs — it does
  NOT rely on HTTP byte-range. Do not "optimize" that away, or you get frozen-at-
  frame-0 on hosts that don't serve ranges.

## Posters — from the ENCODED clip's first frame

The still is 3:2, the clip is a 16:9 re-render — if the still is the loading
poster, the video paints with a visible crop/render pop on the first scene a
visitor sees. Hand off the actual frame:

```bash
ffmpeg -v error -y -ss 0 -i "$ASSETS/vid/$n.mp4" -frames:v 1 -q:v 2 poster.png
cwebp -quiet -q 84 poster.png -o "$ASSETS/$n-poster.webp"   # → sections[k].poster
```

Keep the source still too: it's the reduced-motion artwork and the no-clip
fallback.

## Seam QA — SSIM gate (before any eyeballing)

Seamlessness is the product; don't ship it on a squint. A true actual-frame
handoff scores SSIM ≥0.95 even after encoding.

```bash
seam_ssim() { # clipA clipB  — last frame of A vs first of B
  ffmpeg -v error -y -sseof -0.05 -i "$1" -frames:v 1 _a.png
  ffmpeg -v error -y -ss 0      -i "$2" -frames:v 1 _b.png
  ffmpeg -v info -i _a.png -i _b.png -lavfi ssim -f null - 2>&1 | grep -o 'All:[0-9.]*' | cut -d: -f2
}
# ≥0.90 pass · 0.75–0.90 warn (crossfade usually hides it) · <0.75 FAIL:
# an endpoint was a still, not the neighbour's frame — regenerate, don't rationalize.
```

Re-run after every re-roll: replacing one clip can silently break BOTH its seams.

## Chain architecture — A vs B

- **A — one continuous forward take.** Legs chained from actual last frames, no
  pull-back, no end-image. Use for any grounded walkthrough. No rewind risk.
- **B — dive + connector interleave.** More cinematic, but if a connector's
  camera **velocity reverses** (dive pushes in, connector pulls back out) it
  reads as a rewind even with a perfect frame-match seam. Inherent to B — keep
  connector motion continuing forward, or use A.

## Mobile & iOS — the hard gotchas (the engine handles these; don't undo them)

- **Frozen / stuck at frame 0** → host isn't serving byte ranges → blob URLs (engine does).
- **Blank/black scene on iOS** → a muted video never played won't paint a seeked
  frame. Engine keeps the poster up until a real frame paints and **primes** each
  clip (muted play→pause) on first touch. Don't hide the poster on
  `loadedmetadata`; don't strip `playsinline`/`muted`.
- **Frozen on iOS Low Power Mode** → LPM rejects even muted `play()` and
  `currentTime` scrubbing dies — no video technique survives it. Engine detects
  the rejected prime and flips the whole page to **stills-with-crossfades**. Keep
  that `.catch()` fallback if you adapt it.
- **Phone stutters on a fast flick** → seeks pile up. Engine **coalesces seeks**
  (never issues a new `currentTime` while `seeking`); ship the `-m.mp4` tier as
  the other half.
- **iPad gets blurry 720p** → tier by **screen short side** (≤600 CSS px = phone),
  never by pointer type or UA (iPadOS lies on both).
- **Page jumps while scrolling** → the mobile URL bar fires `resize`; the engine
  ignores height-only resizes (relayout on width change / `orientationchange`).
- **Copy behind the notch/URL bar** → engine uses `env(safe-area-inset-bottom)` +
  `dvh`; keep `<meta viewport … viewport-fit=cover>`.

## Accessibility / SEO (auteur floor — enforced)

- `prefers-reduced-motion`, data-saver → **stills mode** (no video load/decode),
  cross-dissolving as you scroll. Never blank.
- The engine builds its DOM in JS, so put a plain-markup copy block marked
  `data-sw-seo` in the container — crawlers, link previews, and no-JS visitors
  read it; the engine hides it on mount. **Readable with JS off.**

## Config pacing knobs

- `sections[k].scroll` — per-scene scroll distance (more = slower dwell).
- `sections[k].linger` (0..1) — remaps scroll→time so the camera **settles
  mid-scene** where the copy peaks and moves quicker at the seams. Keep ≤0.6.
- `diveScroll` / `connScroll`, `crossfade` (seam dissolve width), `scrollMobileFactor`.

## Upgrade path — canvas frame-sequence (Apple's actual technique)

For butter on low-end devices or when the blob payload (~8 MB × clips) is too
heavy: pre-extract N frames per clip (webp/avif), draw to `<canvas>` (WebCodecs
to decode ahead). Frame paint becomes deterministic — no decoder seek latency.
The seam doctrine, chain math, and pacing knobs all carry over; only the "paint
frame at time t" primitive swaps. More build tooling, more requests — reach for
it only when video-scrub genuinely stutters on the target hardware.

---

*Technique & engine adapted from **scroll-world** by cyw
(github.com/cth9191/scroll-world), MIT. auteur pairs it with grok/Gemini asset
generation and its own slopscan / motionqa gates.*
