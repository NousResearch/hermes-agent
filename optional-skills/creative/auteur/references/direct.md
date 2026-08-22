> **Hermes adaptation note:** upstream auteur generated assets through local agent CLIs (`agy`, `codex`, `grok`). In Hermes, read every such invocation as a call to the built-in `image_generate` tool with the same prompt (then move the returned file into the project's `assets/gen/` path), use the `terminal` tool for `ffmpeg`/`node`/`npx`, and `browser_exec` or Playwright-via-terminal for screenshot loops. The per-CLI routing/strength tables below are upstream reference material — the taste guidance transfers, the CLI names do not.

# direct.md — the cinematic register

In this register the page is a film: the viewport is the frame, scroll is the timeline, sections are scenes. You are the director, and directors do not start by shooting — they start with a script. Every phase below ends with a gate; do not cross a gate that fails.

## Phase 0a — Intake (one message)

Ask once, compactly: product & what it does · audience · the ONE feeling a visitor should leave with (awe / calm / hunger / trust / momentum...) · brand constraints (colors, fonts, logo — if any) · assets that already exist (photos, video, 3D, none) · where it will be hosted (static vs framework). If working autonomously, derive answers from available materials and write every derived answer into the STORYBOARD header's `Assumptions made` field — not into your own reasoning, where the next session cannot see it.

## Phase 0a.5 — Recon (steal like a director)

Before writing the screenplay, spend one short bounded pass gathering live reference — the reflex table in taste.md tells you what to avoid; recon tells you what's currently *alive*. Load `references/recon.md` and run both legs:

```bash
node scripts/refscout.mjs --from awwwards --limit 8          # → design/refs/REFERENCES.md + shots
node scripts/moodboard.mjs "<feeling>" "<treatment>" --limit 24   # → design/moodboard/contact-sheet.png
```

- **refscout** profiles live award-level sites: their real stack, pinned scenes, scroll budget, fonts and painted palette, plus screenshots. You are hunting for *mechanics*, not skins. Look at every shot — the numbers describe the machinery, only your eyes judge the film.
- **moodboard** answers the other question: what should this *feel* like. Two or three queries on different axes (subject / treatment / graphic language), then fill the four read-lines in `MOODBOARD.md`; they feed commit-sheet fields 2 and 4 and every scene-sheet's `lighting:`.
- Note in the storyboard header (`References taken`): 2–3 named references and the ONE mechanic taken from each ("madewithgsap.com — section title pinned while cards scroll through it"). Stolen ideas get adapted to this brand, never copied wholesale — a reference is a starting camera position, not a set.
- Sites the tool marks **NO CAPTURE** withheld their CSS/JS from the headless browser; open them yourself or drop them, never quote a fingerprint it refused to give. No playwright / no network → skip recon without guilt; the reflex table + transition library carry you. Never cite a reference you didn't actually see.
- Whatever recon shows five times IS the category's first-order reflex — that finding belongs in commit-sheet field 6a, and your peak has to deviate from it.

## Phase 0b — Screenplay

Copy `templates/STORYBOARD.md` into the project (`design/STORYBOARD.md`) and write the film:

**Structure: 5–7 scenes, classic arc.**

| Beat | Role | Typical scenes |
|---|---|---|
| Hook | stop the scroll, set the world | 1 (the hero) |
| Rising | develop the promise | 1–2 |
| **Peak** | the ONE wow moment | exactly 1 |
| Proof | make it credible | 1–2 |
| Door | the CTA, land the feeling | 1 |

**Dramaturgy rules:**
- Score each scene's intensity 1–10. Exactly one scene ≥8 (the peak). The hero hooks at 6–7 — if the hero *is* the peak, the rest of the page must consciously de-escalate (harder to pull off; prefer the peak at 40–70% depth).
- Two adjacent scenes must not share the same layout family or the same motion family. The cut between scenes is part of the film — pick every transition deliberately (library in scroll-cinema.md).
- The feeling from intake is the film's key. Every scene either builds it or contrasts it deliberately; a scene that does neither gets cut. Fewer, better scenes beat more scenes.

**Scene-sheet — fill every field for every scene:**

```
### Scene N — <name>            | beat: hook|rising|peak|proof|door | intensity: 1-10
purpose:      what the viewer must FEEL and LEARN here (one line each)
subject:      the single visual subject (product | image | typography | data | scene)
layout_family: full-bleed-media | split-asymmetric | centred-type | stacked-cards |
              editorial-columns | pinned-canvas | marginal-notes   (must differ from both neighbours)
motion_family: scroll-scrub | pinned-stage | entrance-reveal | parallax-depth | kinetic-type |
              ambient-loop | none   (≤3 distinct families page-wide; must differ from both neighbours)
camera:       POV & framing — eye-level / low-angle (heroic) / high-angle (overview) /
              macro (detail) / orbital (show all sides) / static
lighting:     mood of the frame — hard contrast / golden / dusk / studio / neon / paper-flat
motion:       what moves, in one sentence, incl. what drives it (scroll-scrub | entrance | loop | hover)
transition_in / transition_out:  from the library (cut / wipe-mask / curtain / letterbox /
              shutter / depth-parallax / displacement / view-transition)
scroll_len:   how much scroll this scene owns (100vh–400vh; peak usually 300–400vh pinned)
copy:         the headline + subline that live in this scene (write the actual words)
media:        the director's shot spec for this scene's asset (→ the asset plan; route via assets.md):
              · type:  still | A→B morph | video | sequence | element/texture | 3D model | HDRI | none (type-led)
              · route: SOURCE or GENERATE — assets.md §0.5 decides. Geometry, IBL lighting and tiling
                       materials are SOURCE (source.mjs, CC0); the peak keyframe and the hero video are
                       always GENERATE; stock video is never the peak
              · tool:  codex (peak photoreal) | grok-4.5 (color hero + ANYTHING that becomes video) | agy (volume/elements)
                       | source.mjs hdri|model|texture|icon|image|video
              · frame prompt: the literal keyframe prompt = subject + camera + lighting + palette anchor (write it now)
              · motion prompt: (video/morph only) what moves — e.g. "grok image_to_video on frame A: slow push-in + steam, 6s". Controlled A→B state change = WebGL displacement morph (two frames), NOT a video model
              · score:  (peak/ambient scenes only) mood/tempo for MiniMax music, or "none"
fallback:     what this scene is when WebGL/video/motion is unavailable (static frame + one line)
```

`camera` and `lighting` matter even for pure-CSS scenes: they discipline composition (low-angle → oversized subject, viewer looks up; macro → crop tighter than comfortable) and they become literal prompt parameters when the asset is AI-generated.

**GATE 0:** storyboard complete; exactly one peak; adjacent scenes differ in layout & motion family; every scene has a fallback and real copy (not lorem). If the user is present, get the storyboard approved — cheapest possible moment to change the film. Then fill the commit-sheet (SKILL.md) — the storyboard feeds it.

## Phase 0c — Style gate (mockup before the shoot)

Changing the art direction after six scenes are built costs a rebuild; changing it on one static screen costs minutes. Before asset production:

1. Build ONE static hero screen as a throwaway HTML file (`design/mockup-hero.html`): real headline copy, the commit-sheet palette and type, the grid break — **no animations, no assets** (a solid-color placeholder block where the generated keyframe will live). 15–30 minutes of work, not more.
2. Screenshot it at 1440 and 390 (shoot.mjs on the file), look at it, and run the taste.md §8 self-check on the *image*.
3. Run `node scripts/slopscan.mjs design/` on the mockup. It costs nothing and this is the moment to catch a banned gradient or a contrast failure — *before* these tokens become the project tokens.
4. User present → show the screenshots and get a yes/no on the art direction (offer 2 variants only if genuinely torn — a director proposes, not a menu). Autonomous → self-check against the commit-sheet and record the verdict in the storyboard header.
5. Three verdicts, not two:
   - **approved** → the mockup's CSS custom properties become the project tokens verbatim.
   - **approved with carried notes** → good enough to build on, but with named problems you are knowingly carrying (a rule that will lose against real photography, a provisional typeface). Write them into the storyboard header's `Style gate verdict` field and resolve them by GATE 2. Undeclared carried notes become permanent.
   - **rejected** → cheap redo of phase 0c, not of the film.

## Phase 1 — Asset production

Load `references/assets.md` and derive the asset plan from the storyboard's `media:` blocks. Split it in two before spending anything: everything routed SOURCE is fetched first (`source.mjs`, minutes and free), because a real CC0 mesh or HDRI often changes what the generated frames around it need to be.

Order of operations (cost discipline):
1. List every needed asset with its scene, target resolution, and technique (from the selection table in scroll-cinema.md).
2. Generate ONE keyframe first (the peak scene's frame A). Check it against `lighting`/`camera` of the scene-sheet. Only after it's right, produce its frame B via *edit* and the remaining scenes' assets — this catches a wrong art direction at 1 image of cost, not 12.
3. Optimize everything (recipes in assets.md), verify weights against budget (hero ≤2MB video / ≤300KB poster / ≤150KB per sequence frame at 1440w).

**GATE 1:** every scene's asset exists on disk at final path, weights within budget, frame A/B pairs verified same-scene-same-camera (open both, compare eyes-on), poster/fallback image exists for every heavy asset.

## Phase 2 — Assembly

Load `references/scroll-cinema.md`. Build order is not negotiable (it prevents the classic "everything jitters" rebuild):

1. **Foundation first:** Lenis + GSAP ScrollTrigger integration skeleton (or CSS `animation-timeline` for simple scenes with the `@supports` fallback). No scene work until smooth scroll runs clean.
2. **Hero scene** — sets the technical pattern for everything after it.
3. **Scenes top-to-bottom**, one at a time; `ScrollTrigger.refresh()` after each. Wire each scene's `transition_in/out` from the library as you go — transitions are scene work, not polish.
4. **Text reveals** on headings/paragraphs per motion.md numbers — this stitches the film together.
5. **Reduced-motion cut**: implement the alternative art direction now (static posters, soft opacity rhythm, no pinning, no scrub), not as an afterthought. It's a *cut of the same film*, and it's also your no-JS/weak-device story.
6. **Mobile pass**: pinned scenes shorten or unpin (`scroll_len` × 0.6), assets swap to 720p/cropped variants, hover-driven moments get touch equivalents or graceful absence.

Performance discipline while assembling: only `transform`/`opacity`; `will-change` only on actively animated layers (and removed after); heavy scenes lazy-init via IntersectionObserver; one `requestAnimationFrame` loop owner (GSAP's ticker) — never parallel rAF loops.

**GATE 2:** every scene works at 390/768/1440; scrolling up replays cleanly (scrub is bidirectional — test it); no console errors; reduced-motion cut watchable end-to-end.

## Phase 3 — Verification

Load `references/verify.md`, run the full pipeline (slopscan → shoot → rubric), and fill `templates/CINEMA-QA.md`. The film ships when QA is all PASS — and after you have *watched your own film*: one uninterrupted slow scroll top to bottom, then one fast. Jank you can feel beats any metric.

## Phase 4 — Lock the style (DESIGN.md)

A shipped film gets sequels: "add a testimonials scene", "swap the pricing". Without a locked style contract, every later edit — by you, another model, or another session — drifts back toward the mode. After QA passes, fill `templates/DESIGN.md` into `design/DESIGN.md`: the actual tokens, type system, motion vocabulary, section-opening patterns, and this project's own ban additions. Every future edit starts by reading it (the `edit` route in SKILL.md enforces this). This is what makes the style *survive you*.

## Sound (optional scene layer)

Only if the brief asks for atmosphere: one ambient loop, off by default, visible mute/unmute toggle, starts only on user gesture (autoplay policies), volume ≤0.3, `prefers-reduced-motion` implies silent default. Policy details in motion.md §Sound. Sound is seasoning — a silent film that wows silently is complete.
