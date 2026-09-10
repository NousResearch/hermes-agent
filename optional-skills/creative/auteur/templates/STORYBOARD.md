# STORYBOARD — <project>

## Film meta
- **Product:**
- **Audience:**
- **The one feeling:** <!-- awe / calm / hunger / trust / momentum … -->
- **Peak scene:** <!-- exactly one, intensity ≥8 -->
- **Assets available up front:** <!-- footage / photos / 3D / none -->
- **Sourced-asset findings:** <!-- what a sourced asset turned out to make possible that you did not plan for. Inspect before you write: a glTF's node names, an HDRI's actual light direction, a texture's tiling. "the mesh has 8 named nodes" is the kind of fact that changes the film, and it must not live only in one session's reasoning. -->
- **Assumptions made:** <!-- if intake was autonomous, every answer you derived instead of asking goes HERE (direct.md §0a) -->
- **References taken:** <!-- 2–3 from design/refs/REFERENCES.md: "site — the ONE mechanic — how it changes here". Only sites you looked at. -->
- **Moodboard read:** <!-- one line: palette relationship + light character taken from design/moodboard, and the one thing on that sheet to avoid -->
- **Style gate verdict:** <!-- approved / approved with carried notes / rejected+redone. Carried notes = known problems you are deliberately building on; list them so they get resolved, not forgotten. -->

## Arc
<!-- hook → rising → PEAK → proof → door. 5–7 scenes. Adjacent scenes must differ in layout family AND motion family — the columns below are what makes Gate 0 checkable instead of a vibe. -->

| # | Scene | Beat | Intensity (1–10) | layout family | motion family |
|---|-------|------|------------------|---------------|---------------|
| 1 |       | hook |                  |               |               |
| 2 |       |      |                  |               |               |
| … |       |      |                  |               |               |

<!-- layout family: full-bleed-media / split-asymmetric / centred-type / stacked-cards / editorial-columns / pinned-canvas / marginal-notes …
     motion family: scroll-scrub / pinned-stage / entrance-reveal / parallax-depth / kinetic-type / ambient-loop / none
     Motion families used across the whole page must be ≤3 (commit-sheet field 5); the same family may
     repeat, just never in two adjacent scenes. -->

---

### Scene 1 — <name>   | beat: hook | intensity: N
- **purpose:** feel: … / learn: …
- **subject:**
- **layout_family / motion_family:** <!-- must match the Arc table -->
- **camera:** <!-- eye-level / low-angle / high-angle / macro / orbital / static -->
- **lighting:** <!-- hard contrast / golden / dusk / studio / neon / paper-flat -->
- **motion:** <!-- what moves, driven by scroll-scrub | entrance | loop | hover -->
- **transition_in / out:** <!-- cut / wipe-mask / curtain / letterbox / shutter / depth-parallax / displacement / view-transition -->
- **scroll_len:** <!-- 100vh–400vh. Literal for pinned and peak scenes; for content-height scenes write "content" rather than a number you will not honour. -->
- **copy:** H: "…" / sub: "…"  <!-- real words, not lorem -->
- **media:** <!-- the director's shot spec — route via assets.md §0.5 (source) and §0 (generate) -->
  - type: <!-- still | A→B morph | video | sequence | element/texture | 3D model | HDRI | none (type-led) -->
  - route: <!-- SOURCE (source.mjs hdri|model|texture|icon|image|video) or GENERATE (codex | grok-4.5 | agy) — assets.md §0.5 decides -->
  - frame prompt: <!-- the literal keyframe prompt = subject + camera + lighting + palette anchor. Write it NOW, not in phase 1. -->
  - motion prompt: <!-- video/morph only. A controlled A→B state change is the WebGL displacement morph, NOT a video model. -->
  - score: <!-- peak/ambient scenes only: mood/tempo for MiniMax music, or "none" -->
- **fallback:** <!-- the scene when WebGL/video/motion is unavailable -->

<!-- repeat per scene -->

## Gate 0 checklist
- [ ] exactly one scene with intensity ≥8
- [ ] no two adjacent scenes share layout family or motion family (check the Arc table)
- [ ] ≤3 distinct motion families across the whole page
- [ ] every scene has real copy and a fallback
- [ ] every `media:` block is filled: type, route, and — for a GENERATED scene — a literal frame prompt; for a SOURCED or live-3D scene, the equivalent direction to the renderer (subject + camera + lighting + palette), which is what you tune the light rig against; for a type-led scene, `none`. A one-line "generate something" is not a producible ask
- [ ] storyboard approved (user) / self-reviewed against the one feeling (autonomous)
