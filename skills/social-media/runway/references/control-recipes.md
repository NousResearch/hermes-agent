# Control recipes + two worked SHOT briefs

## Pick Runway vs its siblings (route via ai-video)
| Job | Pick | Why |
|---|---|---|
| Consistent character/object across shots | **Runway** (References) | signature consistency |
| Edit/restyle existing footage | **Runway** (Aleph) | video-to-video, no regen |
| Specific camera move / branded ad / previz | **Runway** (Camera Control) | director-grade control |
| Clean cinematic clip **with native audio**, one pass | **veo-3** | Runway has no native audio |
| Native 4K / multi-shot sequence / motion transfer | **kling** (sibling) | resolution + motion leader |
| Cinematic HDR/mood shot (silent, sound in post) | **luma** (sibling) | the Dream Machine lane |
| Talking-head presenter | **heygen** | avatar, not generative scene |

## Recipes
- **Consistent character across a sequence:** lock a Gen-4 Reference → storyboard 5 shots → generate
  each with the same reference + distinct camera move → assemble in the edit → ai-voiceover + captions.
- **Branded product ad:** Gen-4.5 hero shot with a defined camera push-in + brand lighting; test on
  Gen-4 Turbo first; 4K export; clean (no watermark).
- **Fix/restyle existing footage:** Aleph — "relight to golden hour," "remove the sign," "make it
  painterly" — on a 2–30s clip, preserving motion. Review the result.
- **Performance on a character (consented):** Act-Two with a consented driving performance + character
  reference; disclose.

## Worked example 1 — product hero ad (blunt indie-founder voice, ~12s)
```
SHOT: matte-black invoicing app on concrete; slow 180° orbit; hard top light; 9:16
MODEL: Gen-4.5 (hero); tested on Gen-4 Turbo first   REFERENCES: brand product still locked
ASSEMBLE: 10s hero + 1 short cutaway (~12s total); VO via ai-voiceover ("Boring. Works."); captions burned in
CLEAN: 4K, no watermark. DISCLOSURE: "AI-generated" label.
```

## Worked example 2 — calm explainer sequence (warm studio voice, 3 shots)
```
SHOT A: tidy desk, soft morning light, slow push-in (Gen-4.5)
SHOT B: same hands, same mug (Gen-4 References to hold consistency), gentle pan
SHOT C: Aleph restyle of an existing webinar clip to match the warm grade
ASSEMBLE: stitch A-B-C (~30s); ai-voiceover narration; captions via captions-and-clipping
REFERENCES: locked across A/B. DISCLOSURE: "AI-assisted video".
```

Both: read brand-profile + voice-builder first; directorial brief; References for consistency; short
(2–10s) shots stitched in the edit; **no native audio → ai-voiceover**; captions via
captions-and-clipping; consent + disclosure handled. Publish → scheduling-and-queue → WoopSocial.
Avatars → heygen.
