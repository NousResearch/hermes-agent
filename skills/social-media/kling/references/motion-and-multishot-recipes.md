# Motion, multi-shot + two worked examples

## Pick Kling vs its siblings (route via ai-video)
| Job | Pick | Why |
|---|---|---|
| Native 4K / 60fps output | **Kling** | resolution leader |
| Multi-shot sequence (up to 6 cuts) in one clip | **Kling** | AI Director |
| Transfer motion from a reference video | **Kling** (Motion Control) | unique differentiator |
| Strong image-to-video from a still | **Kling** | 3D recon, less warping |
| Production-grade dialogue/audio | **veo-3** | Kling audio ~3/5 |
| Edit existing footage / camera control / consistency | `runway` (via ai-video) | edit-grade controls |
| Cinematic HDR / atmospheric mood shot (silent) | **luma** | 16-bit HDR, Ray3 |
| Talking-head presenter at scale | **heygen** | purpose-built avatars |

## Recipes
- **4K product ad:** image-to-video from a clean product still → describe a slow orbit + brand
  lighting → 4K → audio as guide track (final VO via ai-voiceover). Verify text/reflections.
- **Multi-shot hook (6 cuts):** storyboard 6 beats (establish → detail → reveal → reaction → product →
  CTA) in one ~15s clip; keep the subject consistent; route the on-screen line to the platform writer.
- **Motion-transfer:** clean solo **reference video** of the motion + a **character/product reference**
  → Motion Control applies the move to your subject. Consented reference only.
- **Image-to-video:** animate a `nano-banana`/`ideogram` still — Kling adds motion; you keep exact
  brand/text control from the source frame.

## Worked example 1 — 4K product orbit (blunt indie-founder voice, ~8s)
```
INPUT: clean product still (image-to-video)   TIER: Kling 3.0 4K
SHOT: slow 180° orbit of a matte-black invoicing app on concrete; hard top light; 9:16
MOTION: smooth orbit, subtle parallax; in-frame text "Post less. Reach more." rendered on screen
FINISH: audio = guide track; final VO via ai-voiceover ("Boring. Works."); captions burned in
VERIFY: text legibility, reflections. DISCLOSURE: "AI-generated" label.
```

## Worked example 2 — multi-shot hook (warm studio voice, 6 cuts, ~15s)
```
TIER: Kling 3.0 Multi-Shot (6 cuts, auto-continuity)
CUTS: 1 desk wide → 2 hands on app → 3 calendar filling → 4 founder small smile →
      5 phone publishing burst → 6 logo card
IDENTITY: same founder + same mug across cuts (reference image locked)
FINISH: ai-voiceover narration over the cuts; captions via captions-and-clipping
KEEP: no crowd; verify hands in cut 2. DISCLOSURE: "AI-assisted video".
```

Both: read brand-profile + voice-builder first; image-to-video / references to hold identity; ~15s
planned (multi-shot, not one long take); **audio = guide track → ai-voiceover**; captions via
captions-and-clipping; Draft Mode to iterate; consent + disclosure handled. Publish →
scheduling-and-queue → WoopSocial. Talking-head → heygen/synthesia; edit-grade control → runway
(tool, via ai-video).
