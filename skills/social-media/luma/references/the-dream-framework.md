# The DREAM framework — cinematic AI shots that are on-budget and on-rights

Luma's leverage is per-shot cinematic quality; the expert separators are the draft-first economics and knowing
that generations are **shots, not films**. Each letter is a gate; most failures happen at E (credit burn) or M
(the rights check).

## D — Define the shot
Prompt like a director writing a shot brief, not a wish: **subject (concrete detail) + action + setting +
lighting/time + mood + camera move** ("slow push-in", "handheld", "orbit"). One shot per generation — a
generation is 5–10 seconds, so a "brand film" is a shot list first (route to the script/storyboard skills).
Ray3's reasoning rewards specific, coherent briefs; vague prompts pay the variance tax twice.

## R — Reference with stills + frames
The control layer that separates production from slot-pulling:
- **Still-first:** generate/choose a brand-true still (flux / Photon — exact palette, product fidelity), then
  **image-to-video** — far more brand control than praying text-to-video lands it.
- **Keyframes:** start/end frames define transitions — and stitch shots into sequences cleanly.
- **Character reference** (base Ray3, not 3.14) locks identity across shots; **Modify (V2V)** restyles real
  footage while keeping the performance. Lock a canonical reference set once; vary only the scene per shot.
  (A consistent "character" is never an unpermitted real person's likeness.)

## E — Economize with drafts
The signature workflow: **iterate in Draft Mode** (prompt, composition, camera) at a fraction of the cost, and
**Hi-Fi master only the winner.** Budget ~3 attempts per usable clip (4–5 for consistency shots). Reserve HDR
for graded final deliverables (it runs ~16× standard 720p at 1080p); default Ray3.14 for volume, Ray3 for
HDR/identity work. Spend monthly credits (they expire at reset); top-ups roll over; the API wallet is separate.

## A — Assemble the sequence
Shots become films **in an editor, not in the generator**: Extend chains toward ~30s but quality degrades past
the initial clip (Luma's own docs) — so cut generated shots together in `capcut`, pace them to the script's
beats, and add **sound in post** (no native audio — `ai-music-and-sound` / licensed tracks; a silent cinematic
clip is half-finished). Keyframe-matched end/start frames make the cuts invisible. Side-by-side drift review on
any multi-shot set before a client sees it.

## M — Mind the rights
Before anything ships: **commercial tier confirmed** (free = watermarked non-commercial; the uncontested floor
is Plus-level — verify in-app) ✓ · **Luma's license-back understood** (they may use your generations in
marketing; Enterprise for NDA/privacy) ✓ · **no unpermitted likeness**, no photoreal fake events targeting real
entities ✓ · **AI-disclosure** where platform/region requires (EU AI Act; C2PA) ✓ · then the human judges the
final clip and **WoopSocial publishes** the finished file (it does not generate video).
