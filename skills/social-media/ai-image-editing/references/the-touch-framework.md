# The TOUCH framework — targeted edits that stay honest

Editing's promise is surgical: fix the 5% and keep the 95%. Its hazard is that edited *real* photos cross into
misrepresentation faster than generated ones. Each letter is a gate; most failures happen at U (the honesty
line) or C (unchecked seams).

## T — Task first, tool second
Name the job precisely before touching an engine: **remove** (object/person/blemish) · **replace** (swap an
element) · **erase background** · **extend** (outpaint for aspect ratios) · **upscale** (faithful vs creative —
different tools!) · **restore** (scratches/denoise/faces) · **restyle/relight.** Then route: complex-mask
quality → FLUX Kontext/Fill; indemnified client work → Firefly; in-workflow quick fixes → Canva Magic tools;
conversational multi-step → nano-banana; product batches → Claid-class; faithful upscale → Topaz-class;
creative upscale (art only) → Magnific-class. The task defines the tool — never the reverse.

## O — One change at a time
Small, precise masks beat large irregular ones; one targeted instruction per pass; chain small edits and
**re-check global coherence every few steps** (drift accumulates across chained generative edits). Work from
the best source you have — and if the source is low-res, **upscale before inpainting** so the model has
context. Preserve the 95%: an edit that regenerates what was already right is a re-roll in disguise.

## U — Uphold the real (the honesty gate)
Editing a real photo edits a **claim**:
- **Products:** faithful mode only; no hallucinated textures/details; **never conceal actual defects** (the
  FTC net-impression standard — the photo IS the claim).
- **People:** body-shape retouching in commercial content is **label-required in several markets** (France,
  Norway incl. influencers, Israel — verify applicability); face enhancement on real people stays conservative
  (generative modules can drift identity); consent for edits to someone else's image.
- **Results/before-after:** route to `before-after-and-transformation`'s FTC rules — an edited "after" is a
  fabricated result.
- **Provenance:** never remove watermarks (license evasion) or strip provenance metadata; **AI-disclosure**
  where platform/region requires. Creative hallucination belongs to **art**, not evidence.

## C — Check the seams
QA at **100% zoom**, always: edge blending and halo artifacts · lighting/shadow direction coherence ·
perspective and scale of replaced objects · **faces** (eyes/teeth/skin — the uncanny tells) · **in-image text**
(upscalers mangle it — re-typeset) · consistency across a batch (margins, shadows, background tone) ·
identity drift on chained face work (compare against the original). A seam the editor misses, the audience
finds.

## H — Hand off with rights
Before publishing: source-image license confirmed (you can only edit what you're licensed to edit) ✓ · engine
terms honored (Firefly indemnification vs FLUX outputs-vs-service — see `flux`) ✓ · disclosure labels applied
where required ✓ · **master exported PNG/TIFF** (derivatives compressed per platform) ✓ → the human judges the
final at 100% → `scheduling-and-queue` → **WoopSocial publishes** the finished image (it does not edit).
