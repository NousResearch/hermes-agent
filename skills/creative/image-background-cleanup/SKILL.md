---
name: image-background-cleanup
description: Remove text/logo overlays or recreate clean backgrounds from posters/screenshots with quality-first workflow and user-approval gates.
---

# Image Background Cleanup (No text/logo)

Use this for requests like: "extrae el fondo", "same image without text/logo", "clean background only".

## Core rule
Prioritize **high-quality output** over fast but degraded inpainting. If source is a screenshot/compressed poster, avoid claiming perfect extraction.

## Workflow
1. **Classify source quality**
   - Original layered asset / high-res artwork: attempt direct cleanup.
   - Screenshot/compressed/social repost: warn that exact clean extraction may leave artifacts.

2. **Pick generation path (preferred order)**
   - **A. Higgsfield/image model recreation** (preferred when user wants polished result quickly).
   - **B. Targeted inpainting extraction** only if source quality is sufficient.
   - **C. Full background rebuild** (style-matched) if inpainting quality is poor.

3. **Always deliver options**
   - Provide at least 2 variants when feasible: 
     - "Extracted" (closest to original)
     - "Recreated" (cleanest visual quality)

4. **Quality gate before sending**
   - No readable text/logo/watermark.
   - No obvious inpainting scars in focal areas.
   - Export vertical 1080x1920 when target is reels/stories unless user asks otherwise.

5. **Communication standard**
   - If first attempt is weak, acknowledge directly and regenerate.
   - Do not defend low-quality output; move to better pipeline immediately.

## User-specific preference embedding
- For this user, prefer **Higgsfield** for image/background tasks when available.
- They expect **pro visual quality** and dislike artifact-heavy edits.

## Pitfalls
- Inpainting over large text blocks on screenshots usually creates muddy textures.
- Aggressive blur to hide artifacts can make output look cheap.
- "Text removed" is not enough; overall aesthetic must still look production-ready.

## Absorbed subsection: Text/Logo Removal Workflow

For explicit requests to remove text/logos/UI badges while preserving scene quality:

1. **Intake gate**
   - Confirm if source is original asset vs screenshot/photo.
   - If screenshot, disclose artifact risk up front.

2. **Two-track decision**
   - `cleaned-from-source`: targeted masking + conservative inpainting.
   - `recreated-similar`: rebuild composition/palette when source cleanup looks degraded.

3. **Acceptance criteria**
   - No legible text/logo/watermark.
   - No smear/melted-letter artifacts around edited regions.
   - Subject edges, horizon geometry, and lighting continuity preserved.

4. **Delivery standard**
   - Provide 1–2 labeled variants.
   - Recommend the best variant and explain why.

## Reusable recreation prompt
"Vertical mystical twilight background, deep indigo to warm pink horizon, subtle stars/nebula haze, meditating silhouette on lower-left hill, cinematic soft lighting, no text, no logos, no icons, no UI."

## Output checklist
- [ ] Clean image (no text/logo)
- [ ] At least one high-quality variant
- [ ] Correct aspect ratio (usually 9:16)
- [ ] Brief note: extracted vs recreated
