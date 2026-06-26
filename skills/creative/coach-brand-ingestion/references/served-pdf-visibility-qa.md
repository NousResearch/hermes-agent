# Served PDF visibility QA

Durable lesson from the Congressional / True Swing-class brand-doc work:

## Problem shape
A branded HTML artifact can look correct in the source browser view or local screenshot while the final exported/served PDF silently drops critical visual identity surfaces such as:
- logo panels
- inline identity images
- color swatches / fills
- other asset-backed cover elements

## Acceptance rule
For premium brand docs, the acceptance surface is the exact artifact the operator will open:
- the served PDF
- or the final proof file delivered to review

Do not claim success from:
- source HTML screenshots
- local browser previews of the HTML alone
- assumptions that a referenced image path means the exported PDF contains the asset

## Minimum QA sequence
1. Export the PDF.
2. Serve or open the exact review artifact.
3. Render the delivered PDF back to an image/page preview.
4. Verify that required visible elements survived export:
   - logo / wordmark / icon
   - color swatches
   - typography samples
   - required section structure
5. If the round-tripped render drops those surfaces, mark the artifact NO-GO and rebuild.

## Durable tactics
- Prefer deterministic local raster variants when SVG or remote asset rendering is inconsistent.
- Keep render-critical assets local to the composed artifact directory when using file-based export.
- If necessary, flatten critical cover compositions so the visible identity is baked into the page surface before final PDF handoff.

## Operator-facing truth standard
If the user says they still do not see the logo/swatches, treat that as a hard failure until the served/exported proof itself visibly disproves the complaint.
