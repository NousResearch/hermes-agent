# Known limitations

Each of these is measured, not theoretical.

## The soft contact shadow is destroyed

Every matting model treats a semi-transparent shadow as background. Measured on
3D icon art: recovered alpha in the shadow region was 0.03–0.09 against a true
0.16.

Consequence: on a dark surface it is invisible. On a **light** surface the
static art is subtly grounded and the animated version floats. If the icon sits
on a light sheet, look at it there before shipping.

Workaround, not implemented: composite the original still's shadow layer back
underneath the animation as a static element.

## File size scales with frame count, nearly linearly

Inter-frame differencing buys very little, because on these loops the whole
object moves. Real numbers, 288px, a 5s loop:

| frames | fps | animated WebP | VP9 WebM |
|---|---|---|---|
| 122 | 24 | 857 KB | 254 KB |
| 61 | 12 | 429 KB | — |
| 41 | 8 | 288 KB | — |

WebM is dramatically smaller but is a video: it needs a player, not an `<img>`.
For an app icon the WebP is usually the right call despite the size.

## There is no transparent MP4

Transparent video in Safari/iOS means HEVC with an alpha layer. `libx265` as
packaged by most Linux distributions reports *"Loaded libx265 does not support
alpha layer encoding"* — the alpha layer needs a custom build. The `--mp4`
output is therefore flattened onto a solid colour.

Use the WebP (works everywhere an image works) or the WebM (Chrome, Android,
web) for transparency.

## Animated WebP ignores reduced-motion

Nothing can pause an animated WebP once it is on screen. If you respect
`prefers-reduced-motion` / `useReducedMotion`, keep the still and swap to it —
the pipeline writes both.

## Do not stack a code animation on top

A baked loop plus a spring or drift in code reads as fidgeting rather than
life. Pick one. If the art animates itself, let it.

## Verified against, and why

`rembg` model comparison against ground-truth alpha on glossy 3D icon art:

| model | MAE | note |
|---|---|---|
| birefnet-general | 0.0041 | 7–30× slower |
| **isnet-general-use** | **0.0043** | the default; best speed at equal accuracy |
| bria-rmbg | 0.0045 | |
| u2net | 0.0050 | visible background leak |

`backgroundremover` is not used because it ships only u2net variants.
