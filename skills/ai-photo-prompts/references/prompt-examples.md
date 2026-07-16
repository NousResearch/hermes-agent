# AI Photo Prompt Examples — Real Sessions

## Example 1: Aquarium Portrait (Restoration + Composition)

**Task:** Restore and enhance old photo of elderly woman in front of aquarium. Preserve face exactly.

**Final prompt:**
```
Apply precise color correction, subtle composition adjustment, and eye catchlights to this portrait photo. The person MUST remain completely unchanged.

ABSOLUTE RULE - DO NOT CHANGE THE PERSON:
- The woman's face shape, features, skin texture, wrinkles, expression, hair, and body must remain EXACTLY as in the original. Do NOT make her younger, smoother, slimmer, or different in any way.
- Her outfit (white turtleneck, cream crocheted/lace vest, light trousers) must remain identical.
- Do NOT alter her pose, stance, or the position of her arms/hands.

CHANGES TO APPLY:

1. COLOR CORRECTION (remove aquarium blue cast):
- Neutralize the strong blue/cyan spill from the aquarium that is washing over the woman's face, hair, and clothing.
- Restore warm, natural skin tones to her face, neck, and hands. The skin should look healthy and naturally lit, not cold or tinted by the blue water.
- Keep the aquarium water itself vibrant blue and turquoise — only remove the color bleed ONTO the subject and foreground.
- Warm the off-white/cream tones of her turtleneck and knitted vest so they look natural under warm/neutral light, not pale blue.
- Maintain contrast: do NOT flatten the image. Keep the drama between subject and background.

2. COMPOSITION ADJUSTMENT:
- Slightly re-crop or adjust framing to give the fish and aquatic elements on the LEFT more "breathing room" and visual weight.
- Create a stronger visual balance: the woman anchors the right side, while the fish, eel, and water details on the left should feel like an intentional, balanced counterpart, not empty space.
- If re-cropping, keep the woman in the right third. Do NOT move her or change her size/scale relative to the frame.
- The wooden beam frame of the aquarium and the visible fish species must remain recognizable and sharp.

3. CATCHLIGHTS IN EYES:
- Add subtle, natural catchlights (small bright reflections) in both of the woman's eyes to give them life and dimension.
- The catchlights should look like a soft photographic light source, not artificial or painted-on.
- Do NOT change her eye color, eye shape, or expression — only add the reflection.

OUTPUT:
A photorealistic environmental portrait with professional composition. The woman must be instantly recognizable as the exact same person. The photo should feel professionally shot in a warm studio with the aquarium as a vivid backdrop, rather than an amateur snapshot dominated by blue ambient light.
```

**Result:** GPT respected face preservation. Background aquarium remained authentic.

---

## Example 2: Red Square Travel Portrait (One-Photo Enhancement)

**Task:** Sharpen faces, remove one mid-ground person, keep all authentic background elements.

**What went wrong first:**
- Initial prompt said "remove distractions" → GPT removed the white bus and replaced it with an extra Kremlin tower.
- User corrected: "keep the bus, keep all background exactly as is, remove ONLY the pink-jacket woman."

**Final prompt:**
```
Enhance the quality of this travel portrait photo while preserving both subjects completely unchanged.

SUBJECTS (MUST remain completely unchanged):
- SERGEY (right side, closer to camera): short beard, dark gray/black puffer jacket, dark pants, gray sneakers. Looking directly at camera.
- BOGDAN (left side, slightly behind and higher): smiling, beard, black puffer jacket with hood, blue jeans, black shoes. Looking at camera.

ABSOLUTE RULES:
- Both men's faces, features, skin texture, expressions must remain EXACTLY as in original.
- Faces must remain pixel-perfect identical to originals. Only edit background, lighting, and sharpness. Do not touch faces.

ROTATION:
Photo rotated 90° counter-clockwise. Rotate to correct upright orientation first.

1. FACE SHARPNESS (most important):
- Sharpen both faces significantly while keeping all original features intact.
- Increase resolution on eyes, eyebrows, nose, mouth, beard.
- Enhance eye catchlights slightly.
- Do NOT smooth skin or alter any facial feature.

2. REMOVE ONLY THIS SPECIFIC PERSON:
- Remove ONLY the woman in the pink puffer jacket and pink beanie in mid-ground.
- Do NOT remove bus, signs, lamp posts, distant pedestrians, or any other background.

3. BACKGROUND — KEEP EVERYTHING EXACTLY AS IS:
- Keep the white bus/van/coach, all street signs, lamp posts, vehicles.
- Keep St. Basil's Cathedral, Kremlin walls, overcast sky, hexagonal pavement exactly as original.
- Do NOT redraw, replace, or idealize the background.
- Do NOT add extra domes or towers not in original.

4. LIGHTING:
- Add subtle warm key light on faces from front-left.
- Add gentle contrast. Keep photorealistic.

5. COLOR:
- Warm skin tones naturally.
- Make St. Basil's domes vivid.
- Enrich Kremlin red brick.
- Keep pavement neutral gray.

6. QUALITY:
- Remove compression artifacts and phone softness.
- Increase clarity and micro-contrast.
- Output high-resolution and crisp.

OUTPUT:
One photorealistic travel portrait. Instantly recognizable subjects. No plastic skin. Background must be the real Red Square — not idealized.
```

**Result:** Face sharpening worked. Background remained authentic. User preferred this over an earlier attempt where GPT had invented architecture.

---

## Lessons from these sessions

1. **"Remove distractions" = danger zone.** Always name the exact object/person to remove.
2. **"Keep everything as is" must be explicit.** GPT defaults to "improving" backgrounds by replacing them.
3. **Pixel-perfect face protection is essential.** Without it, every model smooths skin.
4. **One-photo workflow is reliable.** Two-photo merge requests fail with current gen-AI models.
5. **User correction is the strongest signal.** When user says "it drew a tower instead of the bus," immediately add KEEP section and DO NOT REMOVE section.
