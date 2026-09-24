# Worlds

The art direction the whole page lives inside. Pick one, write it as a **style
preamble**, and paste that preamble verbatim at the top of every image and video
prompt. Reusing it word for word is what makes eight separately generated assets
look like one shoot. Paraphrasing it is what makes them look like eight prompts.

---

## The default is photographic

Soft matte low-poly clay diorama, isometric miniature, tilt-shift toy world:
**banned as a default.** It is the house style of AI scroll sites, it announces
that nothing on the page is real, and for any brand selling an actual product it
actively works against the sale. A clay render of a can does not make anyone
thirsty.

Use an illustrated world only when the brand is genuinely illustrated: a
children's product, a game, a brand whose existing identity is drawn. Then
commit to it properly, matched to the brand's real illustration style, not to
the generic diorama look.

---

## The eight

Each is a starting preamble. Tune the light, lens and grade to the brand; keep
the structure.

### 1. Low-key cinematic: the default
Dark, controlled, one light source. Works for almost anything premium: spirits,
coffee, tools, apparel, software, professional services.

> Cinematic product photography shot on 35mm anamorphic lenses. Shallow depth of
> field, high dynamic range, true blacks, matte film grain. Low-key lighting:
> one warm key, cool ambient fill, deep falloff into shadow. Colour grade of deep
> charcoal, warm amber highlights, desaturated mid-tones. Photographic realism.
> NOT 3D render, NOT clay, NOT illustration, NOT CGI, no digital glow, no plastic
> sheen.

### 2. High-key editorial
Bright, airy, shadowless. Wellness, skincare, home goods, healthcare, fintech
that wants to feel calm.

> Editorial still-life photography on a seamless bone-white cyclorama. Large soft
> overhead source, huge white bounce, near-shadowless with one soft contact
> shadow. High key, gentle contrast, colour grade of warm white and pale
> neutrals. Medium-format sharpness, fine grain. Photographic realism, no CGI.

### 3. Natural documentary
Real people, real places, available light. Service businesses, trades,
restaurants, anything where trust comes from "these are actual humans."

> Documentary photography, available light only, handheld 35mm. Natural skin
> tones, honest imperfect surfaces, slight motion in the frame. Muted realistic
> grade, no colour cast, visible grain. Candid, unposed, nobody looking at
> camera. Absolutely not stock-photo styling, no fake smiles, no CGI.

### 4. Hard-light graphic
Direct sun, saturated ground, sharp shadows as composition. Streetwear, energy
drinks, sports, youth brands.

> Studio product photography with a single hard undiffused source. Crisp
> high-contrast shadows used as graphic shapes. Saturated seamless colour
> backdrop. Punchy contrast, slight halation on specular highlights. Shot on
> digital medium format, sharp throughout. Photographic, not rendered.

### 5. Macro texture
Closer than the eye gets. Food, drink, materials, ingredients, craft.

> Extreme macro photography, 100mm macro lens at high magnification. Razor-thin
> plane of focus, everything else falling to soft black. Backlit so edges glow.
> Visible surface texture, condensation, grain of the material. Near-black
> negative space. Photographic realism, no CGI, no illustration.

### 6. Architectural
Space, scale, geometry, almost no people. Real estate, agencies, manufacturing,
B2B infrastructure.

> Architectural photography, tilt-shift corrected verticals, wide 24mm. Vast
> negative space, strong linear geometry, raking daylight through structure.
> Cool neutral grade with one warm accent. Long exposure stillness. Photographic,
> no render, no CGI.

### 7. Nocturne
Night, practical lights, wet surfaces, reflection. Nightlife, automotive,
gaming, security, anything with edge.

> Night photography, practical light sources only: neon, sodium, screen glow.
> Wet reflective surfaces doubling every light. Deep blue-black shadows, warm
> point highlights, heavy atmosphere. Anamorphic flare, visible grain.
> Photographic, cinematic, not rendered.

### 8. Technical drawing
The one non-photographic world that reads as premium rather than cheap, because
it is honest about being a diagram. Engineering, hardware, complex services.

> Precise technical illustration in the style of a patent drawing or exploded
> assembly diagram. Fine consistent line weight, no fills, monochrome ink on
> warm paper ground, dimension lines and leader callouts. Orthographic
> projection. Restrained, engineered, no shading, no gradients, no 3D render.

---

## Writing your own

Every preamble names five things. Miss one and the set drifts.

1. **Medium and lens**: "35mm anamorphic", "100mm macro", "handheld 35mm".
   This is what sets depth of field and perspective.
2. **Light**: count the sources and place them. "One warm key, cool ambient
   fill" is directable. "Beautiful lighting" is not.
3. **Grade**: the colour story in three words. "Deep charcoal, warm amber,
   desaturated mid-tones."
4. **Texture**: grain, halation, condensation, imperfection. This is what makes
   an image read as photographed rather than generated. Skipping it is the
   single biggest cause of the plastic AI look.
5. **The negative list**: what it must not be. "NOT 3D render, NOT clay, NOT
   illustration, no digital glow, no plastic sheen." Models drift toward
   rendered-looking output; the negative list is what holds them.

---

## If the canvas is light

Every worked example above is dark, and a high-key build inverts four things at
once. Getting them wrong costs a full iteration:

1. **The scrim washes toward the canvas, it does not darken.** Over footage on a
   paper ground the density has to go *up*, not down, or the type has nothing to
   sit on.
2. **The type over media stays ink.** Light text on a light page over a bright
   frame is unrecoverable.
3. **`--sc-edge` is an ink-tinted inset highlight** tuned for a near-black
   ground. On paper it reads as dirt along the top lip. Invert it to white, or
   drop it and carry the raise with a hairline.
4. **The `--sc-e1/2/3` shadow alphas are tuned for a near-black ground** too.
   Halve them and re-tint `--sc-shadow-color` toward the canvas hue.

The contrast direction flips with all of this: dark type fails on the *darkest*
patch under it, not the brightest. The harness picks the direction per line, so
it grades a high-key page correctly (see [verify.md](verify.md)).

---

## Composition, per shot

The preamble sets the world. Each shot prompt then names the subject, the frame,
and **where the empty space is**.

**Name the empty space in every scene prompt, not only in the preamble.** This
is a requirement, not advice. It is the single highest-leverage line in a prompt:
seven stills across three aspect ratios came back on-grade and cohesive with zero
rerolls on the build that did it every time. Copy sits on these images, so
composition has to leave room for it:

- "large empty shadowed space across the upper left of the frame"
- "the subject low and to the right, negative space above"
- "centred with even empty space on both sides"

Generate the space, do not crop for it later. And never ask for text in the
image: markup is selectable, translatable, sharp at every density, and editable
after the fact.

---

## Cohesion checks

Lay every generated asset side by side and look for:

- **One light direction** across the set, or a deliberate reason it changes.
- **One grade.** If one image is cooler than the rest, reroll it rather than
  correcting it in CSS; a filter over a full-bleed image flattens it.
- **One level of realism.** A photoreal hero followed by a rendered-looking
  product shot is worse than either style used consistently.
- **The brand object identical everywhere.** Pass the real packaging, logo or
  product shot as `--ref` on every prompt that includes it. A label that drifts
  between shots is the thing a client notices first.
