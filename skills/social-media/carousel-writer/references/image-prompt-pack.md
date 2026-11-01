# Image Prompt Pack — generating carousel slides

A carousel is visual, so this pack turns the slide text into on-brand slide images. It bundles
prompts for the current image generators that render **legible in-image text** — the make-or-break
requirement for carousel slides. (This is the image equivalent of the Veo pack in `reels-script`.)
For the full tool drivers, the `nano-banana` and `ideogram` mini-skills go deeper.

> Tools and pricing move fast — re-verify quarterly. Use whichever the user has; the prompt
> structure transfers.

## Which tool for slides

- **Ideogram** — best-in-class at rendering **accurate, legible text inside images**. Default for
  text-heavy slides.
- **Nano Banana (Gemini-class image)** — strong text + excellent **consistency across images**
  (keep characters/style stable slide to slide). Great for a cohesive set.
- **Recraft** — vector/brand-graphic style; clean, on-brand layouts; good for designed-looking
  slides and reusable styles.
- **Canva / a designer / a template** — the non-AI route; often the most reliable for exact brand
  type and layout. Always a valid fallback.

> Reality check: AI image text is improving fast but still imperfect. For slides where exact
> wording and brand type matter, a **template (Canva/Figma)** with the text typed in is often more
> reliable than generation. Recommend that when precision matters; use generation for speed/volume
> or illustrative slides.

## The slide prompt skeleton

Brief the generator like a designer — and **specify the exact slide text in quotes**:

```
A {square 1:1 | portrait 4:5} social carousel slide.
Style: {minimal/editorial/bold} ; brand palette: {colors} ; typography: {clean sans, large headline}.
Layout: large headline top, short supporting line below, slide number "{n}/{total}" small in a corner.
Headline text (render exactly): "{headline}"
Supporting text (render exactly): "{supporting line}"
Background: {solid brand color / subtle texture} ; one accent element in {accent color}.
High contrast, mobile-legible, generous margins, no clutter.
```

Fill it per slide from the text spec produced in `slide-craft.md`.

## Keeping the set consistent (the hard part)

Slides must look like one set, not ten random images:

- **Reuse the same prompt skeleton** for every slide — only the headline/supporting text and slide
  number change.
- **Lock the style**: same palette, type treatment, layout, margins, slide-number position.
- Use the tool's **consistency features** (a reference image, a fixed seed, or "match the style of
  this") so slide 7 matches slide 1.
- Generate, then **eyeball for legibility and text accuracy** — regenerate any slide where the text
  renders wrong (common failure). Fix typos by re-prompting or editing in a template.

## Compliance — AI-disclosure

AI-generated slide images are AI-generated content:

- WoopSocial **auto-discloses on TikTok**. For Instagram/LinkedIn, follow platform rules and the
  brand's disclosure policy; attach Content Credentials where the tool supports them.
- Never put **fabricated stats or fake claims** on a slide — only verifiable numbers per the brand
  profile.

## Output to the user

Deliver, per slide: the **exact text**, a **design note**, and a **ready-to-paste image prompt** (or
a template spec). Plus the cover prompt (the most important image). If the user prefers Canva/a
designer, give the layout spec instead and skip generation.
