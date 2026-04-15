# Style Prompt Guide

Based on the Narrative Style Creation & Transfer methodology. This guide defines how to create, validate, and use Style Prompts that are "content-pure."

---

## The Fundamental Rule

**Style and Content must be separated.** A Style Prompt describes *how* an image looks. A Content Prompt describes *what* is in the image. Never mix them.

If a Style Prompt mentions a specific object ("a raven," "a castle"), it has content bleed and must be cleaned. If a Content Prompt mentions a color, lighting direction, or material quality, it has style bleed and must be cleaned.

---

## The Five Semantic Categories

Every Style Prompt must be organized into these five sections:

### 1. Perspective & Composition
- Camera angle (eye-level, low-angle, bird's-eye, worm's-eye)
- Lens characteristics (wide-angle distortion, telephoto compression, macro)
- Spatial arrangement (flat, layered, deep recession)
- Subject isolation vs. environmental context

### 2. Lighting & Shadow
- Light source(s) and direction(s)
- Shadow quality (hard, soft, diffused, absent)
- Volumetric effects (haze, god-rays, caustics, subsurface scattering)
- Ambient vs. directed light ratio
- Time-of-day indicators (golden hour, blue hour, midday harsh)

### 3. Color Palette
- Primary, secondary, and accent colors (name them specifically)
- Saturation profile (muted, vivid, desaturated, selective saturation)
- Color temperature (warm, cool, neutral)
- Tonal range (high-key, low-key, full-range)
- Complementary or analogous color relationships

### 4. Brushwork & Technique
- Medium simulation (oil, watercolor, digital, photographic, charcoal)
- Texture quality (smooth, impasto, grain, noise)
- Edge quality (sharp, soft, sfumato, hard-edged graphic)
- Detail distribution (all-over detail, selective detail, low-detail zones)
- Surface quality (matte, glossy, translucent, crystalline)

### 5. Image Framing & Balance
- Subject placement (centered, rule-of-thirds, golden ratio, off-axis)
- Negative space ratio (generous, minimal, none)
- Symmetry vs. asymmetry
- Visual weight distribution
- Cropping (tight, environmental, full-scene)

---

## Creating a Style Prompt

### Method 1: From an Existing Image (Extraction)

Use a multi-modal LLM with this extraction prompt:

```
Analyze this image and describe its visual style in exhaustive detail. You must cover: camera perspective and composition, lighting and shadow characteristics, the complete color palette with specific color names, the brushwork or surface technique, and the framing and spatial balance. Do NOT describe any objects, subjects, or content — only the aesthetic and technical visual properties. Aim for approximately 3000 characters of pure style description.
```

Then reorganize the raw output into the five semantic categories above.

### Method 2: From an Artist or Movement (Research)

Ask the LLM to describe an artist's or movement's style, then apply the **Cleanup Prompt**:

```
Review this style description. Remove all references to: specific artists by name, specific artworks or titles, specific cultural or religious content, any subject matter or objects. Replace named references with their visual equivalents (e.g., "Vermeer's lighting" becomes "single-window diffused directional light with cool shadows"). Ensure the result is a pure visual-technical description organized into five categories: Perspective & Composition, Lighting & Shadow, Color Palette, Brushwork & Technique, Image Framing & Balance.
```

### Method 3: From a Previously Generated Image

Use the extraction method (Method 1) on an image you previously generated. This is useful for "locking in" an emergent style that the model produced by accident.

### Method 4: From Scratch (Manual Construction)

Build each of the five categories manually. This is the most controlled method but requires strong visual vocabulary. Start by choosing a "seed" concept (e.g., "oppressive," "serene," "kinetic") and let that word drive decisions in each category.

---

## Validation: The Style Test

After creating any Style Prompt, you **must** validate it:

1. Generate an image of a **simple, unrelated subject** (e.g., "a single apple on a flat surface," "a sphere," "a chess piece").
2. Open a **new session** (no conversation context) and generate the same test subject with the same Style Prompt.
3. Compare — if the two outputs share the same visual "soul" despite different generation contexts, the Style Prompt is robust.

**Why a new session?** LLMs carry conversation context. If you discussed ravens earlier in the session, a Style Test may accidentally produce raven-adjacent imagery. A fresh session proves the prompt works on its own.

---

## Content Prompt Rules

When writing Content Prompts to pair with a Style Prompt:

1. **DO** describe: objects, their count, their relative scale, their placement/arrangement, the scene's narrative action.
2. **DO NOT** describe: colors, materials (unless structural — e.g., "glass" is structural, "shimmering glass" is style), lighting, atmosphere, texture quality, or any visual quality already covered by the Style Prompt.
3. **AVOID** adjectives that overlap with style — "dark," "bright," "colorful," "detailed," "rough," "smooth" are all style descriptors.

**Good Content Prompt:** "A raven perched on a craggy outcrop, wings half-spread, facing left. Three smaller birds in the background sky, mid-flight."

**Bad Content Prompt:** "A majestic dark raven with glossy feathers perched on a dramatic rocky cliff under moody lighting." — Every word after "raven" is style.

---

## Combining Style + Content

The final generation prompt concatenates Style + Content in this order:

```
[Style Prompt]. [Content Prompt].
```

Style first gives it more weight in most models. If the model supports negative prompts, add any known content-bleed terms from the Style Prompt as negatives.

**Character limit warning:** DALL-E 3 has a ~4000 character limit. If your combined prompt exceeds this, the model will silently truncate. For DALL-E, compress the Style Prompt to fit. Flux (Pollinations.ai) has a higher effective limit (~8000+), so this is less of an issue.