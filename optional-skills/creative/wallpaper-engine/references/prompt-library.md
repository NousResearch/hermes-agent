# Wallpaper Prompt Library

A curated collection of aesthetic prompt templates for SDXL wallpaper
generation.  Each template includes a **positive prompt** (what you want) and a
**negative prompt** (what you want to avoid).  The agent should pick from these
based on the user's stated preferences, then layer in any additional guidance
from the user's memory profile.

All templates assume SDXL.  For Flux Dev, swap the negative prompt style to the
Flux convention (Flux ignores negatives for most compositions).

---

## Nature & Landscapes

### Mountain Vista
- **Positive:** breathtaking mountain range at golden hour, cinematic lighting, 4K wallpaper, epic scale, misty valleys below, alpine glow, sharp focus, photorealistic
- **Negative:** blurry, low quality, people, text, watermark, frame, border, split screen

### Ocean Sunset
- **Positive:** tranquil ocean sunset, gentle waves, warm orange and pink sky, silhouette palm trees, ultra detailed, 4K, wallpaper composition, serene atmosphere
- **Negative:** blurry, low quality, boats, people, text, watermark, harsh lighting, overexposed

### Forest Depth
- **Positive:** enchanted forest, sun rays through canopy, moss covered ground, depth of field, cinematic, 4K wallpaper, rich greens, photorealistic, peaceful
- **Negative:** blurry, low quality, people, animals, text, watermark, frame, dark shadows

---

## Abstract & Minimalist

### Fluid Gradients
- **Positive:** smooth flowing abstract gradient shapes, soft pastels, depth, 4K, clean composition, minimalist aesthetic, gentle curves, digital art
- **Negative:** sharp edges, text, watermark, busy composition, clutter, noise, grain

### Geometric Harmony
- **Positive:** balanced geometric composition, soft ambient lighting, matte finish, 4K wallpaper, abstract architecture, harmonious colors, clean lines
- **Negative:** text, watermark, chaotic, cluttered, noise, harsh shadows, people

---

## Dark & AMOLED

### Dark Mountains
- **Positive:** dark silhouette of mountains against starry night sky, deep blacks, subtle glow, 4K wallpaper, AMOLED friendly, moody, atmospheric, minimal light
- **Negative:** bright areas, text, watermark, people, noise, grain, overexposed

### Void Geometry
- **Positive:** minimalist dark geometric shapes, neon edge glow, deep black background, 4K, AMOLED, cyber minimal, clean, precise, abstract
- **Negative:** bright backgrounds, text, watermark, clutter, noise, grain

---

## Sci-Fi & Futuristic

### Cyberpunk City
- **Positive:** futuristic cyberpunk city skyline at night, neon reflections, rain-slick streets, volumetric fog, cinematic, 4K wallpaper, blade runner aesthetic
- **Negative:** people, text, watermark, blurry, daylight, cartoon, anime, low quality

### Sci-Fi Megastructures *(well-tested — produced great results)*
- **Positive:** futuristic sci-fi cityscape, massive neon-lit megastructures reaching into the sky, flying vehicles between towering spires, holographic billboards and data streams, deep purple and cyan lighting, atmospheric volumetric haze, cinematic composition, ultra detailed
- **Negative:** people, text, watermark, daylight, cartoon, anime, low quality, flat lighting, empty sky

### Nebula Space
- **Positive:** deep space nebula, vibrant cosmic colors, scattered stars, ethereal gas clouds, 4K wallpaper, astronomical photography, vast scale
- **Negative:** text, watermark, spaceships, earth, frame, border, low quality

---

## Seasonal

### Autumn Forest
- **Positive:** autumn forest, golden and red foliage, soft morning light, mist, depth of field, 4K wallpaper, warm tones, peaceful path, photorealistic
- **Negative:** people, text, watermark, summer greens, snow, blurry, low quality

### Winter Silence
- **Positive:** snowy mountain peak, fresh powder, clear blue sky, crisp winter light, 4K wallpaper, pristine, minimal, serene cold atmosphere, photorealistic
- **Negative:** people, text, watermark, mud, dark shadows, overcast, flat light

---

## Usage Notes

1. **Always add user preferences on top.** If MEMORY.md says "user loves
   purple tones and hates busy compositions," append that to the template.

2. **For higher resolutions** (e.g. 4K 3840x2160), set `width: 3840, height:
   2160` in the workflow args. Generation will be slower but yields desktop-
   native resolution. SDXL natively works at 1024x1024 and upscales internally;
   going above 1920x1200 may produce repetition artifacts. For true 4K without
   artifacts, add an upscale stage or use Flux Dev.

3. **Negative prompt hygiene.** The base negative prompt in the workflow JSON
   already excludes common defects. Add style-specific negatives (e.g. "no
   people" for landscapes) but don't accumulate so many that the prompt becomes
   noise — 6-8 negative tokens is the sweet spot for SDXL.

4. **Seed -1** means random; the agent can request "similar but different" by
   using the same seed with slight prompt or CFG changes.
