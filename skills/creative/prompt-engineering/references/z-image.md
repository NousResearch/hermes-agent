# Z Image Turbo

Tongyi-MAI's ultra-fast 6B parameter model. Sub-second generation, bilingual (English/Chinese) text rendering.

## When to Use

- Speed is critical (1-10 second generation)
- Bilingual text rendering needed (English + Chinese)
- Photorealistic portraits with natural skin
- Complex scenes with multiple objects
- Running on consumer hardware (16GB VRAM)

## Strengths

- **Speed**: 1-10 seconds generation, 8 steps optimal
- **Bilingual text**: Accurate English and Chinese typography
- **Photorealism**: Natural skin, fabrics, lighting
- **Complex scenes**: Multiple objects blend smoothly
- **Low VRAM**: Runs on 16GB consumer GPUs

## Prompt Structure

Layer details like a short story: start broad, add specifics.

**Subject** → **Environment** → **Technical details** → **Style/mood**

## Example Prompts

**Portrait with micro-textures:**
```
Hyper-detailed close-up portrait of a young woman with sun-freckled olive skin
and loose waves of dark hair pinned back by sleek metallic clips, her face
half-veiled by overlapping pine boughs that dapple her cheeks with feathery shadows.
Dew-kissed needles glisten nearby, subtle veins trace her eyelids. Ethereal
backlighting with soft god rays, 8K, Nikon Z8 with 105mm macro lens.
```

**Complex scene:**
```
Ultra-photorealistic lifestyle photo of a young woman working late in a modern
Shanghai apartment at night. Sleek silver laptop open on wooden desk, minimalist
white ceramic mug with faint steam rising. Warm orange indoor light mixes with
cool blue city lights from large window, soft-focus skyline and neon signs.
Natural skin texture, detailed hair strands, realistic reflections, shallow depth of field.
```

## Technical Specifications

- **Resolution**: Up to 2048x2048
- **Steps**: 8-10 recommended for speed/quality balance
- **Denoising strength** (I2V): 0.6 default; increase to 0.75 for more change, decrease to 0.45 for subtle edits

## Bilingual Text

Renders clean text in both languages. For posters/graphics:
```
Minimal vertical poster with clean dark background. Big bold Chinese title at top:
"生成式未来高峰论坛". Small English subtitle below: "GENERATIVE FUTURE SUMMIT 2025".
Single glowing abstract sphere in center made of blue and purple light trails.
```

## Image-to-Image

- **Low strength** (0.3-0.4): Upscaling, detail enhancement
- **High strength** (0.7+): Major transformations using input as loose inspiration
- Enable `prompt_expansion` for automatic prompt enhancement

## Camera-First Prompting

Lead with camera specs for photorealism:
```
Shot on Hasselblad X2D, 85mm f/1.4, shallow depth of field, natural window light...
```

## Compared to Nano Banana

- Faster generation (sub-second vs several seconds)
- Better bilingual text
- Less reasoning/intent understanding
- Similar photorealistic quality

## Common Pitfalls

1. **Too few steps** — Below 8 steps quality drops noticeably
2. **Ignoring denoising strength** — 0.6 is safe default; adjust consciously
3. **Weak camera grounding** — Leading with camera specs dramatically improves photorealism
4. **Over-reliance on prompt expansion** — Manual prompting still outperforms auto-expansion for precise control
