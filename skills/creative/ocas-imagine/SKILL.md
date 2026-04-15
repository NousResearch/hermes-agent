---
name: ocas-imagine
description: >
  Imagine: a professional art-direction engine for text-to-image generation.
  Based on the Narrative Style Creation & Transfer methodology, Imagine
  separates 'Style' from 'Content' to ensure aesthetic consistency across
  image series. It supports a custom library of styles, image-to-style 
  extraction via multi-modal LLMs, and a strict prompt-separation workflow 
  to prevent "style drift."
metadata:
  author: Indigo Karasu
  email: mx.indigo.karasu@gmail.com
  version: "1.0.1"
  hermes:
    tags: [creative, images, generative-ai]
    category: visual
  openclaw:
    skill_type: system
    visibility: public
    filesystem:
      read:
        - "{agent_root}/commons/data/ocas-imagine/"
        - "{agent_root}/commons/journals/ocas-imagine/"
      write:
        - "{agent_root}/commons/data/ocas-imagine/"
        - "{agent_root}/commons/journals/ocas-imagine/"
---

# Imagine

Imagine is an art-direction engine that treats image generation as a two-part process: **Style Prompting** and **Content Prompting**. By decoupling the aesthetic DNA of an image from its subject matter, Imagine allows for the creation of a consistent visual series where the content changes but the "soul" of the art remains identical.

## When to use

- Create a series of images with a consistent visual identity (comics, storyboards, concept art).
- Translate a specific artistic style from a reference image into a reusable "Style Prompt."
- Generate high-quality images using the Flux model via the Pollinations.ai API.
- Art-direct an LLM to produce a prompt that is "style-pure" (no content bleed).

## The Core Methodology: Style-Content Separation

The fundamental invariant of Imagine is that **Style** and **Content** must never be mashed into a single descriptive paragraph.

1. **The Style Prompt:** A high-detail technical description of the aesthetic: Perspective, Lighting, Color Palette, Brushwork/Texture, and Framing.
2. **The Content Prompt:** A pure description of the objects, their relative scale, their placement, and the scene's narrative components.

**The Rule:** If a Content Prompt describes a color, material, or lighting effect that is already handled by the Style Prompt, it is an "overspecification" and must be removed to avoid confusing the model.

## Operational Flows

### Flow 1: Image Generation (Existing Style)
Use when the user wants an image in a known or predefined style.

1. **Select Style:** Retrieve a style definition from `references/default_styles.md` or a previously saved Style Prompt.
2. **Expand Content:** Take the user's subject request and expand it into a detailed **Content Prompt**. 
    - *Constraint:* Describe "what" is in the scene, but NEVER "how" it looks (no colors, no style keywords, no lighting directions).
3. **Synthesis:** Combine the two into the final API call.
4. **Execute:** Call the Pollinations.ai API.

### Flow 2: Style Extraction (Image $\rightarrow$ Style)
Use when the user provides an image and wants to "capture its soul" for future use.

1. **Visual Analysis:** Use a multi-modal LLM (via `vision_analyze` or similar) to analyze the reference image.
2. **Exhaustive Extraction:** Use the specific extraction prompt defined in `references/style_prompt_guide.md`. 
    - *Requirement:* Describe the style in exhaustive detail without mentioning any objects or content.
3. **Semantic Organization:** Organize the raw description into the five standard semantic sections:
    - Perspective & Composition
    - Lighting & Shadow
    - Color Palette
    - Brushwork & Technique
    - Image Framing & Balance
4. **Verification:** Generate a "Style Test" image of an unrelated, simple subject to ensure the prompt is robust and free of "content bleed."
5. **Save:** Store the resulting Style Prompt in `commons/data/ocas-imagine/styles.jsonl` or as a reference file.

## Commands

- `imagine.generate --style <name|prompt> --content <description>` — generates an image using a specific style.
- `imagine.extract --image <path/url>` — analyzes an image and produces a structured Style Prompt.
- `imagine.library.list` — lists all available predefined styles.
- `imagine.style.save --name <name> --prompt <prompt>` — saves a new custom style to the library.
- `imagine.journal` — records the run, including the final synthesized prompt and the resulting image URL.

## Implementation Details

### API Integration
Imagine primarily uses the **Pollinations.ai** endpoint for its zero-cost, high-quality Flux implementation:
`https://image.pollinations.ai/prompt/{prompt}?width=1024&height=1024&model=flux&nologo=true`

### Storage Layout
```
{agent_root}/commons/data/ocas-imagine/
  styles.jsonl      # Custom user-created style prompts
  history.jsonl     # Log of generation prompts and results
  
{agent_root}/commons/journals/ocas-imagine/
  YYYY-MM-DD/
    {run_id}.json
```

## Support File Map

| File | Purpose |
|---|---|
| `references/style_prompt_guide.md` | The "Bible" of style extraction and prompt separation logic. |
| `references/default_styles.md` | A curated library of high-performance baseline styles. |
| `references/api_reference.md` | Technical specs for the image generation endpoints. |
| `references/journal.md` | Schema for recording imagine runs. |
