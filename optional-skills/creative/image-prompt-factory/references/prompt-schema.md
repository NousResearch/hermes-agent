# Prompt Pack Schema and Craft Reference

## The prompt scaffold (every prompt, both variants)

Build each prompt from these blocks, in order. Omit a block only when the
brief genuinely has nothing for it.

```text
Use case: <corpus category and asset destination>
Template: <selected template_id>
Primary request: <operator main request>
Input references: <brand/subject reference roles, or none>
Scene/backdrop: <environment>
Subject: <main subject and placement>
Style/medium: <photo, illustration, 3D, and so on>
Composition/framing: <wide, close, top-down; placement; reserved empty zones for overlay>
Lighting/mood: <lighting and mood>
Color palette: <palette notes, from the brand reference when supplied>
Text handling: <baked verbatim text, OR text-free with a forbid-text clause>
Constraints: <must keep and must avoid>
Avoid: <negative constraints>
```

## brief.json

Written by the agent after parsing the operator request:

```json
{
  "request": "hero image for a coffee subscription landing page",
  "render_mode": "baked",
  "aspect": "1:1",
  "count": 2,
  "subject_mode": "generic",
  "exact_text": "",
  "category_hint": ""
}
```

- `render_mode`: `baked` (text rendered inside the image — the default for
  short English ad copy) or `overlay` (text-free scene, copy composited later).
- `subject_mode`: `generic` (invent a neutral subject) or `placeholder` (the
  pack must NOT invent a subject — see the sentinel rule below).
- `count`: 1-8 concepts.

## selection.json

Written by the agent after browsing the corpus (`stats`, `template`):

```json
{
  "template_id": "product-commerce-visual",
  "category": "Products & E-commerce",
  "style_tags": ["photorealistic", "studio"],
  "scene_tags": ["product"],
  "example_case_ids": [101, 205]
}
```

`ground` resolves this into `grounding.local.json`. The `.local.json` file is
agent-eyes-only reference material: learn structure and quality from its
exemplars, then write fresh wording. Never paste exemplar text into the pack,
never cite a case id the grounding did not resolve.

## prompt-pack.json

```json
{
  "prompt_count": 2,
  "example_case_ids": [101, 205],
  "prompt_engine": "gpt-image-2-style-library",
  "corpus_pin": "<from grounding>",
  "corpus_source": "<from grounding>",
  "corpus_sha256": "<from grounding>",
  "license": "MIT",
  "concepts": [
    {
      "concept_id": "concept-01",
      "template_id": "product-commerce-visual",
      "baked_prompt": "...",
      "overlay_prompt": "...",
      "copy": { "eyebrow": "", "headline": "", "subhead": "", "cta": "" },
      "aspect": "1:1",
      "panel_side": "top-left"
    }
  ]
}
```

Provenance rules (enforced by `scripts/pack_validate.py`):

- `grounded: true` in the grounding → stamp `prompt_engine`, `corpus_pin`,
  `corpus_source`, `corpus_sha256`, `license`, and the RESOLVED
  `example_case_ids` (never unresolved ones), copied exactly.
- `grounded: false` → OMIT all of those keys and set `"self_authored": true`.
  Never cite a source you did not read.

## The two variants (always write both)

1. **`baked_prompt`** — text rendered inside the image. If `exact_text` is
   set, quote it verbatim with strict placement and legibility guidance.
   Modern image models render short English copy letter-perfect; this is the
   one-file deliverable an operator posts directly.
2. **`overlay_prompt` + `copy`** — the scene is TEXT-FREE. The prompt must
   forbid baked text explicitly ("no text, no words, no letters, no numbers,
   no logos, no watermarks, no lettering of any kind") and reserve generous
   empty negative space where a composited copy panel will land. The
   correctly-spelled message lives in the separate `copy` object. Use this
   discipline for exact brand fonts, long or legal copy, and non-Latin text.

`render_mode` selects which variant gets rendered. It never drops the other
variant from the pack.

## Subject discipline

- `subject_mode=generic`: neutral world, identity-stable generic traits. Do
  not invent brand names, slogans, people, data, or claims.
- `subject_mode=placeholder`: the `Subject:` field of EVERY prompt (both
  variants) must begin with the literal token
  `[SUBJECT SUPPLIED AT RENDER TIME]`, followed only by placement and posture
  notes plus: "preserve the attached reference subject's identity, features,
  and proportions exactly; do not invent, describe, or restyle the subject."
  Never write physical traits anywhere. A downstream renderer replaces the
  token with its own reference-locked subject. This exists because an
  appended identity LOSES to a pack's own invented `Subject:` line —
  reference images alone do not save you.

## Category craft notes (condensed discipline cards)

| Category | The load-bearing rule |
|---|---|
| Product / e-commerce | Hero product tack-sharp, honest materials, silhouette never cropped; overlay variant reserves an empty panel that never crosses the product. |
| Realistic photography | Name lens, distance, depth of field, and one motivated light source; forbid plastic skin and HDR halos. |
| Brand identity | Palette and geometry from the brand reference role only — never name the brand in the prompt. |
| Campaign / social visual | One focal message per image; aspect from the destination (1:1 feed, 4:5 portrait, 9:16 story). |
| Typography poster | Baked variant only when copy is short English; otherwise overlay with reserved type area. |
| Infographic / explainer | Structure beats decoration: number the zones, cap at 5 information blocks, forbid fake data. |
| People / character | Pose, wardrobe, and expression are scene direction; identity comes from `subject_mode`, never from invented traits. |

## Render handoff

The pack is renderer-agnostic. Feed a chosen variant to any image backend the
install has (the `image_gen` plugin family, the `comfyui` skill, the `codex`
skill, or any external tool). For transparent-background assets prefer a flat
chroma-key background plus local removal over model-native transparency.
