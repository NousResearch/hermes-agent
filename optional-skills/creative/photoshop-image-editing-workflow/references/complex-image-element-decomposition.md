# Complex Image Element Decomposition

Use this reference when a benchmark or target image is too complex for a single image-generation pass. The goal is to preserve detail and editability by decomposing the design into independently generated or sourced assets, then assembling them in Photoshop.

## Decision Rule

Do not force one-shot GPT Image generation when the image contains many dependent elements.

```text
3 or fewer elements
  -> one-shot image generation can be acceptable

4-6 elements
  -> generate background/main visual, build text/cards/CTA in Photoshop

7 or more elements, exact Korean text, product proof, reviews, prices, or benchmark-detail matching
  -> element decomposition + Photoshop composite is required
```

Treat these as high-risk in one-shot generation:

- exact Korean typography
- prices, quantities, discounts, dates, or statistics
- product shape/color/material preservation
- review/proof screenshots
- multiple benefit cards
- CTA buttons
- icon grids
- complex backgrounds with foreground objects
- hand/face-heavy visuals
- layouts that must match a benchmark closely

## Operating Principle

```text
GPT Image model = mood, background, visual objects, decorative elements
Photoshop = composition, text, cards, CTA, numbers, proof assets, final alignment
Hermes = orchestration, measurement, prompt assets, QA, documentation
```

Never ask the generator to produce the final image when exact text, proof content, or product fidelity are critical.

## Multi-Agent Production Roles

Use a small role chain. Keep at most 2-3 truly parallel workers; serialize anything that touches the same PSD or final file.

### 1. Director

Mission: define scope, output size, complexity level, stopping conditions, and final deliverables.

Deliverable:

```yaml
production_brief:
  purpose: "PDP section / ad banner / thumbnail / card news"
  output_ratio: "1:1 / 4:5 / 9:16 / 16:9 / long PDP"
  complexity_level: "quick / standard / pro-composite"
  final_outputs: [final.psd, final.png, final.jpg, qa_report.md]
  pass_criteria:
    - no fake text
    - exact Korean text in editable layers
    - product/proof assets preserved
    - QA sheet created
```

### 2. Reference Analyst

Mission: analyze the benchmark image without copying brand marks or exact protected details.

Deliverable:

```yaml
reference_analysis:
  ratio: "4:5"
  layout_type: "top headline + center visual + bottom cards + CTA"
  color_palette:
    background: "soft warm white"
    primary: "dark navy"
    accent: "mint"
  text_zones:
    headline: "top 20%"
    body_cards: "bottom 30%"
    cta: "bottom center"
  risky_elements:
    - exact Korean headline
    - 3-card grid
    - CTA button
```

### 3. Layout Architect

Mission: create a coordinate-based master layout for Photoshop assembly.

Deliverable:

```yaml
canvas:
  width: 1080
  height: 1350
  ratio: "4:5"
layout:
  headline_area: {x: 80, y: 80, w: 920, h: 220}
  main_visual_area: {x: 120, y: 310, w: 840, h: 470}
  benefit_cards: {x: 80, y: 830, w: 920, h: 300, count: 3}
  cta_area: {x: 180, y: 1180, w: 720, h: 100}
  safe_margin: 64
```

### 4. Asset Decomposer

Mission: split the final design into independent assets and choose the safest production method for each one.

Deliverable:

```yaml
asset_plan:
  background:
    method: gpt-image
    file: 01_background.png
  main_visual:
    method: source-product-or-gpt-image
    file: 02_main_visual.png
  decorations:
    method: gpt-image-or-photoshop-shape
    file: 03_decorations.png
  cards:
    method: photoshop-shapes
    layer_group: 30_cards
  icons:
    method: svg-or-single-icon-generation
    files: [icon_01.png, icon_02.png, icon_03.png]
  text:
    method: photoshop-text-layers
    layer_group: 50_typography
  cta:
    method: photoshop-shape-and-text
    layer_group: 60_cta
  proof_or_reviews:
    method: real-source-sanitized-composite
    rule: never hallucinate proof screenshots
```

### 5. Prompt Engineer

Mission: write one prompt per generated asset. The prompt must keep that asset independent and text-safe.

Background prompt pattern:

```text
Create a clean premium ecommerce background layer.
This will be used as a Photoshop background layer.
Use soft commercial lighting, subtle depth, clean negative space, and the requested brand color mood.
No readable text. No letters. No fake typography. No logos. No people. No products.
Output a background-only image at the requested aspect ratio.
```

Main visual prompt pattern:

```text
Create a single isolated main visual object for a commercial ecommerce composition.
Subject: {subject}
Use clean edges, centered composition, soft realistic shadow, and no text.
Do not add logos, captions, labels, badges, or random marks.
Use a simple background that can be removed in Photoshop.
```

Icon prompt pattern:

```text
Create one minimal rounded line icon.
Topic: {icon topic}
Style: consistent 2px stroke, dark navy line, no text, no background, centered.
```

### 6. Image Generation Worker

Mission: generate or regenerate individual assets. Never create the whole final design in this mode.

Output structure:

```text
03_generated/
  background_v1.png
  background_v2.png
  main_visual_v1.png
  icon_01_v1.png
04_selected/
  01_background.png
  02_main_visual.png
  03_icon_01.png
```

### 7. Photoshop Compositor

Mission: assemble the selected assets into a layered PSD. This stage should be serialized to avoid PSD conflicts.

Recommended layer tree:

```text
00_reference_hidden
01_layout_guides
10_background
20_main_visual
30_cards
40_badges
50_typography
60_cta
90_adjustments
```

Rules:

- keep text editable
- keep CTA editable
- keep card backgrounds as shapes when possible
- place product/proof assets from real sources, not generated approximations
- preserve original source files separately

### 8. Typography Agent

Mission: turn the approved Korean copy into a Photoshop typography plan.

Deliverable:

```yaml
typography_plan:
  headline:
    copy: "초보셀러도 바로 쓰는 AI 자동화 자료"
    font: "Pretendard ExtraBold or Noto Sans KR Bold"
    role: primary
  subheadline:
    copy: "상품등록부터 상세페이지까지 한 번에 정리"
    role: support
  cards:
    - title: "상세페이지 자동화"
      body: "복붙 가능한 실전 템플릿"
  cta:
    copy: "자료 구성 확인하기"
```

### 9. QA Agent

Mission: inspect both generated assets and the final composite.

Asset QA:

- no fake text or pseudo-letters
- no broken hands/faces/objects
- enough blank space for Photoshop text
- style consistency across icons/decorations

Final QA:

- benchmark structure reflected without direct copying
- Korean text exact and readable
- mobile-downscaled preview readable
- product shape/color/material preserved
- no unverified price, rank, certification, discount, review, or medical claims
- PSD/export files exist
- QA sheet/report exists

### 10. Ops Logger

Mission: record the brief, asset plan, prompts, final files, and QA results.

Minimum log fields:

```yaml
production_log:
  source_reference: "..."
  final_psd: "..."
  final_png: "..."
  qa_sheet: "..."
  prompt_folder: "..."
  model: "gpt-image-*"
  photoshop_status: "opened / manual-pass-needed / not-opened"
  qa_status: "pass / needs-revision"
```

## Production Folder

```text
image-production/
  00_brief/
    user_request.md
    reference_analysis.yaml
    layout_plan.yaml
    asset_plan.yaml
    typography_plan.yaml
  01_source/
    reference.png
    product.png
    logo.png
    review_capture.png
  02_prompts/
    background_prompt.md
    main_visual_prompt.md
    icon_prompt.md
    decoration_prompt.md
  03_generated/
  04_selected/
  05_photoshop/
    working.psd
    final.psd
  06_output/
    final.png
    final.jpg
    final_mobile_preview.jpg
  07_qa/
    comparison_sheet.jpg
    qa_report.md
  08_log/
    production_log.md
```

## Gates

### Gate 1 — Brief Ready

Pass when purpose, output ratio, target platform, core copy, and benchmark file are known.

### Gate 2 — Decomposition Ready

Pass when every visual element has a method: generated, sourced, Photoshop shape, SVG, or editable text.

### Gate 3 — Generated Asset QA

Pass when selected generated assets have no fake text, broken objects, or composition blockers.

### Gate 4 — Photoshop Composite QA

Pass when text, cards, CTA, alignment, and mobile readability pass.

### Gate 5 — Delivery

Pass when PNG/JPG/PSD, QA sheet, and production log are verified on disk.

## Common Pitfalls

1. One-shot generation of a complex design creates uneditable mistakes.
2. Generating Korean text, prices, reviews, or proof screenshots introduces accuracy and legal risk.
3. Generating all icons at once can create inconsistent icon styles; generate one at a time or use SVG.
4. Parallel workers should not edit the same PSD or output folder names.
5. A visually nice background can still fail if it lacks clean negative space for text.
6. Treating real product images as generative subjects can distort the product; use source images and masking instead.

## Verification Checklist

- [ ] Complexity level selected.
- [ ] Reference analysis written.
- [ ] Coordinate layout written.
- [ ] Asset plan separates GPT assets from Photoshop/text/proof assets.
- [ ] One prompt per generated asset.
- [ ] Generated assets QA passed before compositing.
- [ ] PSD layer groups are named and editable.
- [ ] Korean text, numbers, CTA, and proof content are Photoshop/source-based.
- [ ] Final QA sheet and report exist.
- [ ] Production log recorded and read back.
