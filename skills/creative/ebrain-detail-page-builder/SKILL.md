---
name: ebrain-detail-page-builder
description: "Use when the user provides a Korean marketplace/wholesale product link and wants Ebrain Detail Page Builder to analyze product info/images, plan a SmartStore/Coupang B2C 상세페이지, generate Korean text-in-image section assets, QA them, and produce exactly two final images: one thumbnail/main image and one stitched full detail-page image."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [ecommerce, korean-marketplace, smartstore, coupang, product-link, pdp, image-generation, obsidian]
    related_skills: [ocr-and-documents]
---

# Ebrain Detail Page Builder

## Overview

Ebrain Detail Page Builder is a Korean ecommerce 상세페이지 production workflow for external students, sellers, and operators. Use this workflow when the user gives a product URL such as 도매꾹, 네이버 스마트스토어, 쿠팡, 11번가, G마켓, ESM/옥션, or another Korean marketplace/wholesale page and asks to create a complete 상세페이지/PDP.

The goal is not just to summarize the link. The Ebrain Detail Page Builder end-to-end deliverable is:

1. product fact extraction
2. product image/source-image recovery
3. visual/product-shape analysis
4. B2C SmartStore/Coupang 상세페이지 section plan
5. section-by-section Korean copy and image prompts
6. GPT Image / ChatGPT image generation with Korean text when requested; prefer the user's configured ChatGPT/OpenAI OAuth/OAuth2-capable runtime when available, otherwise produce a prompt pack for manual or configured image generation
7. OCR/vision QA and regeneration for failed sections
8. intermediate section images used for production and QA
9. final deliverable image 1: one product thumbnail/main image
10. final deliverable image 2: one stitched long detail-page image
11. optional Obsidian notes: plan, prompt pack, QA report, output index

This skill is based on a proven Korean marketplace workflow: product link → product info/images → product-analysis and PDP-copy frameworks → section plan → Korean text-in-image section generation → QA → final thumbnail plus one long PDP image.

## When to Use

Use when the user says things like:

- `이 링크로 상세페이지 만들어줘`
- `도매꾹 제품으로 상세페이지 기획하고 이미지까지 만들어줘`
- `링크 입력하면 제품 정보와 이미지를 파악해서 상세페이지 완성하는 단계까지 해줘`
- `스마트스토어/쿠팡용 상세페이지 이미지 만들어줘`
- `각 파트별 이미지 생성하고 마지막에 한 장으로 합쳐줘`
- `제품 원본 이미지는 사이트에 있는 걸 활용해줘`

Do not use this skill for:

- only naming/title/keyword generation
- only product compliance checking without PDP generation
- dense legal/spec table final rendering where the user does not want images
- products where access requires unauthorized bypass; only use pages and assets the user can legitimately access

## Operating Principles

1. Act directly. If a product URL is provided, inspect it with tools instead of asking for repeated confirmation.
2. Treat all seller/marketplace claims as unverified until extracted from the source page or images.
3. Do not invent certification, battery capacity, runtime, dimensions, review counts, rankings, safety claims, or warranty.
4. For Korean ecommerce, keep copy B2C and consumer-facing unless the user explicitly asks for B2B/wholesale/판촉물.
5. For SmartStore/Coupang, distinguish:
   - main/thumbnail image: product-only, clean background, usually no text for Coupang
   - detail images: vertical mobile slices with clear Korean copy
6. Prefer short Korean text inside GPT Image outputs. Long copy, notice tables, shipping/return terms, prices, and legal copy should be deterministic text composition or summarized as checklist sections.
7. Always QA every generated section for Korean OCR, product fidelity, unsafe claims, and platform risk before saving as accepted.
8. Save outputs in a durable project folder, normally inside the user’s Obsidian vault.

## Default Project Folders

Default to the user's configured workspace or vault. In WSL/Windows setups, convert Windows paths like `C:\Users\<User>\Documents\Obsidian Vault` to `/mnt/c/Users/<User>/Documents/Obsidian Vault`.

Recommended project structure:

```text
<workspace>/
  YYYY-MM-DD <상품명> 상세페이지 기획안.md
  YYYY-MM-DD <상품명> 이미지 세분화 제작 기획서.md
  YYYY-MM-DD <상품명> GPT Image 한글 상세페이지 파트별 프롬프트팩.md
  YYYY-MM-DD <상품명> 섹션별 이미지 생성 QA 결과.md
  Assets/<상품코드>_PDP_Working_Sections/   # 작업/QA용 중간 섹션 이미지
  Assets/<상품코드>_PDP_Final/              # 최종 납품: 썸네일 1장 + 합본 상세페이지 1장
```

If the user specifies a folder, save exactly there.

## End-to-End Workflow

### 1. Intake

Minimum input:

- product URL

Infer or extract:

- product name
- marketplace/platform
- product number/code
- wholesale price or visible price if relevant
- category
- core features
- options/colors
- components
- certification/safety notices
- shipping/return constraints
- target platform: SmartStore, Coupang, or both
- target customer: B2C individual one-unit purchase unless told otherwise

Ask only if the missing choice changes the action materially. If ambiguous, default to SmartStore/Coupang B2C single-purchase 상세페이지.

### 2. Extract Product Facts

Use a layered extraction strategy:

1. `web_extract(url)` for text snapshot.
2. `browser_navigate(url)` for dynamic pages.
3. `browser_console` for `document.body.innerText`, links, script variables, and embedded product data.
4. `browser_get_images` for thumbnails and long detail images.
5. If page detail body is a remote image, look for sources such as:
   - `ai.esmplus.com`
   - `gi.esmplus.com`
   - `cdn*.domeggook.com`
   - marketplace CDN URLs
6. `vision_analyze` the main product thumbnail and long detail image(s) to recover information not in text.

Record facts with source confidence:

```text
confirmed_from_page_text
confirmed_from_source_image_ocr
confirmed_from_visual_inspection
inferred_for_planning
seller_confirmation_required
```

### 3. Product Image Analysis

Run vision analysis on at least:

- main thumbnail/product packshot
- long detail image, if available
- close-up/spec/component images, if separate

Extract:

- product silhouette and color
- visible components
- strap/cable/accessory presence
- logo/label/printed text
- B2B-only visual elements to crop/mask, e.g. `레이저 각인 인쇄`, OEM, 판촉물, 대량구매, 행사 담당자
- use scenes shown by seller
- certification/notice OCR
- features that must not be hallucinated

For exact marketplace fidelity, prefer using real product crop/cutout as the anchor. GPT Image can generate background/scene/design direction, but final marketplace product identity must not drift materially.

### 4. Compliance and Claim Ledger

Create a compact claims ledger before writing copy.

For every claim, mark one of:

- `confirmed`: visible in page text or seller image OCR
- `visual`: visible from product photo only
- `safe_general`: generic consumer language, no hard claim
- `needs_confirmation`: cannot be used as a factual statement yet
- `blocked`: do not use

Common blocked/unsafe phrases for electronics and small appliances:

```text
무소음
초강력
완전 안전
아이에게 안전
KC 인증 완료
장시간 사용
하루종일 사용
배터리 용량/사용시간 without source
국내 최저가
1위
후기 만족도/재구매율 without real reviews
열사병 예방 or medical/safety effect
방수 without source
```

For battery/electronic portability sections, avoid copy implying unsafe storage. Example:

- Risky: `차 안에 하나`
- Safer: `외출 가방에 쏙, 필요한 순간 꺼내 쓰세요.`

### 5. B2C PDP Strategy

Default target:

```text
SmartStore/Coupang B2C, one person buying one unit for personal use.
```

Translate product facts into customer language:

- Feature → Advantage → Benefit → Scene → One-line copy
- Customer problem → product mechanism → proof/visual → buying confidence
- Use scenes should be specific and everyday: 출퇴근, 등하교/캠퍼스, 산책, 야외활동, 책상, 사무실, 보관
- Avoid B2B/판촉 language unless user requested it: 대량구매, 단체선물, 행사 담당자, 인쇄 가능, OEM

### 6. Adaptive Section Planning

Do not force a fixed 17-section layout. Build the section count and order from the product, category, available source images, claim risk, target marketplace, and conversion goal.

Use the recommended flow below as a menu, not a mandatory checklist:

1. product-only main thumbnail
2. hero stopper
3. problem empathy
4. comparison / product difference
5. product reveal
6. benefit summary
7. strongest feature proof
8. feature detail 1
9. feature detail 2
10. use case 1
11. use case 2
12. indoor/desk use
13. portability/storage
14. detail/components
15. trust/checkpoints
16. FAQ
17. notice/checklist
18. final CTA

Adapt this by category; do not force all sections if the product needs a different route. A simple low-risk product may need fewer sections, while a high-consideration or regulated product may need additional proof, comparison, ingredient/spec, safety, usage, or notice sections.

### 7. Product-Specific Section Plan Template

For each section, define:

```text
section_no:
file_name:
section_title:
objective:
scroll_role: stop | deepen | prove | compare | reduce_risk | convert
customer_question:
main_copy:
sub_copy:
labels_or_cards:
visual_direction:
required_source_image:
ai_generation_allowed:
product_fidelity_risk:
claim_risk:
qa_criteria:
```

Example product-specific section module set. This is an example menu, not a required section count:

```text
00_main_thumbnail_product_only.png
01_hero_stopper.png
02_problem_or_use_context.png
03_product_solution.png
04_key_benefit_summary.png
05_feature_proof_or_detail.png
06_comparison_or_difference.png
07_use_case_scene.png
08_components_or_options.png
09_specs_or_care_summary.png
10_trust_checkpoints.png
11_faq_or_notice.png
12_final_cta.png
```

### 8. Prompt Pack Creation

For each GPT Image section prompt, include:

```text
DELIVERABLE
CRITICAL PRODUCT REFERENCE
COMPOSITION
STRICT KOREAN TEXT RULES
TEXT TO RENDER
NEGATIVE CLAIMS
AVOID
```

Korean text rules:

```text
- Render ONLY the exact Korean text below.
- No extra Korean words.
- No random Korean or pseudo-Korean.
- No misspellings.
- Use large bold Korean sans-serif typography.
- Keep text mobile-readable.
```

Product correction block must be concrete:

```text
- exact product category and silhouette from the source image
- visible color/material/finish
- key components and accessories visible in the seller images
- important buttons, labels, ports, packaging, or texture details
- what the product is NOT, to prevent category drift
- no invented logos, certifications, awards, prices, or review counts
```

Rewrite this block for every product from source image analysis.

### 9. Image Generation Strategy

Image generation is runtime-dependent. The skill should not hard-code private API keys or assume one universal renderer. Preferred order:

1. If the user's Hermes/OpenClaw environment has ChatGPT/OpenAI OAuth/OAuth2 image generation available, use that OAuth-authenticated runtime for GPT Image / ChatGPT image outputs.
2. If a configured Hermes image-generation tool is available, use it with the approved section prompt.
3. If direct image generation is unavailable, still produce the product-specific prompt pack so the user can paste it into ChatGPT Image manually.

In all cases, keep the same QA gate: generated images must pass Korean text, product fidelity, unsupported-claim, and platform-risk checks before acceptance.

Recommended generation order:

1. 3-slice pilot first if the style is untested:
   - hero
   - strongest feature/use-case
   - benefit/spec summary
2. If approved, continue 2 parts at a time or generate all remaining sections when the user asks.
3. For each section:
   - generate one image
   - run vision/OCR QA
   - if failed, regenerate only failed section with tighter text/product constraints
   - save accepted output

When the user explicitly wants direct Korean text in GPT Image, do it, but keep section copy short.

### 10. QA Protocol

For each generated image, ask vision QA:

```text
Check exact Korean text against this list: <expected text list>.
Report:
- typo/missing text/extra text
- random Korean or pseudo-Korean
- product shape drift
- unsafe or unsupported claim
- platform risk
- use 가능 / 조건부 / 재생성 필요
```

Regenerate if any of the following occurs:

- typo in key Korean copy
- pseudo-Korean or random text
- product becomes wrong category
- visible fake certification/award/review/price/ranking
- unsupported performance/safety/medical claim
- unsafe storage implication, e.g. hot-car battery storage
- severe layout crop, watermark, tool UI

Conditionally accept when:

- small text is less readable but not critical
- product is stylized but still clearly the same category
- a section is for internal review, not final upload

### 11. Save and Verify Working Section Files

Save accepted section images to the requested working folder for QA and stitching. These section files are production intermediates, not the final result package unless the user explicitly asks for all slices.

Verify with:

```bash
file "<path>"
du -h "<path>"
```

Recommended working filenames are product-specific. Do not force every product to use a fixed section count:

```text
00_main_thumbnail_product_only_gpt2.png
01_hero_stopper_gpt2.png
02_problem_or_use_context_gpt2.png
...
NN_final_cta_gpt2.png
```

Final customer-facing result should be reduced to exactly two image files:

```text
<상품코드>_thumbnail.png or .jpg
<상품코드>_full_detail_page.png or .jpg
```

If older pilot filenames differ, keep them in the working folder but maintain a final ordered list for stitching.

### 12. Create the Final Two Image Deliverables

After all working sections pass QA, create exactly two final image deliverables:

1. **Thumbnail/main image** — one square product-first image for the listing thumbnail. For Coupang, use a clean white-background product-only image with no text, badges, price, fake marks, or lifestyle props. For SmartStore, product-first is still preferred unless the user asks for a styled thumbnail.
2. **Full detail-page image** — one long stitched JPG/PNG made by vertically combining the approved section images.

Keep the individual section images in the working folder for revision and traceability, but do not present them as the final result unless requested.

Use PIL in WSL. If Pillow is missing or PEP 668 blocks install, create a local venv in the project folder:

```bash
python3 -m venv .venv-pdp
.venv-pdp/bin/python -m pip install pillow
```

Stitch logic:

```python
from PIL import Image
from pathlib import Path

folder = Path("/path/to/section/folder")
outdir = Path("/path/to/stitched/folder")
outdir.mkdir(parents=True, exist_ok=True)

files = [
    folder/"00_main_thumbnail_product_only_gpt2.png",
    folder/"01_hero_stopper_gpt2.png",
    # ... product-specific ordered section list ...
    folder/"NN_final_cta_gpt2.png",
]

target_w = 1024
images = []
for p in files:
    im = Image.open(p).convert("RGB")
    if im.width != target_w:
        new_h = round(im.height * target_w / im.width)
        im = im.resize((target_w, new_h), Image.LANCZOS)
    images.append(im)

canvas = Image.new("RGB", (target_w, sum(i.height for i in images)), "white")
y = 0
for im in images:
    canvas.paste(im, (0, y))
    y += im.height

canvas.save(outdir/"full_detail_page.png", optimize=True)
canvas.save(outdir/"full_detail_page.jpg", quality=92, optimize=True, progressive=True)
```

Verify:

```bash
file full_detail_page.png full_detail_page.jpg
du -h full_detail_page.png full_detail_page.jpg
```

Then run vision QA on both final images:

Thumbnail QA:

```text
Is this a clean product-first thumbnail?
For Coupang: is it square, white-background, product-only, and free of text/badges/price/fake certification/lifestyle clutter?
Is the product shape close enough to the source product?
```

Full detail-page QA:

```text
Are all sections ordered naturally?
Any severe crop, black gap, broken image, watermark, tool UI?
Are main Korean lines readable?
Any unsafe claims visible?
Is this usable as SmartStore/Coupang review preview?
```

### 13. Obsidian Notes

When the workflow is reusable or substantial, save notes:

1. Product analysis and PDP plan
2. Image section breakdown
3. GPT Image prompt pack
4. Section generation QA report
5. Output index with file paths

Use wikilinks to source notes when relevant. Do not store secrets, session cookies, passwords, or private authentication values.

## Platform-Specific Guidance

### Coupang

- Main image should be square, clean, product-only, white background.
- Avoid text, badges, props, lifestyle elements in main image.
- Detail-page long image can include text, but check for platform-sensitive button-like CTAs, unsupported claims, or fake certification marks.

### SmartStore

- Main image can be more flexible than Coupang, but clean product-first still works best.
- Detail sections should be mobile-readable.
- Strong Korean headlines are useful, but body text should not be too small.

### Wholesale to Retail Conversion

When source is wholesale/판촉물 page, remove or reframe:

- 대량구매
- 단체선물
- 행사/판촉 담당자
- 인쇄 가능
- OEM/custom print
- B2B-only price/quantity logic

Replace with B2C language:

- 개인 사용 장면
- 여름 외출
- 가방 보관
- 출퇴근/산책/책상
- 구매 전 확인사항

## Example Safe Copy Patterns

Hero:

```text
필요한 순간,
제품의 장점이 바로 보이게
```

Problem:

```text
고객이 불편을 느끼는 순간을
짧고 구체적으로 보여주세요
```

Product reveal:

```text
이 제품이 어떤 방식으로
문제를 줄여주는지 보여주세요
```

Benefit trio:

```text
핵심 장점 1
핵심 장점 2
구매 전 확인 포인트
```

Trust/check:

```text
인증 정보
상세페이지 하단 확인

배송 정보
판매 페이지 기준 확인

최종 구매 전 상세 정보를 확인하세요
```

FAQ fallback when specs are missing:

```text
사용 시간은?
상세 스펙 확인 필요
```

Notice:

```text
구매 전 꼭 확인해주세요
제품명·모델명
인증 정보
배송·교환·반품
사용 전 주의사항
```

## Output Report Format

When finished, report in Korean:

```text
완료했습니다.

저장 위치:
<Windows path>

생성 파일:
1. 썸네일 이미지 1장
<path>
2. 합쳐진 전체 상세페이지 이미지 1장
<path>

작업/QA용 섹션 이미지 폴더:
<path>

파일 정보:
- 크기: <width x height>
- 용량: <sizes>
- 포함 결과: 썸네일 1장 + 전체 상세페이지 합본 1장

QA 결과:
- 주요 한국어 문구: 통과 / 일부 확인 필요
- 제품 형태: 통과 / 일부 보정 필요
- 위험 표현: 없음 / 확인 필요
- 플랫폼 사용성: SmartStore/Coupang 검토용 가능

주의:
- 실제 인증/모델명/사용시간/배송/교환/반품은 판매 페이지 기준 최종 확인 필요
- 쿠팡 대표 이미지는 별도 product-only square image 사용 권장
```

## Common Pitfalls

1. **Only reading page text and missing detail images.** Korean product pages often put specs and notices inside long remote images. Always inspect image URLs and OCR/vision them.

2. **Inventing missing specs.** If runtime, battery, dimensions, certification, charging method, components, or warranty are not visible, mark `확인 필요` or avoid the claim.

3. **Using B2B copy for B2C pages.** Wholesale pages often emphasize printing, gifts, quantity, or unit price. Convert to individual-use consumer scenes.

4. **Accepting pretty but wrong products.** GPT Image may make a generic product. QA product silhouette and regenerate with concrete visual constraints.

5. **Trusting Korean text without OCR QA.** Always vision-check exact Korean phrases. Regenerate if key text is wrong.

6. **Putting dense legal/spec copy into GPT Image.** Use checklist summary for GPT Image, deterministic text composition for exact legal tables.

7. **Unsafe electronics storage copy.** Avoid implying lithium/battery products should be stored in hot cars. Use general portability language.

8. **Forgetting platform differences.** Coupang main image usually needs product-only on white. Detail pages can be more designed.

9. **Not verifying output files.** Always run `file` and `du -h`; then QA the actual stitched image, not just source sections.

10. **Deleting working section images too early.** Final delivery is only two images, but keep working section images internally so the page can be revised, re-QA’d, or re-stitched.

11. **Reporting every section image as the final output.** The final user-facing output should be exactly one thumbnail image and one full stitched detail-page image unless the user asks for slices.

## Verification Checklist

Before telling the user the PDP is complete:

- [ ] Product URL inspected with web/browser tools.
- [ ] Product text facts extracted.
- [ ] Main product image and long detail image analyzed with vision.
- [ ] Claims ledger created or mentally applied; unsupported claims removed.
- [ ] B2C SmartStore/Coupang direction confirmed or defaulted.
- [ ] Section plan created with filenames and copy.
- [ ] Prompt pack or prompts generated with strict Korean text rules.
- [ ] Every section image generated or composed.
- [ ] Every section QA’d for Korean text, product fidelity, and risk claims.
- [ ] Failed sections regenerated or clearly marked as conditional.
- [ ] Working section files saved to durable folder for revision/QA.
- [ ] Final thumbnail image 1장 created and verified.
- [ ] Final full detail-page stitched image 1장 created and verified.
- [ ] `file` and `du -h` verified dimensions and size for the two final images.
- [ ] Vision QA run on both the thumbnail and final stitched image.
- [ ] User received Windows/WSL paths and final cautions.
