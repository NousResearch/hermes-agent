---
name: chinese-menu-image-analysis
description: "Use when analyzing Chinese restaurant menu photos/screenshots: split hard-to-read regions, OCR/VLM each section, add pinyin for Chinese names, translate, summarize prices and recommendations, optionally build a clean HTML menu report, add experimental expected-dish photos, and provide a temporary mobile/browser URL when requested."
version: 1.4.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [ocr, vision, chinese, menu, pinyin, translation, restaurants, html-report, mobile-share, dish-images]
    related_skills: [ocr-and-documents]
---

# Chinese Menu Image Analysis

## Overview

Use this workflow to analyze Chinese restaurant menu photos, especially low-resolution screenshots, angled menu boards, or dense bilingual menus. The goal is to produce a practical Korean explanation with:

- OCR/VLM text extraction by menu section
- Chinese dish names preserved in Hanzi
- Pinyin for each Chinese dish name so the user can read/order it
- Korean explanation of the dish
- Prices, charges, branch/location notes, and practical recommendations
- Clear uncertainty markers when the image is blurry or occluded
- Optional polished HTML report containing the uploaded photo, cropped evidence, OCR menu sections, pinyin, Korean explanations, prices, and recommendations
- Optional mobile/browser sharing link via a report-folder-only local server plus Cloudflare Tunnel when explicitly requested
- Optional **experimental** expected-dish photo cards in the HTML report, using clearly labeled representative imagery rather than claiming an exact restaurant photo
- Optional menu-item image enrichment: for selected dishes, search/collect representative images, attach source/attribution, fall back to generated/placeholder cards, and verify all image assets load
- Optional restaurant/review enrichment from Chinese-language sources (OpenRice, Dianping/大眾點評, Google/Maps snippets, official sites) when the menu shows a restaurant/branch name

This skill is optimized for Discord/Slack image attachments and the user's preference for Korean output. When the user wants a shareable/viewable result, create an HTML artifact and attach it with `MEDIA:/absolute/path/to/file.html`. For external/mobile viewing, follow `references/mobile-html-cloudflared-sharing.md`. For Hong Kong restaurant identity/review enrichment, OpenRice extraction, and safe use of review photos, follow `references/hk-restaurant-review-enrichment.md`.

## When to Use

Use when the user asks things like:

- "이 메뉴판 분석해봐"
- "한자 읽을 수 있게 pinyin 같이 표시해줘"
- "잘 안 보이는 부분은 분할해서 OCR/VLM 해줘"
- "중국 음식점 메뉴판 번역/추천해줘"
- "이 가격/메뉴 구성 비교해줘"
- "업로드한 사진과 메뉴를 HTML로 보기 좋게 정리해줘"
- "분석 결과를 공유 가능한 파일/페이지로 만들어줘"
- "HTML 링크줘" / "html 링크줘" — attachable HTML already exists and the user wants a browser-accessible URL
- "메뉴별 예상 사진도 같이 보여줘"
- "음식 사진 예시를 HTML에 넣어줘"
- "중국쪽 웹에서 이미지/리뷰 찾아봐"
- "이 가게 찾아서 리뷰도 추가해줘"

Don't use for:

- Text already pasted in full; then translate/analyze directly.
- Non-Chinese menus unless pinyin/Hanzi are still relevant.
- Legal/medical document OCR; use `ocr-and-documents` instead.

## Required Workflow

### 0. If the user sends an app/deep link or restaurant-share link

When the user sends a restaurant/menu app link (e.g. Dianping/大众点评, Meituan/美团, OpenRice, WeChat mini-program share URL, Google Maps restaurant link) and asks to translate/recommend/show as HTML:

1. **Open/fetch the link first** with `terminal`/HTTP tooling or browser tooling if available. Do not assume it is the same menu/report as a previous chat.
2. **Identify the target** from the fetched page: restaurant name, branch, city/address, cuisine, rating/price, current/open status, and source URL. Put these in the HTML summary.
3. **Extract whatever menu data is actually exposed**:
   - Full menu/photos/prices when available.
   - Recommended dishes, dish photos, review-photo menus, or masked names when the platform hides the full menu behind the app.
   - If the page exposes only masked dish names like `低**肉`, preserve the masks and label the report as “accessible app-link data,” not a full menu translation.
4. **Download/copy images into the report folder** rather than hotlinking. For public/mobile HTML, prefer inline `data:image/...;base64,...` images when practical so Discord/mobile browsers, Cloud Run `/r/{token}` paths, and anti-hotlinking do not break images.
5. **Build and publish the HTML automatically** when the user asked for HTML or “정리해서 보여줘/배포해줘.” Use the report publishing workflow if configured; otherwise attach the standalone `MEDIA:` HTML.
6. **Verify after publishing**:
   - Published HTML returns HTTP 200.
   - HTML contains the intended restaurant name, not an old report’s restaurant.
   - Every `img src` is either a `data:image/...` URI or a browser-resolvable token-qualified URL.
   - If using asset URLs, verify each resolves with HTTP 200 from the exact published page context.
7. **Final response** should include the URL/file and a short caveat about source limitations (e.g. app-only full menu, masked names, missing prices). Do not ask for a screenshot unless the app link truly does not expose enough useful data.

### 1. Load Supporting Skill

Load `ocr-and-documents` if not already loaded. It contains OCR/document extraction fallbacks and reminders.

### 2. Inspect the Image

Use `terminal` or Python/PIL to get dimensions:

```bash
python3 - <<'PY'
from PIL import Image
p = '/path/to/image.webp'
im = Image.open(p)
print(im.size, im.mode)
PY
```

Then use `vision_analyze` on the full image for a first pass:

- Identify restaurant name / branch / location
- Identify menu sections
- Note if it is a set menu or à la carte menu
- Note currency and service-charge text

### 3. Create Crops for Readability

For dense menus, crop by section rather than relying on the full screenshot. Use 2-4x upscaling, sharpen, and contrast:

```bash
python3 - <<'PY'
from PIL import Image, ImageEnhance
from pathlib import Path

src = Path('/path/to/image.webp')
out = Path('/Users/aa/.hermes/media_cache/menu_analysis')
out.mkdir(parents=True, exist_ok=True)
im = Image.open(src).convert('RGB')
print('size', im.size)

# Adjust boxes after looking at the screenshot. Coordinates are (left, top, right, bottom).
boxes = {
    'full': (0, 0, im.width, im.height),
    'left_section': (0, 0, im.width//3, im.height),
    'middle_section': (im.width//3, 0, 2*im.width//3, im.height),
    'right_section': (2*im.width//3, 0, im.width, im.height),
    'footer': (0, int(im.height*0.78), im.width, im.height),
}

for name, box in boxes.items():
    crop = im.crop(box)
    scale = 3
    crop = crop.resize((crop.width*scale, crop.height*scale), Image.Resampling.LANCZOS)
    crop = ImageEnhance.Sharpness(crop).enhance(1.8)
    crop = ImageEnhance.Contrast(crop).enhance(1.2)
    path = out / f'{name}.jpg'
    crop.save(path, quality=95)
    print(path, crop.size)
PY
```

Crop strategy:

- Wide menu board: left/middle/right columns + footer + restaurant header.
- Set menu board: A set, B set, bottom notes/branch info.
- **Vertical-writing set menus:** first identify the page's real pixel dimensions after conversion (screenshots may be downscaled). Crop each horizontal band/box containing a set, not arbitrary tall strips. Read each set from the **rightmost vertical column to the leftmost column**; preserve this order in the report. Make a separate footer/price crop for small horizontal text.
- Small dense text: crop the specific subsection again (e.g., soup, dessert, rice/noodles).
- If a section remains ambiguous, make a tighter crop and run `vision_analyze` again.

macOS/WebP fallback:

- If Python Pillow/PIL is unavailable, or `sips` refuses to crop/write a WebP input (`Can't write format: org.webmproject.webp`), convert the uploaded WebP to JPEG first with `ffmpeg`, then crop the JPEG with `sips`:

```bash
OUT=/Users/aa/.hermes/media_cache/menu_analysis
mkdir -p "$OUT/crops"
ffmpeg -y -i /path/to/uploaded.webp "$OUT/source.jpg"
cd "$OUT/crops"
# sips syntax: -c cropHeight cropWidth --cropOffset y x
sips -c 655 300 --cropOffset 45 35 "$OUT/source.jpg" --out left_section.jpg
sips -c 625 310 --cropOffset 40 300 "$OUT/source.jpg" --out middle_section.jpg
sips -c 615 260 --cropOffset 30 820 "$OUT/source.jpg" --out right_section.jpg
for f in *.jpg; do sips -Z 1800 "$f" --out "$f" >/dev/null; done
```

- Remember that `sips --cropOffset` uses `y x` order, while PIL crop boxes use `(left, top, right, bottom)`. Convert coordinates carefully when moving between the two approaches.

### 4. OCR/VLM Each Section

Use `vision_analyze` on each crop with a precise prompt:

```text
이 메뉴판의 <SECTION> 섹션을 OCR해줘. 중국어 메뉴명, 영어 설명, 가격을 줄별로 가능한 정확히 읽어줘. 불확실하면 ? 표시.
```

For follow-up refinement:

```text
전체 메뉴판에서 분할 이미지에서 안 보였던 <SECTION>을 보완해서 OCR해줘. 특히 메뉴명/가격 중심으로.
```

Important:

- Mark uncertain characters/prices as `?`.
- Do not invent missing text.
- Use the English text on bilingual menus to cross-check Chinese dish interpretation.
- Treat OCR/VLM output as fallible; compare across full image and cropped sections.

### 5. Generate Pinyin

Use `pypinyin` for dish names. If missing, install it in the system Python used by `terminal`:

```bash
python3 -m pip install -q pypinyin
```

Then generate pinyin:

```bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
python3 - <<'PY'
# -*- coding: utf-8 -*-
from pypinyin import pinyin, Style
items = ['鮮肉小籠包', '上海粗炒', '生拆蟹粉豆腐']
for s in items:
    print(s + ' — ' + ' '.join(x[0] for x in pinyin(s, style=Style.TONE)))
PY
```

Pinyin notes:

- Menus in Hong Kong often use Traditional Chinese; `pypinyin` gives Mandarin readings, not Cantonese.
- Tell the user: "한자는 번체자, pinyin은 보통화/만다린 기준".
- Verify common OCR confusions manually: `鯽` vs `鱖`, `麵` vs `麺`, `燻` vs `熏`, `羹` vs `湯`.

### 6. Calculate Prices When Helpful

Use `execute_code` for arithmetic, especially service charge/per-person totals:

```python
prices = {'A': 1188, 'B': 1288}
for k, p in prices.items():
    total = p * 1.10
    print(k, total, total / 4)
```

Always use a tool for arithmetic.

### 7. Build an HTML Report When Requested

If the user asks to make the uploaded photo and menu analysis into a nice viewable document, create a standalone HTML file under `~/.hermes/media_cache/...`.

The HTML should include:

- The original uploaded image near the top
- Optional crop/contact-sheet evidence section for hard-to-read regions
- A summary card: restaurant, branch/location, currency, service charge, confidence notes
- Menu sections grouped by category
- For each item: Hanzi, pinyin, Korean explanation, English description if available, price, spicy/premium badges
- Recommendation blocks: signature, safe picks, spicy picks, avoid/caution items
- Uncertainty notes: any `?` OCR characters/prices and why
- Clean mobile-friendly CSS; no external CDN dependency
- Prefer a self-contained HTML file for mobile viewing: inline report images/crops as base64/data URLs when practical, so a single downloaded file renders correctly on phones and avoids Cloud Run `/r/{token}` relative-asset path issues. If the report uses separate assets for public publishing, use token-qualified paths or verify browser-resolved asset URLs after deployment.
- If the user asks for a mobile/browser link, serve only that report directory and provide an optional temporary HTTPS URL (e.g. via Cloudflare Tunnel/ngrok); do not serve all of `~/.hermes/media_cache`

Experimental expected-dish photos:

- If the user asks for expected/representative dish photos, add a clearly labeled image-card section to the HTML.
- Treat this as experimental and explain that images are representative, not actual restaurant plating, unless sourced from the uploaded menu or an official restaurant source.
- Prefer only recommended/signature/unfamiliar dishes by default; do not bloat the report with every menu item unless requested.
- If the user explicitly asks for "메뉴별 이미지" / "메뉴마다 사진" / "같이 보이는" output, include image cards for **every confidently identified dish** where feasible; otherwise cap to recommended/unfamiliar dishes.
- Do not silently embed arbitrary copyrighted restaurant photos. Use official/user-provided images, attributed/licensed public images, generated illustrative images when explicitly accepted, or placeholders.
- Follow `references/experimental-dish-image-cards.md` for source priority, card fields, HTML/CSS pattern, and verification.
- For Wikimedia Commons/API-based representative images, query fallbacks, `User-Agent` handling, single-file data-URI reports, and verification snippets, see `references/representative-dish-image-fetching.md`.

Menu-item image enrichment workflow:

1. Build a canonical dish list after OCR cleanup. Use Hanzi + pinyin + Korean name as the stable key.
2. For each dish, create search/query candidates in this order:
   - exact Hanzi dish name, e.g. `鮮肉小籠包`
   - Simplified/Traditional variant if obvious, e.g. `鲜肉小笼包`
   - pinyin or English/common Korean name, e.g. `xiao long bao`, `샤오롱바오`
   - generic category fallback, e.g. `Shanghai soup dumpling`
3. Source priority:
   - image cropped from the uploaded menu itself when the dish photo is visible
   - official restaurant/menu images if available and attributable
   - Wikimedia/Openverse/other clearly licensed public images with source URL + attribution
   - generated illustrative images only if the user accepts generated examples
   - placeholder card if no reliable image is found
4. Store image metadata beside the menu item:

```python
item['dish_image'] = {
    'path': 'dish_images/xiao_long_bao.jpg',
    'kind': 'representative',  # official | menu-crop | representative | generated | placeholder
    'source': 'https://commons.wikimedia.org/...',
    'credit': 'Author / License if known',
    'note': '실제 매장 이미지 아님',
}
```

5. When downloading images, keep filenames ASCII-safe and deterministic from the dish key. Prefer local copies inside the report folder over hotlinking.
6. In HTML, show the image card inline with or directly under the matching menu item, not only in a separate gallery, when the user asked for menu-by-menu visuals.
7. If an image is generic/category-level rather than the exact dish, label it `대표 이미지` and say which query/category it came from.
9. If the menu photo contains a restaurant or branch name, search Chinese-language/local sources for the exact restaurant before finalizing the report:
   - Search exact Hanzi + branch/location first, e.g. `上海小南國 九龍灣 OpenRice`, `上海小南國 MegaBox 食評`, `上海小南國 九龍灣 菜單`.
   - Prefer local restaurant platforms for the geography: OpenRice for Hong Kong, 大眾點評/美團 for mainland China, official site/social pages when available.
   - Add a compact `가게/리뷰 정보` section to the HTML with status, address, price range, ratings/review count, photo/menu-photo count, recommended dishes, and 2-3 recent review headlines when available.
   - Flag status clearly: if the source says `已結業` / 폐업, say the menu is likely historical/archive material and do not present it as an active restaurant.
   - Do not copy full reviews; summarize metadata/headlines with source link.

Implementation pattern:

```python
from pathlib import Path
import html, shutil

out_dir = Path('/Users/aa/.hermes/media_cache/menu_report')
out_dir.mkdir(parents=True, exist_ok=True)

# Copy source image/crops into the report folder so relative paths work.
source_img = Path('/path/to/uploaded.webp')
img_name = 'uploaded_menu.webp'
shutil.copyfile(source_img, out_dir / img_name)

sections = [
    {
        'name': '點心 / Dim Sum',
        'items': [
            {
                'hanzi': '鮮肉小籠包（4個）',
                'pinyin': 'xiān ròu xiǎo lóng bāo',
                'ko': '샤오롱바오 4개',
                'price': 'HK$48',
                'badges': ['추천'],
            }
        ],
    }
]

cards = []
for section in sections:
    item_html = []
    for item in section['items']:
        badges = ''.join(f'<span class="badge">{html.escape(b)}</span>' for b in item.get('badges', []))
        item_html.append(f'''
        <article class="item">
          <div class="item-main">
            <h3>{html.escape(item['hanzi'])}</h3>
            <p class="pinyin">{html.escape(item['pinyin'])}</p>
            <p>{html.escape(item['ko'])}</p>
          </div>
          <div class="price">{html.escape(item.get('price', ''))}</div>
          <div class="badges">{badges}</div>
        </article>''')
    cards.append(f'<section><h2>{html.escape(section["name"])}</h2>{"".join(item_html)}</section>')

html_doc = f'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Chinese Menu Analysis</title>
<style>
:root {{ color-scheme: light; --bg:#fff8ef; --card:#ffffff; --ink:#24170f; --muted:#77665b; --accent:#b83225; --line:#ead7c6; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }}
.wrap {{ max-width:1100px; margin:0 auto; padding:28px; }}
.hero {{ display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:24px; align-items:start; }}
.hero img {{ width:100%; border-radius:18px; box-shadow:0 10px 30px #0002; background:white; }}
.card, section {{ background:var(--card); border:1px solid var(--line); border-radius:18px; padding:20px; margin:18px 0; box-shadow:0 6px 18px #0000000d; }}
h1 {{ margin:0 0 8px; font-size:32px; }}
h2 {{ margin:0 0 14px; color:var(--accent); }}
.item {{ display:grid; grid-template-columns:1fr auto; gap:12px; padding:14px 0; border-top:1px solid var(--line); }}
.item:first-of-type {{ border-top:0; }}
.item h3 {{ margin:0; font-size:20px; }}
.pinyin {{ margin:4px 0; color:var(--accent); font-style:italic; }}
.price {{ font-weight:800; white-space:nowrap; }}
.badge {{ display:inline-block; margin:4px 4px 0 0; padding:3px 8px; border-radius:999px; background:#ffe3dd; color:var(--accent); font-size:12px; font-weight:700; }}
.muted {{ color:var(--muted); }}
@media (max-width:800px) {{ .wrap {{ padding:16px; }} .hero {{ grid-template-columns:1fr; }} .item {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main class="wrap">
<div class="hero">
  <div><h1>중국어 메뉴판 분석</h1><p class="muted">한자 · pinyin · 한국어 설명 · 가격 정리</p></div>
  <img src="{html.escape(img_name)}" alt="Uploaded menu photo" />
</div>
{''.join(cards)}
</main></body></html>'''

report_path = out_dir / 'menu_analysis.html'
report_path.write_text(html_doc, encoding='utf-8')
print(report_path)
```

Verification:

- Open/read the generated HTML file path exists.
- Ensure image references are relative and copied into the same folder.
- If possible, make a screenshot or visually inspect with browser tooling; otherwise attach the HTML and mention it is a standalone file.
- Final response should include `MEDIA:/absolute/path/to/menu_analysis.html` and a short Korean summary.

Optional mobile sharing URL:

- Only create a public/mobile URL when the user explicitly asks.
- Serve only the single report folder, never `~/.hermes` or the whole media cache.
- Start a local server from the report directory:

```bash
cd /Users/aa/.hermes/media_cache/menu_report
python3 -m http.server 8877 --bind 127.0.0.1
```

- In another process, open a quick Cloudflare Tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:8877
```

- Share `https://<random>.trycloudflare.com/menu_analysis.html` after verifying it returns HTTP 200.
- Remind the user the URL stays available only while both processes are running and is accessible to anyone with the link.
- See `references/mobile-html-cloudflared-sharing.md` for the full safe serving pattern, verification snippet, final-response wording, and cleanup notes.

User-approved output pattern:

- Chat response should be concise: first attach the HTML with `MEDIA:/absolute/path/to/menu_analysis.html`, then give a short Korean summary.
- Put the long menu table/list inside the HTML, not in the chat, unless the user explicitly asks for the full text inline.
- The approved HTML style is: original menu image at top/right, summary cards, recommendation/caution block, section cards, per-item Hanzi + Mandarin pinyin + Korean explanation + English/price, badges such as 추천/매움/고가/주의, and an OCR evidence image section at the bottom.
- If expected-dish photos are requested, add an `예상 음식 사진` section near the recommendations with image cards and the representative-image disclaimer from `references/experimental-dish-image-cards.md`.
- If the menu shows a restaurant/branch name and the user asks about Chinese-side web sources, images, or reviews, add a `가게/리뷰 정보` section using `references/hk-restaurant-review-enrichment.md`: prefer OpenRice/official pages for Hong Kong menus, surface `已結業` status prominently, summarize ratings/review/photo counts and recommended dishes, and cite the source URL instead of copying arbitrary review photos.
- Treat dish photos as opt-in experimental output: keep the chat concise, put visual detail in the HTML, and clearly label representative/generated/official/placeholder images.
- If the user asks for “중국 음식점 메뉴판 사진” or points out that menu photos were not added, do not only provide an analysis report. Add an explicit `추가 메뉴판 사진 예시` / menu-photo gallery section with actual menu-board/menu-page photos. Label these as reference examples unless they are official/user-provided photos of the target restaurant.
- If the user asks to translate/show a menu as HTML after sharing both a restaurant link and an older report/menu URL, first verify which source the HTML represents. Do not silently reuse or republish an old menu report for a different restaurant. If the current restaurant link does not expose menu items and no menu photo is available, say that the existing report is for the previous menu and ask for/locate the actual target menu before producing the final HTML.
- If a public/mobile URL is already running, verify that the local folder you edited is the same folder being served before telling the user it is updated. Check the HTTP server working directory (e.g. `pwdx <pid>` or `lsof -a -p <pid> -d cwd -Fn`) and copy/update assets in that served folder if needed.
- After adding images to a served HTML report, verify both the HTML and every newly referenced image URL return HTTP 200. A page returning 200 is not enough; relative image assets may still be 404 if the wrong folder is being served.
- If the user says images still do not show, fix proactively. Do not rely only on direct `curl /assets/foo` checks: inspect the HTML `src` values as the browser will resolve them. For Cloud Run report URLs like `/r/{token}` with no trailing slash, plain `assets/foo` is risky. Prefer token-qualified image URLs (`/r/{token}/assets/foo`) or, for the most robust mobile/shareable menu reports, inline downloaded dish/menu images as `data:image/...;base64,...`. Re-upload and verify the published HTML has the expected image count, zero bad relative asset refs, and the correct restaurant/menu identity.
- Dianping/Meituan mobile pages may expose only masked recommended dish names such as `低**肉` plus photos while hiding full names/prices behind the app. In that case, do not present a “full menu translation.” Create a clearly labeled “accessible Dianping data” report with photos, masked names, visible-character hints, and a caveat that exact menu names/prices require the app screen or a menu photo.
- If Wikimedia/remote image downloads hit rate limits, use the successfully downloaded images first, update the report with those, and say more can be added later rather than blocking the whole deliverable.
- If the user asks "파일은 어딨어" or similar, reply with the same `MEDIA:` attachment line plus the local absolute path.

### 8. Final Answer Structure

Respond in Korean. For Discord, avoid Markdown tables; use bullets or code-block tables.

Recommended structure:

1. *요약 / 식당 정보*
   - Restaurant name, branch/location, menu type
   - Currency and service charge
   - Important caveats: blurry/occluded text

2. *핵심 분석*
   - What type of cuisine/price level
   - Set vs à la carte
   - Spicy icons / premium ingredients / ethics notes if shark fin appears

3. *OCR 결과 + pinyin*
   - Group by menu section
   - For each item:
     - `*漢字*`
     - `pinyin`
     - Korean explanation
     - price

Example:

```markdown
- *鮮肉小籠包（4個）*
  xiān ròu xiǎo lóng bāo
  샤오롱바오 4개 — HK$48
```

4. *추천 조합*
   - Beginner-safe / signature / spicy / high-end options
   - Explain why

5. *불확실한 부분*
   - List any characters/prices that remain uncertain after crops.

6. *HTML 파일* (if requested)
   - Attach the generated report:
     - `MEDIA:/absolute/path/to/menu_analysis.html`

## Interpretation Guide

Common Shanghai/Jiangnan menu words:

- `本幫` (běn bāng): Shanghainese/local style
- `糖醋` (táng cù): sweet-and-sour
- `紅燒` (hóng shāo): soy-braised
- `蟹粉` (xiè fěn): hairy crab roe/meat mixture
- `獅子頭` (shī zi tóu): large pork meatball
- `小籠包` (xiǎo lóng bāo): soup dumplings
- `生煎包` (shēng jiān bāo): pan-fried soup buns
- `年糕` (nián gāo): rice cakes
- `燻魚` (xūn yú): Shanghai-style smoked/braised fish
- `醃篤鮮` (yān dǔ xiān): Jiangnan soup with cured pork, fresh pork, bamboo shoots
- `河蝦仁` (hé xiā rén): river shrimp meat
- `海蜇頭` (hǎi zhē tóu): jellyfish head
- `花雕` (huā diāo): Shaoxing yellow wine
- `醉雞` (zuì jī): drunken chicken
- `魚翅` / `翅` (yú chì / chì): shark fin; mention ethical issue if relevant

Common units:

- `例` (lì): portion
- `位` (wèi): per person
- `隻` (zhī): whole animal/count, often duck/chicken/pigeon
- `半隻` (bàn zhī): half
- `條` (tiáo): whole fish / long item
- `碗` (wǎn): bowl
- `件` (jiàn): pieces
- `個` (gè): pieces
- `起` (qǐ): minimum order starts from

## Common Pitfalls

1. **Assuming mainland China from Chinese text.** Check branch/address. Hong Kong menus often use Traditional Chinese, HKD, service charge, and English.

2. **Over-trusting one OCR pass.** Always crop dense sections and cross-check with the full image.

3. **Hiding uncertainty.** Use `?` and explicitly state what remains unclear.

4. **Forgetting pinyin basis.** If Traditional Chinese or Hong Kong menu, pinyin is Mandarin pronunciation, not Cantonese.

5. **Using Markdown tables in Discord.** The user prefers Discord tables as code-block tables or bullets. Use bullets for long menus.

6. **Doing arithmetic mentally.** Use `execute_code`/`terminal` for service charge, per-person cost, or price comparisons.

8. **Forgetting the HTML deliverable when requested.** If the user asks for a nice viewable document, create the HTML file and attach it via `MEDIA:`; don't only describe that it could be made.

## Verification Checklist

- [ ] Full image inspected with `vision_analyze`
- [ ] Dense sections cropped and upscaled
- [ ] Each section OCR/VLM checked separately
- [ ] Ambiguous characters/prices marked with `?`
- [ ] Pinyin generated for all Chinese dish names
- [ ] Currency, service charge, branch/location captured
- [ ] Arithmetic done with a tool if totals/per-person prices are mentioned
- [ ] Final answer is in Korean and uses bullets, not Markdown tables
- [ ] If requested, standalone HTML report created under `~/.hermes/media_cache/...`, original image copied beside it, and `MEDIA:` attachment included
