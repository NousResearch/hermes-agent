# Experimental Expected-Dish Image Cards

Use this reference when the user asks to show likely/representative dish photos in an HTML menu report.

## Core rule

These images are **not evidence from the restaurant** unless they came from the uploaded menu or an official restaurant source. Label them as representative/expected appearance only.

Recommended Korean disclaimer:

```text
사진은 실제 매장 제공 이미지가 아니라 메뉴명 기반의 예상/대표 이미지입니다. 실제 제공 모양과 다를 수 있습니다.
```

## Source priority

1. Official restaurant/menu images if the user supplied them or they are already present in the uploaded photo.
2. User-approved web search results with usable licenses/citations.
3. Locally generated illustrative images when the user explicitly accepts generated examples.
4. Fallback placeholder cards when no reliable image is available.

Do **not** silently scrape arbitrary copyrighted restaurant photos into a report. If using public web images, keep source URL/attribution/license notes in the HTML.

## When to include cards

Include image cards only for a limited set unless the user requests every menu item:

- signature/recommended dishes
- unfamiliar dishes where a visual helps ordering
- spicy/high-end/caution dishes when useful
- up to ~8–12 items by default to keep HTML size reasonable

If the user explicitly asks for “메뉴별 이미지”, “메뉴마다 사진”, or “같이 보이는” output, treat image cards as part of the deliverable rather than a bonus: attempt every confidently identified dish, but use placeholders for uncertain/unavailable images instead of blocking the report.

## Image search/query strategy

For each OCR-confirmed dish, build queries from most-specific to generic:

1. Exact Chinese name from the menu.
2. Simplified/Traditional variant if obvious.
3. Common English/Korean name or pinyin.
4. Regional/category fallback, e.g. “Shanghai braised pork belly” or “Sichuan mapo tofu”.

Keep image metadata with the menu item, not in a separate untracked list. Recommended source priority:

1. `menu-crop`: cropped from the user-provided menu/photo when visible.
2. `official`: official restaurant/menu/social image with URL.
3. `representative`: licensed or clearly attributable public image, with source/credit.
4. `generated`: generated illustration, only when the user explicitly accepts generated examples.
5. `placeholder`: no reliable image found.

## Card fields

For each image card, store:

```python
{
    'hanzi': '鮮肉小籠包',
    'pinyin': 'xiān ròu xiǎo lóng bāo',
    'ko': '샤오롱바오',
    'image': 'dish_images/xiao_long_bao.jpg',
    'image_kind': 'representative',  # official | representative | generated | placeholder
    'source': 'Generated illustration' or 'https://...',
    'note': '실제 매장 이미지 아님',
}
```

## HTML pattern

Add a section near the recommendations:

```html
<section>
  <h2>예상 음식 사진</h2>
  <p class="note">사진은 실제 매장 제공 이미지가 아니라 메뉴명 기반의 예상/대표 이미지입니다.</p>
  <div class="dish-grid">
    <article class="dish-card">
      <img src="dish_images/xiao_long_bao.jpg" alt="鮮肉小籠包 representative image">
      <h3>鮮肉小籠包</h3>
      <p class="pinyin">xiān ròu xiǎo lóng bāo</p>
      <p>샤오롱바오</p>
      <span class="badge">대표 이미지</span>
    </article>
  </div>
</section>
```

CSS:

```css
.dish-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; }
.dish-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:12px; }
.dish-card img { width:100%; aspect-ratio:4/3; object-fit:cover; border-radius:12px; background:#f4e7dc; }
```

## Placeholder fallback

If no suitable image is available, use a text placeholder instead of inventing a source:

```html
<div class="dish-placeholder">이미지 없음<br><small>대표 사진 확보 필요</small></div>
```

## Verification checklist

- [ ] Each image is labeled as official/representative/generated/placeholder.
- [ ] No arbitrary copyrighted images are embedded without attribution/license context.
- [ ] Report states representative images may differ from the restaurant's actual plating.
- [ ] HTML still works on mobile.
- [ ] If images are external URLs, they load over HTTPS; otherwise copy them into the report folder or inline them.
