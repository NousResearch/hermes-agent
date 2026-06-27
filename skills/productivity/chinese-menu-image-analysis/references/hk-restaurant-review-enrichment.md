# Hong Kong Restaurant Review Enrichment

Use this reference when a Chinese/HK menu report should include restaurant identity, reviews, photos, or current status.

## Trigger

- User asks to find images from Chinese-language web sources.
- User asks to find the restaurant and add reviews.
- The uploaded menu includes a likely Hong Kong restaurant name, branch, mall, phone, or OpenRice-style menu screenshot.

## Search strategy

Prefer Traditional Chinese queries first for Hong Kong menus:

```text
"<restaurant name>" OpenRice
"<restaurant name>" "<district/mall>" 食評
"<restaurant name>" 菜單
"<restaurant name>" 香港
```

Useful examples:

```text
"上海小南國" 九龍灣 OpenRice
"上海小南國" MegaBox 食評
"南島雞飯" 香港
```

If generic web search is noisy, try OpenRice directly:

```text
https://www.openrice.com/zh/hongkong/restaurants?what=<urlencoded restaurant name>
https://www.openrice.com/en/hongkong/restaurants?what=<urlencoded English name>&where=<district>
```

## OpenRice extraction pattern

OpenRice pages often contain useful rendered text plus structured data. Fetch with a browser-like User-Agent and `Accept-Language: zh-HK`.

Useful facts to extract:

- restaurant name and English name
- status, especially `已結業` / closed
- address and mall/landmark
- cuisine and price range
- rating / review count
- smile/OK/cry counts
- photo count and menu-photo count
- bookmarked count
- opening hours, seats, payment methods, service charge
- OpenRice introduction text
- recommended dishes (`食家推介`, `招牌菜`)
- recent review headlines, ratings, dates, and URLs

### Direct page URL from search result

Search pages may show a closed restaurant block like:

```html
<a href="/zh/hongkong/r-...-r20361/" class="pois-closed-restaurant-cell">
  <span>上海小南國 (MegaBox)</span><span> (已結業)</span>
</a>
```

Resolve the relative URL against `https://www.openrice.com` and fetch the restaurant page.

### JSON-LD

Restaurant pages may embed Schema.org JSON-LD with aggregate rating and a few review headlines:

```python
import json, re, urllib.request
url = 'https://www.openrice.com/zh/hongkong/r-...'
req = urllib.request.Request(url, headers={
    'User-Agent': 'Mozilla/5.0',
    'Accept-Language': 'zh-HK,zh;q=0.9,en;q=0.8',
})
s = urllib.request.urlopen(req, timeout=20).read().decode('utf-8', 'ignore')
idx = s.find('"aggregateRating"')
start = s.rfind('{"@context"', 0, idx)
end = s.find('</script>', idx)
data = json.loads(s[start:end])
print(data['name'], data.get('priceRange'), data.get('aggregateRating'))
for review in data.get('review', [])[:5]:
    print(review.get('datePublished'), review.get('reviewRating', {}).get('ratingValue'), review.get('headline'), review.get('url'))
```

### `window.__INITIAL_STATE__`

For richer counts, parse the large app state with bracket matching rather than a fragile regex:

```python
idx = s.find('window.__INITIAL_STATE__=')
start = idx + len('window.__INITIAL_STATE__=')
depth = 0; in_str = False; esc = False; end = None
for i, ch in enumerate(s[start:], start):
    if in_str:
        if esc: esc = False
        elif ch == '\\': esc = True
        elif ch == '"': in_str = False
    else:
        if ch == '"': in_str = True
        elif ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
state = json.loads(s[start:end])
```

Then inspect likely paths:

```text
services.PoiCommonService.state.poiHeader
services.PoiDetailPage.services.poiDetail.state.data
```

Common fields seen in OpenRice:

```text
statusText, address, displayAddress, phone, priceMin, priceMax,
scoreOverall, scoreSmile, scoreOk, scoreCry,
photoCount, menuPhotoCount, reviewCount, bookmarkedUserCount,
poiDetail.seatCount
```

## Image policy for review/photo enrichment

Do not silently embed arbitrary OpenRice or review photos into the report. Use them as source links/context unless the user provided the image or the source license/permission is clear.

Recommended report behavior:

- Add a `가게/리뷰 정보` section with source URL and summarized facts.
- If the restaurant is closed (`已結業`), surface that prominently near the top.
- For dish images, prefer licensed public representative images, official images, user-provided crops, generated images with explicit consent, or placeholders.
- Keep OpenRice/review URLs as citations rather than copying photos.

## When no restaurant match is found

State that no reliable match was found and keep the report grounded in the uploaded menu OCR. Do not force a weak match. Add a short note such as:

```text
정확한 가게명으로 OpenRice/중국어권 검색을 시도했지만 확실히 매칭되는 식당 상세 페이지는 찾지 못했습니다. 현재 보고서는 업로드한 메뉴판 OCR 기준입니다.
```

## Report fields to add

When a match is reliable, add a concise section:

- source: OpenRice / official site / other
- current status
- address
- rating and review count
- photo/menu-photo count
- price range
- seats / opening hours if available
- recommended dishes from source
- recent review headlines with dates/ratings
- caveat that details may be stale and should be verified before visiting
