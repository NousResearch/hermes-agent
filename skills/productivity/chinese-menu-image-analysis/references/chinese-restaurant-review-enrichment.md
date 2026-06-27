# Chinese Restaurant / Review Enrichment

Use this reference when a Chinese menu photo includes a restaurant name, branch, mall, address, phone number, or QR/source clue. The goal is to enrich the menu report with local restaurant context without overclaiming or scraping copyrighted review content.

## Source priority by geography

- Hong Kong: OpenRice 香港 first; then official site, mall tenant page, Google Maps snippets.
- Mainland China: 大眾點評 / 美團 first when accessible; then official WeChat/site, Baidu Maps, Xiaohongshu snippets.
- Taiwan: Google Maps, official site/social, food blogs.
- Unknown: search exact Hanzi name + city/branch/mall + `食評`, `菜單`, `地址`, `電話`.

## Search queries

Start exact and local-language first:

```text
<restaurant Hanzi> <branch/mall> OpenRice
<restaurant Hanzi> <district> 食評
<restaurant Hanzi> <district> 菜單
<restaurant Hanzi> <mall/address> 地址 電話
<English name> <branch> review
```

If search pages are blocked/noisy, fetch the likely platform URL directly or use lightweight HTML requests with a browser-like User-Agent.

## OpenRice extraction pattern

OpenRice pages often contain enough rendered text plus JSON-LD/`window.__INITIAL_STATE__` for a concise restaurant summary.

Useful fields:

- restaurant name / English name
- `statusText` such as `已結業` (closed)
- address and English address
- cuisine tags and price range
- score / aggregate rating, review count
- smile/ok/cry counts if present
- photo count and menu photo count
- seat count, opening hours, payment notes
- recommended dishes / 食家推介 / 招牌菜
- recent review headlines, dates, star ratings from JSON-LD

Safe summary style:

```markdown
- OpenRice 기준 상태: 已結業 / 영업 중
- 평점/리뷰: 3.286/5, 리뷰 166개
- 가격대: HK$201–400
- 사진/메뉴: 사진 1,087장, 메뉴 사진 26장
- 대표 추천: 椒鹽小黃魚, 招牌小籠包, 外婆紅燒肉
- 최근 리뷰 헤드라인: 「健康美味的」(4/5), 「年度團年飯」(3/5)
```

Do not paste full user review text unless the user explicitly asks and the source terms allow it. Prefer headlines/metadata + a source link.

## HTML report section

Add a compact section near the top, before recommendation blocks:

```html
<section class="caution">
  <h2>가게/리뷰 정보 추가 조사</h2>
  <p><b>중국어권 웹 소스:</b> OpenRice 香港 페이지에서 확인. <a href="SOURCE_URL">원문 보기</a></p>
  <div class="summary">
    <div><b>현재 상태</b><br>已結業 — OpenRice 기준 폐업 표시</div>
    <div><b>주소</b><br>...</div>
    <div><b>평점</b><br>...</div>
    <div><b>사진/메뉴</b><br>...</div>
  </div>
  <p><b>소개 요약:</b> ...</p>
  <p class="muted">주의: 폐업 상태라면 이 메뉴판은 과거/보관용 메뉴일 가능성이 큽니다.</p>
</section>
```

## Image sourcing with Chinese web

Chinese/local review sites are useful for discovering what dishes look like, but do not silently copy user-uploaded review photos into the report. Safer options:

1. Link to the restaurant/review page and summarize dish recommendations.
2. Use official restaurant/menu images if clearly provided by the business.
3. Use licensed public images with source/credit.
4. Use placeholders or generated illustrations only when explicitly accepted.

## Verification checklist

- [ ] Restaurant source URL included.
- [ ] Closed/active status checked and surfaced.
- [ ] Rating/review/photo counts are labeled with source and timestamp if possible.
- [ ] Recent review info is summarized as headlines/metadata, not full copied reviews.
- [ ] The HTML report still has no broken image references.
