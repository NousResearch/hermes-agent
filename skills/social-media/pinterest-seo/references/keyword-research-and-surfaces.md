# Keyword research, surfaces + two worked examples

## Keyword-research method (the agent advises; the user runs the native tools)
```
1. Seed: type a broad term in the Pinterest search bar (e.g. "home office").
2. Autocomplete: every suggestion is a real query -- collect them.
3. Guided Search bubbles: click the coloured pills (modern / budget / for beginners / 2026) -> long-tail variants.
4. Pinterest Trends (trends.pinterest.com): seasonal volume + Predicts for emerging terms.
-> Pick long-tail (3+ words) with clear intent. NEVER fabricate a volume -- cite the method to check it.
```

## Keyword-layering map (publishable vs native)
| Surface | Keyword | Who sets it |
|---|---|---|
| Profile name + bio | broad niche | native (advised) |
| Board title + description | topic, precise not cutesy | native (advised) |
| Pin title | long-tail, primary first (<=100) | **WoopSocial (publishable)** |
| Pin description | natural, keyword-rich, no stuffing | **WoopSocial (publishable)** |
| Board the pin saves to | semantically aligned | **WoopSocial (pinterestBoardId)** |
| Destination link | keyword-aligned page | **WoopSocial (link <=2048)** |
| On-pin overlay text | the keyword (OCR-read) | image tool (ideogram) |
| Alt text + file name | descriptive keyword | native / at-upload |

## Title + description templates
```
TITLE:  [Primary Keyword] | [Secondary Benefit] | [Context]   e.g. "Budget Home Office Ideas | Small Space | 2026 Setup"
DESC:   1-2 key phrases early, natural sentences, 100-200 chars. No stuffing, no hashtag clusters (1-2 niche tags max, or none).
```

## Pin -> Board -> Page checklist
```
[ ] Pin title/description, board name, and landing page all reinforce the SAME keyword phrase
[ ] Board is specific ("Keto Meal Prep for Beginners", not "Healthy Food")
[ ] Fresh pin (new image/URL, not a dupe) saved to the relevant board first
[ ] Site claimed + Rich Pins enabled (native); page delivers the pin's promise (long clicks)
```

## WoopSocial publish block
```
platform: PINTEREST | title: <=100 (keyword-led) | content.text: keyword-rich description | pinterestBoardId: <semantic board>
link: <=2048 (keyword-aligned page). Validate via platform-specs-and-validation (single media + board).
```

## Worked example 1 - blog traffic (blunt indie-founder voice)
```
SEED "home office" -> autocomplete + bubbles -> "budget home office ideas small space". TITLE "Budget Home Office Ideas |
Small Space | 2026". DESC natural, the phrase once early. BOARD "Small Space Home Office" (not "Workspace"). On-pin keyword -> ideogram.
WoopSocial publishes title+desc+board+link. I set the bio/board descriptions + alt text natively. Fresh variations, no stuffing.
```

## Worked example 2 - seasonal product (warm studio voice)
```
Trends shows "fall candle gifts" rising -- so I publish 45-90 days ahead. TITLE "Fall Candle Gift Ideas | Hand-Poured Soy | Cozy".
DESC warm + keyword-rich. BOARD "Fall Gift Ideas". Pin -> Board -> Page all say the same phrase; the page actually sells those candles.
WoopSocial publishes title+desc+board+link; I claim the site + enable Rich Pins natively. Saves + long clicks are what I watch.
```

Both: research real queries (method, no fabricated volume); keyword-led title + natural description (publishable);
semantic board + Pin->Board->Page match; fresh not dupes; WoopSocial publishes title+description+board+link, the rest native/advised.
