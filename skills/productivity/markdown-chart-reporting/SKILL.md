---
name: markdown-chart-reporting
description: "Display charts in markdown reports sent to Telegram and SiYuan — ASCII sparklines, separate PNG photos, and what NOT to do (base64 data URIs)."
version: 1.0.0
author: Fuad Al Fajri
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Markdown, Charts, Telegram, SiYuan, Sparkline, Reports]
    category: productivity
    related_skills: [siyuan-markdown-import]
---

# Charts in Markdown Reports (Telegram & SiYuan)

Techniques for showing charts in `.md` reports sent to Telegram (`sendDocument`) and/or imported into SiYuan — applies to ALL reports (stocks, commodities, monitoring, research), not just one domain.

## When to Use

- User asks to include a chart/visual in a markdown report
- User reports charts not rendering in Telegram or SiYuan
- User needs to generate sparklines or PNG charts for reports

## Format Decision (proven)

| Method | Shows in Telegram | Shows in SiYuan | Size | Recommendation |
|--------|:-----------------:|:---------------:|:----:|:--------------:|
| **ASCII sparkline** (`▁▂▃▄▅▆▇█`) | ✅ text | ✅ text | ~200 B | ✅ **PRIMARY** |
| PNG sent as **separate photo** (`sendPhoto`) | ✅ image | — | ~20 KB | ✅ for visual charts |
| **base64 data URI** (`![x](data:image/png;base64,...)`) | ❌ not rendered | ❌ not rendered | ~29 KB | ❌ NEVER |

## Critical Pitfall: base64 data URIs are NOT rendered

- **SiYuan** (`createDocWithMd`): `![x](data:image/png;base64,...)` is stored as a `NodeImage` with a `NodeLinkDest` containing the string `data:...` — **not an image block**. The `.sy` file is still created (code 0) but the image is blank.
- **Telegram document viewer**: doesn't render ANY image inside a `.md` file — base64 shows as raw text.
- Result: base64 bloats the file to 29 KB and shows nowhere. **Remove it from markdown.**

## Correct Pattern: 2 Messages per Cycle (n8n example)

```
sendDocument (.md + ASCII sparkline)  →  msg 1 (text, renders)
GET /chart-png (bridge serves PNG binary) → prepareBinary → sendPhoto  →  msg 2 (image, renders)
```
- Bridge endpoint `/chart-png` sends **binary** (`send_header Content-Type: image/png` + file bytes), not base64.
- n8n: HTTP Request node with `options.response.response: "file"` to capture binary → Code node grabs `$input.first().binary[firstKey]` → HTTP `sendPhoto` multipart-form-data, `photo` field `parameterType: formBinaryData`, `inputDataFieldName: "data"`.

## ASCII Sparkline (implementation)

```python
chars = "▁▂▃▄▅▆▇█"  # U+2581..U+2588
def sparkline(values, width=30):
    vmin, vmax = min(values), max(values)
    rng = (vmax - vmin) or 1
    n = len(values)
    step = max(1, n // width)
    sampled = values[::step][:width]
    return "".join(chars[int((v - vmin) / rng * 7)] for v in sampled)
```
Show in a markdown table: `| 🥇 Gold | \`▁▇▇▇▇█\` | Rp 2,113,294 → Rp 2,454,345 (+16%) |`
Legend: `▁` = lowest, `█` = highest within the observation range.

## PNG Charts: Dual-Axis REQUIRED for Very Different Scales

- Gold (~Rp 2.4M/gram) vs Silver (~Rp 35K/gram): ~68x difference. One axis → the silver line looks flat at the bottom, misleading.
- **Dual-axis**: gold on the left axis, silver on the right, each with its own scale.
- Grid ticks must align: generate **exactly n+1 ticks** per axis (`nice_ticks(vmin, vmax, n=4)` with padding) so grid lines line up.
- Generate PNG with **Pillow** (`ImageDraw.line`), WITHOUT matplotlib — PEP 668 (Debian/Ubuntu) blocks system `pip install`; use Pillow already present or a venv.

## 7-Day Charts (don't rely on local logs alone)

- Local logs only have data since first recording — don't claim "7 days" from 1 day of intraday data.
- Fetch daily close data from the provider's chart API: `GET http://<proxy-host>:<port>/history?symbol=GOLD&days=7`
- Parse: `data.chart.result[0].timestamp` + `indicators.quote[0].close` → format `YYYY-MM-DD`.
- Convert units as needed (e.g. USD/oz → IDR/gram: `close / 31.1035 * usd_idr`).
- Merge with the latest intraday position from local logs, label the last point "today".

## Common Pitfalls

1. Using base64 data URIs in markdown → renders nowhere, bloats file
2. Single-axis chart for series with very different scales → misleading flat lines
3. Claiming "7-day" trends from intraday-only local data
4. Sending PNG inside the `.md` document instead of a separate `sendPhoto`
5. matplotlib on PEP 668 systems → use Pillow or a venv

## Verification Checklist

- [ ] Sparkline renders as text in Telegram and SiYuan
- [ ] PNG sent as a separate photo message (not embedded in the doc)
- [ ] No `data:image` URIs in the final markdown
- [ ] Dual-axis used when series scales differ greatly
- [ ] 7-day chart based on fetched daily data, not only local intraday logs
