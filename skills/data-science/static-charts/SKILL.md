---
name: static-charts
description: Render supplied data as static chart attachments.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [charts, graphs, matplotlib, seaborn, discord, data-visualization]
    category: data-science
    related_skills: [jupyter-live-kernel]
---

# Static Charts Skill

Render user-supplied numeric data as a legible PNG chart using Matplotlib and Seaborn. The output is static by design, so it works as a native attachment in Discord and other Hermes messaging platforms; it does not provide browser-style hover or zoom interactions.

## When to Use

Use this skill when the user asks to chart, graph, compare, trend, or visualize supplied values. It is appropriate for time series, category comparisons, distributions, proportions, and numeric relationships.

Do not invent missing values or choose a chart when units, periods, or categories are ambiguous. Ask a concise question instead. If the user has not asked for a graph, suggest one when a chart would materially clarify at least three related values; do not generate it unsolicited.

## Prerequisites

Use the `terminal` tool in the Python environment that will render the image. Install the bounded dependencies in this skill before the first render:

```bash
python -m pip install -r ${HERMES_SKILL_DIR}/requirements.txt
```

The renderer requires `HERMES_HOME` so it can save output below the gateway's safe attachment directory: `$HERMES_HOME/cache/images/`. `japanize-matplotlib` bundles IPAexゴシック, so Japanese labels render consistently without relying on the host font set. Its bundled font is covered by the IPA Font License v1.0; install and use it only when that license is acceptable for the deployment.

## How to Run

1. Write a UTF-8 JSON chart specification to a temporary file.
2. Use `terminal` to run the bundled renderer:

```bash
python ${HERMES_SKILL_DIR}/scripts/render_chart.py /tmp/chart.json \
  --output "$HERMES_HOME/cache/images/monthly-sales.png"
```

3. Read the JSON result. On success it contains a `media` value with the exact `MEDIA:/absolute/path.png` directive.
4. Give the user a short interpretation, then include that exact `MEDIA:` directive on its own line in the final response. Hermes strips the directive and attaches the PNG natively in Discord.

## Quick Reference

Use one of five chart types: `line`, `bar`, `scatter`, `histogram`, or `pie`.

For line, bar, and scatter charts, every series needs `name`, `x`, and `y`:

```json
{
  "type": "line",
  "title": "月次売上",
  "x_label": "月",
  "y_label": "売上（万円）",
  "source": "ユーザー提供データ",
  "series": [
    {"name": "売上", "x": ["1月", "2月", "3月"], "y": [120, 148, 136]}
  ]
}
```

For a histogram, use `values` instead of `x` and `y`; `bins` is optional (2–100):

```json
{
  "type": "histogram",
  "title": "応答時間の分布",
  "x_label": "秒",
  "y_label": "件数",
  "bins": 12,
  "series": [{"name": "応答時間", "values": [1.2, 1.5, 1.9, 2.4]}]
}
```

For a pie chart, provide exactly one series with `labels` and non-negative `values`:

```json
{
  "type": "pie",
  "title": "支出の内訳",
  "series": [
    {"name": "支出", "labels": ["家賃", "食費", "交通"], "values": [90000, 35000, 12000]}
  ]
}
```

## Procedure

1. Restate the data source, time range, and unit when they matter to interpretation.
2. Select a line chart for changes over ordered time, a bar chart for category comparison, a scatter chart for relationships, a histogram for distributions, or a pie chart only for a small whole-part breakdown.
3. Keep labels, units, and titles explicit. Use at most 12 slices in a pie chart and avoid pie charts when a bar chart would make close values easier to compare.
4. Render the PNG under `$HERMES_HOME/cache/images/` using the script. Do not place generated output in a source repository or a credential/config directory.
5. Confirm the renderer returned `success: true`; if it did not, fix the specification or missing dependency before responding.
6. State the key takeaway and any assumption in prose. Place the `MEDIA:` directive on its own final line so the gateway attaches the chart.

## Pitfalls

- Do not graph values copied from an uncertain source without marking the source or uncertainty in the response.
- Do not compare values with incompatible units, periods, or denominators.
- The renderer accepts at most eight series and 500 points per series to keep a Discord image legible.
- Static PNGs cannot preserve Plotly-style hover, filtering, or zoom. Publish an interactive page separately only when the user explicitly needs exploration.
- If Japanese text renders as boxes, confirm that `japanize-matplotlib` was installed into the same Python environment as the renderer and that `japanize()` runs after the Seaborn theme; do not replace the user’s Japanese labels with romanized text.

## Verification

Verify that the renderer exits with code 0, returns `success: true`, and reports an absolute `.png` path below `$HERMES_HOME/cache/images/`. In Discord, verify the response contains the attached image rather than a visible `MEDIA:` path, and verify that the title, units, and series labels are readable.
