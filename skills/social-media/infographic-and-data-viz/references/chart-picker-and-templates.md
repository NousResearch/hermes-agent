# Chart-picker, checklists & worked examples

## Chart-picker (message → form)
| You want to show… | Use | Watch out for |
|---|---|---|
| A trend over time | Line | ≤5-6 series; consistent intervals; true order |
| Comparison across categories | Bar (sorted) | start at zero; horizontal if labels long |
| Parts of a whole | Stacked bar / donut (sparingly) | avoid many-slice pies |
| Distribution | Histogram | bars touch (continuous) |
| Relationship between two vars | Scatter | label outliers directly |
| One or two numbers | Big stat / table | a chart adds no value here |

## The honest-scale checklist (run before rendering)
Bars start at zero ✓ · no 3D / area / perspective distortion ✓ · no dual-axis or cherry-picked range ✓ · full
relevant range shown ✓ · context/denominator included ✓ · title matches what the data shows ✓ · data is real +
sourced ✓.

## Infographic anatomy (multi-fact)
Takeaway headline → 1 hero stat/chart → 2-3 supporting points (each its own clean mini-viz) → source + date +
logo → a citable footer. Keep one narrative; don't cram. Mobile-legible type; generous spacing.

## What renders it (the agent specs, a tool draws)
The agent produces the spec (chart type, takeaway title, data, honest scale, palette, direct labels, alt text).
A chart/design tool (Canva, Flux/Nano-Banana for layout, a charting library) renders the final image; the human
approves. WoopSocial doesn't generate media — it publishes the finished file.

## Worked example A — "Blunt indie founder" (honest bar)
Spec: horizontal bar, two segments, zero baseline. Title (takeaway): "Customers who finish onboarding churn 3×
less." Direct labels (9% vs 27%), one accent color on the low-churn bar, source: "our data, n=4,120, Jan–Jun
2026." Alt text: "Bar chart of churn by onboarding completion where finishers churn about three times less."
No 3D, no truncated axis.

## Worked example B — "Warm bookkeeping studio" (stat card vs chart)
"We almost made a chart of two numbers — then didn't. It's just clearer to say it: clients on our monthly system
filed on time 96% of the time last year, vs 61% the year before. Same people, one small habit. (Real numbers from
our 2025 client base — happy to show the breakdown.)" — chose a big stat over a needless chart, sourced, honest.

## Never
Truncate a bar axis · use 3D/area distortion · cherry-pick the flattering range · chart two numbers for show ·
rely on color alone · fabricate a data point or source · visualize an unsourced YMYL claim · bury the takeaway in
a chart-name title · claim WoopSocial generates the image (a design/chart tool does).
