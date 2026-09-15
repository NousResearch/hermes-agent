# NOVA Commercial & AWS Deployment Playbook

The source for **`NOVA_Commercial_and_AWS_Deployment_Playbook.pdf`** — an internal master
document covering what NOVA can honestly do, which UK market to sell it into, what to
charge, how to get the first customer, and how to deploy it to AWS step by step.

It is a **commercial artefact, not part of the product.** Nothing here imports NOVA, and
nothing in `nova/` was changed to produce it. Where this document and the repository
disagree, **the repository is right.**

## Regenerating

```bash
pip install reportlab matplotlib pymupdf     # once
cd docs/commercial
python3 charts.py        # rebuild the charts and the architecture diagram
python3 build.py         # build the PDF
python3 check_pdf.py     # render every page to _pages/ and report layout faults
```

`charts.py` resolves its output path relative to the repository root, so run it from
either location — `python3 docs/commercial/charts.py` works too.

## Files

| File | What it is |
|---|---|
| `build.py` | Document scaffold: page templates, cover, running header and footer, the two-pass table of contents. Run this to produce the PDF. |
| `theme.py` | The visual system — palette, typography, and every custom flowable (`Panel`, `SectionOpener`, `Checklist`, `Flow`, `Timeline`, `table`, `callout`, `status_chip`). |
| `layout.py` | Helpers shared by the scaffold and the content (`section`, `h2`, `img`, `caption`). Separate because `build.py` imports the content and the content needs these — keeping them together is a circular import. |
| `content.py` | Assembles the story from the two content modules. |
| `content_a.py` | Parts 1–12: audit method, what NOVA is, capability catalogue, market, offer, ROI, pricing, acquisition, prospecting, marketing, messaging, website. |
| `content_b.py` | Parts 13–25: demo, pilot, sales partner, best practice, the 36-step AWS guide, cost control, production checklist, roadmap, revenue scenarios, positioning, risks, next actions, sources. |
| `charts.py` | Every chart and the architecture diagram, rendered to `assets/*.png` at 220 dpi. |
| `check_pdf.py` | Renders each page to `_pages/pNNN.png` and reports margin overflow, images outside the column, near-empty pages and large trailing whitespace. |
| `assets/` | Generated PNGs. Committed so the PDF can be rebuilt without matplotlib. |

## Rules the content is written under

1. **Every capability claim traces to the repository.** §3 grades each one on the
   repository's own ladder (`declared → wired → enforced → tested → live-proven`) and
   names the file behind it. Claims that would need a real AWS account or a real model
   provider are marked unproven rather than softened.
2. **Every external number carries a source**, listed in §25. Numbers that are assumptions
   for the sake of an example say "illustrative assumption" next to them, not in a
   footnote.
3. **No AWS prices are quoted.** The official pricing pages were unreachable from the
   build environment (the egress proxy blocks `aws.amazon.com`), so §18 gives cost
   structure and the official URLs instead of figures.

## Keeping it true

Regenerate after any change that moves a capability up or down the ladder in §3 — and
update the sales material and the sales partner's capability schedule (§15) at the same
time. The capability schedule is the annex that stops a commission-only partner promising
something NOVA cannot do.

Bump `VERSION` in `build.py` and set `COMMIT` to the commit the audit was performed
against; both appear on the cover and in the page footer.

## Charts

`charts.py` follows the repository's data-viz method: magnitude gets one hue, identity
gets the fixed categorical order, and the two chart hues (`#3987E5`, `#D95926`) were run
through the validator against this document's own panel colour `#141C2E` rather than the
default dark surface — all five checks pass, worst adjacent CVD ΔE 26.8. Both economic
charts carry `ILLUSTRATIVE` in the plot title itself so the label survives being
screenshotted out of the document.
