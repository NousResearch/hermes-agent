---
name: logo-design
description: Design geometric SVG logos from brief to delivery kit.
version: 1.0.0
author: kaankiziltug (adapted by Nous Research)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [logo, branding, identity, svg, wordmark, monogram, favicon, brand-guidelines, design]
    category: creative
    related_skills: [ip-as-logo, mono-color, draw-your-font]
    ported_from: kaankiziltug/logo-design-skill@0ecf52e9a4b3ac92b714f7cc6e3148ab8c774134
---

# Logo Design Skill

Act as a senior identity designer: take a brand from discovery brief to a tested, hand-constructed SVG logo and a delivery kit (colour, lockups, favicon/app-icon set, presentation board, usage guide). A logo is an **identifier, not an explanation** — one clear idea that works at 16 px and on a building, in one colour, for decades. The skill ships dependency-free Python scripts (audit, HTML test sheets, concept sheet, presentation board, PNG/ICO export) and a searchable library of 1,400+ real-world SVG logos, fetched on first use. It does NOT do mascot/character logos via image generation (use `ip-as-logo`) and it does NOT produce raster artwork or illustrations: every mark is geometric SVG you write by hand.

## When to Use

- The user wants a logo, logotype, wordmark, monogram, brand mark, symbol, app icon or favicon designed, redesigned, refreshed, critiqued or compared.
- They ask for logo ideas/concepts, a design brief, logo guidelines, lockups or an identity system.
- They mention branding a new company, product, app or project — even without the word "logo".
- Don't use for: cute mascot/character marks generated as images (`ip-as-logo`), general illustration, raster icon packs, or recolouring an existing finished logo (`mono-color`).

Reply in the user's language. Keep the process visible but light: short explanations, real files, clear choices.

## Prerequisites

- Python 3 on PATH (`python3`; on Windows `python3` is often the Store stub — use `python` or `py -3`). All scripts are stdlib-only.
- **Reference library (one-time, ~10 MB):** `python3 <skill-dir>/scripts/fetch_library.py` downloads the pinned upstream library into `<skill-dir>/assets/library/`. `<skill-dir>` is the directory this SKILL.md was loaded from. Without it `search_library.py`, `preview_sheet.py --refs-industry` and the complexity comparison in `svg_audit.py` cannot work; everything else runs.
- **Optional PNG renderers** for `render_png.py` / `concept_sheet.py` / `presentation_board.py --png-dir`: cairosvg (`pip install cairosvg`), `rsvg-convert` (librsvg), Inkscape, or headless Chrome/Chromium (also used to screenshot HTML sheets). `python3 <skill-dir>/scripts/render_png.py --which` lists what is installed. Without any renderer you still get SVGs, HTML sheets and audits — but you cannot look at your work (see Pitfalls).
- Hermes tools used: `write_file` for SVG files, `terminal` for scripts, `vision_analyze` to look at rendered PNGs, `read_file` for library SVG source, `browser_exec` to open HTML sheets when a browser view is wanted.

## How to Run

**Step 0 — fetch the library and read its JSON.** `terminal(command="python3 <skill-dir>/scripts/fetch_library.py")` prints one JSON line: `{"status": "fetched"|"present", "library_dir": ..., "svg_count": N, "sha": ...}`. Proceed only when `svg_count >= 1400`; on `{"status":"error",...}` tell the user the library is unavailable and continue without the research/peer-test steps.

**Step 1 — pick the mode.**

| The user wants… | Mode | Start with |
|---|---|---|
| A new logo | **Design** (full or fast track) | Procedure phase 1 |
| Feedback on a logo | **Critique** | `references/critique.md` |
| To modernise/replace a logo | **Redesign** | `references/redesign.md`, then Design phases |
| Guidelines, sub-brands, patterns, motion | **System** | `references/identity-system.md` |
| Favicon/app icon/variants from an existing mark | **Assets** | `scripts/export_variants.py` |

**Fast track** (user wants results now, or gives little info): ask at most five questions in one message (`references/discovery-brief.md` §2), or skip questions, state your assumptions and go straight to three concepts. Iterate after they react.

**Step 2 — the concept checkpoint rule.** Every Design and Redesign run pauses after the concepts are built and tested (phase 6): show the concept overview image, one line per concept and your recommendation, then *offer* the full logo kit and wait. Build the kit (phase 7) only after the user picks a direction and says yes. Skip the pause only when the user explicitly says not to check in ("don't ask, just deliver everything"). If the user cannot reply at all, stop at the checkpoint anyway and describe what the kit would contain.

**Step 3 — look at your work.** Drawing in SVG code is drawing blind. After writing or changing a logo: `python3 <skill-dir>/scripts/render_png.py concept-a.svg concept-b.svg --out-dir renders --size 512`, then `vision_analyze` each PNG and actually look. It picks the best available renderer (cairosvg, rsvg-convert, Inkscape, headless Chrome/Chromium, or macOS Quick Look). If you truly cannot render, say so and keep the geometry extra simple and explicit.

## Quick Reference

Invoke as `python3 <skill-dir>/scripts/<name>.py …` (Windows: `python` / `py -3`). Flags are upstream's, verbatim.

| Script | Use it to | Flags |
|---|---|---|
| `fetch_library.py` | Download the pinned reference library into `<skill-dir>/assets/library/` (idempotent; JSON on stdout) | `--force`, `--dest DIR` |
| `search_library.py` | Find reference logos; `--summary` shows a category's conventions; `--format paths` gives files to read; `--list-values` lists every filter value | `--type`, `--symbol-type`, `--technique`, `--geometry`, `--industry`, `--color`, `--primary-color`, `--max-colors`, `--min-colors`, `--aspect`, `--variant`, `--type-style`, `--case`, `--mood`, `--subject`, `--query`, `--exemplary`, `--no-gradient`, `--limit`, `--format {table,paths,json}`, `--summary`, `--list-values` |
| `svg_audit.py` | Check an SVG: live text, rasters, filters, colour count, gradients, strokes, near-miss angles, tiny details, centring, complexity vs. the library | `files`, `--json`, `--bg` |
| `preview_sheet.py` | HTML test sheet: size ladder, 16/32 px pixel test, backgrounds, one-colour, squint blur, mirror/rotate, favicon/app-icon/header/card contexts, side-by-side, shelf test vs. competitors | `files`, `-o/--out`, `--name`, `--names`, `--brand-color`, `--refs`, `--refs-industry`, `--refs-count`, `--compare-only` |
| `concept_sheet.py` | One-image concept overview (large mark, lockup, true 64/32/16 px sizes, name, one-line idea, recommendation) — what you show at the checkpoint | `files`, `--lockups`, `--names`, `--notes`, `--title`, `--subtitle`, `--recommend`, `--greyscale`, `--width`, `-o/--out` |
| `render_png.py` | Render SVG → transparent PNG at exact sizes; screenshot HTML sheets/boards (Chrome); build `favicon.ico`; `--which` lists renderers | `files`, `-o/--out`, `--out-dir`, `--size`, `--width`, `--height`, `--padding`, `--bg`, `--backend`, `--ico`, `--ico-sizes`, `--which` |
| `export_variants.py` | Black, white, brand-mono, square, favicon and app-icon SVGs; `--png` sizes; `--web-icons` = favicon.ico + PNG icon set + webmanifest + `<head>` snippet; `--favicon-source` for a simplified small-size drawing | `master`, `--out-dir`, `--name`, `--title`, `--mono`, `--icon-bg`, `--icon-fg`, `--icon-scale`, `--optical-offset`, `--favicon-source`, `--keep-white`, `--only`, `--png`, `--web-icons` |
| `presentation_board.py` | Client presentation from a JSON spec (`templates/presentation-spec.example.json`): brief, each concept with rationale + 6 **industry-specific** mockups (cup, packaging, payment card, README, terminal, signage…), comparison, recommendation | `spec`, `-o/--out`, `--list-mockups`, `--png-dir` |

Upstream's `build_catalog.py` (maintainers-only catalog rebuild) is not vendored; the catalog ships inside the fetched library.

## Procedure

1. **Discovery → brief.** Learn: name (exact spelling), what they do, audience, 3–5 brand adjectives, competitors, constraints (colours, equity, where it must work), decision-maker. Write a short brief (template `references/discovery-brief.md` §5) and list your assumptions. Adjectives are the most valuable input — they become visual cues. Done when: brief + assumptions written.
2. **Research & strategy.** (a) See what the category looks like so you can avoid blending in: `python3 <skill-dir>/scripts/search_library.py --industry <closest> --summary`, then `read_file` a few of the paths. The library skews to tech; for other sectors search by `--subject`/`--query` and lean on your own knowledge. (b) List the category's **clichés** explicitly (fintech: blue, upward arrows, shields, globes; coffee: beans, steam, cups) — off-limits unless given a genuinely fresh form. (c) **Word map** (`references/discovery-brief.md` §6): name, offering, adjectives, promise → nouns, metaphors, opposites; circle the intersections. (d) Choose candidate **mark types** with `references/mark-types.md` §12; explore at least two different types. Done when: cliché list, word map and ≥2 mark types are on paper.
3. **Concepts.** Write **8–12 one-sentence concepts** spread across mark types; each needs an ownable twist — a sentence that could describe a competitor's logo is not a concept. Score quickly (idea clarity, distinction, simplicity, relevance, small-size strength) and pick the **three strongest and most different**. Show the longlist only if it helps the user steer. Before building, `search_library.py --subject <thing>` to make sure you are not recreating an existing mark, and study 3–5 exemplary files using your technique (`--exemplary --format paths`) to see how the geometry is built. Build only the three. Done when: three concepts chosen, originality search run.
4. **Build in SVG (black first).** Describe the construction in words first (primitives, radii, angles, grid unit), then `write_file` the SVG per `references/svg-construction.md`. Canvas `viewBox="0 0 256 256"` for symbols; lockups keep height 256. Solid black on white, no colour yet. Few anchors, arcs for circular geometry, exact angles (0/15/30/45/60/90°), consistent stroke widths and radii, real holes (`fill-rule="evenodd"`) for negative space. No `<text>` in finished marks — construct letterforms as paths; for exploration `<text>` is allowed but flag it. Save every meaningful iteration (`concept-a-v1.svg`, `-v2.svg`…) instead of overwriting. Done when: three black SVGs exist and have been rendered and looked at.
5. **Test & refine (loop at least twice).**
   ```bash
   python3 <skill-dir>/scripts/svg_audit.py concept-a.svg concept-b.svg concept-c.svg
   python3 <skill-dir>/scripts/preview_sheet.py concept-a.svg concept-b.svg concept-c.svg --refs-industry <industry> -o preview.html
   ```
   Open (`render_png.py preview.html` + `vision_analyze`, or `browser_exec`) and look. Fix what fails, re-run. Key refinements (`references/visual-techniques.md`): **Scale** — the idea survives 16–24 px; gaps and strokes big enough, else simplify or add a small-size version. **Optical corrections** — overshoot round/pointed forms (~1–3 %), fix the bone effect on rounded shapes, thin horizontals slightly, centre optically (slightly above geometric centre), thin the reversed version. **Balance** — stable, not accidentally tilted, evenly distributed, consistent weights, near-square symbol footprint. **Readings** — mirror, rotate 180°, view tiny; check for unintended shapes or meanings. **Distinction** — shelf test against competitors; familiarity test (if it feels familiar and isn't yours, it's someone else's). **Craft pass** (where AI-drawn marks usually fall short): (1) *Letter test* — every modified letter still reads as the intended letter at first glance; if a K reads as an h, revise. (2) *Junctions* — inspect every place strokes meet: no accidental notches, slivers, lumps or ink traps. (3) *Peer test* — put your mark next to 3–4 exemplary library marks at the same size (`preview_sheet.py yours.svg --refs <files from search_library.py --exemplary --format paths>`); it should look equally resolved. (4) *Literalness* — if a concept is simply the product drawn (a cup for coffee), push it further or drop it. Full list: `references/testing-checklist.md`. Done when: audit is clean and two loops are logged.
6. **Show the concepts, then stop (checkpoint).**
   ```bash
   python3 <skill-dir>/scripts/concept_sheet.py a-symbol.svg b-symbol.svg c-symbol.svg --lockups a-lockup.svg b-lockup.svg c-lockup.svg --names "Name A" "Name B" "Name C" --notes "One-line idea A" "…" "…" --recommend 1 --greyscale -o concepts.png
   ```
   `vision_analyze` the image yourself, then show it to the user (attach or display the PNG; if you can't share files, give the path) with the chat format below. Greyscale first — colour triggers taste debates; a small colour hint for your recommendation is fine. End with the kit offer and **wait for the answer**:

   > Want me to prepare the full logo kit for the direction you choose? It includes: the colour palette with one-colour and reversed versions, horizontal and stacked lockups, a small-size cut, favicon + app-icon + web-icon set, a presentation board with mockups for your industry, and a one-page usage guide.

   Checkpoint message shape: `<concept overview image>` → `### A — <Name> · <mark type> ← recommended` with **Idea:** (one sentence) and **Why it fits:** (2–3 bullets tied to the brief's adjectives/audience/competition) → same for B and C → **My recommendation:** (one or two sentences, one honest risk per concept if relevant) → **Next:** pick a direction, kit offer, one-line list of kit contents. Keep it short: the image does the work; don't attach variants, boards or icon sets yet. If they want changes, iterate (phases 4–5) and show the sheet again. Done when: the user has answered.
7. **Build the kit (only after the user says yes).** (1) **Refine the chosen direction**: final geometry, optical corrections, small-size cut, thinned reversed version. (2) **Colour**: 1–2 colours ideally, ownable in the category, reproducible (HEX/RGB/CMYK/Pantone), accessible; one-colour and greyscale versions must still work (`references/color.md`). (3) **Typography & lockups**: type study, custom letters for ownership, optical spacing, max two families (`references/typography.md`); horizontal, stacked, symbol-only, wordmark-only — lock relative sizes and spacing. (4) **Presentation board**: copy `templates/presentation-spec.example.json`, set `"industry"` or an explicit `"mockups"` list (a café gets cups and bags, a dev tool gets a README and terminal); for multi-colour marks pass `symbol_on_tile` / `lockup_on_dark` artwork (and optionally `tile_color`) so mockups keep the colours instead of forcing the mark to white; `--png-dir slides` exports every slide. Guidance: `references/presentation-delivery.md`. (5) **Export the files**:
   ```bash
   python3 <skill-dir>/scripts/export_variants.py final-symbol.svg --title "Brand logo" --mono "#HEX" --icon-bg "#HEX" --web-icons --favicon-source final-symbol-small.svg
   python3 <skill-dir>/scripts/export_variants.py final-horizontal.svg --title "Brand logo" --only black white mono --mono "#HEX" --png 1200
   ```
   (6) **Guidelines & handover**: compact guidelines from `templates/brand-guidelines-template.md` (clear space from a logo element, minimum sizes, colour codes, approved backgrounds, misuse), masters (SVG; PDF/AI/EPS if the user has the tools), handover notes (rationale, test results, open items). Run the final checklist in `references/process.md` §7. For bigger brands extend into a system (`references/identity-system.md`). Done when: every kit item above exists on disk and was rendered and looked at.

## Principles

1. **Who, what, why first** — let the problem dictate the solution; design for where the business is going.
2. **Identify, don't explain** — one signpost, not a catalogue of services.
3. **Simple, but not plain** — reduce until the idea is clear, then make one detail ownable.
4. **Relevant, not literal** — evoke the attitude; don't draw the product. Fresh forms of familiar signs work.
5. **Distinct** — know the category; depart from what blends together.
6. **Memorable** — shape and colour are the first memory hooks; one defining feature.
7. **One idea** — explainable in one sentence; a light puzzle is good, a riddle is not.
8. **Small and large** — 16 px and 16 m; one colour; reversed; embroidered.
9. **Timeless over trendy** — build on a concept, not an effect.
10. **Foundation of a system** — never judge a logo in a void; it must seed patterns, icons, motion.
11. **Craft matters** — geometry, optical corrections, spacing; the invisible details separate good from great.
12. **Be strong, not stubborn** — defend the strategy, stay open on details, value feedback.

Deeper reasoning: `references/principles.md`.

## Red flags

Fix before showing anything:

- Clip-art literalism (a tooth for a dentist, a house for real estate) or category clichés with no twist.
- Generic initials in an unmodified stock font; default-weight geometric sans with nothing ownable.
- More than three colours without a conceptual reason; gradients or shadows used to rescue a weak form.
- Details smaller than ~1/48 of the mark, hairlines, tight gaps that close at small sizes.
- Near-miss angles, lumpy curves from too many anchors, inconsistent stroke weights.
- Live `<text>`, embedded rasters, filters or masks in a "final" file.
- A concept that needs a paragraph to understand.
- Anything that looks like an existing logo — including the library's. The library is for learning, never tracing.

Honesty and limits: you cannot guarantee trademark clearance — recommend a professional search (trademark databases, reverse image search). Font licences must allow logo use; say which fonts you assumed and that outlines were constructed or must be made. If you couldn't render and inspect a file, say so; don't claim tests you didn't run.

## Trademark and licence notice

- The skill text, references, templates and scripts are MIT (upstream `LICENSE.txt`, pinned to the SHA in `ported_from`).
- The 1,400+ library logos are **third-party trademarks of their owners**. The MIT licence does NOT cover them (`references/TRADEMARKS.md`). They are not shipped with this skill: `fetch_library.py` downloads them at use time from the pinned upstream commit into the user's local skill copy and places a copy of the notice beside them.
- Use the library to **study** conventions, geometry and technique only. Never trace, adapt or recolour a library mark into a client deliverable.

## vs ip-as-logo

| | `logo-design` (this skill) | `ip-as-logo` |
|---|---|---|
| Output | Hand-constructed geometric SVG: symbol, wordmark, lockups, favicon/app-icon set, guidelines | Cute mascot / character mark generated with `image_generate` |
| Method | Brief → concepts → SVG geometry → audit/test loop → kit | Prompted image generation and iteration |
| Pick when | Brand identity that must work at 16 px, in one colour, embroidered, for years | A friendly character/IP face for an app, stream or community |

If the user wants a character mascot *and* a wordmark/system, do the mascot with `ip-as-logo` and the wordmark, lockups, testing and guidelines here.

## Pitfalls

1. **Drawing blind.** Never judge an SVG from its source. Render with `render_png.py` and `vision_analyze` the PNG after every change; the test sheets exist to be looked at, not just generated.
2. **`qlmanage` (macOS Quick Look).** Don't call it directly: it crops non-square SVGs, shrinks files that set width/height, and has no transparency. `render_png.py` only falls back to it when nothing better exists.
3. **Windows `python3`.** It is often the Microsoft Store stub that opens the Store; use `python` or `py -3` in every command.
4. **`<text>` in finals.** `svg_audit.py` flags live text; finished marks must have letterforms as paths. Exploration files may keep `<text>` if you say so.
5. **Library not fetched.** `search_library.py` errors and `preview_sheet.py --refs-industry` has nothing to show until `fetch_library.py` reports `svg_count >= 1400`. Re-run with `--force` if the directory is half-written.
6. **Skipping the 16 px test.** A mark that reads at 512 px can collapse at 16; the pixel test row in `preview_sheet.py` and the true-size row in `concept_sheet.py` are the deciders, not the hero render.
7. **Blocked by the checkpoint.** Do not build the kit, variants or board before the user chose a direction unless they explicitly waived the check-in.
8. **Renderer missing.** `render_png.py --which` returning nothing means no PNGs and no concept sheet image; install cairosvg or Chrome, or hand the user the SVGs/HTML and state that you could not inspect them.

## Verification

- `python3 <skill-dir>/scripts/fetch_library.py` prints JSON with `"status": "fetched"` (first run) or `"present"` (later runs) and `"svg_count"` ≥ 1400.
- `python3 <skill-dir>/scripts/search_library.py --industry payments-fintech --summary` prints "N logos match (of 1432)" and mark-type percentages.
- `python3 <skill-dir>/scripts/svg_audit.py <one path from search_library.py --format paths --limit 1>` prints a viewBox line, colour count and a production-readiness score.
- `python3 <skill-dir>/scripts/render_png.py --which` lists at least one backend; if it lists none, PNG steps are unavailable on this host.
- On your own concept: `svg_audit.py` reports no ERROR items, `preview_sheet.py … -o preview.html` writes the file, `export_variants.py … --only black white` writes `<name>-black.svg` and `<name>-white.svg`.

## Reference map

| Read | When |
|---|---|
| `references/principles.md` | Justifying decisions, resolving debates, deep critique |
| `references/discovery-brief.md` | Questions, brief template, word mapping |
| `references/mark-types.md` | Choosing the type of mark; pros/cons; decision guide |
| `references/visual-techniques.md` | Geometry, grids, balance, optical corrections, negative space, gradients, paradoxes |
| `references/color.md` | Palette strategy, harmony, reproduction, accessibility, library colour data |
| `references/typography.md` | Type study, custom letterforms, spacing, lockups, licensing |
| `references/process.md` | Stage-by-stage process, exploration, vector development, final checklist |
| `references/svg-construction.md` | Writing clean SVG logos, recipes, what to avoid |
| `references/testing-checklist.md` | Everything to test before presenting or delivering |
| `references/presentation-delivery.md` | Presenting, feedback, committees, deliverables, working relationship |
| `references/identity-system.md` | Kit of parts, sub-brands, dynamic identities, patterns, motion, guidelines, rollout |
| `references/redesign.md` | Refresh vs. rebrand, equity audit, refresh techniques |
| `references/critique.md` | Structured logo critique with scorecard and fixes |
| `references/library-guide.md` | What's in the 1,400+ logo library, insights, curated examples by technique |
| `references/TRADEMARKS.md` | Ownership of the library logos; what the MIT licence does not cover |
