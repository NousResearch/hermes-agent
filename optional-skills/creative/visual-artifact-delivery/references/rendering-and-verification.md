# Rendering and verification in depth

The short version is in `SKILL.md`. This file is the detail behind three steps: how to get
the image out of an authored artifact, how to read the geometry checker, and how to check
the pixels without guessing.

## 1. Render routes

Pick by what the install has. All three produce a file on disk; none of them require the
user to open anything.

**First, prove the page is the artifact.** Open it, then ask the page:

```
js("document.querySelector('svg') ? 1 : 0")            # browser_exec
browser_cdp → Runtime.evaluate → "document.querySelector('svg') ? 1 : 0"
```

A `0`, or a tab URL of `chrome-error://chromewebdata/`, means the browser never read your
file — a sandboxed or cloud browser cannot open `file://` paths from the agent's machine.
Two routes that always work for a self-contained artifact:

- `data:text/html;charset=utf-8,<percent-encoded artifact>` — no filesystem access at all,
  and self-contained HTML/SVG is exactly what the authoring skills emit.
- serve the directory and open the served URL, when the browser can reach the agent's host.

Only capture once the check returns `1`.

### `browser_vision` — the portable route

```
browser_navigate → the artifact URL                 # call this first; the tool requires it
browser_vision   → question="Read every label in the diagram and say whether any overlap."
```

The result carries `screenshot_path`, already persisted under the Hermes screenshot cache,
and the tool's own contract is that this path is what `MEDIA:<screenshot_path>` should point
at. Use it as the default: one navigate, one call, a delivered file, and a description of
what the model saw in the pixels.

It captures the *visible viewport*. Make the artifact fit the viewport (a fixed
`width`/`height` no larger than the window), or move to the CDP route for taller pages.

### `browser_cdp` — when you need the whole page or a chosen pixel size

```
browser_cdp → method="Page.captureScreenshot"
              params={"format": "png",
                      "captureBeyondViewport": true,
                      "clip": {"x": 0, "y": 0, "width": 1200, "height": 800, "scale": 1}}
```

The reply's `data` is base64, so write it out before delivering it:

```bash
printf '%s' "$DATA" | base64 -d > artifact.png
```

`clip.scale` **multiplies the display's device pixel ratio**. On a 2× display, `scale: 1` on
a 720×380 clip produced a 1440×760 PNG in testing; `scale: 2` would have produced 2880×1520
and timed the capture out. Ask for the size you need, read the PNG header instead of
assuming, and keep the capture near 2×.

`browser_cdp` needs a CDP-connected session (`/browser connect`, `BROWSER_CDP_URL`,
`browser.cdp_url`, or a CDP-backed cloud session); on a session without one, fall back to
`browser_vision`.

### `browser_exec` — browser-use backend, one call for the whole pipeline

`browser_exec` pre-imports the helpers (`new_tab`, `goto_url`, `js`, `cdp`, `page_info`,
`wait_for_load`, `capture_screenshot`) and runs Python, so open → check → wait → capture fits
in one call. Variables do **not** survive between calls even though the browser session does —
anything you need in the capture must be set up in the same call.

```python
import base64, struct, urllib.parse
html = open("/abs/path/artifact.html").read()
new_tab("data:text/html;charset=utf-8," + urllib.parse.quote(html))
wait_for_load()
assert js("document.querySelector('svg') ? 1 : 0") == 1, "the artifact is not in the DOM"
js("document.fonts.ready.then(() => true)")
shot = cdp("Page.captureScreenshot", format="png", captureBeyondViewport=True,
           clip={"x": 0, "y": 0, "width": 720, "height": 380, "scale": 1})
open("/abs/path/artifact.png", "wb").write(base64.b64decode(shot["data"]))
```

`capture_screenshot()` alone takes the viewport, `capture_screenshot(full=True)` the whole
page; neither gives you clip/scale control, so drive `Page.captureScreenshot` when the exact
pixel size matters. Note that on a desktop install a plain screenshot can span the whole
window rather than the artifact — clipping to the canvas is what keeps the deliverable clean.

## 2. Reading the geometry checker

```bash
python3 "$HERMES_SKILL_DIR/scripts/svg_geometry_check.py" diagram.svg
python3 "$HERMES_SKILL_DIR/scripts/svg_geometry_check.py" page.html --json
python3 "$HERMES_SKILL_DIR/scripts/svg_geometry_check.py" diagram.svg \
        --tolerance 2 --min-overlap 64 --strict-boundaries
```

What it measures: `rect`, `circle` and `ellipse` primitives, with ancestor
`translate()` offsets accumulated, skipping hidden elements. What it cannot measure
(`path`, `polygon`, `polyline`, `line`, `text`, `image`, and anything under a
rotate/scale/matrix/skew or inside `clipPath`/`mask`/`symbol`) is named in the `note` lines.
Treat those notes as a to-look-at list for the render, not as a clean bill of health.

| Finding | Severity | Meaning |
|---|---|---|
| `collision` | fail | two shapes partially overlap |
| `duplicate` | fail | two shapes share a box (within `--tolerance`) |
| `outside-canvas` | fail | a shape extends past the `viewBox` / `width`×`height` |
| `stacked` | warn | one shape all but covers another of similar size |
| `boundary` | warn | a dashed-stroke frame overlaps a filled shape |
| `contained` | info | a small shape inside a much larger one (a panel holding its contents) |

Flags: `--tolerance` is the pixels of shift/overlap ignored (default 2); `--min-overlap` is
the area below which an overlap is noise (default 64 px²); `--strict-boundaries` promotes
`boundary` to a failure when a diagram's dashed rules are supposed to sit outside content.

Exit `0` means "nothing needs a fix". Exit `1` means at least one `fail` finding (plus
`warn`, under `--strict-boundaries`). Exit `2` means the input could not be parsed: no
`<svg>` element, or malformed XML. `--json` prints the same report with
`canvas`, `shapes_checked`, `notes`, `findings` and a `failures` count.

Worked example — a legend drawn over a toolbar:

```
fail collision      rect#legend (40,300 220x60) ∩ rect#toolbar (120,290 300x40) overlap=9400px^2 — shapes overlap
2 shape(s) checked on viewBox 800x600; 1 finding(s), 1 to fix
```

Read `overlap_px` to judge intent: 9400 out of the legend's 13200 px² is a real
occlusion, whereas a handful of pixels on a shared border is usually two panels drawn to
abut. The fix is a text edit in the SVG — move the legend's `y`, or raise the canvas
`viewBox` — then re-run the checker; only render once it is clean.

## 3. Checking the pixels

Source geometry clean does not mean the picture is right. Render, then look at the file:

1. `vision_analyze` on the whole image: "List every label you can read and say whether any
   shape overlaps another." This catches collisions the box arithmetic cannot see (a
   `<text>` wider than its box) and broken text (missing glyphs, fallback fonts).
2. If any label is illegible, call `vision_analyze` again with `region=[x1, y1, x2, y2]` in
   **original-image** pixel coordinates. The crop keeps full resolution, so an unreadable
   label becomes readable without touching the render. Zoom before you re-render.
3. Cross-check the two verdicts: a `stacked`/`boundary` finding the vision read did not
   mention is usually deliberate; a visual overlap the checker never reported is a `path`
   or `text` case that lives in the `note` lines.

If the render is wrong, fix the source and go back to step 2 of the procedure — never patch
up the image.

## 4. Delivery

- `MEDIA:/absolute/path.png` on platforms with an attachment channel; the bare path in text
  as well, so a text-only surface is still usable.
- Delivery is refused for `/etc`, `/proc`, `/sys`, `/dev`, `/root`, `/boot`, `/var/log`,
  `/var/lib`, `/var/run`, and for `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.docker`,
  `~/.config`, `~/.azure`, `~/.gcloud`, `~/Library/Keychains`. Render into a normal output
  directory and deliver an absolute path.
- Deliver in the same turn as the answer. "I'll send it in a moment" is a broken promise if
  the turn ends first.
