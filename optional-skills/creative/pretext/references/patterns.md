# Pretext Patterns

Copy-pasteable snippets for the most common pretext demo shapes. Each pattern is self-contained — drop into an HTML `<script type="module">` after importing from `https://esm.sh/@chenglou/pretext@0.0.6`.

## 1. Flow around an obstacle (variable-width column)

The signature pretext move. Row-by-row ask "how wide is the corridor here?" and let pretext break lines accordingly.

```js
const prepared = prepareWithSegments(TEXT, FONT);
const LINE_H = 24;

function drawFlow(ctx, obstacle /* {x,y,r} */, COL_X, COL_W, H) {
  let cursor = { segmentIndex: 0, graphemeIndex: 0 };
  let y = 72;
  while (y < H - 40) {
    const dy = y - obstacle.y;
    const inBand = Math.abs(dy) < obstacle.r;
    let x = COL_X, w = COL_W;
    if (inBand) {
      const half = Math.sqrt(obstacle.r ** 2 - dy ** 2);
      const leftW  = Math.max(0, (obstacle.x - half) - COL_X);
      const rightW = Math.max(0, (COL_X + COL_W) - (obstacle.x + half));
      if (leftW >= rightW) { x = COL_X;                 w = leftW  - 12; }
      else                 { x = obstacle.x + half + 12; w = rightW - 12; }
      if (w < 40) { y += LINE_H; continue; } // skip rather than squeeze
    }
    const range = layoutNextLineRange(prepared, cursor, w);
    if (!range) break;
    const line = materializeLineRange(prepared, range);
    ctx.fillText(line.text, x, y);
    cursor = range.end;
    y += LINE_H;
  }
}
```

**Obstacle variants:** circles (above), rectangles (use `Math.max(0, …)` on the row-segment), multiple obstacles (sort segments and emit the wider remaining lane), animated obstacles (recompute every frame — pretext is fast enough).

## 2. Text-as-geometry game (word-bricks with collision)

Use `layoutWithLines` to get stable line rects, then treat each word as an axis-aligned box for physics.

```js
const prepared = prepareWithSegments(WORDS.join(" "), FONT);
const { lines } = layoutWithLines(prepared, FIELD_W, 28);

// Build brick rects: split each line on spaces and measure word-by-word.
const bricks = [];
let y = 50;
for (const line of lines) {
  let x = 10;
  for (const word of line.text.split(" ")) {
    const wPx = ctx.measureText(word).width; // or use walkLineRanges per word
    bricks.push({ x, y, w: wPx, h: 24, text: word, hp: 1 });
    x += wPx + ctx.measureText(" ").width;
  }
  y += 28;
}
```

Collision: standard AABB vs the ball. When `hp` drops to 0, the brick is "eaten." For the aesthetic: fade brick opacity with hp, trail particles from the letters on impact.

## 3. Shatter / explode typography

Use `walkLineRanges` + a manual grapheme walk to get `(x, y)` for every glyph, then spawn particles.

```js
const prepared = prepareWithSegments(TEXT, FONT);
const particles = [];
let y = 100;
walkLineRanges(prepared, COL_W, (line) => {
  // materialize so we get per-grapheme positions
  const range = materializeLineRange(prepared, line);
  const seg = new Intl.Segmenter(undefined, { granularity: "grapheme" });
  let x = COL_X;
  for (const { segment } of seg.segment(range.text)) {
    const w = ctx.measureText(segment).width;
    particles.push({ ch: segment, x, y, vx: 0, vy: 0, homeX: x, homeY: y });
    x += w;
  }
  y += LINE_H;
});

// On click, kick particles outward from click point; ease them back to (homeX, homeY).
canvas.addEventListener("click", (e) => {
  for (const p of particles) {
    const dx = p.x - e.clientX, dy = p.y - e.clientY;
    const d = Math.hypot(dx, dy) || 1;
    const force = 400 / (d * 0.2 + 1);
    p.vx += (dx / d) * force;
    p.vy += (dy / d) * force;
  }
});

function tick(dt) {
  for (const p of particles) {
    p.vx *= 0.92; p.vy *= 0.92;
    p.vx += (p.homeX - p.x) * 0.06;
    p.vy += (p.homeY - p.y) * 0.06;
    p.x += p.vx * dt; p.y += p.vy * dt;
  }
}
```

## 4. ASCII mask as moving obstacle

The "cool demos" money pattern: rasterize an ASCII logo, sprite, or bitmap into a cell buffer, then convert the occupied cells into per-row obstacle spans. Pretext lays the paragraphs around those spans, so the text actually opens around the moving ASCII object instead of being visually overpainted.

See `templates/donut-orbit.html` in this skill for a full implementation. Treat it as an example, not the canonical scene: it shows how to derive spans from an ASCII logo, project a wire shape into obstacle rows, keep text selectable in a DOM layer, and hide tuning controls behind `?dev`. Key structure:

```js
const CELL_W = 12, CELL_H = 15;
const cols = Math.ceil(W / CELL_W), rows = Math.ceil(H / CELL_H);
const asciiMask = new Uint8Array(cols * rows);
const obstacleRows = Array.from({ length: rows }, () => []);

function rasterizeLogo(time) {
  asciiMask.fill(0);
  for (const r of obstacleRows) r.length = 0;

  for (const block of logoBlocks(time)) {
    const r0 = Math.floor(block.y0 / CELL_H);
    const r1 = Math.ceil(block.y1 / CELL_H);
    for (let r = r0; r <= r1; r++) {
      obstacleRows[r]?.push([block.x0 - 18, block.x1 + 22]);
      // Fill asciiMask cells here for drawing.
    }
  }

  mergeRowSpans(obstacleRows);
}

function drawParagraphs(prepared) {
  let cursor = { segmentIndex: 0, graphemeIndex: 0 };
  for (let y = yStart; y < yEnd; y += LINE_H) {
    const spans = obstacleRows[Math.floor(y / CELL_H)];
    for (const [x0, x1] of freeIntervalsAround(spans)) {
      const range = layoutNextLineRange(prepared, cursor, x1 - x0);
      if (!range) return;
      ctx.fillText(materializeLineRange(prepared, range).text, x0, y);
      cursor = range.end;
    }
  }
}
```

The important bit is that the ASCII geometry is not decorative only. The same moving spans that draw the logo or draggable object also carve the line intervals passed to `layoutNextLineRange`.

### Measured spans beat magic padding

When a logo or bitmap is rasterized into cells, measure the actual occupied cells per row and then add a small halo. Do not use one giant bounding box. Tight measured spans make the text read as if it is flowing around the letter shapes.

```js
const rowMin = new Float32Array(rows).fill(Infinity);
const rowMax = new Float32Array(rows).fill(-Infinity);

for (const cell of visibleCells) {
  rowMin[cell.row] = Math.min(rowMin[cell.row], cell.x);
  rowMax[cell.row] = Math.max(rowMax[cell.row], cell.x + CELL_W);
}

for (let row = 0; row < rows; row++) {
  if (!Number.isFinite(rowMin[row])) continue;
  obstacleRows[row].push([rowMin[row] - halo, rowMax[row] + halo]);
}
```

For sharp pixel-art letters, smooth adjacent rows before pushing spans. A 1-2 row halo usually prevents code/prose from touching corners without losing the letter silhouette.

### Morphing shapes need morphing obstacles

If the visible object morphs (sphere to cube, logo to particles, etc.), tween the collision field too. A convincing demo uses the same `mix` value for both the rendered buffer and the pretext obstacle rows.

```js
function pushMorphedRows(aRows, bRows, mix) {
  for (let row = 0; row < rows; row++) {
    const a = aRows[row] ?? [centerX, centerX];
    const b = bRows[row] ?? [centerX, centerX];
    obstacleRows[row].push([
      a[0] + (b[0] - a[0]) * mix,
      a[1] + (b[1] - a[1]) * mix,
    ]);
  }
}
```

Without this, the artwork may morph while the text still wraps around the old shape, which breaks the pretext effect.

### Separate visual layers from collision

Use separate canvases when visual treatment should not affect layout. For example, fade an ASCII object with CSS opacity on its own canvas layer, but keep its obstacle rows controlled by explicit shape state. Fading glyph intensity or scaling obstacle spans often looks like the object is shrinking instead of fading.

## 5. Editorial multi-column with shared cursor

Classic magazine layout: three columns, text flows from the end of column 1 into the top of column 2, etc. Pretext makes this trivial because the cursor is portable between `layoutNextLineRange` calls.

```js
const prepared = prepareWithSegments(ARTICLE, FONT);
let cursor = { segmentIndex: 0, graphemeIndex: 0 };

for (const col of [COL1, COL2, COL3]) {
  let y = col.y;
  while (y < col.y + col.h) {
    const range = layoutNextLineRange(prepared, cursor, col.w);
    if (!range) return;
    const line = materializeLineRange(prepared, range);
    ctx.fillText(line.text, col.x, y);
    cursor = range.end;
    y += LINE_H;
  }
}
```

Add pull quotes by treating them as obstacles in the middle column and using pattern #1 around them.

## 6. Multiline shrink-wrap (tightest-fitting card)

Given a max width, find the **smallest** container width that still produces the same line count. Useful for chat bubbles, quote cards, tooltip sizing.

```js
const prepared = prepareWithSegments(text, FONT);
const { lineCount, maxLineWidth } = measureLineStats(prepared, MAX_W);
// card width = maxLineWidth + padding; card height = lineCount * LINE_H + padding
```

For a demo that *visualizes* this, render the card shrinking from `MAX_W` down to `maxLineWidth` over a second — the line count stays constant but the right edge pulls in.

## 7. Kinetic typography

Animate per-line transforms over time. `layoutWithLines` gives you stable lines; index `i` drives the timing offset.

```js
const { lines } = layoutWithLines(prepared, W - 80, 40);
function frame(t) {
  for (let i = 0; i < lines.length; i++) {
    const phase = t * 0.001 - i * 0.15;
    const y = 100 + i * 40 + Math.sin(phase) * 12;
    const opacity = 0.4 + 0.6 * Math.max(0, Math.sin(phase));
    ctx.globalAlpha = opacity;
    ctx.fillText(lines[i].text, 40, y);
  }
}
```

Variants: Star Wars crawl (perspective skew per line), wave (sine y-offset), bounce (ease-in-out arrival), glitch (per-glyph random offset using `Intl.Segmenter`).

## 8. Font stack patterns

| Vibe | Font string | Palette hint |
|------|-------------|--------------|
| Editorial / serious | `17px/1.4 "Iowan Old Style", Georgia, serif` | bone `#e8e6df` on charcoal `#0c0d10` |
| CRT / terminal | `600 13px "JetBrains Mono", ui-monospace, monospace` | amber `hsl(38 60% 62%)` on `#07070a` |
| Humanist / modern | `500 17px Inter, ui-sans-serif, system-ui, sans-serif` | off-white `#f3efe6` on deep-navy `#0b1020` |
| Display / poster | `700 64px "Playfair Display", serif` | hot-red `#ff4130` on cream `#f0ebe0` |
| Engineering | `14px "IBM Plex Mono", monospace` | neon-green `#7cff7c` on near-black `#0a0a0c` |

Always load the web font explicitly (Google Fonts link tag or `@font-face`) so the canvas measurement matches the CSS render.
