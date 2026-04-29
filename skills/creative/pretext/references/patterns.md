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

## 4. Proportional ASCII surface (donut / sphere / wave)

The "cool demos" money pattern. Sample a parametric 3D surface, use classic luminance → glyph picking, but replace the monospace grid with a **z-buffer keyed by screen cell** and pull glyphs from a real corpus in reading order.

See `templates/donut-orbit.html` in this skill for the full implementation. Key structure:

```js
const CELL = 9; // px bucket
const cols = Math.ceil(W / CELL), rows = Math.ceil(H / CELL);
const zbuf = new Float32Array(cols * rows);
const chbuf = new Array(cols * rows);

// Sample the surface
for (let j = 0; j < PHI_STEPS; j++) {
  for (let i = 0; i < THETA_STEPS; i++) {
    const { sx, sy, ooz, L } = projectSurfacePoint(i, j);
    if (L <= 0) continue;
    const ci = (sx / CELL) | 0, ri = (sy / CELL) | 0;
    const idx = ri * cols + ci;
    if (ooz > zbuf[idx]) {
      zbuf[idx] = ooz;
      chbuf[idx] = GLYPHS[glyphIdx++ % GLYPHS.length];
    }
  }
}

// Draw once
for (let i = 0; i < chbuf.length; i++) if (chbuf[i]) ctx.fillText(chbuf[i], ...);
```

The `GLYPHS` array comes from pretext:

```js
const prepared = prepareWithSegments(CORPUS, FONT);
const { lines } = layoutWithLines(prepared, 260, 16);
const GLYPHS = [];
for (const line of lines) {
  const seg = new Intl.Segmenter(undefined, { granularity: "grapheme" });
  for (const { segment } of seg.segment(line.text)) GLYPHS.push(segment);
}
```

Why not just `[...CORPUS]`? Because pretext gives you **reading-order graphemes after line-break decisions** — which makes the surface glyphs follow the corpus's natural rhythm, including non-Latin scripts and soft-hyphen-resolved breaks.

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
