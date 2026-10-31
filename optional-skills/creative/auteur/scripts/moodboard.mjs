#!/usr/bin/env node
/**
 * moodboard.mjs — visual reference search → downloaded images → one contact sheet you can actually look at.
 *
 * Usage:
 *   node scripts/moodboard.mjs "<query>" ["<query2>" ...] [options]
 *
 * Options:
 *   --source bing,pinterest,arena   which sources (default all three)
 *   --limit N                       images kept in total (default 24, cap 60)
 *   --out DIR                       output dir (default design/moodboard)
 *   --cols N                        contact-sheet columns (default 5)
 *   --min-kb N                      skip images smaller than this (default 15)
 *
 * Writes DIR/img/NN.<ext>, DIR/contact-sheet*.png (LOOK AT THESE), DIR/MOODBOARD.md.
 * Exit 0 if any image landed.
 *
 * Source notes (probed live, 2026-08):
 *  · bing      — most reliable. Indexes pinimg / dribbble / behance CDNs and serves
 *                hotlinkable originals through a browser context. This is the workhorse.
 *  · pinterest — logged-out search renders a partial grid behind a login wall: sometimes
 *                ~15 pins, sometimes zero. Best-effort, never load-bearing. Thumbnails are
 *                rewritten 236x → 736x (originals/ is 403 for hotlinks).
 *  · arena     — api.are.na public search, no auth. Curated designer taste, lower volume.
 */

import { mkdir, writeFile } from 'fs/promises';
import { resolve, join } from 'path';

const args = process.argv.slice(2);
if (!args.length || args.includes('--help')) {
  console.log(`moodboard.mjs — image search → contact sheet

  node scripts/moodboard.mjs "<query>" ["<query2>" ...] [--source bing,pinterest,arena]
                             [--limit 24] [--out design/moodboard] [--cols 5] [--min-kb 15]`);
  process.exit(0);
}
const get = (f, d) => { const i = args.indexOf(f); return i !== -1 ? args[i + 1] : d; };
const flagIdx = args.findIndex(a => a.startsWith('--'));
const queries = (flagIdx === -1 ? args : args.slice(0, flagIdx)).filter(Boolean);

const sources = get('--source', 'bing,pinterest,arena').split(',').map(s => s.trim());
const limit   = Math.min(parseInt(get('--limit', '24'), 10) || 24, 60);
const outDir  = resolve(get('--out', 'design/moodboard'));
const cols    = parseInt(get('--cols', '5'), 10) || 5;
const minKb   = parseInt(get('--min-kb', '15'), 10) || 15;

if (!queries.length) { console.error('[moodboard] no query given.'); process.exit(1); }

let chromium;
try { ({ chromium } = await import('playwright')); }
catch { console.error('playwright not found. Install: npm i -D playwright && npx playwright install chromium'); process.exit(2); }

const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';
const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ userAgent: UA, viewport: { width: 1440, height: 1000 }, ignoreHTTPSErrors: true });

async function open(page, url, settle = 4000) {
  await page.goto(url, { waitUntil: 'commit', timeout: 30_000 });
  const dcl = await page.waitForLoadState('domcontentloaded', { timeout: 15_000 }).then(() => true).catch(() => false);
  await page.waitForTimeout(settle);
  if (!dcl) await page.evaluate(() => window.stop()).catch(() => {}); // stream that never closes
}

// --- sources ------------------------------------------------------------------
async function fromBing(q, want) {
  const page = await ctx.newPage();
  const out = [];
  try {
    await open(page, 'https://www.bing.com/images/search?q=' + encodeURIComponent(q) + '&form=HDRSC2', 3500);
    for (let pass = 0; pass < 3 && out.length < want; pass++) {
      const items = await page.evaluate(() =>
        [...document.querySelectorAll('a.iusc')].map(a => { try { return JSON.parse(a.getAttribute('m')); } catch { return null; } }).filter(Boolean));
      for (const it of items) {
        if (out.length >= want) break;
        if (it.murl && !out.some(o => o.img === it.murl)) out.push({ img: it.murl, thumb: it.turl, page: it.purl, source: 'bing' });
      }
      await page.evaluate(() => window.scrollBy(0, innerHeight * 2));
      await page.waitForTimeout(1800);
    }
  } catch (e) { console.error(`[moodboard] bing "${q}": ${e.message.split('\n')[0]}`); }
  finally { await page.close(); }
  return out;
}

async function fromPinterest(q, want) {
  const page = await ctx.newPage();
  const out = [];
  try {
    await open(page, 'https://www.pinterest.com/search/pins/?q=' + encodeURIComponent(q) + '&rs=typed', 6000);
    for (let pass = 0; pass < 4 && out.length < want; pass++) {
      const items = await page.evaluate(() =>
        [...document.querySelectorAll('img')]
          .filter(i => /i\.pinimg\.com\/\d+x\//.test(i.src))
          .map(i => ({ src: i.src, pin: i.closest('a[href^="/pin/"]')?.getAttribute('href') || null })));
      for (const it of items) {
        // 736x is the largest size that stays hotlinkable; /originals/ returns 403.
        const img = it.src.replace(/\/\d+x\//, '/736x/');
        if (out.length >= want) break;
        if (!out.some(o => o.img === img)) out.push({ img, thumb: it.src, page: it.pin ? 'https://www.pinterest.com' + it.pin : 'https://www.pinterest.com/search/pins/?q=' + encodeURIComponent(q), source: 'pinterest' });
      }
      await page.evaluate(() => window.scrollBy(0, innerHeight * 1.5));
      await page.waitForTimeout(2000);
    }
    if (!out.length) console.error('[moodboard] pinterest returned 0 (login wall) — bing covers it.');
  } catch (e) { console.error(`[moodboard] pinterest "${q}": ${e.message.split('\n')[0]}`); }
  finally { await page.close(); }
  return out;
}

async function fromArena(q, want) {
  const out = [];
  try {
    const res = await ctx.request.get(`https://api.are.na/v2/search?q=${encodeURIComponent(q)}&per=${Math.min(want * 2, 40)}`, { timeout: 20_000 });
    const json = await res.json();
    for (const b of json.blocks || []) {
      const img = b.image?.large?.url || b.image?.display?.url || b.image?.original?.url;
      if (!img || out.length >= want) continue;
      out.push({ img, thumb: b.image?.thumb?.url, page: `https://www.are.na/block/${b.id}`, source: 'arena', title: b.title });
    }
  } catch (e) { console.error(`[moodboard] are.na "${q}": ${e.message.split('\n')[0]}`); }
  return out;
}

// --- collect ------------------------------------------------------------------
const perSourcePerQuery = Math.ceil(limit / (sources.length * queries.length)) + 4;
let candidates = [];
for (const q of queries) {
  for (const s of sources) {
    const fn = s === 'bing' ? fromBing : s === 'pinterest' ? fromPinterest : s === 'arena' ? fromArena : null;
    if (!fn) { console.error(`[moodboard] unknown source "${s}"`); continue; }
    const got = await fn(q, perSourcePerQuery);
    process.stderr.write(`[moodboard] ${s} · "${q}" → ${got.length}\n`);
    candidates.push(...got.map(g => ({ ...g, query: q })));
  }
}
// dedupe by url, then interleave sources so one source can't own the whole sheet
const seen = new Set();
candidates = candidates.filter(c => !seen.has(c.img) && seen.add(c.img));
const bySource = sources.map(s => candidates.filter(c => c.source === s));
const ordered = [];
for (let i = 0; ordered.length < candidates.length; i++) {
  let added = false;
  for (const bucket of bySource) if (bucket[i]) { ordered.push(bucket[i]); added = true; }
  if (!added) break;
}

// --- download -----------------------------------------------------------------
await mkdir(join(outDir, 'img'), { recursive: true });
const kept = [];
const sizes = new Set();
for (const c of ordered) {
  if (kept.length >= limit) break;
  let buf = null, ct = '';
  for (const url of [c.img, c.thumb].filter(Boolean)) {
    try {
      const r = await ctx.request.get(url, { timeout: 20_000, headers: { Referer: c.page || '' } });
      if (!r.ok()) continue;
      ct = r.headers()['content-type'] || '';
      if (!/^image\//.test(ct)) continue;
      const b = await r.body();
      if (b.length < minKb * 1024) continue;
      buf = b; c.used = url; break;
    } catch {}
  }
  if (!buf) continue;
  if (sizes.has(buf.length)) continue;      // cheap byte-identical dedupe across sources
  sizes.add(buf.length);
  const ext = /png/.test(ct) ? 'png' : /webp/.test(ct) ? 'webp' : /gif/.test(ct) ? 'gif' : 'jpg';
  const n = String(kept.length + 1).padStart(2, '0');
  const file = `img/${n}.${ext}`;
  await writeFile(join(outDir, file), buf);
  kept.push({ ...c, n, file, kb: Math.round(buf.length / 1024) });
}

// --- contact sheet ------------------------------------------------------------
// Rendering an HTML grid and screenshotting it needs no extra dependency and copes
// with mixed aspect ratios — one image the model reads instead of 24 separate ones.
const PER_SHEET = 20;
// A remainder of one or two tiles becoming its own 95%-empty page costs a second look for nothing,
// and the doctrine promises reading a moodboard costs one look. Balance the sheets instead.
const sheetCount = Math.max(1, Math.round(kept.length / PER_SHEET));
const perSheet = Math.ceil(kept.length / sheetCount);
const sheets = [];
for (let s = 0; s * perSheet < kept.length; s++) {
  const slice = kept.slice(s * perSheet, (s + 1) * perSheet);
  const html = `<!doctype html><meta charset="utf-8"><style>
    body{margin:0;background:#111;font:13px/1.2 ui-monospace,monospace;color:#eee}
    .g{display:grid;grid-template-columns:repeat(${cols},1fr);gap:10px;padding:10px}
    figure{margin:0;background:#1c1c1c;border-radius:4px;overflow:hidden}
    img{display:block;width:100%;height:210px;object-fit:contain;background:#0a0a0a}
    figcaption{padding:4px 6px;color:#9a9a9a}
    b{color:#fff}
  </style><div class=g>${slice.map(k =>
    `<figure><img src="img/${k.file.split('/')[1]}"><figcaption><b>${k.n}</b> ${k.source}</figcaption></figure>`).join('')}</div>`;
  const f = join(outDir, `_sheet${s}.html`);
  await writeFile(f, html, 'utf8');
  const page = await ctx.newPage();
  await page.setViewportSize({ width: cols * 320, height: 1000 });
  await page.goto('file:///' + f.replace(/\\/g, '/'), { waitUntil: 'load', timeout: 20_000 }).catch(() => {});
  await page.waitForTimeout(1200);
  const name = sheetCount === 1 ? 'contact-sheet.png' : `contact-sheet-${s + 1}.png`;
  await page.screenshot({ path: join(outDir, name), fullPage: true });
  await page.close();
  sheets.push(name);
}

await browser.close();

// --- index --------------------------------------------------------------------
const md = [
  `# MOODBOARD — ${queries.map(q => `"${q}"`).join(' · ')}`,
  '',
  sheets.length
    ? `**Look at ${sheets.map(s => `\`${s}\``).join(', ')} first** — every tile is numbered; refer to tiles by number below.`
    : '_no contact sheet: nothing downloaded._',
  '',
  'Reference images are for *direction* — palette, light, composition, texture, type energy.',
  'They are not assets: never ship a downloaded image, and never reproduce one composition.',
  'Name what you take from each tile you use, then throw the rest away.',
  '',
  '| # | source | kb | origin page |',
  '|---|---|---|---|',
  ...kept.map(k => `| ${k.n} | ${k.source} | ${k.kb} | ${k.page ? `[link](${k.page})` : '—'} |`),
  '',
  '## Read (fill in)',
  '- dominant palette across the tiles you like: ',
  '- light / contrast character: ',
  '- composition move worth stealing: ',
  '- what to explicitly avoid from these: ',
  '',
].join('\n');
await writeFile(join(outDir, 'MOODBOARD.md'), md, 'utf8');

console.log(`\n=== moodboard ===`);
console.log(`queries : ${queries.join(' | ')}`);
console.log(`kept    : ${kept.length} images (${sources.map(s => `${s}:${kept.filter(k => k.source === s).length}`).join(' ')})`);
console.log(`sheets  : ${sheets.map(s => join(outDir, s)).join('\n          ') || '—'}`);
console.log(`index   : ${join(outDir, 'MOODBOARD.md')}`);
process.exit(kept.length ? 0 : 1);
