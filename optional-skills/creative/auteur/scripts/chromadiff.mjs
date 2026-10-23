#!/usr/bin/env node
/**
 * chromadiff.mjs — the cheerful-drift gate.
 * Given a near-monochrome reference, a model still returns a friendlier, bluer, more saturated
 * version of it, and it does so with the reference on screen. Eyes forgive that drift because each
 * frame looks fine alone; side by side it is the difference between a photograph and game graphics.
 * This measures it: mean OKLCH chroma of your screenshot vs the reference that set the direction.
 * Second mode, one file + --target-l: check the page against the background lightness the commit-sheet
 * committed to. This skill drifts to near-black across unrelated projects (taste.md §2.5) and every
 * such page argues convincingly for itself, so the number goes in the sheet BEFORE the build and gets
 * checked after, when "it just felt right dark" is no longer admissible evidence.
 * Feed --target-l a FULL-PAGE capture (shoot.mjs --full). The sheet commits to the page's mean
 * lightness, and the first viewport is never the mean: measured on one build, the hero frame read
 * 0.876 against a page that actually sat at 0.740 — a false FAIL produced entirely by the crop.
 * Usage: node chromadiff.mjs <yours.png> <reference.png|jpg> [--max-delta 0.02] [--json]
 *        node chromadiff.mjs <full-page.png> --target-l 0.55 [--tolerance 0.12] [--json]
 * ponytail: decodes via the playwright chromium that shoot.mjs already installs — no image deps.
 */
import { chromium } from 'playwright';
import { resolve, extname } from 'path';
import { existsSync } from 'fs';
import { readFile } from 'fs/promises';

const argv = process.argv.slice(2);
const VALUE_FLAGS = new Set(['--max-delta', '--target-l', '--tolerance']);
const files = argv.filter((a, i) => !a.startsWith('--') && !VALUE_FLAGS.has(argv[i - 1]));
const jsonMode = argv.includes('--json');
const num = (flag, def) => { const i = argv.indexOf(flag); return i >= 0 && argv[i + 1] ? +argv[i + 1] : def; };
const MAX_DELTA = num('--max-delta', 0.02);   // 0.02 OKLCH C is a visible step
const TARGET_L = argv.includes('--target-l') ? num('--target-l', NaN) : null;
const TOLERANCE = num('--tolerance', 0.12);   // wide on purpose: this catches drift, not art direction

if (TARGET_L !== null && !(TARGET_L >= 0 && TARGET_L <= 1)) {
  process.stderr.write('--target-l takes the committed mean background lightness, 0..1\n');
  process.exit(2);
}
if (!(files.length === 2 || (files.length === 1 && TARGET_L !== null))) {
  process.stderr.write('Usage: node chromadiff.mjs <yours.png> <reference.png> [--max-delta 0.02] [--json]\n' +
                       '       node chromadiff.mjs <yours.png> --target-l 0.55 [--tolerance 0.12] [--json]\n');
  process.exit(2);
}
const [mine, ref] = files.map(f => resolve(f));
for (const f of [mine, ref].filter(Boolean)) if (!existsSync(f)) { process.stderr.write(`not found: ${f}\n`); process.exit(2); }

// A file:// image will not decode inside an about:blank page (opaque origin), so the bytes travel
// as a data URL instead — which also keeps this working against any page the browser is showing.
const MIME = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.avif': 'image/avif' };
const toUrl = async p => {
  const mime = MIME[extname(p).toLowerCase()];
  if (!mime) { process.stderr.write(`unsupported image type: ${p}\n`); process.exit(2); }
  return `data:${mime};base64,${(await readFile(p)).toString('base64')}`;
};

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.goto('about:blank');

// One decode pass per image, downsampled to 256px on the long edge: chroma is a distribution
// statistic, and averaging 65k pixels answers it as well as averaging 4 million.
const measure = url => page.evaluate(async src => {
  const img = new Image();
  img.src = src;
  await img.decode();
  const scale = 256 / Math.max(img.width, img.height);
  const w = Math.max(1, Math.round(img.width * scale)), h = Math.max(1, Math.round(img.height * scale));
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const cx = cv.getContext('2d', { willReadFrequently: true });
  cx.drawImage(img, 0, 0, w, h);
  const d = cx.getImageData(0, 0, w, h).data;

  const lin = c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  let sumC = 0, sumL = 0, saturated = 0, dark = 0, n = 0;
  for (let p = 0; p < d.length; p += 4) {
    const r = lin(d[p] / 255), g = lin(d[p + 1] / 255), b = lin(d[p + 2] / 255);
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    const L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
    const A = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
    const B = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
    const C = Math.hypot(A, B);
    sumC += C; sumL += L; if (C > 0.1) saturated++; if (L < 0.5) dark++; n++;
  }
  return { chroma: sumC / n, lightness: sumL / n, saturatedShare: saturated / n, darkShare: dark / n };
}, url);

const a = await measure(await toUrl(mine));
const b = ref ? await measure(await toUrl(ref)) : null;
await browser.close();

const r3 = x => +x.toFixed(3);
const fails = [];

// Lightness mode: one frame against the number the commit-sheet committed to.
if (TARGET_L !== null) {
  const dl = a.lightness - TARGET_L;
  if (Math.abs(dl) > TOLERANCE)
    fails.push(`background lightness ${r3(a.lightness)} vs committed ${TARGET_L} (${dl > 0 ? '+' : ''}${r3(dl)}, tolerance ±${TOLERANCE})` +
               (dl < 0 ? ' — the page went darker than the sheet says; taste.md §2.5 tell #1' : ''));
  const summaryL = { yours: { chroma: r3(a.chroma), lightness: r3(a.lightness), darkShare: r3(a.darkShare) }, targetL: TARGET_L, tolerance: TOLERANCE, fails };
  if (jsonMode) process.stdout.write(JSON.stringify(summaryL, null, 2) + '\n');
  else {
    process.stdout.write(`chromadiff: L ${r3(a.lightness)} (committed ${TARGET_L}) - C ${r3(a.chroma)} - ${Math.round(a.darkShare * 100)}% dark pixels\n`);
    for (const f of fails) process.stdout.write(`FAIL ${f}\n`);
    process.stdout.write(fails.length ? 'FAIL — decide it or change the sheet; do not let it drift\n' : 'PASS\n');
  }
  process.exit(fails.length ? 1 : 0);
}

const delta = a.chroma - b.chroma;
// Only the upward direction fails. Ending up quieter than the reference is a defensible choice
// (and often the right one); ending up more colourful than it is the reflex this gate exists for.
if (delta > MAX_DELTA) fails.push(`chroma +${r3(delta)} over reference (${r3(a.chroma)} vs ${r3(b.chroma)}, limit +${MAX_DELTA})`);
if (a.saturatedShare - b.saturatedShare > 0.15)
  fails.push(`${Math.round((a.saturatedShare - b.saturatedShare) * 100)}pp more saturated pixels than the reference (C>0.1)`);

const summary = {
  yours: { chroma: r3(a.chroma), lightness: r3(a.lightness), saturatedShare: r3(a.saturatedShare) },
  reference: { chroma: r3(b.chroma), lightness: r3(b.lightness), saturatedShare: r3(b.saturatedShare) },
  chromaDelta: r3(delta), maxDelta: MAX_DELTA, fails,
};
if (jsonMode) process.stdout.write(JSON.stringify(summary, null, 2) + '\n');
else {
  process.stdout.write(`chromadiff: yours C ${r3(a.chroma)} L ${r3(a.lightness)} - reference C ${r3(b.chroma)} L ${r3(b.lightness)} - delta ${delta >= 0 ? '+' : ''}${r3(delta)}\n`);
  for (const f of fails) process.stdout.write(`FAIL ${f}\n`);
  process.stdout.write(fails.length ? 'FAIL — this is the drift, not a lighting difference: pull chroma back to the committed palette\n' : 'PASS\n');
}
process.exit(fails.length ? 1 : 0);
