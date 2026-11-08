#!/usr/bin/env node
/**
 * shoot.mjs — screenshot journey for scroll-driven websites
 * Usage: node shoot.mjs <url> [--stops 7] [--out shots] [--breakpoints 390,768,1440] [--dpr 1] [--reduced-motion] [--full]
 */

import { mkdir, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import { resolve, join } from 'path';

// --- CLI parsing ---
const args = process.argv.slice(2);
if (!args.length || args[0] === '--help') {
  console.log('Usage: node shoot.mjs <url> [<url>...] [--stops 7] [--out shots] [--breakpoints 390,768,1440] [--dpr 1] [--reduced-motion] [--full]');
  console.log('Writes: <out>/bp<width>-stop<NN>.png  (also bp<width>-rm-stop<NN>.png with --reduced-motion, bp<width>-full.png with --full)');
  process.exit(0);
}

// Multi-screen products need every route shot, not a sample — take all leading non-flag args.
const rawTargets = [];
for (const a of args) { if (a.startsWith('--')) break; rawTargets.push(a); }
const get = (flag, def) => {
  const i = args.indexOf(flag);
  return i !== -1 ? args[i + 1] : def;
};
const has = flag => args.includes(flag);

const stopsCount  = parseInt(get('--stops', '7'), 10);
const outDir      = resolve(get('--out', 'shots'));
const bpArg       = get('--breakpoints', '390,768,1440');
const reducedMotion = has('--reduced-motion');
const fullPage    = has('--full');
// Layout bugs read fine at 1x, so that stays the default (30 frames at 2x cost 4x the bytes for
// nothing). Shoot --dpr 2 when the question is rendering rather than layout: hairline borders that
// vanish, text that only looks crisp at 1x, moiré in a fine pattern, an image whose real resolution
// is half what the layout claims.
const dpr         = parseFloat(get('--dpr', '1')) || 1;

const breakpoints = bpArg.split(',').map(Number);
const heights     = { 390: 844, 768: 1024, 1440: 900 };
const getHeight   = w => heights[w] ?? 900;

// Convert local paths → file:// URLs
const toUrl = u => (u.startsWith('http://') || u.startsWith('https://') || u.startsWith('file://'))
  ? u
  : 'file:///' + resolve(u).replace(/\\/g, '/');
const targets = rawTargets.map(toUrl);
// One route keeps the historical bp<width>-stop<NN>.png names; several get a per-route prefix.
// Truncating the tail collides for routes that share a long suffix and silently overwrites frames,
// so keep the distinguishing end AND an index that cannot repeat.
const labelFor = (u, i) => targets.length === 1 ? '' :
  `${String(i).padStart(2, '0')}_${(u.replace(/^https?:\/\//, '').replace(/^file:\/\/\//, '')
    .replace(/[^\w]+/g, '_').replace(/^_+|_+$/g, '').slice(-24)) || 'r'}-`;

// --- Dynamic import playwright with friendly error ---
let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch {
  console.error('playwright not found. Install: npm i -D playwright && npx playwright install chromium');
  process.exit(2);
}

// --- Heuristic: is this screenshot likely blank? ---
// Checks byte-entropy of the PNG buffer — a nearly-uniform image compresses to
// a much smaller ratio vs a varied one. ponytail: approximate, not pixel-perfect.
function likelyBlank(buf) {
  if (buf.length === 0) return true;
  // PNG files with almost all identical pixels compress extremely well.
  // Heuristic: if buffer is <2KB for any resolution it's suspicious.
  const VERY_SMALL = 2048;
  if (buf.length < VERY_SMALL) return true;
  // Sample 512 bytes spread across the buffer, count unique byte values.
  const samples = new Set();
  const step = Math.max(1, Math.floor(buf.length / 512));
  for (let i = 33; i < buf.length; i += step) samples.add(buf[i]); // skip PNG header
  return samples.size < 8; // fewer than 8 distinct byte values → very uniform
}

// --- Zero-pad helper ---
const pad = (n, len = 2) => String(n).padStart(len, '0');

// --- Main ---
await mkdir(outDir, { recursive: true });

let totalWritten = 0;
let anySucceeded = false;
const summary = [];

for (const [ti, pageUrl] of targets.entries()) {
const prefix = labelFor(pageUrl, ti);
for (const width of breakpoints) {
  const vpHeight = getHeight(width);
  const bpErrors = [];
  const bpFiles  = [];

  let browser;
  try {
    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
      viewport: { width, height: vpHeight },
      deviceScaleFactor: dpr,
    });
    const page = await context.newPage();

    // Collect page console errors
    page.on('console', msg => { if (msg.type() === 'error') bpErrors.push(msg.text()); });
    page.on('pageerror', err => bpErrors.push(err.message));

    // Navigate with networkidle, fall back to load on timeout
    try {
      await page.goto(pageUrl, { waitUntil: 'networkidle', timeout: 30_000 });
    } catch {
      await page.goto(pageUrl, { waitUntil: 'load', timeout: 15_000 });
    }

    const scrollHeight = await page.evaluate(() => document.documentElement.scrollHeight);
    const maxScroll    = Math.max(0, scrollHeight - vpHeight);

    // Compute evenly spaced stop positions
    const stops = Array.from({ length: stopsCount }, (_, i) =>
      stopsCount === 1 ? 0 : Math.round((i / (stopsCount - 1)) * maxScroll)
    );

    // Screenshot each stop (incremental scroll so animations fire)
    let currentY = 0;
    for (let si = 0; si < stops.length; si++) {
      const target = stops[si];
      // Scroll incrementally toward target in chunks so IntersectionObserver triggers
      const CHUNK = 200;
      while (Math.abs(currentY - target) > CHUNK) {
        const next = currentY < target ? Math.min(currentY + CHUNK, target) : Math.max(currentY - CHUNK, target);
        await page.evaluate(y => window.scrollTo({ top: y, behavior: 'smooth' }), next);
        await page.waitForTimeout(120);
        currentY = next;
      }
      await page.evaluate(y => window.scrollTo({ top: y, behavior: 'smooth' }), target);
      await page.waitForTimeout(700); // let scroll-triggered animations settle

      const fname = `${prefix}bp${width}-stop${pad(si)}.png`;
      const fpath = join(outDir, fname);
      const buf   = await page.screenshot({ path: fpath, fullPage: false });
      bpFiles.push(fname);
      totalWritten++;
      if (likelyBlank(buf)) console.warn(`  [warn] ${fname} possibly blank`);
    }

    // --reduced-motion: redo first + last stop
    if (reducedMotion) {
      const rmContext = await browser.newContext({
        viewport: { width, height: vpHeight },
        deviceScaleFactor: dpr,
        reducedMotion: 'reduce',
      });
      const rmPage = await rmContext.newPage();
      try {
        await rmPage.goto(pageUrl, { waitUntil: 'networkidle', timeout: 30_000 });
      } catch {
        await rmPage.goto(pageUrl, { waitUntil: 'load', timeout: 15_000 });
      }
      // Every stop, not just first and last: the peak lives at 40-70% depth, so shooting only the
      // ends captures exactly the frames the reduced-motion cut was never going to break.
      for (const [idx, stop] of stops.map((s, i) => [i, s])) {
        await rmPage.evaluate(y => window.scrollTo({ top: y, behavior: 'smooth' }), stop);
        await rmPage.waitForTimeout(700);
        const fname = `${prefix}bp${width}-rm-stop${pad(idx)}.png`;
        const fpath = join(outDir, fname);
        const buf   = await rmPage.screenshot({ path: fpath, fullPage: false });
        bpFiles.push(fname);
        totalWritten++;
        if (likelyBlank(buf)) console.warn(`  [warn] ${fname} possibly blank`);
      }
      await rmContext.close();
    }

    // --full: one full-page screenshot (and its reduced-motion twin when both flags are on)
    if (fullPage) {
      const fname = `${prefix}bp${width}-full.png`;
      const fpath = join(outDir, fname);
      const buf   = await page.screenshot({ path: fpath, fullPage: true });
      bpFiles.push(fname);
      totalWritten++;
      if (likelyBlank(buf)) console.warn(`  [warn] ${fname} possibly blank`);
    }

    anySucceeded = true;
    summary.push({ url: pageUrl, width, scrollHeight, files: bpFiles, errors: bpErrors });
  } catch (err) {
    console.error(`[${pageUrl} bp ${width}] failed: ${err.message}`);
    summary.push({ url: pageUrl, width, failed: true, error: err.message });
  } finally {
    await browser?.close();
  }
}
}

// --- Print summary ---
console.log('\n=== shoot.mjs summary ===');
console.log(`Output dir : ${outDir}`);
console.log(`Files written: ${totalWritten}`);
for (const s of summary) {
  const where = targets.length > 1 ? `${s.url} ` : '';
  if (s.failed) {
    console.log(`  ${where}bp${s.width}: FAILED — ${s.error}`);
  } else {
    console.log(`  ${where}bp${s.width}: scrollHeight=${s.scrollHeight}px  files=${s.files.length}`);
    if (s.errors.length) console.log(`    page errors: ${s.errors.join(' | ')}`);
  }
}

process.exit(anySucceeded ? 0 : 1);
