#!/usr/bin/env node
/**
 * refscout.mjs — reference scouting: find live award-level sites, then take them apart.
 *
 * Usage:
 *   node scripts/refscout.mjs <url> [<url>...]        analyse specific sites
 *   node scripts/refscout.mjs --from awwwards          harvest the current gallery
 *   node scripts/refscout.mjs --from awwwards:scrolling --limit 6
 *   node scripts/refscout.mjs --from awwwards --search "coffee brand" --limit 5
 *
 * Options:
 *   --limit N       how many sites (default 6, hard cap 15)
 *   --out DIR       output dir (default design/refs)
 *   --shots N       screenshots per site (default 3: hero + 2 scroll stops)
 *   --no-shots      fingerprint only, no browser screenshots
 *   --width N       viewport width for shots (default 1440)
 *
 * Writes DIR/REFERENCES.md (read this), DIR/refs.json, DIR/shots/*.png.
 * Exits 0 if at least one site was profiled, 1 otherwise.
 */

import { mkdir, writeFile } from 'fs/promises';
import { resolve, join } from 'path';

const args = process.argv.slice(2);
if (!args.length || args.includes('--help')) {
  console.log(`refscout.mjs — scout live website references and fingerprint their mechanics

  node scripts/refscout.mjs <url> [<url>...]
  node scripts/refscout.mjs --from awwwards[:<tag>] [--search "<text>"] [--limit 6]

  --limit N  --out DIR  --shots N  --no-shots  --width N`);
  process.exit(0);
}
const get = (f, d) => { const i = args.indexOf(f); return i !== -1 ? args[i + 1] : d; };
const has = f => args.includes(f);

const limit    = Math.min(parseInt(get('--limit', '6'), 10) || 6, 15);
const outDir   = resolve(get('--out', 'design/refs'));
const shotsN   = has('--no-shots') ? 0 : (parseInt(get('--shots', '3'), 10) || 3);
const width    = parseInt(get('--width', '1440'), 10) || 1440;
const from     = get('--from', null);
const search   = get('--search', null);
const urls     = args.filter(a => /^https?:\/\//.test(a));

let chromium;
try { ({ chromium } = await import('playwright')); }
catch { console.error('playwright not found. Install: npm i -D playwright && npx playwright install chromium'); process.exit(2); }

const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';

/**
 * Award-level sites are heavy streaming-SSR / WebGL pages: 'load' and 'networkidle'
 * frequently never fire. Commit + a timed settle is the only navigation that survives them.
 * Returns false when DOMContentLoaded never fired — those pages need freeze() before capture.
 */
async function open(page, url, settle = 7000, onStatus) {
  const resp = await page.goto(url, { waitUntil: 'commit', timeout: 30_000 });
  onStatus?.(resp?.status?.() ?? 0);
  const dcl = await page.waitForLoadState('domcontentloaded', { timeout: 12_000 }).then(() => true).catch(() => false);
  await page.waitForTimeout(dcl ? settle : settle + 5000); // no DCL → give hydration longer
  return dcl;
}

/**
 * A page whose HTML stream never closes never yields a compositor frame either, so every
 * screenshot on it times out (measured on basement.studio and lusion.co). `window.stop()`
 * ends the stream and capture drops to ~40ms. Call it only AFTER the settle wait — stopping
 * early aborts the CSS/JS the fingerprint needs and leaves you reading Times-New-Roman
 * defaults off a half-loaded page.
 */
async function freeze(page, dcl) {
  if (!dcl) await page.evaluate(() => window.stop()).catch(() => {});
}

// --- library signatures, matched against the concatenated JS the page actually loaded ---
// Bundlers hide globals (window.gsap is empty on virtually every modern site), but
// minifiers keep string literals and class names, so the bundle text still tells the truth.
const LIBS = [
  ['GSAP',            /gsap\.registerPlugin|GreenSock|_gsap|gsap\.timeline|gsap\.to\(/],
  ['ScrollTrigger',   /ScrollTrigger/],
  ['ScrollSmoother',  /ScrollSmoother/],
  ['SplitText',       /SplitText|SplitType/],
  ['Lenis',           /\blenis\b|new Lenis|lenis-smooth/i],
  ['Locomotive',      /locomotive-scroll|LocomotiveScroll|has-scroll-smooth/],
  ['three.js',        /THREE\.WebGLRenderer|WebGLRenderer|three\.module|\bTHREE\b/],
  ['R3F',             /react-three|@react-three\/fiber|useFrame/],
  ['OGL',             /\bogl\b|new Renderer\(\{.*dpr/],
  ['PixiJS',          /PIXI\.|pixi\.js/],
  ['Framer Motion',   /framer-motion|motion-dom|framerAppearId/],
  ['Motion One',      /@motionone|motion\.dev/],
  ['Barba',           /barba\.init|@barba|new Barba/],
  ['Swup',            /new Swup|@swup/],
  ['Matter.js',       /Matter\.Engine/],
  ['Rive',            /@rive-app|rive\.wasm/],
  ['Lottie',          /lottie-web|bodymovin/],
  ['Spline',          /splinetool|spline-viewer/],
  ['Theatre.js',      /@theatre\/core/],
  ['Swiper',          /new Swiper|swiper-bundle/],
  ['Splitting',       /Splitting\(/],
];

function classifyPeak(f) {
  const out = [];
  if (f.videos.some(v => !v.autoplay && v.preload !== 'none') && f.scrollRatio > 4) out.push('likely scroll-scrubbed video');
  if (f.webgl && f.scrollRatio > 3) out.push('WebGL scene driven by scroll');
  else if (f.webgl) out.push('WebGL hero');
  if (f.pinSpacers > 0) out.push(`${f.pinSpacers} pinned scene(s) (GSAP ScrollTrigger)`);
  if (f.stickies > 2) out.push(`${f.stickies} sticky layers (stack/pin effect)`);
  if (f.cssTimeline) out.push('CSS scroll-driven animations (animation-timeline)');
  if (f.libs.includes('Lenis') || f.libs.includes('Locomotive') || f.libs.includes('ScrollSmoother')) out.push('smoothed/virtualised scroll');
  if (f.mixBlend > 0) out.push(`${f.mixBlend} mix-blend-mode layer(s)`);
  if (f.cursorNone) out.push('custom cursor');
  if (f.videos.length && !out.length) out.push('looping video texture');
  return out;
}

async function fingerprint(ctx, url) {
  const page = await ctx.newPage();
  const js = [];
  let jsBytes = 0;
  page.on('response', async res => {
    if (jsBytes > 4_000_000) return;
    const ct = res.headers()['content-type'] || '';
    if (!/javascript|ecmascript/.test(ct)) return;
    try { const t = await res.text(); jsBytes += t.length; js.push(t); } catch {}
  });

  let httpStatus = 0;
  try {
    const dcl = await open(page, url, 8000, s => { httpStatus = s; });
    // one scroll pass so pinning / lazy scenes actually initialise before we look
    await page.evaluate(() => window.scrollTo({ top: innerHeight * 1.6, behavior: 'instant' }));
    await page.waitForTimeout(2500);

    const dom = await page.evaluate(() => {
      const cs = getComputedStyle;
      // A blocked / challenge / download navigation can leave us with no body at all.
      // Bail with an explicitly empty reading rather than throwing away the whole site.
      if (!document.body) return { title: (document.title || '').slice(0, 90), sheets: 0, readyState: document.readyState, globals: [], htmlClass: '', pinSpacers: 0, stickies: 0, canvases: 0, webgl: false, videos: [], mixBlend: 0, cursorNone: false, cssTimeline: false, fontFaces: [], display: null, bodyFont: '', bg: '', palette: [], sections: 0, scrollH: 0, vh: window.innerHeight, framework: 'unknown' };
      const all = [...document.querySelectorAll('*')].slice(0, 4000);
      const canvases = [...document.querySelectorAll('canvas')];
      let cssTimeline = false;
      const faces = new Set();
      for (const ss of document.styleSheets) {
        try {
          for (const r of ss.cssRules) {
            const t = r.cssText || '';
            if (t.includes('animation-timeline') || t.includes('view-timeline')) cssTimeline = true;
            if (r.style && r.constructor.name === 'CSSFontFaceRule') faces.add((r.style.fontFamily || '').replace(/["']/g, ''));
          }
        } catch {}
      }
      // Chromium reports modern colour syntaxes verbatim (`lab(48.5 0 0)`, `oklch(...)`), which is
      // unreadable as a palette. Round-trip every value through a canvas so the report shows a
      // colour a human can picture, keeping the original alongside.
      const cx = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
      const toHex = v => {
        try {
          cx.clearRect(0, 0, 1, 1); cx.fillStyle = '#000'; cx.fillStyle = v;
          cx.fillRect(0, 0, 1, 1);
          const [r, g, b, a] = cx.getImageData(0, 0, 1, 1).data;
          const hex = '#' + [r, g, b].map(n => n.toString(16).padStart(2, '0')).join('');
          return a < 250 ? `${hex}@${(a / 255).toFixed(2)}` : hex;
        } catch { return null; }
      };
      // sample the real palette from what is painted, not from the token file
      const colors = {};
      for (const el of all) {
        const s = cs(el);
        for (const c of [s.backgroundColor, s.color]) {
          if (!c || c === 'rgba(0, 0, 0, 0)') continue;
          colors[c] = (colors[c] || 0) + 1;
        }
      }
      const palette = Object.entries(colors).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([c, n]) => {
        const hex = toHex(c);
        return hex && !/^(#|rgb)/.test(c) ? `${hex} (${c}) ×${n}` : `${hex || c} ×${n}`;
      });
      // The display face is the biggest thing actually painted, not the first h1 — a hidden or
      // fallback-styled heading reports whatever the CSS cascade left there and will happily
      // claim an awwwards winner ships Inter. Measure instead: largest visible rendered text.
      let display = null, bestPx = 0;
      for (const e of all) {
        if (e.children.length || (e.textContent || '').trim().length < 2) continue;
        const s = cs(e);
        if (s.visibility === 'hidden' || s.display === 'none' || +s.opacity === 0) continue;
        const r = e.getBoundingClientRect();
        if (r.width < 4 || r.height < 4) continue;
        const px = parseFloat(s.fontSize) || 0;
        if (px > bestPx) { bestPx = px; display = { font: s.fontFamily.slice(0, 60), px: Math.round(px), weight: s.fontWeight, sample: e.textContent.trim().replace(/\s+/g, ' ').slice(0, 28) }; }
      }
      return {
        title: (document.title || '').slice(0, 90),
        sheets: document.styleSheets.length,
        readyState: document.readyState,
        globals: ['gsap', 'ScrollTrigger', 'Lenis', 'THREE', 'locomotiveScroll', 'barba', 'Swup', 'PIXI'].filter(k => k in window),
        htmlClass: document.documentElement.className.slice(0, 100),
        pinSpacers: document.querySelectorAll('.pin-spacer').length,
        stickies: all.filter(e => cs(e).position === 'sticky').length,
        canvases: canvases.length,
        webgl: canvases.some(c => { try { return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch { return false; } }),
        videos: [...document.querySelectorAll('video')].slice(0, 4).map(v => ({
          autoplay: v.autoplay, loop: v.loop, muted: v.muted, preload: v.preload,
          src: (v.currentSrc || v.src || '').split('/').pop().slice(0, 40),
        })),
        mixBlend: all.filter(e => cs(e).mixBlendMode !== 'normal').length,
        cursorNone: cs(document.body).cursor === 'none',
        cssTimeline,
        fontFaces: [...faces].filter(Boolean).slice(0, 8),
        display,
        bodyFont: cs(document.body).fontFamily.slice(0, 70),
        bg: cs(document.body).backgroundColor,
        palette,
        sections: document.querySelectorAll('section, main > div, [class*="section"]').length,
        scrollH: document.documentElement.scrollHeight,
        vh: window.innerHeight,
        framework:
          document.querySelector('[data-framer-name],[data-framer-root]') ? 'Framer' :
          document.querySelector('[data-wf-page]') ? 'Webflow' :
          document.querySelector('#__next,[id^=__next]') ? 'Next.js' :
          document.querySelector('astro-island,[data-astro-cid]') ? 'Astro' :
          document.querySelector('[data-sveltekit-preload-data]') ? 'SvelteKit' :
          document.querySelector('#app,[data-v-app]') ? 'Vue/Nuxt' :
          document.querySelector('[data-shopify],[id^=shopify]') ? 'Shopify' : 'unknown',
      };
    });

    const blob = js.join('\n');
    const libs = LIBS.filter(([, re]) => re.test(blob)).map(([n]) => n);
    for (const g of dom.globals) if (!libs.some(l => l.toLowerCase().includes(g.toLowerCase().slice(0, 5)))) libs.push(`${g} (global)`);

    const f = { url, ...dom, libs, jsBytes, scrollRatio: +(dom.scrollH / Math.max(dom.vh, 1)).toFixed(1) };
    f.mechanics = classifyPeak(f);
    // Some sites (Vercel-edge streamers, bot-walled studios) hand a headless client the SSR
    // HTML and then never deliver CSS or JS. Everything we'd read off that page — fonts,
    // palette, libs — is browser default, i.e. a confident lie. Detect it and say so.
    // Chrome's unstyled body is Times New Roman with zero web fonts loaded; a real site that
    // merely leaves body at the default still has @font-face rules from its stylesheet.
    const unstyled = /^"?Times New Roman/.test(dom.bodyFont) && !dom.fontFaces.length;
    // An error page has a title, a palette and a page shape, and will happily be written up as a
    // reference with a blank `steal:` line waiting to be filled. It is not a design.
    f.httpStatus = httpStatus;
    f.errorPage = httpStatus >= 400 || /^\s*(4\d\d|5\d\d)|bad gateway|not found|forbidden|service unavailable|internal server error/i.test(dom.title || '');
    f.thin = dom.sheets === 0 || unstyled || f.errorPage;

    // Screenshots are best-effort on purpose: a capture that fails must never throw away
    // a fingerprint we already paid for.
    f.shots = [];
    // A site that never got its CSS has nothing to photograph: take one frame as evidence the
    // capture failed, not a full journey of identical unstyled pages.
    const n = f.thin ? Math.min(1, shotsN) : shotsN;
    if (n > 0) {
      await freeze(page, dcl);
      const slug = url.replace(/^https?:\/\//, '').replace(/[^\w.-]+/g, '_').slice(0, 40);
      const max = Math.max(0, dom.scrollH - dom.vh);
      for (let i = 0; i < n; i++) {
        const y = n === 1 ? 0 : Math.round((i / (n - 1)) * max);
        const name = `${slug}-${String(i).padStart(2, '0')}.png`;
        try {
          await page.evaluate(t => window.scrollTo({ top: t, behavior: 'instant' }), y);
          await page.waitForTimeout(1600);
          await page.screenshot({ path: join(outDir, 'shots', name), timeout: 20_000 });
          f.shots.push(`shots/${name}`);
        } catch (e) {
          (f.shotErrors ??= []).push(`stop ${i}: ${e.message.split('\n')[0]}`);
        }
      }
      f.shotsWanted = n;
    }
    return f;
  } catch (e) {
    return { url, error: e.message.split('\n')[0] };
  } finally {
    await page.close();
  }
}

// --- gallery harvesting -------------------------------------------------------
// awwwards is the one gallery whose listing AND detail pages both render reliably headless
// and expose the outbound site URL. Other galleries either need JS we can't wait out or bury
// the real URL behind affiliate redirects — for those, find URLs with WebSearch and pass them
// positionally.
async function harvestAwwwards(ctx, tag, text, n) {
  const page = await ctx.newPage();
  const base = 'https://www.awwwards.com/websites/';
  const url = text ? `${base}?text=${encodeURIComponent(text)}` : tag ? `${base}${tag}/` : base;
  const found = [];
  try {
    await open(page, url, 6000);
    const slugs = await page.evaluate(() =>
      [...new Set([...document.querySelectorAll('a[href*="/sites/"]')].map(a => a.getAttribute('href')).filter(h => h && h.startsWith('/sites/')))]);
    if (!slugs.length) console.error(`[refscout] awwwards listing returned no cards for ${url}`);
    for (const slug of slugs.slice(0, n)) {
      try {
        await open(page, 'https://www.awwwards.com' + slug, 3500);
        const d = await page.evaluate(() => {
          const visit = [...document.querySelectorAll('a[href^="http"]')]
            .find(a => /visit site/i.test(a.innerText) || /button/.test(a.className));
          const ext = visit?.href || [...document.querySelectorAll('a[href^="http"]')]
            .map(a => a.href).find(h => !/awwwards|twitter|x\.com|facebook|instagram|linkedin|behance|dribbble/i.test(h));
          return {
            site: ext,
            name: (document.querySelector('h1')?.innerText || document.title).trim().slice(0, 60),
          };
        });
        if (d.site) found.push({ ...d, via: 'awwwards' + slug });
      } catch {}
    }
  } catch (e) {
    console.error(`[refscout] awwwards harvest failed: ${e.message.split('\n')[0]}`);
  } finally { await page.close(); }
  return found;
}

// --- run ----------------------------------------------------------------------
await mkdir(join(outDir, 'shots'), { recursive: true });
const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ userAgent: UA, viewport: { width, height: 900 }, ignoreHTTPSErrors: true });

let targets = urls.map(u => ({ site: u, name: null, via: 'direct' }));
if (from) {
  const [gallery, tag] = from.split(':');
  if (gallery === 'awwwards') targets = targets.concat(await harvestAwwwards(ctx, tag, search, limit));
  else console.error(`[refscout] unknown gallery "${gallery}" — supported: awwwards. Pass URLs directly instead (find them with WebSearch).`);
}
targets = targets.slice(0, limit);

if (!targets.length) {
  console.error('[refscout] no targets. Pass URLs, or --from awwwards.');
  await browser.close();
  process.exit(1);
}

const results = [];
for (const t of targets) {
  process.stderr.write(`[refscout] profiling ${t.site} …\n`);
  const f = await fingerprint(ctx, t.site);
  results.push({ ...t, ...f });
}
await browser.close();

// --- report -------------------------------------------------------------------
const ok = results.filter(r => !r.error);
const lines = [];
lines.push('# REFERENCES — scouted ' + new Date().toISOString().slice(0, 10));
lines.push('');
lines.push('Mechanics, not skins. For each reference name the ONE idea you are taking and how it');
lines.push('changes for this brand. Copying a reference wholesale is slop with better taste.');
lines.push('');
lines.push('| # | site | stack detected | page shape | mechanics |');
lines.push('|---|---|---|---|---|');
ok.forEach((r, i) => {
  const cell = r.errorPage ? '_error page — not a design_' : r.thin ? '_no capture — see below_' : null;
  lines.push(`| ${i + 1} | [${r.name || r.url.replace(/^https?:\/\//, '')}](${r.url}) | ${cell ?? (r.libs.slice(0, 4).join(', ') || '—')} | ${cell ?? `${r.scrollRatio}× vh, ${r.sections} sections`} | ${cell ?? (r.mechanics.slice(0, 3).join('; ') || '—')} |`);
});
lines.push('');

ok.forEach((r, i) => {
  lines.push(`## ${i + 1}. ${r.name || r.title || r.url}`);
  lines.push(`- url: ${r.url}${r.via && r.via !== 'direct' ? `  ·  via ${r.via}` : ''}`);
  if (r.errorPage) {
    lines.push(`- ⚠ **NOT A DESIGN** — this URL returned an error page${r.httpStatus >= 400 ? ` (HTTP ${r.httpStatus})` : ''}, title "${r.title}". Nothing here is a reference. Drop it, or find the site's real address.`);
    lines.push('');
    return;
  }
  if (r.thin) {
    lines.push(`- ⚠ **NO CAPTURE** — this site served the headless browser bare HTML and never delivered its CSS/JS (${r.sheets} stylesheets, readyState \`${r.readyState}\`). Any font, colour or stack reading here would be the browser's defaults, so none is reported. Open the URL yourself, or describe it from the gallery write-up — and check the page title above, a dead link from the gallery lands here too.`);
    if (r.shots.length) lines.push(`- unstyled shots (evidence that the capture failed, not a look at the design): ${r.shots.map(s => `\`${s}\``).join(' ')}`);
    lines.push('');
    return;
  }
  lines.push(`- stack: ${r.libs.join(', ') || 'nothing detected'}${r.framework !== 'unknown' ? `  ·  built with ${r.framework}` : ''}`);
  lines.push(`- mechanics: ${r.mechanics.join('; ') || '—'}`);
  lines.push(`- page: ${r.scrollRatio}× viewport tall · ${r.sections} sections · ${r.canvases} canvas${r.webgl ? ' (WebGL live)' : ''} · ${r.videos.length} video`);
  lines.push(`- type: display \`${r.display ? `${r.display.font} ${r.display.px}px/${r.display.weight}` : '?'}\`${r.display ? ` (largest painted text: "${r.display.sample}")` : ''} · body \`${r.bodyFont}\`${r.fontFaces.length ? ` · @font-face: ${r.fontFaces.join(', ')}` : ''}`);
  lines.push(`- palette (most painted): ${r.palette.join(' · ')}`);
  if (r.shots.length) lines.push(`- shots: ${r.shots.map(s => `\`${s}\``).join(' ')}  ← look at the hero; look at the rest only if you take a steal from this one`);
  if (r.shotErrors?.length) lines.push(`- ⚠ ${r.shotErrors.length} of ${r.shotsWanted} shots failed (${r.shotErrors.join('; ')}) — the fingerprint above still holds.`);
  lines.push('- **steal:** _<one mechanic, one line — fill this in>_');
  lines.push('');
});

const failed = results.filter(r => r.error);
if (failed.length) {
  lines.push('## Not reached');
  failed.forEach(r => lines.push(`- ${r.url} — ${r.error}`));
  lines.push('');
}

await writeFile(join(outDir, 'REFERENCES.md'), lines.join('\n'), 'utf8');
await writeFile(join(outDir, 'refs.json'), JSON.stringify(results, null, 2), 'utf8');

console.log(`\n=== refscout ===`);
console.log(`profiled : ${ok.filter(r => !r.thin).length}/${results.length} usable`);
if (ok.some(r => r.thin)) console.log(`no-css   : ${ok.filter(r => r.thin).length} (site withheld CSS/JS from headless — reported as unknown, not guessed)`);
if (failed.length) console.log(`failed   : ${failed.length} — ${failed.map(r => r.url).join(', ')}`);
const lostShots = ok.reduce((n, r) => n + (r.shotErrors?.length || 0), 0);
if (lostShots) console.log(`shots    : ${lostShots} capture(s) failed — flagged per site in the report`);
console.log(`report   : ${join(outDir, 'REFERENCES.md')}`);
console.log(`shots    : ${ok.reduce((n, r) => n + r.shots.length, 0)} in ${join(outDir, 'shots')}`);
process.exit(ok.length ? 0 : 1);
