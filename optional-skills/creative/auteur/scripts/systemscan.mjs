#!/usr/bin/env node
/**
 * systemscan.mjs — the multi-screen gate: does this product still have ONE design system?
 *
 * slopscan reads source and catches defaults. shoot.mjs photographs one page. Neither can see the
 * failure mode of a real app: drift. Screen 1 has three button variants, screen 7 invents a fourth,
 * and nobody notices because every screen looks fine on its own. This crawls every route, reads what
 * the browser ACTUALLY PAINTED, and reports the system as built rather than as documented.
 *
 * Usage:
 *   node scripts/systemscan.mjs <url> [<url>...]
 *   node scripts/systemscan.mjs http://localhost:3000 --routes /,/settings,/billing
 *
 * Options:
 *   --routes a,b,c   paths to append to the first url (instead of listing full urls)
 *   --out DIR        output dir (default design/system)
 *   --max-variants B  variant budget: a number, or per kind — `button=5,select=1,4` (default 4)
 *   --width N        viewport width (default 1440)
 *   --no-shots       skip the component contact sheet
 *
 * Writes DIR/SYSTEM-REPORT.md (read it), DIR/system.json, DIR/components.png.
 * Exit 1 when the system is provably broken: a control type over budget, an interactive element with
 * no visible focus state, or a token used on exactly one route.
 */

import { mkdir, writeFile } from 'fs/promises';
import { resolve, join } from 'path';

const args = process.argv.slice(2);
if (!args.length || args.includes('--help')) {
  console.log(`systemscan.mjs — cross-route design-system drift gate

  node scripts/systemscan.mjs <url> [<url>...]
  node scripts/systemscan.mjs http://localhost:3000 --routes /,/settings,/billing

  --out DIR  --max-variants 4|button=5,select=1  --width 1440  --no-shots`);
  process.exit(0);
}
const get = (f, d) => { const i = args.indexOf(f); return i !== -1 ? args[i + 1] : d; };
const has = f => args.includes(f);

const outDir = resolve(get('--out', 'design/system'));
// Per-kind budgets: `--max-variants button=5,select=1` (a bare number sets the default for the rest).
// One global number meant a sheet declaring button:5 and select:1 had to pass 5, leaving every
// smaller kind unpoliced — the sheet's central promise enforced for exactly one control kind.
const rawBudget = get('--max-variants', '4');
const budget = { default: 4, byKind: {} };
for (const part of String(rawBudget).split(',').map(s => s.trim()).filter(Boolean)) {
  const m = part.match(/^([a-z]+)\s*=\s*(\d+)$/i);
  if (m) budget.byKind[m[1].toLowerCase()] = +m[2];
  else if (/^\d+$/.test(part)) budget.default = +part;
  else console.error(`[systemscan] ignoring unparseable budget "${part}" — use 4 or button=5,select=1`);
}
const budgetFor = kind => budget.byKind[kind] ?? budget.default;
const width = parseInt(get('--width', '1440'), 10) || 1440;
const shots = !has('--no-shots');

let urls = args.filter(a => /^https?:\/\//.test(a) || a.startsWith('file://'));
const routes = get('--routes', null);
if (routes && urls.length) {
  const base = urls[0].replace(/\/$/, '');
  urls = routes.split(',').map(r => base + (r.startsWith('/') ? r : '/' + r));
}
if (!urls.length) { console.error('[systemscan] no urls. Pass them, or a base url + --routes /a,/b'); process.exit(1); }

let chromium;
try { ({ chromium } = await import('playwright')); }
catch { console.error('playwright not found. Install: npm i -D playwright && npx playwright install chromium'); process.exit(2); }

// Everything below is read off getComputedStyle, i.e. the system as PAINTED. A token that exists in
// the stylesheet but is never rendered is not part of the system; a one-off inline style is.
const COLLECT = () => {
  const cs = getComputedStyle;
  const px = v => Math.round(parseFloat(v) || 0);
  const vis = el => {
    const s = cs(el);
    if (s.display === 'none' || s.visibility === 'hidden' || +s.opacity === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };
  const all = [...document.querySelectorAll('*')].slice(0, 6000).filter(vis);

  const controlKind = el => {
    const t = el.tagName.toLowerCase();
    const role = el.getAttribute('role');
    if (t === 'button' || role === 'button') return 'button';
    if (t === 'a' && /(^|\s)(btn|button)/i.test(el.className || '')) return 'button';
    if (t === 'a') return 'link';
    if (t === 'input') return ['checkbox', 'radio'].includes(el.type) ? 'toggle' : 'input';
    if (t === 'select') return 'select';
    if (t === 'textarea') return 'input';
    if (role === 'tab') return 'tab';
    return null;
  };

  // The signature is what a user can SEE. Two buttons with different class names but identical
  // paint are one variant; two with the same class and different paint are two.
  const sig = el => {
    const s = cs(el);
    return [
      s.backgroundColor, s.color, s.borderColor,
      `${px(s.borderTopWidth)}b`, `${px(s.borderTopLeftRadius)}r`,
      `${px(s.fontSize)}/${s.fontWeight}`,
      `${px(s.paddingTop)}x${px(s.paddingLeft)}`,
      s.boxShadow === 'none' ? 'noshadow' : 'shadow',
    ].join(' | ');
  };

  // A STATE is not a variant. A disabled secondary button paints differently from an enabled one by
  // design — that is the state matrix system.md demands, and counting it as a fifth button punishes
  // the team that built it while a system with no disabled state at all sails through. Same for a
  // control that inverts because the row it sits in is in an alert state: the component did not
  // multiply, its container changed colour underneath it. Both are counted and reported separately.
  const stateOf = el => {
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') return 'disabled';
    if (el.getAttribute('aria-current') || el.getAttribute('aria-selected') === 'true') return 'current';
    for (let n = el.parentElement, hops = 0; n && hops < 4; n = n.parentElement, hops++) {
      const st = n.getAttribute?.('data-state');
      if (st && st !== 'default') return `in-${st}`;
    }
    return null;
  };

  const controls = {};
  const states = {};
  let shotId = 0;
  for (const el of all) {
    const kind = controlKind(el);
    if (!kind) continue;
    const state = stateOf(el);
    if (state) { (states[kind] ??= {})[state] = ((states[kind] ??= {})[state] || 0) + 1; continue; }
    const k = sig(el);
    (controls[kind] ??= {});
    if (!controls[kind][k]) {
      // Tag the exemplar NOW and re-select it by attribute in the screenshot pass. Handing a
      // positional index between two independent DOM walks produced tiles that showed a parent or a
      // sibling rather than the control they were captioned with — which destroys the sheet's whole
      // purpose, because a tile showing the wrong element looks different for the wrong reason.
      const tag = `ss${shotId++}`;
      el.setAttribute('data-ss-shot', tag);
      controls[kind][k] = { count: 0, sample: (el.innerText || el.value || el.type || '').trim().slice(0, 24), tag };
    }
    controls[kind][k].count++;
  }

  const tally = (map, key) => { map[key] = (map[key] || 0) + 1; };
  const colors = {}, type = {}, radii = {}, shadows = {}, space = {};
  for (const el of all) {
    const s = cs(el);
    if (s.backgroundColor && s.backgroundColor !== 'rgba(0, 0, 0, 0)') tally(colors, s.backgroundColor);
    // `color` on a wrapper paints no glyph. Tallying it unfiltered made <html>'s inherited initial
    // colour the most common "colour in the product" and inflates the headline count for everyone.
    const leafText = el.children.length === 0 && (el.textContent || '').trim();
    if (s.color && leafText) tally(colors, s.color);
    if (leafText) {
      tally(type, `${px(s.fontSize)}px/${s.fontWeight}/${s.fontFamily.split(',')[0].replace(/["']/g, '')}`);
    }
    const r = px(s.borderTopLeftRadius); if (r) tally(radii, `${r}px`);
    if (s.boxShadow && s.boxShadow !== 'none') tally(shadows, s.boxShadow.slice(0, 60));
    for (const v of [s.paddingTop, s.paddingLeft, s.marginTop]) { const n = px(v); if (n) tally(space, `${n}px`); }
  }

  return {
    controls, states, colors, type, radii, shadows, space,
    focusable: all.filter(el => el.matches('a[href],button,input,select,textarea,[tabindex]:not([tabindex="-1"])')).length,
    hasDisabled: all.some(el => el.matches('[disabled],[aria-disabled="true"]')),
    title: document.title.slice(0, 60),
    landmarks: ['header', 'nav', 'main', 'footer', 'aside'].filter(t => document.querySelector(t)).join(','),
    h1: document.querySelectorAll('h1').length,
  };
};

/**
 * Focus state cannot be read from a stylesheet: a `:focus-visible` rule may exist and be overridden,
 * and `:focus-visible` itself is a heuristic that programmatic `.focus()` does not reliably trigger.
 * So drive the real thing — press Tab and look at what the browser actually paints. Tabbing also
 * skips disabled and unfocusable elements for free, which a `.focus()` loop reports as failures.
 */
const PROBE_FOCUS = async (page, maxStops = 40) => {
  const paint = () => {
    const cs = getComputedStyle;
    // Only properties that actually PAINT. outline-offset alone draws nothing, and including it
    // let an element that kills its outline still read as "focus state changed".
    const sig = el => { const s = cs(el); return `${s.outlineStyle === 'none' ? 'no-outline' : s.outline} ${s.boxShadow} ${s.borderColor} ${s.backgroundColor} ${s.color} ${s.textDecorationLine}`; };
    const els = [...document.querySelectorAll('a[href],button,input,select,textarea,summary,[tabindex]:not([tabindex="-1"])')];
    els.forEach((el, i) => el.setAttribute('data-ss-i', String(i)));
    return els.map(sig);
  };
  const unfocused = await page.evaluate(paint);
  await page.evaluate(() => (document.activeElement || document.body).blur?.());

  const bad = [], seen = new Set();
  for (let i = 0; i < maxStops; i++) {
    await page.keyboard.press('Tab');
    const stop = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el || el === document.body || !el.hasAttribute?.('data-ss-i')) return null;
      const s = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      return {
        i: +el.getAttribute('data-ss-i'),
        sig: `${s.outlineStyle === 'none' ? 'no-outline' : s.outline} ${s.boxShadow} ${s.borderColor} ${s.backgroundColor} ${s.color} ${s.textDecorationLine}`,
        label: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + ' “' + (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().slice(0, 20) + '”',
        offscreen: r.width < 1 || r.height < 1,
      };
    });
    if (!stop) continue;
    if (seen.has(stop.i)) break;                 // wrapped around the tab ring
    seen.add(stop.i);
    if (!stop.offscreen && unfocused[stop.i] === stop.sig) bad.push(stop.label);
  }
  return { probed: seen.size, noFocusRing: bad };
};

const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width, height: 900 } });
await mkdir(outDir, { recursive: true });

const perRoute = [];
for (const url of urls) {
  const page = await ctx.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  page.on('console', m => m.type() === 'error' && errs.push(m.text()));
  try {
    process.stderr.write(`[systemscan] ${url}\n`);
    const resp = await page.goto(url, { waitUntil: 'commit', timeout: 30_000 });
    const dcl = await page.waitForLoadState('domcontentloaded', { timeout: 15_000 }).then(() => true).catch(() => false);
    await page.waitForTimeout(dcl ? 2500 : 6000);
    if (!dcl) await page.evaluate(() => window.stop()).catch(() => {});
    const data = await page.evaluate(COLLECT);
    const focus = await PROBE_FOCUS(page);
    const status = resp?.status?.() ?? 0;
    // A route that renders nothing contributes nothing to the counts, so a mistyped route list made
    // the gate QUIETER instead of louder. A gate that goes green on a 404 is worse than no gate.
    const empty = !data.landmarks && !data.h1 && !data.focusable;
    perRoute.push({ url, ...data, focus, errors: errs, status, empty });
    if (status >= 400) console.error(`[systemscan] ${url} → HTTP ${status}`);
    else if (empty) console.error(`[systemscan] ${url} → rendered nothing measurable (no landmark, no h1, no focusable)`);

    if (shots) {
      // one exemplar screenshot per distinct control variant, so the report has a picture of the
      // drift and not only a count of it
      for (const [kind, variants] of Object.entries(data.controls)) {
        let i = 0;
        for (const [, v] of Object.entries(variants)) {
          if (!v.tag) continue;
          const name = `${kind}-${perRoute.length}-${i++}.png`;
          try {
            const box = await page.$(`[data-ss-shot="${v.tag}"]`);   // the element we actually measured
            if (box) { await box.scrollIntoViewIfNeeded({ timeout: 4000 }); await box.screenshot({ path: join(outDir, 'components', name), timeout: 8000 }); v.shot = `components/${name}`; }
          } catch { /* an element that will not sit still is not worth failing the run over */ }
        }
      }
    }
  } catch (e) {
    perRoute.push({ url, error: e.message.split('\n')[0] });
  } finally { await page.close(); }
}

// ---- aggregate across routes -------------------------------------------------
const ok = perRoute.filter(r => !r.error);
if (!ok.length) { console.error('[systemscan] no route could be read.'); await browser.close(); process.exit(1); }

// A document scanned twice (`/settings` and `/settings#state-error` are the same DOM) must not count
// twice — summing across URLs meant scanning MORE thoroughly hid genuine one-off variants, which is
// the opposite of what `system.md` tells you to do. Count per document, take the max, then sum.
const docOf = u => u.split('#')[0];
const docs = [...new Set(ok.map(r => docOf(r.url)))];

const mergeCount = key => {
  const m = {};
  for (const r of ok) for (const [k, n] of Object.entries(r[key] || {})) {
    (m[k] ??= { perDoc: {} });
    const d = docOf(r.url);
    m[k].perDoc[d] = Math.max(m[k].perDoc[d] || 0, n);
  }
  return Object.entries(m)
    .map(([k, v]) => ({ k, total: Object.values(v.perDoc).reduce((a, b) => a + b, 0), routes: Object.keys(v.perDoc).length }))
    .sort((a, b) => b.total - a.total);
};

const controlVariants = {};
for (const r of ok) for (const [kind, vars] of Object.entries(r.controls || {})) {
  (controlVariants[kind] ??= {});
  for (const [sig, v] of Object.entries(vars)) {
    const slot = (controlVariants[kind][sig] ??= { perDoc: {}, sample: v.sample, shot: v.shot });
    const d = docOf(r.url);
    slot.perDoc[d] = Math.max(slot.perDoc[d] || 0, v.count);
    slot.shot ??= v.shot;
  }
}
for (const vars of Object.values(controlVariants)) for (const v of Object.values(vars)) {
  v.count = Object.values(v.perDoc).reduce((a, b) => a + b, 0);
  v.routes = new Set(Object.keys(v.perDoc));
}

// States, gathered the same way but never counted against the variant budget.
const controlStates = {};
for (const r of ok) for (const [kind, st] of Object.entries(r.states || {}))
  for (const [name, n] of Object.entries(st)) {
    ((controlStates[kind] ??= {})[name] ??= 0);
    controlStates[kind][name] += n;
  }

const fails = [], warns = [];
for (const [kind, vars] of Object.entries(controlVariants)) {
  const n = Object.keys(vars).length, b = budgetFor(kind);
  if (n > b) fails.push(`${kind}: ${n} distinct rendered variants (budget ${b}). A variant nobody can name is drift.`);
  for (const [sig, v] of Object.entries(vars)) {
    if (v.count === 1 && n > 1) warns.push(`${kind} variant used exactly once ("${v.sample}") — either promote it into the system or delete it: ${sig}`);
  }
}
const blank = perRoute.filter(r => !r.error && (r.status >= 400 || r.empty));
if (blank.length) fails.push(`${blank.length} route(s) rendered nothing measurable or returned an error status — a gate that goes green on a 404 is worse than no gate: ${blank.map(r => `${r.url}${r.status >= 400 ? ` (HTTP ${r.status})` : ''}`).join(', ')}`);
const noFocus = ok.flatMap(r => (r.focus?.noFocusRing || []).map(t => `${r.url} → ${t}`));
if (noFocus.length) fails.push(`${noFocus.length} interactive element(s) paint identically when focused — keyboard users cannot see where they are`);

const colors = mergeCount('colors'), type = mergeCount('type'), radii = mergeCount('radii'), shadows = mergeCount('shadows');
if (docs.length > 1) {
  for (const [label, list] of [['colour', colors], ['type step', type], ['radius', radii]]) {
    const singles = list.filter(x => x.routes === 1 && x.total >= 3);
    if (singles.length) warns.push(`${singles.length} ${label}(s) appear on exactly one route and nowhere else — that is where the system is splitting: ${singles.slice(0, 4).map(s => s.k).join(' · ')}`);
  }
}
if (type.length > 12) warns.push(`${type.length} distinct type steps across the product — a scale nobody can hold in their head is not a scale`);
if (!ok.some(r => r.hasDisabled)) warns.push('no disabled control appeared on any route — the disabled state is probably undesigned, not absent');

// ---- component contact sheet -------------------------------------------------
let sheet = null;
const tiles = Object.entries(controlVariants).flatMap(([kind, vars]) =>
  Object.entries(vars).filter(([, v]) => v.shot).map(([, v], i) => ({ kind, i, ...v })));
if (shots && tiles.length) {
  const html = `<!doctype html><meta charset="utf-8"><style>
    body{margin:0;background:#141414;font:12px/1.3 ui-monospace,monospace;color:#ddd}
    .g{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:12px}
    figure{margin:0;background:#1e1e1e;border-radius:4px;overflow:hidden;padding:10px}
    img{display:block;max-width:100%;margin:0 auto 8px}
    b{color:#fff}
  </style><div class=g>${tiles.map(t =>
    `<figure><img src="${t.shot}"><figcaption><b>${t.kind}</b> ×${t.count} · ${t.routes.size} route(s)</figcaption></figure>`).join('')}</div>`;
  const f = join(outDir, '_components.html');
  await writeFile(f, html, 'utf8');
  const p = await ctx.newPage();
  await p.setViewportSize({ width: 1200, height: 900 });
  await p.goto('file:///' + f.replace(/\\/g, '/'), { waitUntil: 'load', timeout: 20_000 }).catch(() => {});
  await p.waitForTimeout(900);
  // shoot the grid, not the viewport — a fullPage shot of a two-row sheet is mostly empty canvas
  const grid = await p.$('.g');
  await (grid || p).screenshot({ path: join(outDir, 'components.png') });
  await p.close();
  sheet = 'components.png';
}
await browser.close();

// ---- report ------------------------------------------------------------------
const L = [];
L.push(`# SYSTEM-REPORT — ${ok.length} route(s), ${new Date().toISOString().slice(0, 10)}`);
L.push('');
L.push('The system as **painted**, not as documented. A token in the stylesheet that never renders is');
L.push('not part of the system; a one-off inline style is. Read this against `design/DESIGN.md` — every');
L.push('number below that DESIGN.md does not account for is drift.');
L.push('');
if (sheet) L.push(`**Look at \`${sheet}\`** — one tile per distinct rendered control variant. Two tiles that look the same to you but appear separately are the drift.\n`);

L.push('## Controls');
L.push('');
L.push('| kind | distinct variants | budget | total instances |');
L.push('|---|---|---|---|');
for (const [kind, vars] of Object.entries(controlVariants)) {
  const n = Object.keys(vars).length;
  L.push(`| ${kind} | **${n}** | ${budgetFor(kind)} | ${Object.values(vars).reduce((s, v) => s + v.count, 0)} |`);
}
L.push('');
for (const [kind, vars] of Object.entries(controlVariants)) {
  L.push(`### ${kind}`);
  for (const [sig, v] of Object.entries(vars)) L.push(`- ×${v.count} on ${v.routes.size} route(s) — "${v.sample}" — \`${sig}\``);
  if (controlStates[kind]) {
    const st = Object.entries(controlStates[kind]).map(([n, c]) => `${n} ×${c}`).join(' · ');
    L.push(`- *states (not counted as variants): ${st}*`);
  }
  L.push('');
}
if (Object.keys(controlStates).length) {
  L.push('> States — disabled, current, and controls inside a row carrying a `data-state` — are');
  L.push('> excluded from the variant budget. A disabled button paints differently on purpose; a');
  L.push('> product that has no disabled state at all should not score better than one that does.');
  L.push('');
}

L.push('## Tokens as rendered');
L.push('');
for (const [label, list] of [['Colour', colors], ['Type step', type], ['Radius', radii], ['Shadow', shadows]]) {
  L.push(`**${label}** (${list.length} distinct)`);
  for (const x of list.slice(0, 10)) L.push(`- \`${x.k}\` — ${x.total}× on ${x.routes} route(s)`);
  if (list.length > 10) L.push(`- …and ${list.length - 10} more`);
  L.push('');
}

L.push('## Per route');
L.push('');
L.push('| route | landmarks | h1 | focusable | console errors |');
L.push('|---|---|---|---|---|');
for (const r of ok) L.push(`| ${r.url} | ${r.landmarks || '—'} | ${r.h1} | ${r.focusable} | ${r.errors.length} |`);
for (const r of perRoute.filter(r => r.error)) L.push(`| ${r.url} | **unreachable** — ${r.error} | | | |`);
L.push('');

if (fails.length) { L.push('## FAIL'); fails.forEach(f => L.push(`- ${f}`)); L.push(''); }
if (noFocus.length) { L.push('### Elements with no visible focus state'); noFocus.slice(0, 20).forEach(t => L.push(`- ${t}`)); L.push(''); }
if (warns.length) { L.push('## WARN'); warns.forEach(w => L.push(`- ${w}`)); L.push(''); }
if (!fails.length && !warns.length) L.push('No drift detected. The product renders one system.\n');

await writeFile(join(outDir, 'SYSTEM-REPORT.md'), L.join('\n'), 'utf8');
await writeFile(join(outDir, 'system.json'), JSON.stringify({
  routes: perRoute.map(r => ({ ...r, controls: undefined })),
  controls: Object.fromEntries(Object.entries(controlVariants).map(([k, v]) =>
    [k, Object.entries(v).map(([sig, x]) => ({ sig, count: x.count, routes: [...x.routes], sample: x.sample }))])),
  colors, type, radii, shadows, fails, warns,
}, null, 2), 'utf8');

console.log(`\n=== systemscan ===`);
console.log(`routes  : ${ok.length}/${perRoute.length} read (${docs.length} distinct document(s))`);
console.log(`controls: ${Object.entries(controlVariants).map(([k, v]) => `${k} ${Object.keys(v).length}`).join(' · ') || '—'}`);
console.log(`tokens  : ${colors.length} colours · ${type.length} type steps · ${radii.length} radii · ${shadows.length} shadows`);
console.log(`report  : ${join(outDir, 'SYSTEM-REPORT.md')}${sheet ? `\nsheet   : ${join(outDir, sheet)}` : ''}`);
for (const f of fails) console.log(`FAIL ${f}`);
for (const w of warns.slice(0, 5)) console.log(`WARN ${w}`);
process.exit(fails.length ? 1 : 0);
