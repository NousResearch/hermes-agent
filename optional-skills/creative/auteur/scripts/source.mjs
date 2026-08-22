#!/usr/bin/env node
/**
 * source.mjs — find and download licence-clean assets you don't have to generate.
 *
 * Generation is not always the right tool. You cannot generate a glTF mesh, a 16-bit HDRI
 * environment map, or a correctly-hinted variable font — and CC0 versions of all three exist at
 * production quality. This fetches those, records the licence for every file, and refuses to
 * hand you anything whose terms it can't state.
 *
 * Usage:
 *   node scripts/source.mjs <kind> "<query>" [options]
 *
 *   kind = hdri | model | texture | icon | font | image | video
 *
 * Options:
 *   --limit N      how many assets (default 5, cap 20)
 *   --out DIR      output dir (default assets/sourced)
 *   --res 1k|2k|4k|8k   resolution for hdri/model/texture (default 2k)
 *   --list         print a shortlist to stdout; download nothing and write no ledger
 *
 * Writes DIR/ASSETS-SOURCED.md — the licence ledger — on every real fetch. Read it before you ship.
 *
 * Sources (all no-key, verified live 2026-08):
 *   hdri/model/texture  Poly Haven      CC0        no attribution required
 *   icon                Iconify         per set    licence reported per icon (MIT/Apache/CC-BY/OFL)
 *   font                Google Fonts    OFL/Apache free for commercial use, no attribution in UI
 *   image               Openverse       CC-BY/BY-SA ATTRIBUTION REQUIRED — ledger carries the string
 *   video               Coverr          free-use    no redistribution; see the taste warning below
 *
 * The video caveat is not legal boilerplate. Stock footage is generic by construction — the same
 * drone shot is on three thousand landing pages. Auteur's entire thesis is committed, specific
 * assets, so sourced video is allowed as an ambient loop, a texture, or a fallback, and is never
 * the peak scene. If the wow moment is stock, there is no wow moment.
 */

import { mkdir, writeFile, readFile } from 'fs/promises';
import { existsSync } from 'fs';
import { resolve, join, dirname } from 'path';

const args = process.argv.slice(2);
const KINDS = ['hdri', 'model', 'texture', 'icon', 'font', 'image', 'video'];
if (!args.length || args.includes('--help') || !KINDS.includes(args[0])) {
  console.log(`source.mjs — licence-clean asset sourcing

  node scripts/source.mjs <kind> "<query>" [--limit 5] [--out assets/sourced] [--res 2k] [--list]

  kind: ${KINDS.join(' | ')}

  hdri|model|texture  Poly Haven, CC0            — what generation cannot make
  icon                Iconify, licence per set
  font                Google Fonts, OFL/Apache
  image               Openverse, CC — attribution REQUIRED, ledger carries it
  video               Coverr, free-use           — ambient/fallback only, never the peak`);
  process.exit(args.length && !KINDS.includes(args[0]) ? 1 : 0);
}

const kind = args[0];
const get = (f, d) => { const i = args.indexOf(f); return i !== -1 ? args[i + 1] : d; };
const has = f => args.includes(f);
const query = (args[1] && !args[1].startsWith('--') ? args[1] : '').trim();

const limit  = Math.min(parseInt(get('--limit', '5'), 10) || 5, 20);
const outDir = resolve(get('--out', 'assets/sourced'));
const res    = get('--res', '2k');
const listOnly = has('--list');

if (!query) { console.error(`[source] no query. e.g. node scripts/source.mjs ${kind} "coastal dusk"`); process.exit(1); }

const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';
const H = { 'User-Agent': UA, Accept: '*/*' };

async function json(url) {
  const r = await fetch(url, { headers: H, signal: AbortSignal.timeout(45_000) });
  if (!r.ok) throw new Error(`${r.status} ${url.slice(0, 70)}`);
  return r.json();
}
async function text(url) {
  const r = await fetch(url, { headers: H, signal: AbortSignal.timeout(45_000) });
  if (!r.ok) throw new Error(`${r.status} ${url.slice(0, 70)}`);
  return r.text();
}
// Keep pure-digit tokens: Poly Haven names most of its library thing_01 … thing_09, so dropping
// short words made a specific asset unaddressable ("studio small 03" silently returned 09).
const words = q => q.toLowerCase().split(/[\s,]+/).filter(w => w.length > 2 || /^\d+$/.test(w));

/** score a Poly Haven asset against the query across name / tags / categories / attributes */
function phScore(slug, a, ws) {
  const hay = [slug, a.name, ...(a.tags || []), ...(a.categories || []), ...Object.values(a.attributes || {})]
    .join(' ').toLowerCase();
  return ws.reduce((n, w) => n + (hay.includes(w) ? 1 : 0), 0);
}

// ---------------------------------------------------------------- Poly Haven (CC0)
async function polyhaven(type) {
  const all = await json(`https://api.polyhaven.com/assets?t=${type}`);
  const ws = words(query);
  const ranked = Object.entries(all)
    .map(([slug, a]) => ({ slug, a, score: phScore(slug, a, ws) }))
    .filter(x => x.score > 0)
    .sort((x, y) => y.score - x.score || (y.a.download_count || 0) - (x.a.download_count || 0))
    .slice(0, limit);

  if (!ranked.length) { console.error(`[source] Poly Haven has no ${type} matching "${query}". Try broader words (its tags are physical: "coast", "dusk", "concrete", "chair").`); return []; }

  const out = [];
  for (const { slug, a, score } of ranked) {
    const files = await json(`https://api.polyhaven.com/files/${slug}`).catch(() => null);
    if (!files) continue;
    const item = {
      id: slug,
      title: a.name,
      licence: 'CC0',
      attribution: null,                                   // CC0: none required
      credit: Object.keys(a.authors || {}).join(', '),
      landing: `https://polyhaven.com/a/${slug}`,
      note: [a.categories?.join('/'), Object.entries(a.attributes || {}).map(([k, v]) => `${k}=${v}`).join(' ')].filter(Boolean).join(' · '),
      score, files: [],
    };

    if (type === 'hdris') {
      const pick = files.hdri?.[res] || files.hdri?.['2k'] || files.hdri?.['1k'];
      const f = pick?.hdr || pick?.exr;
      if (f) item.files.push({ url: f.url, path: `${slug}_${res}.${f.url.split('.').pop()}`, size: f.size });
    } else if (type === 'models') {
      const g = files.gltf?.[res]?.gltf || files.gltf?.['2k']?.gltf || files.gltf?.['1k']?.gltf;
      if (g) {
        item.files.push({ url: g.url, path: `${slug}/${g.url.split('/').pop()}`, size: g.size });
        // a glTF without its .bin and textures is a broken file, not a smaller one
        for (const [rel, inc] of Object.entries(g.include || {})) item.files.push({ url: inc.url, path: `${slug}/${rel}`, size: inc.size });
      }
    } else {                                                // textures
      const maps = { Diffuse: 'diff', nor_gl: 'nor', Rough: 'rough', arm: 'arm', Displacement: 'disp' };
      for (const [key, tag] of Object.entries(maps)) {
        const f = files[key]?.[res]?.jpg || files[key]?.['2k']?.jpg || files[key]?.['1k']?.jpg;
        if (f) item.files.push({ url: f.url, path: `${slug}/${slug}_${tag}_${res}.jpg`, size: f.size });
      }
    }
    if (item.files.length) out.push(item);
  }
  return out;
}

// ---------------------------------------------------------------- Iconify
async function icons() {
  const collections = await json('https://api.iconify.design/collections').catch(() => ({}));
  // multi-word queries return nothing; the index is single-keyword
  const terms = words(query).length ? words(query) : [query];
  const seen = new Set(), out = [];
  for (const term of terms) {
    if (out.length >= limit) break;
    const r = await json(`https://api.iconify.design/search?query=${encodeURIComponent(term)}&limit=${limit * 4}`).catch(() => null);
    for (const id of r?.icons || []) {
      if (out.length >= limit || seen.has(id)) continue;
      seen.add(id);
      const [set, name] = id.split(':');
      const c = collections[set] || {};
      out.push({
        id, title: `${c.name || set} / ${name}`,
        licence: c.license?.title || 'unknown',
        attribution: /CC.?BY/i.test(c.license?.title || '') ? `Icon "${name}" from ${c.name || set} — ${c.license?.title}` : null,
        credit: c.author?.name || '', landing: c.license?.url || `https://icon-sets.iconify.design/${set}/`,
        note: `set has ${c.total || '?'} icons`,
        files: [{ url: `https://api.iconify.design/${set}/${name}.svg?height=64`, path: `${set}-${name}.svg` }],
      });
    }
  }
  if (!out.length) console.error('[source] Iconify found nothing — its index is single-keyword; try one noun ("bottle", not "whisky bottle").');
  return out;
}

// ---------------------------------------------------------------- Google Fonts
const GF_BANNED = ['Inter', 'Space Grotesk', 'Instrument Serif', 'Playfair Display'];
async function fonts() {
  // Not inside --out: that just relocates 1.5MB from your cwd into your shipped assets tree.
  const cacheFile = join(process.env.TEMP || process.env.TMPDIR || '.', 'auteur-gf-metadata.json');
  let meta;
  if (existsSync(cacheFile)) meta = JSON.parse(await readFile(cacheFile, 'utf8'));
  else {
    meta = JSON.parse((await text('https://fonts.google.com/metadata/fonts')).replace(/^\)\]\}'\n?/, ''));
    // --list promises to write nothing; a 2.6MB metadata cache is still something.
    if (!listOnly) {
      await mkdir(outDir, { recursive: true });
      await writeFile(cacheFile, JSON.stringify(meta));     // fetch it once per project
    }
  }
  // Google's own category names overlap ("Sans Serif" contains "serif"), so a plain substring
  // match on "serif" happily returns Noto Sans Display. Resolve the category first, filter on it,
  // and only then rank the leftover words.
  const CAT = { serif: 'Serif', slab: 'Serif', sans: 'Sans Serif', grotesque: 'Sans Serif', grotesk: 'Sans Serif', mono: 'Monospace', monospace: 'Monospace', display: 'Display', handwriting: 'Handwriting', script: 'Handwriting' };
  const ws = words(query);
  const wantSans = ws.some(w => w === 'sans' || w.startsWith('grotes'));
  const wantCat = wantSans ? 'Sans Serif' : (ws.map(w => CAT[w]).find(Boolean) || null);
  const wantVariable = ws.some(w => w === 'variable' || w === 'vf');
  const rest = ws.filter(w => !CAT[w] && w !== 'variable' && w !== 'vf');

  // "Display", "Mono" and "Sans" are Google categories AND parts of real family names, so a query
  // like "Playfair Display" was read as a category and filtered its own family out. If the category
  // reading finds nothing, the word was part of a name — drop the constraint and search again.
  const inPool = cat => (meta.familyMetadataList || []).filter(f => (!cat || f.category === cat) && (!wantVariable || (f.axes || []).length));
  let pool = inPool(wantCat);
  if (wantCat && !pool.some(f => rest.some(w => f.family.toLowerCase().includes(w)))) {
    const byName = inPool(null).filter(f => rest.some(w => f.family.toLowerCase().includes(w)));
    if (byName.length) pool = byName;
  }
  const ranked = pool
    .map(f => {
      const name = f.family.toLowerCase();
      // A family must actually match something before it is a candidate. Scoring alone did not
      // enforce that — the "keep banned families listed but demoted" filter let every family
      // through, so `font "Bodoni Moda"` downloaded Roboto.
      const hits = rest.reduce((n, w) => n + (name.includes(w) ? 1 : 0), 0);
      const matched = hits > 0 || (wantCat && !rest.length);
      let score = (wantCat ? 1 : 0) + hits * 2;
      if (name === rest.join(' ')) score += 3;               // exact family name wins outright
      if ((f.axes || []).length) score += 0.5;               // variable axes are worth having
      // Google's popularity ranking is exactly what makes a font the AI default; a banned family
      // must never be the top suggestion, but it stays listed (flagged) rather than hidden.
      if (GF_BANNED.includes(f.family)) score -= 100;
      return { f, score, matched };
    })
    .filter(x => x.matched)
    .sort((a, b) => b.score - a.score || (a.f.popularity || 9999) - (b.f.popularity || 9999))
    .slice(0, limit);
  if (!ranked.length) console.error(`[source] no Google font matched. Say the shape: "serif variable", "mono", "display grotesk".`);

  const out = ranked.map(({ f }) => {
    const axes = (f.axes || []).map(a => `${a.tag} ${a.min}–${a.max}`).join(', ');
    const banned = GF_BANNED.includes(f.family);
    const spec = (f.axes || []).find(a => a.tag === 'wght')
      ? `${f.family.replace(/ /g, '+')}:wght@${(f.axes.find(a => a.tag === 'wght')).min}..${(f.axes.find(a => a.tag === 'wght')).max}`
      : f.family.replace(/ /g, '+');
    return {
      id: f.family, title: `${f.family} (${f.category})`,
      licence: 'OFL / Apache-2.0 (Google Fonts)', attribution: null, credit: '',
      landing: `https://fonts.google.com/specimen/${f.family.replace(/ /g, '+')}`,
      note: `${banned ? '⛔ ON THE AUTEUR BAN LIST — pick something else. ' : ''}popularity #${f.popularity ?? '?'} · weights ${Object.keys(f.fonts || {}).length}${axes ? ` · variable: ${axes}` : ' · static only'}`,
      css: `https://fonts.googleapis.com/css2?family=${spec}&display=swap`,
      files: [],                                            // filled below with the real woff2
      family: f.family,
    };
  });

  // Most of what this skill builds must have zero third-party origins, so linking
  // fonts.googleapis.com is not an answer. Resolve the CSS with a browser UA (an old UA gets you
  // ttf), take the latin block, and fetch the actual woff2 so the page can self-host.
  for (const it of out) {
    try {
      const css = await text(it.css);
      // Google emits one @font-face per unicode subset, each preceded by a `/* latin */` comment.
      const latin = (css.match(/\/\*\s*latin\s*\*\/([\s\S]*?)(?=\/\*|$)/) || [, ''])[1] || css;
      const url = (latin.match(/url\((https:\/\/[^)]+\.woff2)\)/) || [])[1];
      if (url) {
        it.files.push({ url, path: `${it.family.replace(/ /g, '')}.woff2` });
        it.selfHost = `@font-face{font-family:'${it.family}';src:url('fonts/${it.family.replace(/ /g, '')}.woff2') format('woff2');font-display:swap;font-weight:${(latin.match(/font-weight:\s*([^;]+)/) || [, '400'])[1].trim()};font-style:normal}`;
      }
    } catch { /* leave it link-only; the ledger will show no file */ }
  }
  return out;
}

// ---------------------------------------------------------------- Openverse (CC)
async function images() {
  // commercial,modification filter is mandatory: the unfiltered index is full of by-nc-nd
  const r = await json(`https://api.openverse.org/v1/images/?q=${encodeURIComponent(query)}&page_size=${limit}&license_type=commercial,modification`);
  return (r.results || []).map(x => ({
    id: x.id, title: x.title || x.id,
    licence: `CC ${(x.license || '').toUpperCase()} ${x.license_version || ''}`.trim(),
    attribution: x.attribution || `"${x.title}" by ${x.creator} is licensed under CC ${(x.license || '').toUpperCase()} ${x.license_version || ''}`,
    credit: x.creator || '', landing: x.foreign_landing_url, note: `${x.width}×${x.height} ${x.filetype || ''} · ${x.provider || ''}`,
    files: [{ url: x.url, path: `${(x.title || x.id).replace(/[^\w-]+/g, '_').slice(0, 40)}.${x.filetype || 'jpg'}` }],
  }));
}

// ---------------------------------------------------------------- stock video (Coverr)
// Mixkit is deliberately not wired in: its search page lists mp4s, but its CDN throttles a
// download to ~15KB in 90s, so every fetch dies on timeout. One working source beats two listed.
const RES_RANK = { '2160p': 4, '1080p': 3, '720p': 2, '360p': 1 };
async function videos() {
  let html;
  try { html = await text(`https://coverr.co/s?q=${encodeURIComponent(query)}`); }
  catch (e) { console.error(`[source] coverr: ${e.message}`); return []; }

  // Keep only the curated library (slug `coverr-…`): `user-ai-generation-…` are user uploads
  // with unclear provenance, and cdn-staging is not a URL to build on.
  const best = new Map();
  for (const u of new Set(html.match(/https:\/\/cdn\.coverr\.co\/videos\/coverr-[^"' ]+\.mp4/g) || [])) {
    const [, slug, res] = u.match(/videos\/([^/]+)\/(\d+p)\.mp4$/) || [];
    if (!slug) continue;
    if (!best.has(slug) || RES_RANK[res] > RES_RANK[best.get(slug).res]) best.set(slug, { u, res });
  }
  const out = [...best.entries()].slice(0, limit).map(([slug, { u, res }]) => ({
    id: slug,
    title: slug.replace(/^coverr-/, '').replace(/-\d+$/, '').replace(/-/g, ' '),
    licence: 'Coverr License — free commercial use, redistribution prohibited',
    attribution: null, credit: 'Coverr',
    landing: `https://coverr.co/videos/${slug}`,
    note: `${res} · ⚠ stock — ambient loop / texture / fallback only, never the peak scene`,
    files: [{ url: u, path: `${slug}-${res}.mp4` }],
  }));
  if (!out.length) console.error('[source] no stock video matched — Coverr indexes simple nouns ("snow", "rain", "city", "smoke").');
  return out;
}

// ---------------------------------------------------------------- run
const fetchers = { hdri: () => polyhaven('hdris'), model: () => polyhaven('models'), texture: () => polyhaven('textures'), icon: icons, font: fonts, image: images, video: videos };
let items = [];
try { items = await fetchers[kind](); }
catch (e) { console.error(`[source] ${kind} lookup failed: ${e.message}`); process.exit(1); }

if (!items.length) { console.error('[source] nothing found.'); process.exit(1); }

// --list is a read loop: print the shortlist and touch nothing. Appending every exploratory
// search to the ledger buried three shipped assets under 260 lines of search history.
if (listOnly) {
  console.log(`
${items.length} match(es) for "${query}" (${kind}):
`);
  for (const it of items) {
    console.log(`  ${it.id}`);
    console.log(`    ${it.title}  ·  ${it.licence}${it.credit ? `  ·  ${it.credit}` : ''}`);
    if (it.note) console.log(`    ${it.note}`);
    if (it.css) console.log(`    css: ${it.css}`);
    console.log(`    ${it.landing}`);
    if (it.files.length) console.log(`    ${it.files.length} file(s), ${Math.round(it.files.reduce((n, f) => n + (f.size || 0), 0) / 1024)}KB`);
    console.log('');
  }
  console.log(`Nothing downloaded and no ledger written (--list). Re-run without --list to fetch.`);
  process.exit(0);
}

await mkdir(outDir, { recursive: true });
let bytes = 0, files = 0;
{
  for (const it of items) {
    for (const f of it.files) {
      const dest = join(outDir, kind, f.path);
      if (existsSync(dest)) { f.skipped = true; continue; }  // never re-download a cached master
      try {
        const r = await fetch(f.url, { headers: { ...H, Referer: it.landing || '' }, signal: AbortSignal.timeout(120_000) });
        if (!r.ok) { f.error = `HTTP ${r.status}`; continue; }
        const buf = Buffer.from(await r.arrayBuffer());
        await mkdir(dirname(dest), { recursive: true });
        await writeFile(dest, buf);
        f.written = buf.length; bytes += buf.length; files++;
      } catch (e) { f.error = e.message.split('\n')[0]; }
    }
    process.stderr.write(`[source] ${it.id} — ${it.files.filter(f => f.written || f.skipped).length}/${it.files.length} files\n`);
  }
}

// ---------------------------------------------------------------- ledger
const ledgerPath = join(outDir, 'ASSETS-SOURCED.md');
const prev = existsSync(ledgerPath) ? await readFile(ledgerPath, 'utf8') : `# ASSETS-SOURCED — licence ledger

Every file below came from someone else. This file is the record of what you may do with it.
**Before shipping:** every entry with an "attribution required" line must have that credit visible
on the site (a credits block in the footer is fine). CC0 and OFL entries need nothing.

Sourced assets are for production use; the moodboard in \`design/\` is not — do not confuse them.
`;
const block = [
  '',
  `## ${kind} — "${query}"  ·  ${new Date().toISOString().slice(0, 10)}${listOnly ? '  (list only, nothing downloaded)' : ''}`,
  '',
  ...items.flatMap(it => {
    const l = [`### ${it.title}`];
    l.push(`- licence: **${it.licence}**${it.credit ? ` · by ${it.credit}` : ''}`);
    if (it.attribution) l.push(`- ⚠ **attribution required** — put this on the page: \`${it.attribution}\``);
    l.push(`- source: ${it.landing}`);
    if (it.note) l.push(`- ${it.note}`);
    if (it.selfHost) l.push(`- self-host (no third-party origin): \`${it.selfHost}\``);
    else if (it.css) l.push(`- css: \`<link rel="stylesheet" href="${it.css}">\`  (woff2 could not be resolved — this links a third-party origin)`);
    for (const f of it.files) l.push(`- file: \`${kind}/${f.path}\`${f.written ? ` (${Math.round(f.written / 1024)}KB)` : f.skipped ? ' (cached)' : f.error ? ` — FAILED ${f.error}` : ' (not downloaded)'}`);
    return [...l, ''];
  }),
].join('\n');
await writeFile(ledgerPath, prev + block, 'utf8');

console.log(`\n=== source (${kind}) ===`);
console.log(`found   : ${items.length} for "${query}"`);
const cached = items.reduce((n, it) => n + it.files.filter(f => f.skipped).length, 0);
if (!listOnly) console.log(`written : ${files} new${cached ? ` + ${cached} already on disk` : ''}, ${(bytes / 1048576).toFixed(1)}MB → ${join(outDir, kind)}`);
const needsCredit = items.filter(i => i.attribution);
if (needsCredit.length) console.log(`⚠ credit: ${needsCredit.length} asset(s) REQUIRE visible attribution — see the ledger`);
if (kind === 'video') console.log(`⚠ stock video is ambient/fallback material. If your peak scene is stock, you have no peak scene.`);
console.log(`ledger  : ${ledgerPath}`);
process.exit(0);
