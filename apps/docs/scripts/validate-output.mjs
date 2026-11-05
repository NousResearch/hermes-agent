#!/usr/bin/env node
// Check the rendered export against its source routes, producer recipes and assets.
// Read-only: no generation, network requests or writes.

import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { parse } from 'parse5';
import { parse as parseYaml } from 'yaml';
import { exportedRoutes, htmlPath, rscPath, redirectEntries, sitemapXml, redirectHtml, redirectScriptAsset } from './postbuild.mjs';
import { LOCALES, SITE_URL, DOCS_BASE_PATH, canonicalUrl, localizedHref, publicHref, splitLocalePath } from '../src/lib/urls.ts';
import { openSearchXml } from './opensearch.mjs';
import { checkExportLinks } from './check-export-links.mjs';
import { readDocumentMetadata, documentSlugs } from '../src/lib/content.mjs';

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OAUTH_CLIENT_ID = 'https://nousresearch.github.io/hermes-agent/docs/oauth/client-metadata.json';
const PLACEHOLDER_HOSTS = /^example\.(?:com|org|net)$/;
const SITE_ORIGIN = new URL(SITE_URL).origin;

function walkFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    return entry.isDirectory() ? walkFiles(full) : [full];
  });
}

function isNonempty(file) {
  const info = statSync(file, { throwIfNoEntry: false });
  return Boolean(info?.isFile() && info.size > 0);
}

function readJson(file) {
  try {
    return { value: JSON.parse(readFileSync(file, 'utf8')) };
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

/** Read a JSON artifact from out/, reporting absence/corruption once. */
function readExportedJson(outDir, fail, relative, origin, kind = 'object') {
  const file = path.join(outDir, relative);
  if (!existsSync(file)) {
    fail(`out/${relative} is missing from the export — ${origin}`);
    return null;
  }
  const parsed = readJson(file);
  if (parsed.error) {
    fail(`out/${relative} is not valid JSON: ${parsed.error}`);
    return null;
  }
  const value = parsed.value;
  if (!value || typeof value !== 'object' || Array.isArray(value) !== (kind === 'array')) {
    fail(`out/${relative} must be a JSON ${kind}`);
    return null;
  }
  return value;
}

// ── Exported pages and export-root artifacts ──────────────────────────────────

function checkExportedPages(appDir, fail, ctx) {
  const manifest = path.join(appDir, '.next', 'prerender-manifest.json');
  if (!existsSync(manifest)) {
    fail(`the prerender manifest is missing at ${manifest} — build the export first (npm run build in apps/docs)`);
    return;
  }
  const routes = exportedRoutes(appDir);
  ctx.routes = routes;
  const outDir = path.join(appDir, 'out');
  for (const route of routes) {
    if (!isNonempty(htmlPath(outDir, route))) {
      fail(`the exported HTML for ${route} is missing or empty`);
    }
    const payload = rscPath(outDir, route);
    if (!isNonempty(payload)) {
      fail(`the RSC payload ${path.relative(outDir, payload)} for ${route} is missing or empty`);
    }
    // Pages live at x/index.html; a flat x.html would be a second URL for the page.
    if (route !== '/') {
      const flat = `${path.dirname(htmlPath(outDir, route))}.html`;
      if (existsSync(flat)) fail(`${path.relative(outDir, flat)} duplicates ${route}; postbuild nests pages as x/index.html`);
    }
  }
}

function checkExportArtifacts(appDir, fail) {
  const outDir = path.join(appDir, 'out');
  if (!isNonempty(path.join(outDir, '404.html'))) {
    fail('out/404.html is missing or empty');
  }
  if (!existsSync(path.join(outDir, '.nojekyll'))) {
    fail('out/.nojekyll is missing — GitHub Pages would skip _next assets without it');
  }
}

// ── Route expectations derived independently from the sources ────────────────

function sourceDocRoutes(appDir, locale) {
  const dir = path.join(appDir, 'content', locale);
  const routes = new Map();
  if (!existsSync(dir)) return routes;
  for (const file of walkFiles(dir)) {
    if (!/\.mdx?$/.test(file)) continue;
    const relative = path.relative(dir, file).split(path.sep).join('/');
    const route = '/' + documentSlugs(relative, readDocumentMetadata(file).slug).join('/');
    routes.set(route, `${locale}/${relative}`);
  }
  return routes;
}

function checkSourceRoutes(appDir, fail, ctx) {
  if (!ctx.routes) return;
  const contentDir = path.join(appDir, 'content');
  if (!existsSync(contentDir)) {
    fail(`the content directory is missing at ${contentDir} — route expectations are derived from it`);
    return;
  }
  const known = new Set(ctx.routes);
  for (const [route, source] of sourceDocRoutes(appDir, 'en')) {
    if (!known.has(route)) fail(`source document ${source} has no exported page at ${route}`);
    const zhRoute = localizedHref(route, 'zh-Hans');
    if (!known.has(zhRoute)) {
      fail(`English document ${source} has no zh-Hans fallback page at ${zhRoute} — every English route must exist in zh-Hans (translated or en-fallback)`);
    }
  }
  for (const [route, source] of sourceDocRoutes(appDir, 'zh-Hans')) {
    const zhRoute = localizedHref(route, 'zh-Hans');
    if (!known.has(zhRoute)) fail(`source document ${source} has no exported page at ${zhRoute}`);
  }
}

function checkCatalogRoutes(appDir, fail, ctx) {
  if (!ctx.routes) return;
  const plugins = readExportedJson(
    path.join(appDir, 'out'), fail, 'api/plugins.json', 'extract-plugins.py generates it in prebuild', 'array',
  );
  ctx.plugins = plugins;
  if (plugins === null) return;
  for (const [index, entry] of plugins.entries()) {
    if (!entry || typeof entry !== 'object' || typeof entry.name !== 'string' || !entry.name) {
      fail(`out/api/plugins.json entry ${index} is missing a non-empty name`);
    }
  }
  const names = plugins.map((entry) => entry?.name).filter((name) => typeof name === 'string' && name);
  const seen = new Set();
  for (const name of names) {
    if (seen.has(name)) fail(`out/api/plugins.json has duplicate entries for ${name}`);
    seen.add(name);
  }
  const known = new Set(ctx.routes);
  for (const locale of LOCALES) {
    // The marketplace owns the catalogs: the edge 308s the hubs, plugin pages and
    // author pages there, so an exported copy would be unreachable and stale.
    for (const hub of ['plugins', 'skills']) {
      const route = localizedHref(`/${hub}`, locale);
      if (known.has(route) || existsSync(htmlPath(path.join(appDir, 'out'), route))) {
        fail(`the ${locale} ${hub} hub page ${route} is exported, but the marketplace owns it`);
      }
    }
    // Any page under /plugins/ (a plugin or author page), whether Next listed it or not.
    const prefix = `${localizedHref('/plugins', locale)}/`;
    const pluginsDir = path.join(appDir, 'out', ...prefix.split('/').filter(Boolean));
    const exported = new Set([...known].filter((route) => route.startsWith(prefix)));
    if (existsSync(pluginsDir)) {
      for (const entry of readdirSync(pluginsDir, { recursive: true }).map(String)) {
        if (path.basename(entry) === 'index.html') exported.add(prefix + path.dirname(entry).split(path.sep).join('/'));
      }
    }
    for (const route of [...exported].sort()) {
      fail(`the ${locale} catalog page ${route} is exported, but the marketplace owns it`);
    }
  }
}

// ── Producer recipe: sitemap and legacy aliases ───────────────────────────────

function checkSitemapRecipe(appDir, fail, ctx) {
  if (!ctx.routes) return;
  const file = path.join(appDir, 'out', 'sitemap.xml');
  if (!existsSync(file)) {
    fail('out/sitemap.xml is missing from the export — scripts/postbuild.mjs generates it');
    return;
  }
  const expected = sitemapXml(ctx.routes);
  if (readFileSync(file, 'utf8') !== expected) {
    fail('out/sitemap.xml does not match the producer recipe for the exported routes (regenerate with scripts/postbuild.mjs)');
  }
  // Docusaurus served a sitemap under each locale root; those URLs keep resolving.
  for (const locale of LOCALES.filter((locale) => locale !== 'en')) {
    const copy = path.join(appDir, 'out', locale, 'sitemap.xml');
    if (!existsSync(copy)) fail(`out/${locale}/sitemap.xml is missing from the export`);
    else if (readFileSync(copy, 'utf8') !== expected) fail(`out/${locale}/sitemap.xml differs from out/sitemap.xml`);
  }
  if (existsSync(path.join(appDir, 'out', '_not-found'))) {
    fail("out/_not-found is exported as a page; postbuild removes Next's internal not-found route");
  }
}

function checkLegacyAliases(appDir, fail, ctx) {
  if (!ctx.routes) return;
  const outDir = path.join(appDir, 'out');
  const asset = redirectScriptAsset();
  const script = path.join(outDir, asset.url);
  if (!isNonempty(script) || readFileSync(script, 'utf8') !== asset.content) {
    fail('the redirect script does not match its producer recipe');
  }
  for (const entry of redirectEntries(ctx.routes)) {
    const file = htmlPath(outDir, entry.from);
    if (!isNonempty(file)) fail(`legacy alias ${entry.from} is missing or empty`);
    else if (readFileSync(file, 'utf8') !== redirectHtml(entry, asset.url)) {
      fail(`legacy alias ${entry.from} does not match its redirect recipe`);
    }
  }
}

// ── Copied artifacts: public/ verbatim, Next static assets, page assets ──────

function checkCopiedTree(source, destination, label, fail) {
  if (!existsSync(source)) {
    fail(`the ${label} source directory is missing at ${source}`);
    return;
  }
  const files = walkFiles(source);
  if (!files.length) fail(`the ${label} source directory is empty`);
  for (const file of files) {
    const relative = path.relative(source, file);
    const exported = path.join(destination, relative);
    if (!statSync(exported, { throwIfNoEntry: false })?.isFile()) {
      fail(`${label} file ${relative} was not copied into the export`);
    } else if (!readFileSync(file).equals(readFileSync(exported))) {
      fail(`${label} file ${relative} differs from its exported copy`);
    }
  }
}

function checkPublicCopied(appDir, fail) {
  const outDir = path.join(appDir, 'out');
  checkCopiedTree(path.join(appDir, 'public'), outDir, 'public', fail);
  checkCopiedTree(path.join(appDir, '.next/static'), path.join(outDir, '_next/static'), 'Next static', fail);
}

function checkLocalizedArtifacts(appDir, fail) {
  const publicDir = path.join(appDir, 'public');
  if (!existsSync(publicDir)) return; // checkPublicCopied reports the missing source.
  for (const file of walkFiles(publicDir)) {
    const relative = path.relative(publicDir, file).split(path.sep).join('/');
    if (!relative.startsWith('api/') && !relative.startsWith('oauth/')
      && relative !== 'llms.txt' && relative !== 'llms-full.txt') continue;
    for (const locale of LOCALES.filter((locale) => locale !== 'en')) {
      const alias = `${locale}/${relative}`;
      const exported = path.join(appDir, 'out', alias);
      if (!statSync(exported, { throwIfNoEntry: false })?.isFile()) {
        fail(`localized artifact ${alias} is missing from the export`);
      } else if (!readFileSync(file).equals(readFileSync(exported))) {
        fail(`localized artifact ${alias} differs from its canonical source`);
      }
    }
  }
}

function srcsetUrls(value) {
  const urls = [];
  let cursor = 0;
  while (cursor < value.length) {
    while (cursor < value.length && /[,\t\n\f\r ]/.test(value[cursor])) cursor++;
    const token = /^[^\t\n\f\r ]+/.exec(value.slice(cursor))?.[0];
    if (!token) break;
    cursor += token.length;
    urls.push(token.replace(/,+$/, ''));
    if (token.endsWith(',')) continue;
    let parentheses = 0;
    while (cursor < value.length) {
      const character = value[cursor++];
      if (character === '(') parentheses++;
      else if (character === ')') parentheses--;
      else if (character === ',' && parentheses === 0) break;
    }
  }
  return urls;
}

function collectAssetRefs(node, refs, searchLinks) {
  for (const child of node.childNodes ?? []) {
    if (child.tagName) {
      const attr = (name) => child.attrs?.find((attribute) => attribute.name === name)?.value;
      if (child.tagName === 'script' && attr('src')) refs.push(attr('src'));
      if (child.tagName === 'img') {
        if (attr('src')) refs.push(attr('src'));
        refs.push(...srcsetUrls(attr('srcset') ?? ''));
      }
      if (child.tagName === 'link') {
        const rel = attr('rel') ?? '';
        if (rel.toLowerCase().split(/\s+/).includes('stylesheet') && attr('href')) refs.push(attr('href'));
        if (rel.toLowerCase().split(/\s+/).includes('search')
          && attr('type') === 'application/opensearchdescription+xml' && attr('href')) {
          searchLinks.push(attr('href'));
          refs.push(attr('href'));
        }
      }
    }
    collectAssetRefs(child, refs, searchLinks);
  }
}

function checkHtmlAssets(appDir, fail, ctx) {
  if (!ctx.routes) return;
  const outDir = path.join(appDir, 'out');
  for (const route of ctx.routes) {
    const file = htmlPath(outDir, route);
    if (!existsSync(file)) continue; // the page checks already reported it
    const refs = [];
    const searchLinks = [];
    collectAssetRefs(parse(readFileSync(file, 'utf8')), refs, searchLinks);
    const expectedSearch = publicHref(localizedHref('/opensearch.xml', splitLocalePath(route).locale));
    if (!searchLinks.includes(expectedSearch)) fail(`page ${route} is missing OpenSearch autodiscovery for ${expectedSearch}`);
    // The browser's base: pages are served slashless, so `x` resolves beside the page.
    const pageBase = new URL(canonicalUrl(route)).pathname;
    for (const ref of refs) {
      if (ref.startsWith('#')) continue;
      try {
        const url = new URL(ref, `${SITE_ORIGIN}${pageBase}`);
        if (url.origin !== SITE_ORIGIN) continue;
        const pathname = url.pathname;
        if (pathname !== DOCS_BASE_PATH && !pathname.startsWith(`${DOCS_BASE_PATH}/`)) {
          fail(`local asset ${ref} on page ${route} is not under the ${DOCS_BASE_PATH} base path`);
          continue;
        }
        const file = path.resolve(outDir, `.${decodeURIComponent(pathname.slice(DOCS_BASE_PATH.length))}`);
        const relative = path.relative(outDir, file);
        if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
          fail(`local asset ${ref} on page ${route} escapes the export boundary`);
        } else if (!isNonempty(file)) {
          fail(`local asset ${ref} on page ${route} resolves to ${pathname}, which is missing or empty in the export`);
        }
      } catch (error) {
        fail(`invalid local asset ${ref} on page ${route}: ${error.message}`);
      }
    }
  }
}

// ── JSON contracts (schemas the producers actually emit) ──────────────────────

function checkSkillsJson(appDir, fail) {
  const outDir = path.join(appDir, 'out');
  const skills = readExportedJson(outDir, fail, 'api/skills.json', 'extract-skills.py generates it in prebuild', 'array');
  if (skills === null) return;
  for (const [index, skill] of skills.entries()) {
    for (const field of ['name', 'category', 'source']) {
      if (!skill || typeof skill !== 'object' || typeof skill[field] !== 'string' || !skill[field]) {
        fail(`out/api/skills.json entry ${index} is missing a non-empty ${field}`);
      }
    }
  }
  const meta = readExportedJson(outDir, fail, 'api/skills-meta.json', 'extract-skills.py generates it in prebuild');
  if (meta !== null) {
    if (typeof meta.totalSkills !== 'number' || meta.totalSkills !== skills.length) {
      fail(`out/api/skills-meta.json totalSkills ${meta.totalSkills} does not match the ${skills.length} entries in skills.json`);
    }
    const bySource = (meta.bySource && typeof meta.bySource === 'object') ? meta.bySource : {};
    const actual = new Map();
    for (const skill of skills) {
      if (skill?.source) actual.set(skill.source, (actual.get(skill.source) ?? 0) + 1);
    }
    for (const [source, count] of Object.entries(bySource)) {
      if (count !== (actual.get(source) ?? 0)) {
        fail(`out/api/skills-meta.json bySource['${source}'] (${count}) does not match ${actual.get(source) ?? 0} entries in skills.json`);
      }
    }
    for (const source of actual.keys()) {
      if (!(source in bySource)) fail(`out/api/skills-meta.json bySource is missing the '${source}' source`);
    }
    if (typeof meta.localSkills === 'number' && typeof meta.externalSkills === 'number'
      && meta.localSkills + meta.externalSkills !== skills.length) {
      fail(`out/api/skills-meta.json localSkills + externalSkills (${meta.localSkills} + ${meta.externalSkills}) does not match the ${skills.length} entries in skills.json`);
    }
  }
  if (skills.length === 0) {
    const sources = countSourceSkills(path.resolve(appDir, '../..'));
    if (sources > 0) {
      fail(`out/api/skills.json is empty but the repository has ${sources} SKILL.md sources — extract-skills.py did not run or wrote its empty fallback`);
    }
  }
  const indexFile = path.join(outDir, 'api', 'skills-index.json');
  if (existsSync(indexFile)) {
    const index = readJson(indexFile);
    if (index.error) fail(`out/api/skills-index.json is not valid JSON: ${index.error}`);
    else if (!index.value || typeof index.value !== 'object' || !Array.isArray(index.value.skills)) {
      fail(`out/api/skills-index.json must have a 'skills' array`);
    } else if (typeof index.value.skill_count === 'number' && index.value.skill_count !== index.value.skills.length) {
      fail(`out/api/skills-index.json skill_count ${index.value.skill_count} does not match its ${index.value.skills.length} skills entries`);
    }
  }
}

function checkPluginsJson(appDir, fail, ctx) {
  const plugins = ctx.plugins ?? [];
  const outDir = path.join(appDir, 'out');
  const meta = readExportedJson(outDir, fail, 'api/plugins-meta.json', 'extract-plugins.py generates it in prebuild');
  if (meta !== null) {
    if (typeof meta.total !== 'number' || meta.total !== plugins.length) {
      fail(`out/api/plugins-meta.json total ${meta.total} does not match the ${plugins.length} entries in plugins.json`);
    }
    const byTier = (meta.byTier && typeof meta.byTier === 'object') ? meta.byTier : {};
    for (const tier of ['official', 'verified', 'community']) {
      const actual = plugins.filter((plugin) => plugin?.tier === tier).length;
      if (typeof byTier[tier] !== 'number') fail(`out/api/plugins-meta.json byTier is missing the '${tier}' tier`);
      else if (byTier[tier] !== actual) {
        fail(`out/api/plugins-meta.json byTier['${tier}'] (${byTier[tier]}) does not match ${actual} entries in plugins.json`);
      }
    }
    const byCategory = (meta.byCategory && typeof meta.byCategory === 'object') ? meta.byCategory : {};
    for (const [category, count] of Object.entries(byCategory)) {
      const actual = plugins.filter((plugin) => plugin?.category === category).length;
      if (count !== actual) {
        fail(`out/api/plugins-meta.json byCategory['${category}'] (${count}) does not match ${actual} entries in plugins.json`);
      }
    }
  }
  const raw = readExportedJson(outDir, fail, 'api/plugin-catalog.json', 'extract-plugins.py generates it in prebuild');
  if (raw !== null) {
    if (!Array.isArray(raw.entries) || !Array.isArray(raw.removed)) {
      fail(`out/api/plugin-catalog.json must have 'entries' and 'removed' arrays`);
    } else {
      const names = new Set(raw.entries.map((entry) => entry?.name));
      for (const plugin of plugins) {
        if (!names.has(plugin?.name)) fail(`out/api/plugin-catalog.json is missing the raw catalog entry for '${plugin.name}'`);
      }
      if (meta && typeof meta === 'object' && typeof meta.removedCount === 'number' && meta.removedCount !== raw.removed.length) {
        fail(`out/api/plugins-meta.json removedCount ${meta.removedCount} does not match the ${raw.removed.length} removed entries in plugin-catalog.json`);
      }
    }
  }
  if (ctx.plugins && plugins.length === 0) {
    const valid = countValidCatalogEntries(path.resolve(appDir, '../..'));
    if (valid > 0) {
      fail(`out/api/plugins.json is empty but plugin-catalog/ has ${valid} valid entries — extract-plugins.py did not run or wrote its empty fallback`);
    }
  }
}

function checkBlueprintsJson(appDir, fail) {
  const blueprints = readExportedJson(
    path.join(appDir, 'out'), fail, 'api/automation-blueprints-index.json', 'extract-automation-blueprints.py generates it in prebuild', 'array',
  );
  if (blueprints === null) return;
  for (const [index, entry] of blueprints.entries()) {
    for (const field of ['key', 'title']) {
      if (!entry || typeof entry !== 'object' || typeof entry[field] !== 'string' || !entry[field]) {
        fail(`out/api/automation-blueprints-index.json entry ${index} is missing a non-empty ${field}`);
      }
    }
  }
}

function checkModelCatalogJson(appDir, fail) {
  const catalog = readExportedJson(
    path.join(appDir, 'out'), fail, 'api/model-catalog.json', 'it is committed in apps/docs/public',
  );
  if (catalog === null) return;
  const providers = catalog.providers;
  if (!providers || typeof providers !== 'object' || Array.isArray(providers) || Object.keys(providers).length === 0) {
    fail('out/api/model-catalog.json has no providers — the committed model catalog is missing or gutted');
    return;
  }
  for (const [provider, data] of Object.entries(providers)) {
    if (!data || typeof data !== 'object' || !Array.isArray(data.models)) {
      fail(`out/api/model-catalog.json provider '${provider}' must have a models array`);
      continue;
    }
    for (const [index, model] of data.models.entries()) {
      if (!model || typeof model !== 'object' || typeof model.id !== 'string' || !model.id) {
        fail(`out/api/model-catalog.json provider '${provider}' model ${index} is missing a non-empty id`);
      }
    }
  }
}

function checkOauthMetadata(appDir, fail) {
  const metadata = readExportedJson(
    path.join(appDir, 'out'), fail, 'oauth/client-metadata.json', 'it is committed in apps/docs/public',
  );
  if (metadata === null) return;
  if (metadata.client_id !== OAUTH_CLIENT_ID) {
    fail(`out/oauth/client-metadata.json client_id must be ${OAUTH_CLIENT_ID} (the published OAuth discovery metadata URL)`);
  }
  const redirects = metadata.redirect_uris;
  if (!Array.isArray(redirects) || redirects.length === 0 || redirects.some((uri) => typeof uri !== 'string' || !uri)) {
    fail('out/oauth/client-metadata.json redirect_uris must be a non-empty array of callback URLs');
    return;
  }
  for (const [index, uri] of redirects.entries()) {
    let url;
    try {
      url = new URL(uri);
    } catch {
      fail(`out/oauth/client-metadata.json redirect_uris entry ${index} is not a valid URL`);
      continue;
    }
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      fail(`out/oauth/client-metadata.json redirect_uris entry ${index} is not an http(s) callback`);
    } else if (PLACEHOLDER_HOSTS.test(url.host)) {
      fail(`out/oauth/client-metadata.json redirect_uris entry ${index} is a placeholder stub (${url.host})`);
    }
  }
}

function checkSearchConfig(appDir, fail) {
  const config = readExportedJson(
    path.join(appDir, 'out'), fail, 'search-config.json', 'it is committed in apps/docs/public',
  );
  if (config === null) return;
  for (const field of ['appId', 'apiKey', 'indexName']) {
    // Never include the field's value in the message — keys stay out of logs.
    if (typeof config[field] !== 'string' || !config[field]) {
      fail(`out/search-config.json field '${field}' must be a non-empty string`);
    }
  }
}

function checkOpenSearch(appDir, fail, ctx) {
  for (const locale of LOCALES) {
    const route = localizedHref('/search', locale);
    if (ctx.routes && !ctx.routes.includes(route)) fail(`full search page ${route} is missing from the export`);
    const relative = localizedHref('/opensearch.xml', locale).slice(1);
    const file = path.join(appDir, 'out', relative);
    if (!isNonempty(file)) fail(`out/${relative} is missing or empty`);
    else if (readFileSync(file, 'utf8') !== openSearchXml(locale)) fail(`out/${relative} does not match its OpenSearch recipe`);
  }
  if (!isNonempty(path.join(appDir, 'out/img/favicon.ico'))) fail('the OpenSearch icon out/img/favicon.ico is missing or empty');
}

function checkLlmsFiles(appDir, fail) {
  for (const name of ['llms.txt', 'llms-full.txt']) {
    if (!isNonempty(path.join(appDir, 'out', name))) {
      fail(`out/${name} is missing or empty — generate-llms-txt.py did not run (prebuild generates it)`);
    }
  }
}

// ── Source-backed expectations: no silent empty fallbacks ──────────────────────

function countSourceSkills(repoRoot) {
  let count = 0;
  for (const dir of ['skills', 'optional-skills']) {
    const base = path.join(repoRoot, dir);
    if (existsSync(base)) count += walkFiles(base).filter((file) => path.basename(file) === 'SKILL.md').length;
  }
  return count;
}

// Match load_catalog_entries: malformed YAML and invalid pins do not produce UI entries.
function countValidCatalogEntries(repoRoot) {
  const dir = path.join(repoRoot, 'plugin-catalog');
  if (!existsSync(dir)) return 0;
  let count = 0;
  for (const file of readdirSync(dir)) {
    if (!file.endsWith('.yaml') || file === 'removed.yaml') continue;
    try {
      const raw = parseYaml(readFileSync(path.join(dir, file), 'utf8'), { intAsBigInt: true });
      if (raw && typeof raw === 'object' && !Array.isArray(raw) && raw.name && raw.repo
        && /^[0-9a-f]{40}$/i.test(String(raw.sha ?? '').trim())) count++;
    } catch {
      // unreadable YAML — skipped, mirroring the producer
    }
  }
  return count;
}

// ── Entry points ──────────────────────────────────────────────────────────────

const CHECKS = [
  checkExportedPages,
  checkExportArtifacts,
  checkSourceRoutes,
  checkCatalogRoutes,
  checkSitemapRecipe,
  checkLegacyAliases,
  checkPublicCopied,
  checkLocalizedArtifacts,
  checkHtmlAssets,
  checkExportLinks,
  checkSkillsJson,
  checkPluginsJson,
  checkBlueprintsJson,
  checkModelCatalogJson,
  checkOauthMetadata,
  checkSearchConfig,
  checkOpenSearch,
  checkLlmsFiles,
];

/**
 * Validate the static export of a docs app directory (default: this script's
 * own apps/docs, so the CLI works from the repo root). Read-only; returns
 * `{ ok, errors, pages }` and never throws for contract violations.
 */
export function validateOutput(appDir = appRoot) {
  const errors = [];
  const ctx = {};
  for (const check of CHECKS) {
    try {
      check(appDir, errors.push.bind(errors), ctx);
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }
  return { ok: errors.length === 0, errors, pages: ctx.routes?.length ?? 0 };
}

if (import.meta.main) {
  const { values } = parseArgs({ options: { 'app-dir': { type: 'string' } } });
  const result = validateOutput(values['app-dir'] ? path.resolve(values['app-dir']) : appRoot);
  if (result.ok) {
    console.log(`[docs validate-output] ${result.pages} exported pages validated`);
  } else {
    for (const error of result.errors) {
      console.error(`[docs validate-output] ${error}`);
    }
    console.error(`[docs validate-output] ${result.errors.length} problem(s) in the static export`);
    process.exitCode = 1;
  }
}
