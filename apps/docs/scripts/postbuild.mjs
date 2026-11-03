#!/usr/bin/env node
import { existsSync, readFileSync, writeFileSync, mkdirSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { canonicalUrl, localizedHref, publicHref, splitLocalePath, LOCALES } from '../src/lib/urls.ts';

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const internalRoutes = new Set(['/_not-found', '/_global-error', '/404', '/500']);

export function htmlPath(outputDir, route) {
  if (!route.startsWith('/') || route.startsWith('//') || /[?#\\]/.test(route)) {
    throw new Error(`Invalid exported route: ${route}`);
  }
  const segments = route.split('/').filter(Boolean);
  for (const segment of segments) {
    const decoded = decodeURIComponent(segment);
    if (decoded === '.' || decoded === '..' || /[\\\0]/.test(decoded)) {
      throw new Error(`Invalid exported route segment: ${route}`);
    }
  }
  return path.join(outputDir, ...segments, 'index.html');
}

export function exportedRoutes(appDir) {
  const manifest = JSON.parse(readFileSync(path.join(appDir, '.next/prerender-manifest.json'), 'utf8'));
  if (!manifest.routes || typeof manifest.routes !== 'object' || Array.isArray(manifest.routes)) {
    throw new Error('The Next prerender manifest has no route map');
  }
  const notFound = new Set(manifest.notFoundRoutes ?? []);
  const routes = Object.keys(manifest.routes).filter((route) => !internalRoutes.has(route) && !notFound.has(route)).sort();
  if (!routes.length) throw new Error('The export contains no prerendered pages');
  for (const route of routes) {
    const info = statSync(htmlPath(path.join(appDir, 'out'), route), { throwIfNoEntry: false });
    if (!info?.isFile()) throw new Error(`Missing exported HTML for ${route}`);
    if (!info.size) throw new Error(`Empty exported HTML for ${route}`);
  }
  return routes;
}

export function escapeMarkup(value) {
  return value.replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]);
}

export const LEGACY_REDIRECTS = {
  '/guides/automation-templates': '/guides/automation-blueprints',
  '/guides/build-a-hermes-plugin': '/developer-guide/plugins',
  '/quickstart': '/getting-started/quickstart',
  '/installation': '/getting-started/installation',
};

const redirectMarker = '<!-- hermes-docs-redirect -->';
const redirectScript = `(() => {
  const target = new URL(document.currentScript.dataset.redirect, window.location.origin);
  if (window.location.search) target.search = window.location.search;
  if (window.location.hash) target.hash = window.location.hash;
  window.location.replace(target.href);
})();
`;

export function redirectEntries(routes) {
  const known = new Set(routes);
  return LOCALES.flatMap((locale) => Object.entries(LEGACY_REDIRECTS).map(([from, to]) => {
    const entry = { from: localizedHref(from, locale), to: localizedHref(to, locale), locale };
    if (!known.has(entry.to)) throw new Error(`Missing legacy redirect target: ${entry.to}`);
    if (known.has(entry.from)) throw new Error(`Legacy redirect would replace a page: ${entry.from}`);
    return entry;
  }));
}

function writeRedirects(outputDir, entries) {
  const digest = createHash('sha256').update(redirectScript).digest('hex').slice(0, 16);
  const asset = `/_next/static/docs-redirect-${digest}.js`;
  const assetFile = path.join(outputDir, asset);
  mkdirSync(path.dirname(assetFile), { recursive: true });
  writeFileSync(assetFile, redirectScript);
  for (const { from, to, locale } of entries) {
    const file = htmlPath(outputDir, from);
    if (existsSync(file) && !readFileSync(file, 'utf8').startsWith(redirectMarker)) {
      throw new Error(`Refusing to replace existing export: ${from}`);
    }
    const target = escapeMarkup(publicHref(to));
    const html = `${redirectMarker}
<!doctype html>
<html lang="${locale}"><head><meta charset="utf-8">
<title>Redirecting</title>
<link rel="canonical" href="${escapeMarkup(canonicalUrl(to))}">
<script src="${publicHref(asset)}" data-redirect="${target}"></script>
<meta http-equiv="refresh" content="0;url=${target}">
</head><body><p>Continue to <a href="${target}">${target}</a>.</p></body></html>
`;
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, html);
  }
}

export function sitemapXml(routes) {
  const knownRoutes = new Set(routes);
  const entries = routes.map((route) => {
    const { path: logicalPath } = splitLocalePath(route);
    const alternates = LOCALES.map((locale) => [locale, localizedHref(logicalPath, locale)])
      .filter(([, href]) => knownRoutes.has(href))
      .map(([locale, href]) => `    <xhtml:link rel="alternate" hreflang="${locale}" href="${escapeMarkup(canonicalUrl(href))}"/>`);
    return ['  <url>', `    <loc>${escapeMarkup(canonicalUrl(route))}</loc>`, ...alternates, '  </url>'].join('\n');
  });
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">',
    ...entries,
    '</urlset>',
    '',
  ].join('\n');
}

export function finishExport(appDir = appRoot) {
  const routes = exportedRoutes(appDir);
  const redirects = redirectEntries(routes);
  const outputDir = path.join(appDir, 'out');
  writeFileSync(path.join(outputDir, 'sitemap.xml'), sitemapXml(routes));
  writeRedirects(outputDir, redirects);
  return { pages: routes.length, redirects: redirects.length };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const { values } = parseArgs({ options: { 'app-dir': { type: 'string' } } });
    console.log(JSON.stringify(finishExport(values['app-dir'] ? path.resolve(values['app-dir']) : appRoot)));
  } catch (error) {
    console.error(`[docs postbuild] ${error.message}`);
    process.exitCode = 1;
  }
}
