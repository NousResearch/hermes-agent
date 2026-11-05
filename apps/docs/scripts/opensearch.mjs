#!/usr/bin/env node
// Browser search-engine descriptors, generated during postbuild.
// Standalone: node scripts/opensearch.mjs [outputDir]

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { LOCALES, SITE_URL, canonicalUrl, localizedHref, publicHref } from '../src/lib/urls.ts';

const SHORT_NAME = 'Hermes Agent';
const DESCRIPTION = 'Search Hermes Agent';

function escapeXml(value) {
  return value.replace(/[&<>"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[character]);
}

/** Locale-scoped home URL, in the slashless form the production edge serves. */
function searchForm(locale) {
  return canonicalUrl(localizedHref('/', locale));
}

export function openSearchXml(locale = 'en') {
  const favicon = new URL(publicHref('/img/favicon.ico'), SITE_URL).href;
  const searchPage = canonicalUrl(localizedHref('/search', locale));
  const descriptor = new URL(publicHref(localizedHref('/opensearch.xml', locale)), SITE_URL).href;
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<OpenSearchDescription xmlns="http://a9.com/-/spec/opensearch/1.1/"',
    '                       xmlns:moz="http://www.mozilla.org/2006/browser/search/">',
    `  <ShortName>${escapeXml(SHORT_NAME)}</ShortName>`,
    `  <Description>${escapeXml(DESCRIPTION)}</Description>`,
    '  <InputEncoding>UTF-8</InputEncoding>',
    `  <Image width="16" height="16" type="image/x-icon">${escapeXml(favicon)}</Image>`,
    `  <Url type="text/html" method="get" template="${escapeXml(`${searchPage}?q={searchTerms}`)}"/>`,
    `  <Url type="application/opensearchdescription+xml" rel="self" template="${escapeXml(descriptor)}" />`,
    `  <moz:SearchForm>${escapeXml(searchForm(locale))}</moz:SearchForm>`,
    '</OpenSearchDescription>',
    '',
  ].join('\n');
}

/** Write one descriptor per locale; English owns the root, others nest by locale. */
export function writeOpenSearch(outputDir) {
  const written = [];
  for (const locale of LOCALES) {
    const file = locale === 'en' ? join(outputDir, 'opensearch.xml') : join(outputDir, locale, 'opensearch.xml');
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, openSearchXml(locale));
    written.push(file);
  }
  return written;
}

if (import.meta.main) {
  const outputDir = process.argv[2] ? resolve(process.argv[2]) : resolve(dirname(fileURLToPath(import.meta.url)), '..', 'out');
  for (const file of writeOpenSearch(outputDir)) {
    console.log(`[docs opensearch] wrote ${file}`);
  }
}
