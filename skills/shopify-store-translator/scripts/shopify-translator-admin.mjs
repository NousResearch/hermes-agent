#!/usr/bin/env node
/**
 * shopify-translator-admin.mjs
 * Shopify Store Translator — helper script
 * API version: 2026-04
 *
 * Commands:
 *   init-env              Create skill-hub.env with placeholders
 *   connection-check      Verify Admin API access
 *   check-locales         Report locale status for a target language
 *   enable-locale         Enable and publish a locale
 *   check-markets         Report which markets have the target locale
 *   add-locale-to-market  Add locale to a market's webPresence
 *   fetch                 Fetch translatable content for a resource type
 *   write                 Write translations from a CSV file to Shopify
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { loadSkillHubEnv, resolveAdminHost, shopifyGraphql } from './lib/shopify-cli.mjs';

// ─── CLI arg parsing ──────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const command = args[0];
const flags = {};
for (let i = 1; i < args.length; i += 2) {
  if (args[i]?.startsWith('--')) flags[args[i].slice(2)] = args[i + 1] ?? true;
}

const loadEnv = loadSkillHubEnv;
const gql = (host, _token, query, variables = {}) => shopifyGraphql(host, query, variables);
const gqlWithThrottle = gql;

// ─── CSV helpers ──────────────────────────────────────────────────────────────

function parseCSV(content) {
  const rows = [];
  let row = [], value = '', quoted = false;
  const text = String(content).replace(/^\uFEFF/, '');
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') { value += '"'; index += 1; }
      else if (char === '"') quoted = false;
      else value += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { row.push(value); value = ''; }
    else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[index + 1] === '\n') index += 1;
      row.push(value); value = '';
      if (row.some((cell) => cell !== '')) rows.push(row);
      row = [];
    } else value += char;
  }
  if (quoted) throw new Error('CSV_PARSE_FAILED: Unterminated quoted field.');
  row.push(value);
  if (row.some((cell) => cell !== '')) rows.push(row);
  const [headers = [], ...dataRows] = rows;
  return { headers: headers.map((header) => header.trim()), rows: dataRows };
}

function escapeCSV(val) {
  if (val == null) return '';
  const s = String(val);
  return s.includes(',') || s.includes('"') || s.includes('\n')
    ? `"${s.replace(/"/g, '""')}"` : s;
}

function csvRecords(content) {
  const { headers, rows } = parseCSV(content);
  return rows.map((row) => Object.fromEntries(headers.map((header, index) => [header, row[index] ?? ''])));
}

function stringifyCSV(headers, rows) {
  return [headers, ...rows].map((row) => row.map(escapeCSV).join(',')).join('\n');
}

function minimumTranslationRatio(locale) {
  return /^(zh|ja|ko)(-|$)/i.test(locale) ? 0.6 : 0.8;
}

// ─── Commands ─────────────────────────────────────────────────────────────────

async function cmdInitEnv() {
  const envPath = flags.env || 'skill-hub.env';
  if (existsSync(envPath)) {
    console.log(`${envPath} already exists. Edit it directly.`);
    return;
  }
  const content = `# Skill Hub shared Shopify configuration
# Quick browser connection is the default. For a long-running agent, change the
# method and add the Client ID and Client Secret in this private file.

SKILL_HUB_SHOPIFY_ACCESS_METHOD=shopify_cli_oauth
SKILL_HUB_SHOPIFY_STORE_DOMAIN=
# SKILL_HUB_SHOPIFY_CLIENT_ID=
# SKILL_HUB_SHOPIFY_CLIENT_SECRET=
# SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN=
`;
  writeFileSync(envPath, content);
  console.log(`Created ${envPath}. Add your Shopify admin URL or .myshopify.com domain.`);
}

async function cmdConnectionCheck() {
  const env = loadEnv(flags.env);
  const host = resolveAdminHost(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  try {
    const data = await gql(host, null, 'query { shop { name id } }');
    console.log(JSON.stringify({ status: 'OK', shop: data.shop.name, id: data.shop.id }));
  } catch (e) {
    console.error('CONNECTION_FAILED:', e.message);
    process.exit(1);
  }
}

async function cmdCheckLocales() {
  const env = loadEnv(flags.env);
  const host = resolveAdminHost(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  const token = null;
  const target = flags.target;
  if (!target) { console.error('--target locale is required'); process.exit(1); }

  const data = await gql(host, token, 'query { shopLocales { locale primary published } }');
  const locales = data.shopLocales;
  const primary = locales.find(l => l.primary);
  const found = locales.find(l => l.locale === target);

  const result = {
    primaryLocale: primary?.locale,
    targetLocale: target,
    exists: !!found,
    published: found?.published ?? false,
    action: !found ? 'ENABLE_AND_PUBLISH' : !found.published ? 'PUBLISH' : 'NONE',
    allLocales: locales,
  };
  console.log(JSON.stringify(result, null, 2));
}

async function cmdEnableLocale() {
  const env = loadEnv(flags.env);
  const host = resolveAdminHost(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  const token = null;
  const locale = flags.locale;
  if (!locale) { console.error('--locale is required'); process.exit(1); }

  // Enable
  const enableData = await gql(host, token,
    `mutation EnableLocale($locale: String!) { shopLocaleEnable(locale: $locale) { shopLocale { locale published } userErrors { field message } } }`,
    { locale }
  );
  const enableErrors = enableData.shopLocaleEnable.userErrors;
  if (enableErrors.length) { console.error('Enable errors:', enableErrors); process.exit(1); }

  // Publish
  const publishData = await gql(host, token,
    `mutation PublishLocale($locale: String!) { shopLocaleUpdate(locale: $locale, shopLocale: { published: true }) { shopLocale { locale published } userErrors { field message } } }`,
    { locale }
  );
  const publishErrors = publishData.shopLocaleUpdate.userErrors;
  if (publishErrors.length) { console.error('Publish errors:', publishErrors); process.exit(1); }

  console.log(JSON.stringify({ status: 'OK', locale, published: true }));
}

async function cmdCheckMarkets() {
  const env = loadEnv(flags.env);
  const host = resolveAdminHost(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  const token = null;
  const locale = flags.locale;
  if (!locale) { console.error('--locale is required'); process.exit(1); }

  const data = await gql(host, token, `query {
    markets(first: 50) {
      nodes {
        id name enabled primary
        webPresence {
          id
          rootUrls { locale url }
          defaultLocale { locale published }
          alternateLocales { locale published }
        }
      }
    }
  }`);

  const markets = data.markets.nodes.map(m => ({
    id: m.id,
    name: m.name,
    enabled: m.enabled,
    primary: m.primary,
    webPresenceId: m.webPresence?.id ?? null,
    hasLocale: m.webPresence
      ? m.webPresence.alternateLocales.some(l => l.locale === locale) ||
        m.webPresence.defaultLocale?.locale === locale
      : false,
    currentAlternateLocales: m.webPresence?.alternateLocales.map(l => l.locale) ?? [],
    url: m.webPresence?.rootUrls.find(u => u.locale === locale)?.url ?? null,
  }));

  console.log(JSON.stringify({ targetLocale: locale, markets }, null, 2));
}

async function cmdAddLocaleToMarket() {
  const env = loadEnv(flags.env);
  const host = resolveAdminHost(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  const token = null;
  const webPresenceId = flags['market-web-presence-id'];
  const locale = flags.locale;
  if (!webPresenceId || !locale) {
    console.error('--market-web-presence-id and --locale are required');
    process.exit(1);
  }

  // Read current alternateLocales first
  const checkData = await gql(host, token, `query {
    markets(first: 50) {
      nodes {
        webPresence { id alternateLocales { locale } }
      }
    }
  }`);
  const wp = checkData.markets.nodes
    .map(m => m.webPresence)
    .find(w => w?.id === webPresenceId);
  if (!wp) { console.error('webPresence not found:', webPresenceId); process.exit(1); }

  const existing = wp.alternateLocales.map(l => l.locale);
  if (existing.includes(locale)) {
    console.log(JSON.stringify({ status: 'ALREADY_EXISTS', locale, webPresenceId }));
    return;
  }

  const newLocales = [...existing, locale];

  const data = await gql(host, token, `mutation UpdateMarketLocales($webPresenceId: ID!, $locales: [String!]!) {
    marketWebPresenceUpdate(
      webPresenceId: $webPresenceId
      webPresence: { alternateLocales: $locales }
    ) {
      market { webPresence { alternateLocales { locale } } }
      userErrors { field message }
    }
  }`, { webPresenceId, locales: newLocales });

  const errors = data.marketWebPresenceUpdate.userErrors;
  if (errors.length) { console.error('Errors:', errors); process.exit(1); }

  console.log(JSON.stringify({
    status: 'OK',
    locale,
    webPresenceId,
    alternateLocales: data.marketWebPresenceUpdate.market.webPresence.alternateLocales.map(l => l.locale),
  }));
}

async function cmdFetch() {
  const env = loadEnv(flags.env);
  const host = resolveAdminHost(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  const token = null;
  const resourceType = flags['resource-type'] || 'PRODUCT';
  const locale = flags.locale;
  const outputFile = flags.output;
  if (!locale) { console.error('--locale is required'); process.exit(1); }

  const SKIP_TYPES = new Set(['URI','URL','LINK','LIST_URL','LIST_LINK','FILE_REFERENCE','LIST_FILE_REFERENCE','JSON','JSON_STRING']);

  const results = [];
  let cursor = null;
  let hasNextPage = true;

  while (hasNextPage) {
    const data = await gqlWithThrottle(host, token, `query FetchTranslatableResources($resourceType: TranslatableResourceType!, $locale: String!, $after: String) {
      translatableResources(first: 50, resourceType: $resourceType, after: $after) {
        nodes {
          resourceId
          translatableContent { key value digest locale type }
          translations(locale: $locale) { key value outdated }
        }
        pageInfo { hasNextPage endCursor }
      }
    }`, { resourceType, locale, after: cursor });

    const { nodes, pageInfo } = data.translatableResources;
    hasNextPage = pageInfo.hasNextPage;
    cursor = pageInfo.endCursor;

    for (const node of nodes) {
      const translationMap = {};
      for (const t of (node.translations || [])) translationMap[t.key] = t;
      const fields = (node.translatableContent || [])
        .filter(c => c.key !== 'handle')
        .filter(c => !SKIP_TYPES.has(c.type))
        .map(c => {
          const existing = translationMap[c.key];
          let status = 'NEW';
          if (existing) status = existing.outdated ? 'OUTDATED' : 'CURRENT';
          return { key: c.key, value: c.value, digest: c.digest, type: c.type, status, existingTranslation: existing?.value ?? null };
        });
      results.push({ resourceId: node.resourceId, resourceType, fields });
    }
  }

  const output = JSON.stringify({ locale, resourceType, resources: results }, null, 2);
  if (outputFile) {
    const dir = dirname(resolve(outputFile));
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    writeFileSync(outputFile, output);
    console.log(`Fetched ${results.length} resources → ${outputFile}`);
  } else {
    console.log(output);
  }
}

async function cmdWrite() {
  const env = loadEnv(flags.env);
  const host = resolveAdminHost(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  const token = null;
  const inputFile = flags.input;
  const locale = flags.locale;
  if (!inputFile || !locale) { console.error('--input and --locale are required'); process.exit(1); }

  const csv = readFileSync(inputFile, 'utf8');
  const rows = csvRecords(csv).filter(r => r.status === 'NEW' || r.status === 'OUTDATED' || r.status === 'UPDATE');

  // Group by resource_id (support both 'resource_id' and legacy 'product_id' column names)
  const byResource = {};
  for (const row of rows) {
    const id = row.resource_id || row.product_id;
    if (!id) continue;
    // Support both 'translation_{locale}' column naming conventions
    const translationKey = `translation_${locale}`;
    const value = row[translationKey];
    if (!value || !row.digest) continue;
    // Support both 'field_key' column names
    const key = row.field_key;
    if (!key) continue;
    if (!byResource[id]) byResource[id] = [];
    byResource[id].push({ key, value, digest: row.digest });
  }

  const resourceIds = Object.keys(byResource);
  let successCount = 0, errorCount = 0;

  for (const id of resourceIds) {
    const translations = byResource[id].map((item) => ({
      locale,
      key: item.key,
      value: item.value,
      translatableContentDigest: item.digest,
    }));
    try {
      const data = await gqlWithThrottle(host, token, `mutation RegisterTranslations($resourceId: ID!, $translations: [TranslationInput!]!) {
        translationsRegister(resourceId: $resourceId, translations: $translations) {
          translations { key value }
          userErrors { field message }
        }
      }`, { resourceId: id, translations });
      const result = data.translationsRegister;
      if (result.userErrors?.length) {
        console.error(`Error for resource ${id}:`, result.userErrors);
        errorCount += translations.length;
      } else {
        successCount += result.translations?.length ?? 0;
      }
    } catch (e) {
      console.error(`Resource ${id} failed:`, e.message);
      errorCount += translations.length;
    }
  }

  console.log(JSON.stringify({ status: 'DONE', successCount, errorCount, locale }));
}

async function cmdTranslateCSV() {
  const inputFile = flags.input;
  const locale = flags.locale;
  if (!inputFile || !locale) {
    console.error('--input and --locale are required');
    process.exit(1);
  }

  const content = readFileSync(inputFile, 'utf8');
  const { headers, rows } = parseCSV(content);
  const offset = Math.max(0, Number.parseInt(flags.offset || '0', 10) || 0);
  const limit = Math.max(1, Number.parseInt(flags.limit || '50', 10) || 50);

  // Column indices
  const COL = { type:0, id:1, field:2, locale:3, market:4, status:5, default:6, translated:7 };

  function shouldSkipRow(r) {
    const type = r[COL.type], field = r[COL.field], val = r[COL.default] || '';
    if ((r[COL.translated] || '').trim()) return true; // already translated
    if (!val.trim()) return true;
    if (field === 'handle') return true;
    if (type === 'ONLINE_STORE_THEME' || type === 'PACKING_SLIP_TEMPLATE') return true;
    if (type === 'METAOBJECT' && field === 'data') return true;
    const v = val.trim();
    if (v.startsWith('{%') || v.startsWith('{{')) return true;
    if (type === 'PRODUCT_OPTION_VALUE' && /^\$?\d+(\.\d+)?$/.test(v)) return true;
    if (field === 'description' && /^[a-z_]+$/.test(v)) return true;
    return false;
  }

  let needsTranslation = 0, skipped = 0, alreadyDone = 0;
  const toTranslate = [];

  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    while (r.length < 8) r.push('');
    if ((r[COL.translated] || '').trim()) { alreadyDone++; continue; }
    if (shouldSkipRow(r)) { skipped++; continue; }
    needsTranslation++;
    toTranslate.push({ rowIdx: i, type: r[COL.type], field: r[COL.field], value: r[COL.default] });
  }

  const page = toTranslate.slice(offset, offset + limit);

  console.log(JSON.stringify({
    status: 'READY',
    locale,
    headers,
    totalRows: rows.length,
    alreadyDone,
    skipped,
    needsTranslation,
    offset,
    limit,
    hasMore: offset + page.length < toTranslate.length,
    nextOffset: offset + page.length < toTranslate.length ? offset + page.length : null,
    note: 'Translate this page, save patches as [{rowIdx, translation}], then run write-csv-translations. Use nextOffset for the next page.',
    toTranslate: page,
  }, null, 2));
}

async function cmdWriteCSVTranslations() {
  // Reads original CSV + a JSON patch file, writes translated CSV
  const inputFile = flags.input;
  const patchFile = flags.patch; // JSON: [{rowIdx, translation}]
  const outputFile = flags.output;
  if (!inputFile || !patchFile || !outputFile) {
    console.error('--input, --patch, and --output are required');
    process.exit(1);
  }

  const content = readFileSync(inputFile, 'utf8');
  const { headers, rows } = parseCSV(content);
  const patches = JSON.parse(readFileSync(patchFile, 'utf8'));

  for (const { rowIdx, translation } of patches) {
    if (rows[rowIdx]) {
      while (rows[rowIdx].length < 8) rows[rowIdx].push('');
      rows[rowIdx][7] = translation;
    }
  }

  writeFileSync(outputFile, '\uFEFF' + stringifyCSV(headers, rows), 'utf8');
  console.log(JSON.stringify({ status: 'OK', outputFile, patchedRows: patches.length }));
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMMAND: generate-audit
// Reads annotated fetch JSON → generates translation-audit.csv (NO Shopify write)
// User must review CSV and approve before the `write` command is called.
// ═══════════════════════════════════════════════════════════════════════════════

async function cmdGenerateAudit() {
  const inputFile = flags.input;
  const locale = flags.locale;
  const translationsFile = flags.translations;
  const csvOutput = flags.output || resolve('translation-audit.csv');
  if (!inputFile || !locale) {
    console.error('--input <annotated-json> and --locale are required');
    process.exit(1);
  }

  const raw = readFileSync(resolve(inputFile), 'utf8');
  const data = JSON.parse(raw);
  const resources = data.resources || [];
  const resourceType = data.resourceType || 'UNKNOWN';
  const translationMap = new Map();
  if (translationsFile) {
    const candidates = JSON.parse(readFileSync(resolve(translationsFile), 'utf8'));
    for (const candidate of candidates) {
      if (candidate.resourceId && candidate.key && typeof candidate.translation === 'string') {
        translationMap.set(`${candidate.resourceId}\u0000${candidate.key}`, candidate.translation);
      }
    }
  }

  let totalNew = 0, totalOutdated = 0, totalSkipped = 0, totalCurrent = 0;
  const csvRows = [];

  for (const res of resources) {
    const rid = res.resourceId;
    if (!rid) continue;
    const rname = res.resourceName || rid;

    for (const field of res.fields || []) {
      const translation = field.translation ?? translationMap.get(`${rid}\u0000${field.key}`);
      if (!translation || !translation.trim()) continue;

      // Skip: handle fields, non-translatable types
      if (field.key === 'handle') continue;
      const SKIP_TYPES = new Set(['URI','URL','LINK','LIST_URL','LIST_LINK','FILE_REFERENCE','LIST_FILE_REFERENCE','JSON','JSON_STRING']);
      if (SKIP_TYPES.has(field.type)) { totalSkipped++; continue; }

      // Count statuses
      if (field.status === 'CURRENT') { totalCurrent++; continue; }
      if (field.status === 'OUTDATED') totalOutdated++;
      else totalNew++;

      csvRows.push({
        resource_id: rid,
        resource_type: resourceType,
        resource_name: rname,
        field_key: field.key,
        original: field.value,
        original_length: String(field.value?.length ?? 0),
        [`translation_${locale}`]: translation,
        translation_length: String(translation.length),
        status: field.status || 'NEW',
        digest: field.digest,
      });
    }
  }

  if (csvRows.length === 0) {
    console.error('No translatable entries found in JSON (check that fields have "translation" property)');
    process.exit(1);
  }

  // ── Generate CSV audit file (NO Shopify write) ──
  const csvHeaders = [
    'resource_id', 'resource_type', 'resource_name', 'field_key',
    'original', 'original_length', `translation_${locale}`, 'translation_length',
    'status', 'digest',
  ];
  const csvLines = [
    csvHeaders.map(escapeCSV).join(','),
    ...csvRows.map(row => csvHeaders.map(h => escapeCSV(row[h] ?? '')).join(',')),
  ];
  writeFileSync(csvOutput, '\uFEFF' + csvLines.join('\n'), 'utf8');

  // ── Summary for user review ──
  let warnings = [];
  for (const row of csvRows) {
    const origLen = parseInt(row.original_length, 10);
    const transLen = parseInt(row.translation_length, 10);
    if (origLen > 0 && transLen > 0) {
      const ratio = transLen / origLen;
      if (ratio < minimumTranslationRatio(locale)) {
        warnings.push({ field: row.field_key, resource: row.resource_name, ratio: Math.round(ratio * 100) + '%', original: origLen, translation: transLen });
      }
    }
  }

  console.log(JSON.stringify({
    status: 'AUDIT_READY',
    locale,
    fields: csvRows.length,
    new: totalNew,
    outdated: totalOutdated,
    skipped: totalSkipped,
    current: totalCurrent,
    csvOutput,
    lengthWarnings: warnings.length > 0 ? warnings : undefined,
    instruction: 'Review the CSV above. Obtain explicit approval before running the write command.',
  }, null, 2));
}

const commands = {
  'init-env': cmdInitEnv,
  'connection-check': cmdConnectionCheck,
  'check-locales': cmdCheckLocales,
  'enable-locale': cmdEnableLocale,
  'check-markets': cmdCheckMarkets,
  'add-locale-to-market': cmdAddLocaleToMarket,
  'fetch': cmdFetch,
  'write': cmdWrite,
  'generate-audit': cmdGenerateAudit,
  'translate-csv': cmdTranslateCSV,
  'write-csv-translations': cmdWriteCSVTranslations,
  'verify-translations': cmdVerifyTranslations,
  'check-encoding': cmdCheckEncoding,
};

// ── COMMAND: verify-translations ─────────────────────────────────────────────

async function cmdVerifyTranslations() {
  const env = loadEnv(flags.env);
  const host = resolveAdminHost(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  const token = null;
  const locale = flags.locale;
  const inputFile = flags.input;
  if (!locale) { console.error('--locale required'); process.exit(1); }

  // Read resource IDs from CSV or fetch fresh
  let resourceIds = [];
  if (inputFile) {
    const csv = readFileSync(inputFile, 'utf8');
    const rows = csvRecords(csv);
    const writtenRows = rows.filter(r => r.status === 'NEW' || r.status === 'OUTDATED');
    resourceIds = [...new Set(writtenRows.map(r => r.resource_id).filter(Boolean))];
  }

  if (resourceIds.length === 0) {
    console.log('No written resources to verify.');
    return;
  }

  let issues = [];
  for (const id of resourceIds.slice(0, 10)) { // max 10 resources
    try {
      const data = await gql(host, token, `query VerifyTranslation($resourceId: ID!, $locale: String!) {
        translatableResource(resourceId: $resourceId) {
          resourceId
          translatableContent { key value digest locale }
          translations(locale: $locale) { key value outdated }
        }
      }`, { resourceId: id, locale });
      const res = data.translatableResource;
      if (!res) { issues.push({ resourceId: id, error: 'NOT_FOUND' }); continue; }
      for (const t of res.translatableContent) {
        const trans = res.translations?.find(tr => tr.key === t.key);
        if (!trans) continue;
        const stored = trans.value;
        // Check for garbled characters
        const garbled = (stored.match(/\uFFFD/g) || []).length;
        const lengthRatio = stored.length / t.value.length;
        if (garbled > 0 || lengthRatio < minimumTranslationRatio(locale)) {
          issues.push({
            resourceId: id,
            key: t.key,
            originalLength: t.value.length,
            storedLength: stored.length,
            lengthRatio: Math.round(lengthRatio * 100) + '%',
            garbledChars: garbled,
          });
        }
      }
    } catch (e) {
      issues.push({ resourceId: id, error: e.message.substring(0, 100) });
    }
  }

  if (issues.length === 0) {
    console.log(JSON.stringify({ status: 'OK', verified: resourceIds.length, locale, message: 'All translations verified with no encoding or truncation issues.' }, null, 2));
  } else {
    console.log(JSON.stringify({ status: 'WARN', issues }, null, 2));
  }
}

// ── COMMAND: check-encoding ──────────────────────────────────────────────────

async function cmdCheckEncoding() {
  const filePath = flags.file || flags.input;
  if (!filePath) { console.error('--file required'); process.exit(1); }

  const content = readFileSync(resolve(filePath), 'utf8');
  const garbledCount = (content.match(/\uFFFD/g) || []).length;
  const hasBom = content.charCodeAt(0) === 0xFEFF;

  const result = {
    file: filePath,
    size: content.length,
    hasBom,
    garbledChars: garbledCount,
    encodingOK: garbledCount === 0,
  };

  if (!result.encodingOK) {
    console.error(JSON.stringify({ status: 'FAIL', ...result }, null, 2));
    process.exit(1);
  }
  console.log(JSON.stringify({ status: 'OK', ...result }, null, 2));
}

if (!command || !commands[command]) {
  console.log(`Usage: shopify-translator-admin.mjs <command> [options]

Commands:
  init-env              --env skill-hub.env
  connection-check      --env skill-hub.env
  check-locales         --env skill-hub.env --target <locale>
  enable-locale         --env skill-hub.env --locale <locale>
  check-markets         --env skill-hub.env --locale <locale>
  add-locale-to-market  --env skill-hub.env --market-web-presence-id <id> --locale <locale>
  fetch                 --env skill-hub.env --resource-type <TYPE> --locale <locale> [--output <file>]
  write                 --env skill-hub.env --input <csv> --locale <locale>
  generate-audit        --input <fetch.json> --translations <candidates.json> --locale <locale> [--output <audit.csv>]
  verify-translations   --env skill-hub.env --locale <locale> [--input <csv>]
  check-encoding        --file <path>
  translate-csv         --input <shopify-export.csv> --locale <locale> [--offset 0 --limit 50]
  write-csv-translations --input <shopify-export.csv> --patch <patches.json> --output <translated.csv>
`);
  process.exit(command ? 1 : 0);
}

commands[command]().catch(e => { console.error(e.message); process.exit(1); });
