#!/usr/bin/env node
/**
 * shopify-market-lang-check.mjs
 * Check language (locale) and market configuration for a Shopify store.
 * Also provides mutation helpers for market/locale setup.
 *
 * Usage:
 *   node shopify-market-lang-check.mjs check --target zh-CN --env skill-hub.env
 *   node shopify-market-lang-check.mjs setup-market ... (future)
 */

import { loadSkillHubEnv, resolveAdminHost, shopifyGraphql } from './lib/shopify-cli.mjs';
const args = process.argv.slice(2);
const command = args[0];
const flags = {};
for (let i = 1; i < args.length; i += 2) {
  if (args[i]?.startsWith('--')) flags[args[i].slice(2)] = args[i + 1] ?? true;
}

const loadEnv = loadSkillHubEnv;
const gql = (host, _token, query, variables = {}) => shopifyGraphql(host, query, variables);

// ── COMMAND: check ─────────────────────────────────────────────────────────

async function cmdCheck() {
  const env = loadEnv(flags.env);
  const host = resolveAdminHost(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  const targetLocale = flags.target;
  if (!targetLocale) { console.error('--target locale required (e.g. zh-CN)'); process.exit(1); }

  const localesData = await gql(host, null, 'query { shopLocales { locale primary published } }');
  const marketsData = await gql(host, null, `query { markets(first: 50) { nodes { id name enabled primary status webPresence { id rootUrls { locale url } defaultLocale { locale } alternateLocales { locale } } } } }`);

  const allLocales = localesData.shopLocales;
  const primaryLocale = allLocales.find(l => l.primary);
  const targetInfo = allLocales.find(l => l.locale === targetLocale);

  const report = {
    primaryLocale: primaryLocale?.locale,
    targetLocale: targetLocale,
    targetExists: !!targetInfo,
    targetPublished: targetInfo?.published ?? false,
    actionNeeded: !targetInfo ? 'ENABLE_AND_PUBLISH' : !targetInfo.published ? 'PUBLISH' : 'NONE',
    localePresenceInMarkets: [],
    marketSummary: []
  };

  for (const m of marketsData.markets.nodes) {
    const entry = {
      id: m.id,
      name: m.name,
      enabled: m.enabled,
      status: m.status,
      primary: m.primary,
      hasWebPresence: !!m.webPresence,
      webPresenceId: m.webPresence?.id ?? null,
      isDefaultLocale: m.webPresence?.defaultLocale?.locale === targetLocale ?? false,
      isAlternateLocale: m.webPresence?.alternateLocales?.some(l => l.locale === targetLocale) ?? false,
      currentAlternateLocales: m.webPresence?.alternateLocales?.map(l => l.locale) ?? [],
      rootUrls: m.webPresence?.rootUrls ?? []
    };
    report.localePresenceInMarkets.push(entry);

    const status = entry.isDefaultLocale ? 'DEFAULT' : entry.isAlternateLocale ? 'ALTERNATE' : 'NOT_PRESENT';
    report.marketSummary.push(`${m.name}: ${status}`);
  }

  const marketsWithLocale = report.localePresenceInMarkets.filter(m => m.isDefaultLocale || m.isAlternateLocale);

  // Build human-readable output
  const lines = [];
  lines.push(`Locale Check Report for: ${targetLocale}`);
  lines.push(`Store primary locale: ${primaryLocale?.locale}`);
  lines.push(`Target locale status: ${targetInfo ? (targetInfo.published ? '✅ Published' : '🟡 Exists but not published') : '❌ Not enabled'}`);
  lines.push('');
  lines.push('Market Presence:');
  for (const m of report.localePresenceInMarkets) {
    if (m.isDefaultLocale) {
      lines.push(`  ✅ ${m.name}: ${targetLocale} is DEFAULT locale`);
    } else if (m.isAlternateLocale) {
      lines.push(`  ✅ ${m.name}: ${targetLocale} is ALTERNATE locale`);
    } else if (m.hasWebPresence) {
      lines.push(`  🟢 ${m.name}: ${targetLocale} NOT present (alternates: ${m.currentAlternateLocales.join(', ') || 'none'})`);
    } else {
      lines.push(`  ⚪ ${m.name}: no web presence configured`);
    }
  }
  lines.push('');
  lines.push('Recommended next steps:');
  if (!targetInfo) {
    lines.push(`  1. Enable locale: shopLocaleEnable(locale: "${targetLocale}")`);
    lines.push(`  2. Publish locale: shopLocaleUpdate(locale: "${targetLocale}", published: true)`);
  } else if (!targetInfo.published) {
    lines.push(`  1. Publish locale: shopLocaleUpdate(locale: "${targetLocale}", published: true)`);
  }
  if (marketsWithLocale.length === 0) {
    lines.push(`  3. Create a market with ${targetLocale} as the default locale and add country regions`);
    lines.push(`  4. Create web presence for the market with subfolderSuffix`);
  } else {
    const missingMarkets = report.localePresenceInMarkets.filter(m => m.hasWebPresence && !m.isDefaultLocale && !m.isAlternateLocale);
    if (missingMarkets.length > 0) {
      lines.push(`  3. Consider adding ${targetLocale} to alternate locales in: ${missingMarkets.map(m => m.name).join(', ')}`);
    }
  }

  console.log(lines.join('\n'));
  // Also output JSON for machine consumption
  console.log('\n---JSON_START---');
  console.log(JSON.stringify(report, null, 2));
  console.log('---JSON_END---');
}

// ── COMMAND: validate-setup ─────────────────────────────────────────────────

async function cmdValidateSetup() {
  const env = loadEnv(flags.env);
  const host = resolveAdminHost(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  const targetLocale = flags.target;
  if (!targetLocale) { console.error('--target and --env required'); process.exit(1); }

  const localesData = await gql(host, null, 'query { shopLocales { locale primary published } }');
  const marketsData = await gql(host, null, `query { markets(first: 50) { nodes { id name enabled primary status webPresence { id rootUrls { locale url } defaultLocale { locale } alternateLocales { locale } } } } }`);

  const targetInfo = localesData.shopLocales.find(l => l.locale === targetLocale);
  const marketsWithLocale = marketsData.markets.nodes.filter(m =>
    m.webPresence && (
      m.webPresence.defaultLocale?.locale === targetLocale ||
      m.webPresence.alternateLocales?.some(l => l.locale === targetLocale)
    )
  );

  const localeOk = targetInfo && targetInfo.published;
  const marketOk = marketsWithLocale.length > 0;

  const result = {
    status: localeOk && marketOk ? 'OK' : 'WARN',
    localeConfigured: localeOk,
    localeAction: !targetInfo ? 'ENABLE' : !targetInfo.published ? 'PUBLISH' : 'OK',
    marketsConfigured: marketOk,
    markets: marketsData.markets.nodes.map(m => ({
      name: m.name,
      hasLocale: marketsWithLocale.some(w => w.id === m.id),
      defaultLocale: m.webPresence?.defaultLocale?.locale,
      alternateLocales: m.webPresence?.alternateLocales?.map(l => l.locale) ?? []
    }))
  };

  if (!localeOk) {
    console.error(`⚠ WARN: Locale ${targetLocale} is not fully configured (enabled=${!!targetInfo}, published=${targetInfo?.published})`);
  }
  if (!marketOk) {
    console.error(`⚠ WARN: Locale ${targetLocale} is not present in any market web presence`);
  }
  if (localeOk && marketOk) {
    console.log(`✅ Locale ${targetLocale} is properly configured.`);
    console.log(`   Present in: ${marketsWithLocale.map(m => m.name).join(', ')}`);
  }

  console.log('\n---JSON_START---');
  console.log(JSON.stringify(result, null, 2));
  console.log('---JSON_END---');
}

const commands = {
  'check': cmdCheck,
  'validate-setup': cmdValidateSetup,
};

if (!command || !commands[command]) {
  console.log(`Usage:
  shopify-market-lang-check.mjs check --target <locale> --env skill-hub.env
  shopify-market-lang-check.mjs validate-setup --target <locale> --env skill-hub.env

Commands:
  check            Check locale and market configuration for a target language
  validate-setup   Quick validation: is locale enabled+published + present in markets?
`);
  process.exit(command ? 1 : 0);
}

commands[command]().catch(e => { console.error(e.message); process.exit(1); });
