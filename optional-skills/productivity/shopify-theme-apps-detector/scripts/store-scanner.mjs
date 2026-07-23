#!/usr/bin/env node
/**
 * store-scanner.mjs
 * Zero-dependency Shopify store evidence collector.
 * Outputs a JSON evidence bundle to stdout.
 * Usage: node store-scanner.mjs <url>
 */

import fs from "node:fs/promises";
import { fetchPublic } from "./public-fetch.mjs";

const TIMEOUT_MS = 15000;
const REQUEST_DELAY_MS = 800;
const MAX_PAGES = 5;
const MAX_RETRIES = 2;
const USER_AGENT = 'Mozilla/5.0 (compatible; ShopifyDetector/1.0; +https://selofy.com)';

const url = process.argv.find((value, index) => index > 0 && !value.startsWith("--"));
const outputFlag = process.argv.indexOf("--output");
const outputPath = outputFlag >= 0 ? process.argv[outputFlag + 1] : null;
if (!url) {
  console.error('Usage: node store-scanner.mjs <url>');
  process.exit(1);
}
try {
  let tempUrl = url.trim();
  if (!/^https?:\/\//i.test(tempUrl)) tempUrl = 'https://' + tempUrl;
  validateSafeUrl(tempUrl);
} catch (e) {
  console.error(`ERROR: ${e.message}`);
  process.exit(1);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function validateSafeUrl(value) {
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error(`Invalid protocol: "${parsed.protocol}". Only HTTP and HTTPS are allowed.`);
    }
    const hostname = parsed.hostname.toLowerCase();
    if (
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "0.0.0.0" ||
      hostname.startsWith("10.") ||
      hostname.startsWith("192.168.") ||
      (hostname.startsWith("172.") &&
        Number(hostname.split(".")[1]) >= 16 &&
        Number(hostname.split(".")[1]) <= 31)
    ) {
      throw new Error(`Access to private address "${hostname}" is blocked.`);
    }
    return parsed.href;
  } catch (err) {
    throw new Error(`Invalid or unsafe URL "${value}": ${err.message}`);
  }
}

function normalizeUrl(raw) {
  let u = raw.trim();
  if (!/^https?:\/\//i.test(u)) u = 'https://' + u;
  const parsed = new URL(u);
  return parsed.origin; // strip path
}

async function fetchWithTimeout(url, opts = {}) {
  return fetchPublic(url, opts, { timeoutMs: TIMEOUT_MS });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function writeHtmlReport(bundle, output) {
  if (!output) return null;
  const template = await fs.readFile(new URL("../templates/report-template.html", import.meta.url), "utf8");
  const domain = (() => { try { return new URL(bundle.storeUrl).hostname; } catch { return "unknown"; } })();
  const theme = bundle.shopifyTheme || {};
  const values = {
    STORE_DOMAIN: domain,
    STORE_URL: bundle.storeUrl,
    SCAN_DATE: bundle.scannedAt,
    THEME_NAME: theme.name || "Unknown",
    THEME_NAME_SHORT: theme.name || "Unknown",
    THEME_SCHEMA_NAME: theme.schema_name || "unknown",
    THEME_ENTITY_ID: theme.id || "unknown",
    THEME_VERSION: theme.version || "unknown",
    THEME_STORE_ID: theme.theme_store_id || "unknown",
    CONFIRMED_COUNT: 0,
    PROBABLE_COUNT: 0,
    CLUES_COUNT: bundle.shopifySignals?.length || 0,
  };
  let html = template.replace(/\{\{([A-Z0-9_]+)\}\}/g, (match, key) => escapeHtml(values[key] ?? ""));
  const safeJson = JSON.stringify(bundle).replaceAll("<", "\\u003c").replaceAll(">", "\\u003e").replaceAll("&", "\\u0026");
  const evidence = `<section style="margin:24px;padding:24px;border:1px solid #dbe3ef;border-radius:12px;background:#fff"><h2>Raw evidence bundle</h2><p>This report contains static scan evidence only.</p><pre style="white-space:pre-wrap;overflow:auto">${escapeHtml(JSON.stringify(bundle, null, 2))}</pre></section>`;
  html = html.replace("</body>", `${evidence}<script type="application/json" id="report-data">${safeJson}</script></body>`);
  await fs.writeFile(output, html, "utf8");
  return output;
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function extractHeaders(res) {
  const out = {};
  for (const [k, v] of res.headers.entries()) out[k.toLowerCase()] = v;
  return out;
}

// ── Shopify signal detectors ──────────────────────────────────────────────────

function detectShopifyFromHeaders(headers) {
  const signals = [];
  if (/shopify/i.test(headers['powered-by'] || '')) signals.push({ type: 'header', key: 'powered-by', value: headers['powered-by'] });
  const st = headers['server-timing'] || '';
  const themeMatch = st.match(/theme;desc="([^"]+)"/);
  if (themeMatch) signals.push({ type: 'header', key: 'server-timing-theme', value: themeMatch[1] });
  return { signals, serverTimingThemeId: themeMatch ? themeMatch[1] : null };
}

function detectShopifyFromHtml(html) {
  const signals = [];
  const htmlLower = html.toLowerCase();
  if (/cdn\.shopify\.com/.test(htmlLower)) signals.push({ type: 'html', key: 'cdn.shopify.com', value: 'Shopify CDN reference found' });
  if (/myshopify\.com/.test(htmlLower)) signals.push({ type: 'html', key: 'myshopify.com', value: 'myshopify.com domain reference found' });
  if (/window\.shopify\s*=/.test(htmlLower)) signals.push({ type: 'html', key: 'window.Shopify', value: 'window.Shopify object found' });
  return signals;
}

function extractShopifyTheme(html) {
  const start = html.search(/Shopify\.theme\s*=\s*\{/);
  if (start === -1) return null;
  const brace = html.indexOf('{', start);
  let depth = 0, i = brace;
  let inString = false, stringChar = null, escapeNext = false;
  for (; i < html.length; i++) {
    const ch = html[i];
    if (escapeNext) { escapeNext = false; continue; }
    if (inString) {
      if (ch === '\\') { escapeNext = true; continue; }
      if (ch === stringChar) { inString = false; }
      continue;
    }
    if (ch === '"' || ch === "'") { inString = true; stringChar = ch; continue; }
    if (ch === '{') { depth++; continue; }
    if (ch === '}') { depth--; if (depth === 0) break; }
  }
  try { return JSON.parse(html.slice(brace, i + 1)); } catch { return null; }
}

function extractShopifyShop(html) {
  const m = html.match(/Shopify\.shop\s*=\s*["']([^"']+)["']/);
  return m ? m[1] : null;
}

function extractFavicon(html, baseUrl) {
  // Try multiple favicon link patterns in priority order
  const patterns = [
    /<link[^>]*rel=["'](?:shortcut )?icon["'][^>]*href=["']([^"']+)["']/i,
    /<link[^>]*href=["']([^"']+)["'][^>]*rel=["'](?:shortcut )?icon["']/i,
    /<link[^>]*rel=["']apple-touch-icon["'][^>]*href=["']([^"']+)["']/i,
    /<link[^>]*href=["']([^"']+)["'][^>]*rel=["']apple-touch-icon["']/i,
  ];
  for (const pat of patterns) {
    const m = html.match(pat);
    if (m) {
      const href = m[1];
      if (/^https?:\/\//i.test(href)) return href;
      if (/^\/\//.test(href)) return 'https:' + href;
      try { return new URL(href, baseUrl).href; } catch { continue; }
    }
  }
  return `${baseUrl}/favicon.ico`;
}

function extractScripts(html, baseOrigin) {
  const external = [];
  const appEmbeds = [];
  const inline = [];
  const re = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    const attrs = m[1];
    const body = m[2];
    const srcMatch = attrs.match(/src=["']([^"']+)["']/i);
    if (srcMatch) {
      const src = srcMatch[1];
      if (/^\/\//.test(src)) {
        external.push('https:' + src);
      } else if (/^https?:\/\//.test(src)) {
        if (/cdn\.shopify\.com\/extensions\//.test(src)) {
          appEmbeds.push(src);
        } else if (!/cdn\.shopify\.com\/s\/files/.test(src) && !/shopifycloud/.test(src)) {
          external.push(src);
        }
      }
    } else if (body.trim()) {
      inline.push(body.slice(0, 600));
    }
  }
  return { external: [...new Set(external)], appEmbeds: [...new Set(appEmbeds)], inline };
}

function extractInlineScriptUrls(html, storeOrigin) {
  // Extract URLs from inline script content (dynamic injection patterns)
  const urls = new Set();
  const re = /<script\b(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
  const urlRe = /https?:\/\/([a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,})(\/[^\s"'`)\]},;]*)?/g;
  let m;
  while ((m = re.exec(html)) !== null) {
    const body = m[1];
    let u;
    while ((u = urlRe.exec(body)) !== null) {
      const full = u[0].replace(/[.,;)}\]]+$/, '');
      const host = u[1].toLowerCase();
      if (/cdn\.shopify\.com|shopifycloud|shopifysvc/.test(host)) continue;
      if (host === new URL(storeOrigin).hostname) continue;
      urls.add(full);
    }
  }
  return [...urls];
}

function extractTrackingIds(html) {
  const ids = {};
  const gtm = [...new Set(html.match(/GTM-[A-Z0-9]{4,}/g) || [])];
  const ga4 = [...new Set((html.match(/\bG-[A-Z0-9]{8,10}\b/g) || []).filter(id => !/^G-[A-Z]{2,}/.test(id)))];
  const aw  = [...new Set(html.match(/\bAW-\d{9,}\b/g) || [])];
  const ua  = [...new Set(html.match(/\bUA-\d{5,}-\d+\b/g) || [])];
  if (gtm.length) ids.gtm = gtm;
  if (ga4.length) ids.ga4 = ga4;
  if (aw.length)  ids.googleAds = aw;
  if (ua.length)  ids.ua = ua;
  return ids;
}

function extractLazyQueueUrls(html, storeOrigin) {
  // Capture src URLs inside lazy-load queue patterns: ffLazyQueue, LazyQueue, lazyLoadScripts, etc.
  const urls = new Set();
  const queueRe = /(?:ffLazyQueue|LazyQueue|lazyLoadScripts|_lazyScripts)\s*[=.]*[^;]{0,50}?\[\s*\{[^}]{0,500}\}/gi;
  const srcRe = /['"](https?:\/\/[^'"]+)['"]/g;
  let m;
  while ((m = queueRe.exec(html)) !== null) {
    const block = m[0];
    let s;
    while ((s = srcRe.exec(block)) !== null) {
      const host = new URL(s[1]).hostname.toLowerCase();
      if (/cdn\.shopify\.com|shopifycloud/.test(host)) continue;
      if (host === new URL(storeOrigin).hostname) continue;
      urls.add(s[1]);
    }
  }
  return [...urls];
}

function extractLinkTags(html, storeOrigin) {
  const dnsPrefetch = [];
  const appEmbedCss = [];
  const re = /<link\b([^>]*)>/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    const attrs = m[1];
    const rel = (attrs.match(/rel=["']([^"']+)["']/i) || [])[1] || '';
    const href = (attrs.match(/href=["']([^"']+)["']/i) || [])[1] || '';
    if (!href) continue;
    if (/dns-prefetch|preconnect/.test(rel)) {
      const host = href.replace(/^https?:\/\//, '').split('/')[0];
      if (host && !/cdn\.shopify\.com|shopifycloud|shopifysvc/.test(host) && host !== new URL(storeOrigin).hostname) {
        dnsPrefetch.push(href);
      }
    }
    if (/stylesheet/.test(rel) && /cdn\.shopify\.com\/extensions\//.test(href)) {
      appEmbedCss.push(href);
    }
  }
  return { dnsPrefetch: [...new Set(dnsPrefetch)], appEmbedCss: [...new Set(appEmbedCss)] };
}

function extractWindowGlobals(html) {
  const globals = new Set();
  const patterns = [
    /\bwindow\.([A-Za-z_$][A-Za-z0-9_$]*)\s*=/g,
    /\bvar\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=/g,
  ];
  const skip = new Set(['Shopify', 'ShopifyAnalytics', 'ShopifyPay', 'ShopifyBuy', 'undefined', 'null', 'true', 'false', 'document', 'window', 'location', 'navigator', 'history', 'screen', 'performance', 'console', 'JSON', 'Math', 'Date', 'Array', 'Object', 'String', 'Number', 'Boolean', 'RegExp', 'Error', 'Promise', 'Symbol', 'Map', 'Set', 'WeakMap', 'WeakSet', 'Proxy', 'Reflect', 'Intl', 'i', 'j', 'k', 'n', 's', 't', 'e', 'r', 'a', 'b', 'c', 'd', 'f', 'g', 'h', 'l', 'm', 'o', 'p', 'q', 'u', 'v', 'w', 'x', 'y', 'z']);
  for (const pat of patterns) {
    let m;
    while ((m = pat.exec(html)) !== null) {
      const name = m[1];
      if (name.length > 2 && !skip.has(name)) globals.add(name);
    }
  }
  return [...globals].slice(0, 80);
}

function extractCssClassNamespaces(html) {
  // Tailwind base utility prefixes — not vendor signals
  const TAILWIND = new Set(['tracking','transition','inset','object','overflow','duration','aspect','border','left','opacity','max','swiper','cursor','pointer','top','shadow','right','bottom','btn','min','ease','shrink','line','peer','grid','space','backdrop','align','group','col','inline','from','auto','row','select','order','whitespace','prose','flex','text','font','bg','p','px','py','pt','pb','pl','pr','m','mx','my','mt','mb','ml','mr','w','h','gap','ring','rounded','leading','justify','items','size','fixed','hidden','block','relative','absolute','static','sticky','z','sr','not','list','table','caption','float','clear','break','box','decoration','underline','italic','normal','antialiased','truncate','wrap','nowrap','scale','rotate','translate','skew','origin','accent','caret','fill','stroke','outline','placeholder','divide','via','to','grow','basis','self','place','content','visible','invisible','pointer','touch','resize','appearance','will','scroll','snap','overscroll','columns','break']);
  const counts = {};
  const re = /\bclass=["']([^"']+)["']/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    for (const cls of m[1].split(/\s+/)) {
      if (!cls.includes('-')) continue;
      const prefix = cls.split('-')[0];
      if (!prefix || prefix.length <= 2 || prefix.includes(':') || prefix.startsWith('!')) continue;
      if (TAILWIND.has(prefix)) continue;
      counts[prefix] = (counts[prefix] || 0) + 1;
    }
  }
  return Object.entries(counts)
    .filter(([, c]) => c >= 3 && c <= 200)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 40)
    .map(([prefix, count]) => ({ prefix, count }));
}

function extractDataAttributes(html) {
  const attrs = new Set();
  const re = /\b(data-[a-z][a-z0-9-]*)/gi;
  let m;
  while ((m = re.exec(html)) !== null) attrs.add(m[1].toLowerCase());
  return [...attrs].filter(a => !['data-id', 'data-src', 'data-href', 'data-url', 'data-type', 'data-value', 'data-name', 'data-key', 'data-index', 'data-target', 'data-action', 'data-toggle', 'data-dismiss', 'data-placement', 'data-content', 'data-title', 'data-original', 'data-lazy', 'data-width', 'data-height', 'data-alt', 'data-class', 'data-style', 'data-text', 'data-label', 'data-icon', 'data-color', 'data-size', 'data-count', 'data-page', 'data-limit', 'data-offset', 'data-sort', 'data-filter', 'data-search', 'data-query', 'data-params', 'data-options', 'data-config', 'data-settings', 'data-attr', 'data-tag', 'data-role', 'data-state', 'data-status', 'data-mode', 'data-format', 'data-locale', 'data-lang', 'data-currency', 'data-price', 'data-product', 'data-variant', 'data-handle', 'data-section', 'data-block', 'data-template', 'data-theme', 'data-shopify'].includes(a)).slice(0, 50);
}

function extractJsonLd(html) {
  const blocks = [];
  const re = /<script[^>]+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    try { blocks.push(JSON.parse(m[1])); } catch { /* skip malformed */ }
  }
  return blocks;
}

function extractHtmlComments(html) {
  const comments = [];
  const re = /<!--([\s\S]*?)-->/g;
  let m;
  while ((m = re.exec(html)) !== null) {
    const c = m[1].trim();
    if (c && c.length > 5 && c.length < 300) comments.push(c);
  }
  return [...new Set(comments)].slice(0, 30);
}

function extractAppBlockComments(html) {
  const slugs = [];
  const re = /<!--\s*BEGIN app block:\s*shopify:\/\/apps\/([^/]+)\//g;
  let m;
  while ((m = re.exec(html)) !== null) slugs.push(m[1]);
  return [...new Set(slugs)];
}

function stripHtmlComments(html) {
  return html.replace(/<!--[\s\S]*?-->/g, '');
}

function extractBodyClasses(html) {
  const m = html.match(/<body\b[^>]*class=["']([^"']+)["']/i);
  return m ? m[1].split(/\s+/).filter(Boolean) : [];
}

function extractMetaTags(html) {
  const metas = [];
  const re = /<meta\b([^>]*)>/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    const attrs = m[1];
    const name = (attrs.match(/name=["']([^"']+)["']/i) || [])[1];
    const content = (attrs.match(/content=["']([^"']+)["']/i) || [])[1];
    if (name && content) metas.push({ name, content });
  }
  return metas.filter(mt => /shopify|theme|generator|platform|smartbanner|apple-itunes-app|google-play-app|app-argument/i.test(mt.name)).slice(0, 20);
}

function detectPasswordPage(html) {
  // Only match actual password-protection pages: a form that submits to /password
  // AND contains a password input field. Avoids false positives from JS strings.
  return /<form[^>]+action=["'][^"']*\/password["'][^>]*>/i.test(html) &&
    /<input[^>]+type=["']password["']/i.test(html);
}

// ── Page scanner ──────────────────────────────────────────────────────────────

async function scanPage(pageUrl, pageType) {
  let res, html, headers;
  let effectiveUrl = pageUrl;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      res = await fetchWithTimeout(pageUrl, {
        headers: { 'User-Agent': USER_AGENT, 'Accept': 'text/html,application/xhtml+xml', 'Accept-Language': 'en-US,en;q=0.9' }
      });
      headers = extractHeaders(res);
      effectiveUrl = res.url;
      if (res.ok) break;
      if ((res.status === 429 || res.status >= 500) && attempt < MAX_RETRIES) {
        await sleep(REQUEST_DELAY_MS * Math.pow(2, attempt));
        continue;
      }
      return { url: pageUrl, effectiveUrl, pageType, status: res.status, error: `HTTP ${res.status}` };
    } catch (err) {
      if (attempt < MAX_RETRIES) {
        await sleep(REQUEST_DELAY_MS * Math.pow(2, attempt));
        continue;
      }
      return { url: pageUrl, pageType, error: err.message, status: null };
    }
  }
  html = await res.text();

  const isPasswordPage = detectPasswordPage(html);
  const shopifyTheme = extractShopifyTheme(html);
  const shopifyShop = extractShopifyShop(html);
  const htmlNoComments = stripHtmlComments(html);
  const { external, appEmbeds, inline } = extractScripts(htmlNoComments, new URL(effectiveUrl).origin);
  const inlineScriptUrls = extractInlineScriptUrls(html, new URL(effectiveUrl).origin);
  const lazyQueueUrls = extractLazyQueueUrls(html, new URL(effectiveUrl).origin);
  const trackingIds = extractTrackingIds(html);
  const { dnsPrefetch, appEmbedCss } = extractLinkTags(html, new URL(effectiveUrl).origin);
  const windowGlobals = extractWindowGlobals(html);
  const cssClassNamespaces = extractCssClassNamespaces(html);
  const dataAttributes = extractDataAttributes(html);
  const jsonLdBlocks = extractJsonLd(html);
  const htmlComments = extractHtmlComments(html);
  const appBlockComments = extractAppBlockComments(html);
  const bodyClasses = extractBodyClasses(html);
  const metaTags = extractMetaTags(html);
  const htmlSignals = detectShopifyFromHtml(html);
  const favicon = extractFavicon(html, effectiveUrl);

  return {
    url: pageUrl,
    effectiveUrl,
    pageType,
    status: res.status,
    isPasswordPage,
    favicon,
    shopifyTheme,
    shopifyShop,
    scripts: { external, appEmbeds, inlineScriptUrls, lazyQueueUrls, inlineSnippets: inline.slice(0, 5) },
    trackingIds,
    dnsPrefetch,
    appEmbedCss,
    windowGlobals,
    cssClassNamespaces,
    dataAttributes,
    jsonLdBlocks: jsonLdBlocks.slice(0, 5),
    htmlComments,
    appBlockComments,
    bodyClasses,
    metaTags,
    htmlSignals,
    htmlSize: html.length,
  };
}

// ── Product page discovery ────────────────────────────────────────────────────

async function discoverProductUrl(origin) {
  try {
    const res = await fetchWithTimeout(`${origin}/products.json?limit=1`, {
      headers: { 'User-Agent': USER_AGENT }
    });
    if (res.status !== 200) return null;
    const data = await res.json();
    const handle = data?.products?.[0]?.handle;
    return handle ? `${origin}/products/${handle}` : null;
  } catch { return null; }
}

async function discoverCollectionUrl(origin) {
  try {
    const res = await fetchWithTimeout(`${origin}/collections.json?limit=1`, {
      headers: { 'User-Agent': USER_AGENT }
    });
    if (res.status !== 200) return null;
    const data = await res.json();
    const handle = data?.collections?.[0]?.handle;
    return handle ? `${origin}/collections/${handle}` : null;
  } catch { return null; }
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  let storeUrl;
  try {
    storeUrl = normalizeUrl(url);
  } catch (err) {
    console.error(JSON.stringify({ error: `Invalid URL: ${err.message}` }));
    process.exit(1);
  }

  // Step 1: HEAD request for headers
  let headHeaders = {};
  let serverTimingThemeId = null;
  try {
    const headRes = await fetchWithTimeout(storeUrl, {
      method: 'HEAD',
      headers: { 'User-Agent': USER_AGENT }
    });
    headHeaders = extractHeaders(headRes);
  } catch { /* fall through to GET */ }

  const { signals: headerSignals, serverTimingThemeId: stId } = detectShopifyFromHeaders(headHeaders);
  serverTimingThemeId = stId;

  // Step 2: Scan homepage
  const homePage = await scanPage(storeUrl, 'home');
  await sleep(REQUEST_DELAY_MS);

  // Step 3: Determine if Shopify
  const allShopifySignals = [
    ...headerSignals,
    ...(homePage.htmlSignals || []),
  ];
  const isShopify = allShopifySignals.length > 0 || !!homePage.shopifyTheme || !!homePage.shopifyShop;

  if (!isShopify) {
    const bundle = {
      storeUrl,
      storeFavicon: homePage.favicon || `${storeUrl}/favicon.ico`,
      isShopify: false,
      shopifySignals: [],
      pages: [homePage],
      responseHeaders: headHeaders,
      errors: [],
      scannedAt: new Date().toISOString(),
    };
    await writeHtmlReport(bundle, outputPath);
    console.log(JSON.stringify(bundle, null, 2));
    return;
  }

  const pages = [homePage];

  // Step 4: Discover and scan product page
  if (pages.length < MAX_PAGES) {
    const productUrl = await discoverProductUrl(storeUrl);
    if (productUrl) {
      await sleep(REQUEST_DELAY_MS);
      const productPage = await scanPage(productUrl, 'product');
      pages.push(productPage);
    }
  }

  // Step 5: Discover and scan collection page
  if (pages.length < MAX_PAGES) {
    const collectionUrl = await discoverCollectionUrl(storeUrl);
    if (collectionUrl) {
      await sleep(REQUEST_DELAY_MS);
      const collectionPage = await scanPage(collectionUrl, 'collection');
      pages.push(collectionPage);
    }
  }

  // Aggregate all external scripts and app embeds across pages
  const allExternalScripts = [...new Set(pages.flatMap(p => p.scripts?.external || []))];
  const allInlineScriptUrls = [...new Set(pages.flatMap(p => p.scripts?.inlineScriptUrls || []))];
  const allLazyQueueUrls = [...new Set(pages.flatMap(p => p.scripts?.lazyQueueUrls || []))];
  const allAppEmbeds = [...new Set(pages.flatMap(p => p.scripts?.appEmbeds || []))];
  const allAppEmbedCss = [...new Set(pages.flatMap(p => p.appEmbedCss || []))];
  const allAppBlockComments = [...new Set(pages.flatMap(p => p.appBlockComments || []))];
  const allWindowGlobals = [...new Set(pages.flatMap(p => p.windowGlobals || []))];
  const allDnsPrefetch = [...new Set(pages.flatMap(p => p.dnsPrefetch || []))];

  // Merge tracking IDs across pages
  const mergedTrackingIds = {};
  for (const p of pages) {
    for (const [k, v] of Object.entries(p.trackingIds || {})) {
      mergedTrackingIds[k] = [...new Set([...(mergedTrackingIds[k] || []), ...v])];
    }
  }
  // Find the best shopifyTheme (prefer product page if available)
  const shopifyTheme = pages.find(p => p.shopifyTheme)?.shopifyTheme || null;
  const shopifyShop = pages.find(p => p.shopifyShop)?.shopifyShop || null;

  const storeFavicon = pages.find(p => p.favicon)?.favicon || `${storeUrl}/favicon.ico`;

  const bundle = {
    storeUrl,
    storeFavicon,
    isShopify: true,
    shopifyShop,
    shopifyTheme,
    serverTimingThemeId,
    shopifySignals: allShopifySignals,
    responseHeaders: headHeaders,
    pages,
    aggregated: {
      externalScripts: allExternalScripts,
      inlineScriptUrls: allInlineScriptUrls,
      lazyQueueUrls: allLazyQueueUrls,
      appEmbedScripts: allAppEmbeds,
      appEmbedCss: allAppEmbedCss,
      appBlockComments: allAppBlockComments,
      windowGlobals: allWindowGlobals,
      dnsPrefetch: allDnsPrefetch,
      trackingIds: mergedTrackingIds,
    },
    errors: pages.filter(p => p.error).map(p => ({ url: p.url, error: p.error })),
    scannedAt: new Date().toISOString(),
  };

  const report = await writeHtmlReport(bundle, outputPath);
  console.log(JSON.stringify({ ...bundle, reportPath: report }, null, 2));
}

main().catch(err => {
  console.error(JSON.stringify({ error: err.message }));
  process.exit(1);
});
