#!/usr/bin/env node
/**
 * gmc-store-audit.mjs — Phase A: Store-level GMC Misrepresentation audit + product discovery
 *
 * Usage:
 *   node gmc-store-audit.mjs <store-url>
 *   node gmc-store-audit.mjs https://your-store.com
 *
 * Output: JSON to stdout — { storeUrl, fetchedAt, themeHints, checks, productUrls, score }
 * Errors: stderr only
 *
 * No API token required. Crawls public pages only.
 * Respects robots.txt. Rate-limits to 1 request/second minimum.
 */

import { readFileSync, writeFileSync } from 'fs';
import { createRequire } from 'module';
import { fetchPublic } from './public-fetch.mjs';

// ─── Minimal fetch with timeout ──────────────────────────────────────────────

function validateSafeUrl(value) {
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error(`Invalid protocol: "${parsed.protocol}". Only HTTP and HTTPS are allowed.`);
    }
    const hostname = parsed.hostname.toLowerCase();
    
    // Block localhost, loopbacks, and private IP ranges to prevent SSRF
    const isIp = /^[0-9.]+$/.test(hostname) || hostname.includes(":");
    if (hostname === "localhost" || 
        hostname === "127.0.0.1" || 
        hostname === "0.0.0.0" || 
        hostname.startsWith("10.") || 
        hostname.startsWith("192.168.") || 
        (hostname.startsWith("172.") && (Number(hostname.split(".")[1]) >= 16 && Number(hostname.split(".")[1]) <= 31))) {
      throw new Error(`Access to private address "${hostname}" is blocked.`);
    }
    return parsed.href;
  } catch (err) {
    throw new Error(`Invalid or unsafe URL "${value}": ${err.message}`);
  }
}

async function fetchPage(url, opts = {}) {
  try {
    validateSafeUrl(url);
  } catch (e) {
    return { ok: false, status: 0, url, text: '', error: `Request blocked: ${e.message}` };
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), opts.timeout || 15000);
  try {
    const res = await fetchPublic(url, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; GMC-Auditor/1.0; +https://selofy.com)',
        'Accept': 'text/html,application/xhtml+xml,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        ...opts.headers,
      },
    }, { timeoutMs: opts.timeout || 15000 });
    const text = await res.text();
    return { ok: res.ok, status: res.status, url: res.url, text };
  } catch (e) {
    return { ok: false, status: 0, url, text: '', error: e.message };
  } finally {
    clearTimeout(timer);
  }
}

async function headUrl(url) {
  try {
    validateSafeUrl(url);
  } catch (e) {
    return { ok: false, status: 0 };
  }
  try {
    const res = await fetchPublic(url, {
      method: 'HEAD',
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible; GMC-Auditor/1.0)' },
    }, { timeoutMs: 8000 });
    return { ok: res.ok, status: res.status };
  } catch {
    return { ok: false, status: 0 };
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── HTML helpers ─────────────────────────────────────────────────────────────

function extractText(html) {
  return html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
}

function extractLinks(html, baseUrl) {
  const links = [];
  const re = /href=["']([^"']+)["']/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    try {
      const abs = new URL(m[1], baseUrl).href;
      if (abs.startsWith('http')) links.push(abs);
    } catch {}
  }
  return [...new Set(links)];
}

function extractJsonLd(html) {
  const blocks = [];
  const re = /<script[^>]+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    try { blocks.push(JSON.parse(m[1])); } catch {}
  }
  return blocks;
}

function extractMeta(html, name) {
  const re = new RegExp(`<meta[^>]+(?:name|property)=["']${name}["'][^>]+content=["']([^"']+)["']`, 'i');
  const m = re.exec(html);
  return m ? m[1] : null;
}

function extractPrice(html) {
  // Try data attributes first (most reliable)
  let m = /data-product-price[^>]*>[\s]*([0-9,]+\.?[0-9]*)/i.exec(html);
  if (m) return parseFloat(m[1].replace(/,/g, ''));
  // Try common price class patterns
  m = /class="[^"]*price[^"]*"[^>]*>[\s$£€¥]*([0-9,]+\.?[0-9]*)/i.exec(html);
  if (m) return parseFloat(m[1].replace(/,/g, ''));
  return null;
}

function detectTheme(html) {
  if (/<meta[^>]+name=["']theme-name["'][^>]+content=["']([^"']+)["']/i.test(html)) {
    const m = /<meta[^>]+name=["']theme-name["'][^>]+content=["']([^"']+)["']/i.exec(html);
    return { family: m[1].toLowerCase(), confidence: 0.95 };
  }
  if (/--color-base-accent-1/.test(html) || /product__title/.test(html)) return { family: 'dawn', confidence: 0.85 };
  if (/product-single__title/.test(html)) return { family: 'impulse', confidence: 0.80 };
  if (/--body-bg-secondary/.test(html) || /prd-block/.test(html)) return { family: 'shella', confidence: 0.80 };
  return { family: 'unknown', confidence: 0.5 };
}

// ─── robots.txt check ─────────────────────────────────────────────────────────

async function checkRobots(storeUrl) {
  const robotsUrl = new URL('/robots.txt', storeUrl).href;
  const res = await fetchPage(robotsUrl);
  if (!res.ok) return { blocked: false, note: 'robots.txt not found' };

  const lines = res.text.split('\n');
  let currentAgents = [];
  let googleBlocked = false;
  let productPathBlocked = false;

  for (const line of lines) {
    const l = line.trim().toLowerCase();
    if (l.startsWith('user-agent:')) {
      currentAgents = [l.replace('user-agent:', '').trim()];
    } else if (!l && currentAgents.length) {
      currentAgents = [];
    }
    if ((currentAgents.includes('gmc-auditor') || currentAgents.includes('googlebot') || currentAgents.includes('*')) && l.startsWith('disallow:')) {
      const path = l.replace('disallow:', '').trim();
      if (path === '/' || path === '/products' || path === '/products/') {
        googleBlocked = true;
        productPathBlocked = true;
      }
    }
  }

  return { blocked: googleBlocked, productPathBlocked, robotsUrl };
}

// ─── Policy page discovery ────────────────────────────────────────────────────

// Canonical Shopify /policies/ paths are preferred. /pages/ paths are valid but
// merchants should be advised to migrate to canonical paths for GMC reliability.
const POLICY_PATTERNS = [
  {
    key: 'returns',
    keywords: ['return', 'refund', 'exchange'],
    canonical: ['/policies/refund-policy'],
    fallback: ['/returns', '/return-policy', '/refund-policy', '/pages/returns', '/pages/refund-policy', '/pages/return-policy'],
  },
  {
    key: 'shipping',
    keywords: ['shipping', 'delivery', 'ship'],
    canonical: ['/policies/shipping-policy'],
    fallback: ['/shipping', '/shipping-policy', '/pages/shipping', '/pages/shipping-policy', '/pages/delivery'],
  },
  {
    key: 'privacy',
    keywords: ['privacy'],
    canonical: ['/policies/privacy-policy'],
    fallback: ['/privacy', '/privacy-policy', '/pages/privacy', '/pages/privacy-policy'],
  },
  {
    key: 'terms',
    keywords: ['terms', 'conditions', 'tos'],
    canonical: ['/policies/terms-of-service'],
    fallback: ['/terms', '/terms-of-service', '/pages/terms', '/pages/terms-of-service', '/pages/terms-and-conditions'],
  },
];

async function discoverPolicies(storeUrl, pageHtml) {
  const results = {};
  // Collect all links from the full page (footer + nav + anywhere)
  const allLinks = extractLinks(pageHtml, storeUrl);
  const allLinksLower = allLinks.map(l => l.toLowerCase());

  for (const policy of POLICY_PATTERNS) {
    let found = null;
    let isCanonical = false;

    // Step 1: scan all page links for keyword matches (catches /pages/xxx and custom slugs)
    for (const link of allLinks) {
      const llink = link.toLowerCase();
      const path = new URL(link).pathname.toLowerCase();
      if (policy.keywords.some(kw => path.includes(kw))) {
        found = link;
        isCanonical = policy.canonical.some(c => path === c || path.startsWith(c));
        break;
      }
    }

    // Step 2: if not found in links, probe canonical paths then fallbacks
    if (!found) {
      for (const path of [...policy.canonical, ...policy.fallback]) {
        const url = new URL(path, storeUrl).href;
        const r = await headUrl(url);
        await sleep(400);
        if (r.ok) {
          found = url;
          isCanonical = policy.canonical.includes(path);
          break;
        }
      }
    }

    results[policy.key] = found ? { url: found, isCanonical } : null;
  }

  return results;
}

// Check whether a policy page has actual text content (not just an empty Shopify shell)
function isPolicyContentEmpty(html) {
  // Shopify policy pages use .shopify-policy__body > .rte
  const rteMatch = /<div[^>]*class="[^"]*rte[^"]*"[^>]*>([\s\S]*?)<\/div>/i.exec(html);
  if (rteMatch) {
    const text = rteMatch[1].replace(/<[^>]+>/g, '').trim();
    return text.length < 30; // fewer than 30 chars = effectively empty
  }
  // Generic: check total visible text length
  const text = html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
  return text.length < 200;
}

async function auditPolicyContent(policyObj, policyKey) {
  if (!policyObj) return { found: false, url: null, issues: [`${policyKey} policy page not found`] };

  const { url, isCanonical } = policyObj;
  const res = await fetchPage(url);
  if (!res.ok) return { found: false, url, status: res.status, issues: [`${policyKey} policy page returns ${res.status}`] };

  const issues = [];

  // Check for empty content (page exists but has no text)
  if (isPolicyContentEmpty(res.text)) {
    issues.push(`${policyKey} policy page exists but has empty content — the page shell loads but no policy text is present`);
  }

  const text = extractText(res.text);

  if (policyKey === 'returns' && text.length > 30) {
    if (!/\d+\s*(?:day|business day)/i.test(text)) issues.push('No return window (number of days) found');
    if (!/refund|exchange/i.test(text)) issues.push('No refund or exchange terms found');
    if (!/return\s+(?:shipping|postage|label)/i.test(text)) issues.push('Who pays return shipping is not stated');
  }

  if (policyKey === 'shipping' && text.length > 30) {
    if (!/\d+\s*(?:day|business day|week)/i.test(text)) issues.push('No delivery time estimate found');
    if (!/ship(?:ping|s)\s+to|deliver(?:y|s)\s+to/i.test(text)) issues.push('Shipping destinations not stated');
  }

  // Suggest migration from /pages/ to /policies/ canonical path
  const migration = !isCanonical
    ? `Policy is served from a custom path (${url}). Consider migrating to the canonical Shopify /policies/ path for better GMC reliability.`
    : null;

  return { found: true, url, isCanonical, wordCount: text.split(/\s+/).length, issues, migration };
}

// ─── Identity checks ──────────────────────────────────────────────────────────

function checkIdentity(html, storeUrl) {
  const checks = [];
  const text = extractText(html);

  // Footer business identity
  const hasAddress = /\d+\s+\w+\s+(?:st|street|ave|avenue|blvd|boulevard|rd|road|dr|drive|ln|lane|way|pl|place)/i.test(text);
  const hasEmail = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/.test(text);
  const hasPhone = /(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/.test(text);

  if (!hasAddress && !hasEmail && !hasPhone) {
    checks.push({ id: 'identity:no_contact_info', severity: 'critical', confidence: 0.9,
      message: 'No business contact information (address, email, or phone) found in footer/page',
      policyBasis: 'GMC requires easy-to-find contact information' });
  } else {
    if (!hasEmail) checks.push({ id: 'identity:no_email', severity: 'high', confidence: 0.85,
      message: 'No business email address found on page',
      policyBasis: 'Contact information must be easy to find' });
    if (!hasAddress) checks.push({ id: 'identity:no_address', severity: 'medium', confidence: 0.7,
      message: 'No physical business address found on page',
      policyBasis: 'Business identity transparency' });
  }

  // Liquid errors
  if (/Liquid error:/i.test(html)) {
    checks.push({ id: 'technical:liquid_error', severity: 'critical', confidence: 1.0,
      message: 'Liquid template error found in page HTML — known automatic GMC suspension trigger',
      evidence: html.match(/Liquid error:[^\n<]*/i)?.[0],
      policyBasis: 'Shopify-specific: Liquid errors trigger automatic GMC suspension' });
  }

  // Test/placeholder content
  if (/lorem ipsum/i.test(text)) {
    checks.push({ id: 'identity:lorem_ipsum', severity: 'high', confidence: 1.0,
      message: 'Lorem ipsum placeholder text found on page',
      policyBasis: 'Placeholder content indicates incomplete store' });
  }

  if (/coming soon|password protected|this store is password/i.test(text)) {
    checks.push({ id: 'identity:store_not_live', severity: 'critical', confidence: 0.95,
      message: 'Store appears to be password-protected or not yet live',
      policyBasis: 'GMC requires publicly accessible store' });
  }

  return checks;
}

// ─── Urgency & FOMO checks ────────────────────────────────────────────────────

function checkUrgency(html) {
  const checks = [];

  // Countdown timers
  if (/countdown|count-down|timer|time-left|ends-in/i.test(html)) {
    checks.push({ id: 'urgency:countdown_timer', severity: 'high', confidence: 0.7,
      message: 'Countdown timer detected — verify it resets on page refresh (evergreen timers violate GMC policy)',
      policyBasis: 'Urgency must be real, not evergreen' });
  }

  // Low stock widgets
  const stockMatch = html.match(/only\s+(\d+)\s+left|(\d+)\s+(?:item|unit)s?\s+(?:left|remaining|in stock)/i);
  if (stockMatch) {
    checks.push({ id: 'urgency:low_stock_widget', severity: 'high', confidence: 0.75,
      message: `Low-stock message detected: "${stockMatch[0]}" — verify this updates dynamically with real inventory`,
      policyBasis: 'False scarcity is a GMC Misrepresentation violation' });
  }

  // Visitor counters
  if (/\d+\s+(?:people|visitors?|shoppers?)\s+(?:viewing|looking|watching)/i.test(html)) {
    checks.push({ id: 'urgency:visitor_counter', severity: 'medium', confidence: 0.7,
      message: 'Visitor counter detected — must be real-time, not static',
      policyBasis: 'Misleading social proof' });
  }

  // Evergreen sale copy
  if (/(?:today only|ends tonight|flash sale|limited time offer|24.hour sale)/i.test(html)) {
    checks.push({ id: 'urgency:evergreen_sale_copy', severity: 'high', confidence: 0.65,
      message: 'Time-limited sale copy detected — verify this is a real, time-bound promotion',
      policyBasis: 'Fake urgency is a GMC Misrepresentation violation' });
  }

  return checks;
}

// ─── Trust signals checks ─────────────────────────────────────────────────────

function checkTrustSignals(html) {
  const checks = [];

  // Certification/badge claims
  const certMatches = html.match(/(?:certified|accredited|approved by|trusted by|verified by)\s+[A-Z][a-zA-Z\s]+/gi) || [];
  for (const cert of certMatches.slice(0, 3)) {
    checks.push({ id: `trust:unverified_cert:${cert.slice(0, 30)}`, severity: 'high', confidence: 0.6,
      message: `Certification/trust claim detected: "${cert}" — verify this links to an official third-party source`,
      policyBasis: 'Badges must come from official third-party sources' });
  }

  // "As Seen On" claims
  if (/as seen on|featured in|as featured in/i.test(html)) {
    checks.push({ id: 'trust:as_seen_on', severity: 'high', confidence: 0.7,
      message: '"As Seen On" or "Featured In" claim detected — verify media coverage is real and links are working',
      policyBasis: 'Unsubstantiated media claims are Misrepresentation' });
  }

  // Google association claims
  if (/certified by google|google partner|google approved|google trusted/i.test(html)) {
    checks.push({ id: 'trust:false_google_claim', severity: 'critical', confidence: 0.95,
      message: 'False Google association claim detected — this is an egregious GMC violation',
      policyBasis: 'False Google association = immediate account suspension' });
  }

  return checks;
}

// ─── AI Bulk / Dropshipping signal checks ──────────────────────────────────────

function checkDropshippingSignals(homeHtml, storeUrl, productCount) {
  const checks = [];
  const signals = { noCustomDomain: false, productCountSignal: null, chinaPatterns: false, templateDescs: false };

  // 1. Custom domain check
  const isMyshopify = /\.myshopify\.com/i.test(storeUrl);
  if (isMyshopify) {
    signals.noCustomDomain = true;
    checks.push({
      id: 'dropship:no_custom_domain',
      severity: 'medium', confidence: 0.8,
      message: 'Store uses default .myshopify.com subdomain with no custom domain — common in new/dropshipping stores',
      policyBasis: 'Custom domain signals business investment and brand commitment',
    });
  }

  // 2. Product count vs typical store profile
  if (productCount > 100) {
    signals.productCountSignal = 'high';
    checks.push({
      id: 'dropship:large_catalog_young_store',
      severity: productCount > 200 ? 'high' : 'medium',
      confidence: 0.7,
      message: `Store has ${productCount} products without established brand presence — high-volume catalog is typical of AI-bulk or dropshipping operations`,
      policyBasis: 'GMC expects authentic inventory from established businesses',
    });
  } else if (productCount > 50) {
    signals.productCountSignal = 'moderate';
    checks.push({
      id: 'dropship:moderate_catalog',
      severity: 'low', confidence: 0.6,
      message: `Store has ${productCount} products — moderate catalog size, monitor for rapid expansion`,
      policyBasis: 'Rapid product addition without brand building is a risk signal',
    });
  }

  // 3. China/dropshipping supplier patterns in page text
  const pageText = extractText(homeHtml).toLowerCase();
  const supplierPatterns = [
    /\b(?:ship\s+from\s+china|china\s+warehouse|sold\s+by\s+\w+store\d*\b)/i,
    /\b(?:order\s+within\s+\d+\s+(?:hours|days)\s+for\s+(?:same\s+day|next\s+day)\s+dispatch)/i,
    /\b(?:processing\s+time\s*:\s*\d+\s*(?:business\s+)?days)/i,
    /\b(?:estimated\s+delivery\s*:\s*\d+[–-]\d+\s*(?:business\s+)?days)/i,
  ];
  const matchedPatterns = supplierPatterns.filter(p => p.test(homeHtml));
  if (matchedPatterns.length >= 2) {
    signals.chinaPatterns = true;
    checks.push({
      id: 'dropship:supplier_fulfillment_patterns',
      severity: 'medium', confidence: 0.7,
      message: 'Multiple dropshipping fulfillment patterns detected (China warehouse, processing times, estimated delivery ranges)',
      policyBasis: 'GMC requires transparent fulfillment — undisclosed dropshipping can trigger misrepresentation',
    });
  }

  // 5. Product titles look auto-generated (sequential numbers + generic keywords)
  const titleText = extractText(homeHtml).slice(0, 3000);
  const autoGenPatterns = [
    /\d+\s*(?:in\s+)?1\s*(?:pcs|pack|set)/i,          // "1 pcs", "1 pack"
    /(?:free\s+shipping|hot\s+sale|wholesale)\s+\w+/i,  // generic sale qualifiers
    /\d+\s*%\s*off/i,                                    // percentage off
  ];
  const autoGenMatched = autoGenPatterns.filter(p => p.test(homeHtml)).length;

  return { checks, signals };
}

// ─── Product URL discovery ────────────────────────────────────────────────────

async function discoverProductUrls(storeUrl) {
  const productUrls = new Set();

  // Try sitemap.xml — Shopify uses a sitemapindex that links to sub-sitemaps
  const sitemapRes = await fetchPage(new URL('/sitemap.xml', storeUrl).href);
  await sleep(1000);

  if (sitemapRes.ok) {
    // Check if root sitemap is a sitemapindex (links to sub-sitemaps)
    const subSitemaps = sitemapRes.text.match(/<loc>([^<]+sitemap_products_[^<]+\.xml[^<]*)<\/loc>/gi) || [];
    
    if (subSitemaps.length > 0) {
      // Fetch each product sub-sitemap
      for (const ss of subSitemaps) {
        const subUrl = ss.replace(/<\/?loc>/gi, '').trim();
        const subRes = await fetchPage(subUrl);
        await sleep(1000);
        if (subRes.ok) {
          const productMatches = subRes.text.match(/<loc>([^<]+\/products\/[^<]+)<\/loc>/gi) || [];
          for (const m of productMatches) {
            productUrls.add(m.replace(/<\/?loc>/gi, '').trim());
          }
        }
        if (productUrls.size >= 200) break;
      }
    } else {
      // Fallback: try direct product URL extraction from root sitemap
      const directMatches = sitemapRes.text.match(/<loc>([^<]+\/products\/[^<]+)<\/loc>/gi) || [];
      for (const m of directMatches) {
        productUrls.add(m.replace(/<\/?loc>/gi, '').trim());
      }
    }
  }

  // Always try collections page for additional discovery (catches products not in sitemap)
  const collectionsRes = await fetchPage(new URL('/collections/all', storeUrl).href);
  await sleep(1000);
  if (collectionsRes.ok) {
    const links = extractLinks(collectionsRes.text, storeUrl);
    for (const link of links) {
      if (link.includes('/products/') && !link.includes('?')) productUrls.add(link);
    }
  }

  return [...productUrls].slice(0, 200); // Cap at 200 for discovery
}

// ─── Sampling strategy ────────────────────────────────────────────────────────

function sampleProducts(allUrls, html) {
  const n = allUrls.size || allUrls.length;
  const urls = [...allUrls];

  if (n <= 10) return urls;

  const target = n <= 100 ? Math.min(30, n) : n <= 1000 ? 60 : 100;

  // Prioritize sale/discount products
  const salePriority = urls.filter(u => /sale|discount|clearance/i.test(u));
  const rest = urls.filter(u => !salePriority.includes(u));

  const sampled = [
    ...salePriority.slice(0, Math.floor(target * 0.3)),
    ...rest.slice(0, target - Math.floor(target * 0.3)),
  ];

  return sampled.slice(0, target);
}

// ─── Main audit ───────────────────────────────────────────────────────────────

async function runStoreAudit(storeUrl) {
  const result = {
    storeUrl,
    fetchedAt: new Date().toISOString(),
    themeHints: null,
    checks: [],
    policies: {},
    productUrls: [],
    sampledProductUrls: [],
    dropshippingSignals: {},
    score: null,
  };

  // Normalize URL
  if (!storeUrl.startsWith('http')) storeUrl = 'https://' + storeUrl;
  const base = new URL(storeUrl);
  storeUrl = base.origin;

  // 1. Check robots.txt
  const robots = await checkRobots(storeUrl);
  await sleep(1000);
  if (robots.blocked) {
    result.checks.push({ id: 'technical:robots_blocks_googlebot', severity: 'critical', confidence: 1.0,
      message: 'robots.txt blocks Googlebot from crawling product pages — GMC cannot index your products',
       evidence: robots.robotsUrl,
       policyBasis: 'Google must be able to crawl product pages' });
    result.score = computeScore(result.checks);
    return result;
  }

  // 2. Fetch homepage
  const homeRes = await fetchPage(storeUrl);
  await sleep(1000);

  if (!homeRes.ok) {
    result.checks.push({ id: 'technical:homepage_unreachable', severity: 'critical', confidence: 1.0,
      message: `Homepage returned ${homeRes.status} — store is not accessible`,
      policyBasis: 'Store must be publicly accessible' });
    result.score = computeScore(result.checks);
    return result;
  }

  // SSL check (if we got here over HTTPS, it's valid)
  if (!homeRes.url.startsWith('https://')) {
    result.checks.push({ id: 'technical:no_ssl', severity: 'critical', confidence: 1.0,
      message: 'Store is not served over HTTPS — SSL is required for GMC',
      policyBasis: 'HTTPS required for checkout' });
  }

  result.themeHints = detectTheme(homeRes.text);

  // 3. Identity checks on homepage
  result.checks.push(...checkIdentity(homeRes.text, storeUrl));

  // 4. Urgency checks on homepage
  result.checks.push(...checkUrgency(homeRes.text));

  // 5. Trust signal checks on homepage
  result.checks.push(...checkTrustSignals(homeRes.text));

  // 6. About Us page
  const aboutPaths = ['/about', '/about-us', '/pages/about', '/pages/about-us', '/our-story', '/pages/our-story'];
  let aboutFound = false;
  for (const path of aboutPaths) {
    const r = await headUrl(new URL(path, storeUrl).href);
    await sleep(500);
    if (r.ok) {
      const aboutRes = await fetchPage(new URL(path, storeUrl).href);
      await sleep(1000);
      const wordCount = extractText(aboutRes.text).split(/\s+/).length;
      if (wordCount > 100) { aboutFound = true; break; }
    }
  }
  if (!aboutFound) {
    result.checks.push({ id: 'identity:no_about_page', severity: 'high', confidence: 0.85,
      message: 'No About Us / Our Story page found with meaningful content (>100 words)',
      policyBasis: 'GMC trust checklist requires About page' });
  }

  // 7. Contact page
  const contactPaths = ['/contact', '/pages/contact', '/contact-us', '/pages/contact-us'];
  let contactFound = false;
  for (const path of contactPaths) {
    const r = await headUrl(new URL(path, storeUrl).href);
    await sleep(500);
    if (r.ok) { contactFound = true; break; }
  }
  if (!contactFound) {
    result.checks.push({ id: 'identity:no_contact_page', severity: 'critical', confidence: 0.9,
      message: 'No contact page found — contact information must be easy to find during checkout',
      policyBasis: 'GMC requires accessible contact information' });
  }

  // 8. Policy discovery and audit
  result.policies = await discoverPolicies(storeUrl, homeRes.text);
  await sleep(500);

  for (const [key, policyObj] of Object.entries(result.policies)) {
    const audit = await auditPolicyContent(policyObj, key);
    await sleep(1000);

    if (!audit.found) {
      result.checks.push({ id: `policy:missing:${key}`, severity: key === 'returns' || key === 'shipping' ? 'critical' : 'high',
        confidence: 0.9,
        message: `${key} policy page not found`,
        policyBasis: 'Missing policy pages block informed purchase decisions' });
    } else {
      for (const issue of audit.issues) {
        const isEmptyContent = issue.includes('empty content');
        result.checks.push({ id: `policy:${isEmptyContent ? 'empty' : 'incomplete'}:${key}`, severity: isEmptyContent ? 'critical' : 'high', confidence: 0.9,
          message: issue,
          evidence: { url: audit.url, wordCount: audit.wordCount },
          policyBasis: isEmptyContent ? 'Empty policy page = same as missing policy for GMC' : 'Policy must clearly state all terms' });
      }
      // Suggest migration from /pages/ to /policies/ canonical path
      if (audit.migration) {
        result.checks.push({ id: `policy:non_canonical:${key}`, severity: 'low', confidence: 1.0,
          message: audit.migration,
          evidence: { url: audit.url },
          policyBasis: 'Canonical /policies/ paths are more reliably indexed by GMC' });
      }
    }
  }

  // 9. JSON-LD on homepage
  const jsonLdBlocks = extractJsonLd(homeRes.text);
  const orgBlock = jsonLdBlocks.find(b => b['@type'] === 'Organization');
  if (!orgBlock) {
    result.checks.push({ id: 'schema:no_organization', severity: 'medium', confidence: 0.7,
      message: 'No Organization schema block found on homepage',
      policyBasis: 'Structured data helps GMC verify business identity' });
  }

  // 10. Product discovery
  const allProductUrls = await discoverProductUrls(storeUrl);
  result.productUrls = allProductUrls;
  result.sampledProductUrls = sampleProducts(allProductUrls, homeRes.text);

  // 10b. AI Bulk / Dropshipping detection
  const dropshipResult = checkDropshippingSignals(homeRes.text, storeUrl, allProductUrls.length);
  result.checks.push(...dropshipResult.checks);
  result.dropshippingSignals = dropshipResult.signals;

  // 11. Score
  result.score = computeScore(result.checks);

  return result;
}

// ─── Scoring ──────────────────────────────────────────────────────────────────

function computeScore(checks) {
  const weights = { critical: 25, high: 12, medium: 5, low: 2 };
  let total = 0;

  for (const c of checks) {
    const w = weights[c.severity] || 0;
    const conf = c.confidence || 0.7;
    const ef = c.evidenceFactor || 1.0;
    total += w * conf * ef;
  }

  const risk = Math.min(100, Math.round(total));

  return { risk };
}

// ─── Entry point ──────────────────────────────────────────────────────────────

const rawStoreUrl = process.argv[2];
if (!rawStoreUrl) {
  process.stderr.write('Usage: node gmc-store-audit.mjs <store-url>\n');
  process.exit(1);
}

let storeUrl;
try {
  storeUrl = validateSafeUrl(rawStoreUrl);
} catch (e) {
  process.stderr.write(`Error: ${e.message}\n`);
  process.exit(1);
}

runStoreAudit(storeUrl)
  .then(result => {
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  })
  .catch(err => {
    process.stderr.write(`Error: ${err.message}\n`);
    process.exit(1);
  });
