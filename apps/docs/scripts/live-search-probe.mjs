// API leg of the live docs search monitor, factored out of
// scripts/check-live-search.mjs so behaviour can be tested against real
// local HTTP fixtures (tests/live-search-monitor.test.mjs). The browser
// modal leg stays in the CLI and is exercised against the real deployed site.
//
// The deployed site serves the Algolia search config at
// ${DOCS_URL}/search-config.json, and index hits carry canonical site URLs.
// A probe at another origin (a preview) still validates those hits: only
// canonical URLs under the docs base path are remapped to the probed origin
// — exactly what the modal's transformItems does in
// src/components/doc-search.tsx — and everything else is a defect in the
// index, not something to paper over.

import { DOCS_BASE_PATH, SITE_URL } from '../src/lib/urls.ts';

// Terms that must always match many pages; one per major area of the docs.
export const QUERIES = ['telegram', 'install', 'skills'];
export const MIN_HITS = 5;
// Every request (body included) gets a deadline, so a slow-drip server fails
// the check with time left to retry and alert instead of hanging.
export const REQUEST_TIMEOUT_MS = 20_000;
export const UA =
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36 hermes-docs-search-monitor';
// Vercel keeps one cache entry per exact Accept-Encoding string, so a request
// without Chrome's value reads a different copy of the page than visitors do.
export const ACCEPT_ENCODING = 'gzip, deflate, br, zstd';

// The app's DSN host by default; tests pass a local base so the real query
// flow can run against an HTTP fixture.
export function algoliaQueryUrl(appId, base) {
  return base ? new URL('/1/indexes/*/queries', base).href : `https://${appId}-dsn.algolia.net/1/indexes/*/queries`;
}

async function get(url, headers = {}) {
  const res = await fetch(url, {
    headers: { 'user-agent': UA, 'accept-encoding': ACCEPT_ENCODING, ...headers },
    redirect: 'follow',
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  return { res, body: await res.text() };
}

// Vercel's firewall/challenge and password protection answer 403/401/429
// with an HTML page of their own; name that instead of "bad status".
function blockedBy(res) {
  if (res.headers.get('x-vercel-mitigated')) return `Vercel firewall (${res.headers.get('x-vercel-mitigated')})`;
  if (res.status === 401) return 'auth wall (401)';
  if (res.status === 403 || res.status === 429) return `HTTP ${res.status}`;
  return null;
}

// The deployed search config: a JSON object whose appId, apiKey and indexName
// are non-empty strings. Failures name the offending field, never its value —
// the key is credential material and must not reach logs or annotations.
export function parseSearchConfig(raw) {
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    // Node's parse errors quote the raw body around the syntax error, and a
    // half-rendered config can put the key literal right there — so this
    // diagnostic never quotes the raw text, only that it is not valid JSON.
    return { error: 'search-config.json is not valid JSON' };
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return {
      error: `search-config.json must be a JSON object with "appId", "apiKey" and "indexName" string fields (got ${
        Array.isArray(parsed) ? 'an array' : `a ${typeof parsed}`
      })`,
    };
  }
  for (const name of ['appId', 'apiKey', 'indexName']) {
    const value = parsed[name];
    if (typeof value !== 'string' || value.trim() === '') {
      const defect = value === undefined ? 'is missing' : typeof value !== 'string' ? `is a ${typeof value}, not a string` : 'is empty';
      return { error: `search-config.json field "${name}" must be a non-empty string, but it ${defect}` };
    }
  }
  return { config: { appId: parsed.appId, apiKey: parsed.apiKey, indexName: parsed.indexName } };
}

export async function readDeployedConfig(docsUrl) {
  const url = `${docsUrl}/search-config.json`;
  let res;
  let body;
  try {
    ({ res, body } = await get(url));
  } catch (error) {
    return { error: `docs search config ${url} cannot be read: ${error.cause?.message ?? error.message}` };
  }
  const block = blockedBy(res);
  if (block || !res.ok) {
    const reason =
      block ??
      (res.status === 404
        ? 'HTTP 404 — the deployed export does not serve search-config.json, so the search modal cannot run; check the export step and redeploy'
        : `HTTP ${res.status}`);
    return { error: `docs search config ${url} is not reachable for visitors: ${reason}` };
  }
  const parsed = parseSearchConfig(body);
  if (parsed.error) return { error: `docs search config ${url}: ${parsed.error}` };
  return {
    config: parsed.config,
    note: `deployed search config from ${url}: appId, apiKey and indexName are set`,
  };
}

// Where a hit URL opens when the probe runs: canonical docs URLs resolve at
// the probed origin (a preview serves the same pages at its own origin/base
// path), exactly like the modal's transformItems. Anything else is rejected —
// remapping an unrelated URL would only turn an index defect into a confusing
// "does not open" failure.
export function hitProbeUrl(hitUrl, docsUrl) {
  let url;
  try {
    url = new URL(hitUrl);
  } catch {
    return { rejected: 'is not a valid URL' };
  }
  if (url.origin !== SITE_URL) return { rejected: `points off-site (${url.origin}); index records must use ${SITE_URL}` };
  if (url.pathname !== DOCS_BASE_PATH && !url.pathname.startsWith(`${DOCS_BASE_PATH}/`)) {
    return { rejected: `is outside the docs base path (${url.pathname})` };
  }
  return { probeUrl: new URL(url.pathname + url.search + url.hash, docsUrl).href };
}

// Query Algolia with the credentials the site is serving. Every hit must be a
// canonical docs URL; the first hit of each query is returned (remapped to the
// probed origin) for checkHitPagesOpen to open. The answer must carry exactly
// one well-formed result page per query: an empty, truncated or malformed
// response is a broken search path, not a passing check.
export async function checkAlgolia({ appId, apiKey, indexName }, { docsUrl, algoliaBase }) {
  const failures = [];
  const notes = [];
  const hitUrls = [];
  const res = await fetch(algoliaQueryUrl(appId, algoliaBase), {
    method: 'POST',
    headers: {
      'x-algolia-application-id': appId,
      'x-algolia-api-key': apiKey,
      origin: new URL(docsUrl).origin,
      referer: `${docsUrl}/`,
      'content-type': 'application/json',
    },
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    body: JSON.stringify({
      requests: QUERIES.map((query) => ({ indexName, query, hitsPerPage: 3, attributesToRetrieve: ['url'] })),
    }),
  });
  const body = await res.text();
  if (!res.ok) {
    // 403 "Invalid Application-ID or API key" is the deleted-key outage. The
    // body is deliberately not quoted: a proxy or provider can hand the
    // request's own API key back in an error page, and credential material
    // must not reach logs or annotations. The HTTP status says enough.
    failures.push(
      res.status === 403
        ? 'Algolia rejected the deployed search key: HTTP 403 — the deployed key may be deleted or rotated'
        : `Algolia rejected the deployed search key: HTTP ${res.status}`,
    );
    return { failures, notes, hitUrls };
  }
  // The answer must be usable before any query counts as checked: a body that
  // is not JSON, a missing results array or the wrong number of result pages
  // (the request asks for exactly one per query) fail the probe outright —
  // an empty answer must never read as "every query passed".
  let results;
  try {
    results = JSON.parse(body)?.results;
  } catch {
    // Never quote the raw body here either: a proxy error page can echo
    // credential material from the request.
    failures.push(`Algolia answered HTTP ${res.status} but the response is not valid JSON`);
    return { failures, notes, hitUrls };
  }
  if (!Array.isArray(results)) {
    failures.push(`Algolia response has no results array (HTTP ${res.status})`);
    return { failures, notes, hitUrls };
  }
  if (results.length !== QUERIES.length) {
    failures.push(`Algolia returned ${results.length} result page(s) for ${QUERIES.length} queries; expected exactly one per query`);
    return { failures, notes, hitUrls };
  }
  for (const [i, result] of results.entries()) {
    const query = QUERIES[i];
    // Each page must carry the shape the request asked for — a nonnegative
    // numeric hit count and a nonempty page of hits with URL strings — or
    // that query reads as broken, not as fine.
    if (typeof result?.nbHits !== 'number' || !Number.isFinite(result.nbHits) || result.nbHits < 0) {
      failures.push(`Algolia result for "${query}" has no valid hit count; the response is malformed`);
      continue;
    }
    if (!Array.isArray(result?.hits) || result.hits.length === 0 || !result.hits.every((hit) => typeof hit?.url === 'string')) {
      failures.push(`Algolia result for "${query}" has no usable hits array; the response is truncated or malformed`);
      continue;
    }
    if (result.nbHits < MIN_HITS) {
      failures.push(`query "${query}" returned ${result.nbHits} hits (< ${MIN_HITS}); index looks empty`);
      continue;
    }
    const probed = result.hits.map((hit) => ({ raw: hit.url, ...hitProbeUrl(hit.url, docsUrl) }));
    const rejected = probed.find((p) => p.rejected);
    if (rejected) {
      failures.push(`query "${query}" hit ${rejected.raw} ${rejected.rejected}`);
    } else {
      hitUrls.push(probed[0].probeUrl);
      notes.push(`query "${query}": ${result.nbHits} hits`);
    }
  }
  return { failures, notes, hitUrls };
}

export async function checkHitPagesOpen(urls) {
  const failures = [];
  const notes = [];
  for (const url of urls) {
    const { res } = await get(url);
    const block = blockedBy(res);
    if (block || !res.ok) failures.push(`search result ${url} does not open: ${block ?? `HTTP ${res.status}`}`);
    else notes.push(`result opens: ${url}`);
  }
  return { failures, notes };
}
