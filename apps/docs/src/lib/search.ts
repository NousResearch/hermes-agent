// Shared modal/page search helpers. Preserve both legacy tag groups: catalog
// landing pages use `default`, while document records carry the version tag.

import { DOCS_BASE_PATH, SITE_URL } from './urls.ts';

export interface SearchConfig {
  appId: string;
  apiKey: string;
  indexName: string;
}

export interface SearchHit {
  objectID?: string;
  url: string;
  anchor?: string;
  type?: string;
  hierarchy?: Record<string, string | undefined> | null;
  _highlightResult?: {
    hierarchy?: Record<string, { value: string } | undefined> | null;
  } | null;
}

export interface HighlightSegment {
  text: string;
  marked: boolean;
}

export interface AlgoliaRequestPlan {
  url: string;
  headers: Record<string, string>;
  body: string;
}

export interface AlgoliaSearchResult {
  hits: SearchHit[];
  page: number;
  nbPages: number;
  nbHits: number;
  query: string;
}

export const SEARCH_HITS_PER_PAGE = 20;
export const HIGHLIGHT_PRE_TAG = '<mark>';
export const HIGHLIGHT_POST_TAG = '</mark>';

/** Deepest hierarchy level first; `lvl0` is the site-section label. */
const HIERARCHY_LEVELS = ['lvl6', 'lvl5', 'lvl4', 'lvl3', 'lvl2', 'lvl1', 'lvl0'];

/**
 * Facet filters shared by the modal and the full-page search: the active
 * locale AND (docs tagged for the Fumadocs cutover OR docs tagged on main).
 */
export function searchFacetFilters(locale: string): (string | string[])[] {
  return [
    `lang:${locale}`,
    ['docusaurus_tag:docs-default-current', 'docusaurus_tag:default'],
  ];
}

/** Read-only search endpoint on the DSN host; the key travels in a header. */
export function algoliaSearchUrl({ appId, indexName }: SearchConfig): string {
  return `https://${appId}-dsn.algolia.net/1/indexes/${encodeURIComponent(indexName)}/query`;
}

/** Request headers and payload for read-only Algolia searches. */
export function algoliaSearchPlan(
  config: SearchConfig,
  query: string,
  locale: string,
  options: { page?: number } = {},
): AlgoliaRequestPlan {
  return {
    url: algoliaSearchUrl(config),
    headers: {
      'X-Algolia-Application-Id': config.appId,
      'X-Algolia-API-Key': config.apiKey,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      facetFilters: searchFacetFilters(locale),
      hitsPerPage: SEARCH_HITS_PER_PAGE,
      page: options.page ?? 0,
      highlightPreTag: HIGHLIGHT_PRE_TAG,
      highlightPostTag: HIGHLIGHT_POST_TAG,
    }),
  };
}

/** Query the deployed index with native fetch. */
export async function algoliaSearch(
  config: SearchConfig,
  query: string,
  locale: string,
  options: { page?: number; signal?: AbortSignal } = {},
): Promise<AlgoliaSearchResult> {
  const plan = algoliaSearchPlan(config, query, locale, options);
  const response = await fetch(plan.url, {
    method: 'POST',
    headers: plan.headers,
    body: plan.body,
    signal: options.signal,
  });
  if (!response.ok) throw new Error(`docs search request failed (HTTP ${response.status})`);
  const data: Record<string, unknown> | null = await response.json();
  if (!data || !Array.isArray(data.hits)
    || typeof data.page !== 'number' || typeof data.nbPages !== 'number' || typeof data.nbHits !== 'number'
    || [data.page, data.nbPages, data.nbHits].some((value) => !Number.isInteger(value) || value < 0)
    || data.hits.some((hit) => !hit || typeof hit.url !== 'string' || transformResultUrl(hit.url) === null)) {
    throw new Error('Invalid search response');
  }
  return {
    hits: data.hits as SearchHit[],
    page: data.page,
    nbPages: data.nbPages,
    nbHits: data.nbHits,
    query: typeof data.query === 'string' ? data.query : query,
  };
}

/** Keep trusted docs hits on the preview origin; reject unrelated destinations. */
export function transformResultUrl(url: string): string | null {
  try {
    const parsed = new URL(url);
    if (parsed.origin !== SITE_URL
      || (parsed.pathname !== DOCS_BASE_PATH && !parsed.pathname.startsWith(`${DOCS_BASE_PATH}/`))) return null;
    return parsed.pathname + parsed.search + parsed.hash;
  } catch {
    return null;
  }
}

const HIGHLIGHT_ENTITIES: Record<string, string> = {
  '&lt;': '<',
  '&gt;': '>',
  '&quot;': '"',
  '&#39;': "'",
  '&amp;': '&',
};

function decodeHighlightEntities(text: string): string {
  return text.replace(/&(?:lt|gt|quot|#39|amp);/g, (entity) => HIGHLIGHT_ENTITIES[entity]);
}

/**
 * Split an Algolia highlight value into plain segments. Only the exact mark
 * tags Algolia was asked to emit become highlights; any other markup in the
 * value stays inert text that React escapes on render, so highlighted titles
 * cannot smuggle in HTML.
 */
export function highlightSegments(value: string): HighlightSegment[] {
  const segments: HighlightSegment[] = [];
  let marked = false;
  for (const token of value.split(/(<\/?mark>)/)) {
    if (token === '<mark>') {
      marked = true;
    } else if (token === '</mark>') {
      marked = false;
    } else if (token) {
      segments.push({ text: decodeHighlightEntities(token), marked });
    }
  }
  return segments;
}

/** Highlighted title: the deepest highlighted hierarchy level, falling back to plain values. */
export function resultTitle(hit: SearchHit): HighlightSegment[] {
  const highlighted = hit?._highlightResult?.hierarchy;
  for (const level of HIERARCHY_LEVELS) {
    const value = highlighted?.[level]?.value;
    if (typeof value === 'string' && value.length) return highlightSegments(value);
  }
  const plain = hit?.hierarchy;
  for (const level of HIERARCHY_LEVELS) {
    const value = plain?.[level];
    if (typeof value === 'string' && value.length) return [{ text: decodeHighlightEntities(value), marked: false }];
  }
  return highlightSegments(typeof hit?.url === 'string' ? hit.url : '');
}

/** Plain-text ancestor trail for anchor hits, outermost page first; empty for page hits. */
export function resultBreadcrumb(hit: SearchHit): string[] {
  const hierarchy = hit?.hierarchy;
  if (!hierarchy) return [];
  const present = HIERARCHY_LEVELS.filter(
    (level) => typeof hierarchy[level] === 'string' && hierarchy[level].length > 0,
  );
  return present
    .slice(1)
    .filter((level) => level !== 'lvl0')
    .map((level) => decodeHighlightEntities(hierarchy[level] as string))
    .reverse();
}
