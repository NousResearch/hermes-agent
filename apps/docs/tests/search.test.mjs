import assert from 'node:assert/strict';
import test from 'node:test';
import {
  SEARCH_HITS_PER_PAGE,
  algoliaSearch,
  algoliaSearchPlan,
  highlightSegments,
  resultBreadcrumb,
  resultTitle,
  searchFacetFilters,
  transformResultUrl,
} from '../src/lib/search.ts';

// Fixture credentials only; never the real deployed config.
const config = { appId: 'TESTAPPID', apiKey: 'test-search-key', indexName: 'test index' };

test('facet filters keep the locale facet and the main-compatible tag OR-group', () => {
  assert.deepEqual(searchFacetFilters('en'), [
    'lang:en',
    ['docusaurus_tag:docs-default-current', 'docusaurus_tag:default'],
  ]);
  assert.deepEqual(searchFacetFilters('zh-Hans'), [
    'lang:zh-Hans',
    ['docusaurus_tag:docs-default-current', 'docusaurus_tag:default'],
  ]);
});

test('the search request plans a POST to the DSN host with credentials in headers', () => {
  const plan = algoliaSearchPlan(config, 'telegram', 'en');
  assert.equal(plan.url, 'https://TESTAPPID-dsn.algolia.net/1/indexes/test%20index/query');
  assert.ok(!plan.url.includes('test-search-key'), 'the API key must not travel in the URL');
  assert.equal(plan.headers['X-Algolia-Application-Id'], 'TESTAPPID');
  assert.equal(plan.headers['X-Algolia-API-Key'], 'test-search-key');
  assert.equal(plan.headers['Content-Type'], 'application/json');
  const body = JSON.parse(plan.body);
  assert.equal(body.query, 'telegram');
  assert.deepEqual(body.facetFilters, searchFacetFilters('en'));
  assert.equal(body.hitsPerPage, SEARCH_HITS_PER_PAGE);
  assert.equal(body.highlightPreTag, '<mark>');
  assert.equal(body.highlightPostTag, '</mark>');
  assert.equal(body.page, 0);
});

test('paged requests move the page into the body', () => {
  const body = JSON.parse(algoliaSearchPlan(config, 'x', 'zh-Hans', { page: 2 }).body);
  assert.equal(body.page, 2);
  assert.deepEqual(body.facetFilters, searchFacetFilters('zh-Hans'));
});

test('canonical-origin results become root-relative links that keep query and hash', () => {
  assert.equal(
    transformResultUrl('https://hermes-agent.nousresearch.com/docs/user-guide/cli?x=1#setup'),
    '/docs/user-guide/cli?x=1#setup',
  );
  assert.equal(transformResultUrl('https://hermes-agent.nousresearch.com/docs/skills/'), '/docs/skills/');
});

test('search results reject foreign, malformed and non-docs destinations', () => {
  for (const url of [
    'https://example.com/path?a=1#b', 'not a url', 'javascript:alert(1)',
    'data:text/html,unsafe', '//example.com/path',
    'https://hermes-agent.nousresearch.com/docsite/page',
    'https://hermes-agent.nousresearch.com/private',
  ]) assert.equal(transformResultUrl(url), null, url);
});

test('malformed search responses fail instead of appearing as empty results', async (t) => {
  const invalid = [null, {}, { hits: [], nbHits: '0', nbPages: 0, page: 0 }, { hits: null, nbHits: 0, nbPages: 0, page: 0 }];
  for (const payload of invalid) {
    const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify(payload)));
    await assert.rejects(algoliaSearch(config, 'test', 'en'), /invalid search response/i);
    fetch.mock.restore();
  }
});

test('highlight values split into marked and unmarked segments', () => {
  assert.deepEqual(highlightSegments('Skills <mark>Hub</mark> guide'), [
    { text: 'Skills ', marked: false },
    { text: 'Hub', marked: true },
    { text: ' guide', marked: false },
  ]);
});

test('highlight entities decode so React re-escapes them exactly once', () => {
  assert.deepEqual(highlightSegments('A &amp; B &lt;tag&gt; <mark>C</mark>'), [
    { text: 'A & B <tag> ', marked: false },
    { text: 'C', marked: true },
  ]);
});

test('markup smuggled into a highlight stays inert text', () => {
  assert.deepEqual(highlightSegments('<script>alert(1)</script>'), [
    { text: '<script>alert(1)</script>', marked: false },
  ]);
  assert.deepEqual(highlightSegments('</mark>'), []);
});

const pageHit = {
  url: 'https://hermes-agent.nousresearch.com/docs/skills/',
  type: 'lvl1',
  hierarchy: { lvl0: 'Documentation', lvl1: 'Skills Hub | Hermes Agent' },
  _highlightResult: {
    hierarchy: {
      lvl0: { value: 'Documentation' },
      lvl1: { value: '<mark>Skills</mark> Hub | Hermes Agent' },
    },
  },
};

test('result titles come from the deepest highlighted hierarchy level', () => {
  assert.deepEqual(resultTitle(pageHit), [
    { text: 'Skills', marked: true },
    { text: ' Hub | Hermes Agent', marked: false },
  ]);
  assert.deepEqual(resultBreadcrumb(pageHit), []);
});

test('anchor hits show their parent page as the breadcrumb', () => {
  const anchorHit = {
    url: 'https://hermes-agent.nousresearch.com/docs/user-guide/messaging/telegram#setup',
    type: 'lvl2',
    hierarchy: { lvl0: 'Documentation', lvl1: 'Telegram | Hermes Agent', lvl2: 'Setup' },
    _highlightResult: {
      hierarchy: {
        lvl0: { value: 'Documentation' },
        lvl1: { value: 'Telegram | Hermes Agent' },
        lvl2: { value: '<mark>Setup</mark>' },
      },
    },
  };
  assert.deepEqual(resultTitle(anchorHit), [{ text: 'Setup', marked: true }]);
  assert.deepEqual(resultBreadcrumb(anchorHit), ['Telegram | Hermes Agent']);
});

test('plain titles and breadcrumbs decode crawler HTML entities once', () => {
  assert.deepEqual(resultTitle({ url: '/docs/nix', hierarchy: { lvl1: 'Nix &amp; NixOS' } }), [
    { text: 'Nix & NixOS', marked: false },
  ]);
  assert.deepEqual(resultBreadcrumb({ hierarchy: { lvl1: 'Nix &amp; NixOS', lvl2: 'Install' } }), ['Nix & NixOS']);
});

test('hits without highlight payloads fall back to plain hierarchy values', () => {
  const plainHit = {
    url: 'https://hermes-agent.nousresearch.com/docs/skills/',
    type: 'lvl1',
    hierarchy: { lvl0: 'Documentation', lvl1: 'Skills Hub | Hermes Agent' },
  };
  assert.deepEqual(resultTitle(plainHit), [{ text: 'Skills Hub | Hermes Agent', marked: false }]);
});
