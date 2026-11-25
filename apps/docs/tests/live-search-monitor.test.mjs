// Behaviour tests for the live-search monitor's API leg
// (apps/docs/scripts/live-search-probe.mjs, used by scripts/check-live-search.mjs).
//
// The monitor runs every 6 hours against the deployed site: the Algolia
// config is served at ${DOCS_URL}/search-config.json, and index hits carry
// canonical site URLs that a preview probe has to validate before it opens
// them. Every fixture below is a real local HTTP server — the browser modal
// leg stays a real-browser check in the CLI against the deployed site and is
// deliberately not faked here.
//
// Run: npm run test   (or: node --experimental-strip-types --test tests/live-search-monitor.test.mjs)

import assert from 'node:assert/strict';
import test from 'node:test';
import http from 'node:http';
import {
  algoliaQueryUrl,
  checkAlgolia,
  checkHitPagesOpen,
  hitProbeUrl,
  MIN_HITS,
  QUERIES,
  readDeployedConfig,
  parseSearchConfig,
} from '../scripts/live-search-probe.mjs';
import { SITE_URL } from '../src/lib/urls.ts';

const CONFIG = { appId: 'TESTAPPID1', apiKey: '0123456789abcdef'.repeat(2), indexName: 'hermes-agent' };
// Stand-in for credential material in leak tests: no failure message, note or
// annotation may ever contain it.
const SENTINEL = 'S3N7INEL9';

// A fixture server is a map of exact URL → { status, headers?, body? }; any
// other path 404s like a static export would. `seen` records every request so
// tests can prove which result pages the probe actually opened.
async function withServer(routes, run) {
  const seen = [];
  const server = http.createServer((req, res) => {
    seen.push(`${req.method} ${req.url}`);
    const route = routes[req.url] ?? { status: 404, body: 'not found' };
    res.writeHead(route.status, route.headers ?? {});
    res.end(route.body ?? '');
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    await run({ base: `http://127.0.0.1:${server.address().port}`, seen });
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

const json = (status, value, headers = {}) => ({ status, headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(value) });
const page = (status = 200) => ({ status, body: `<!DOCTYPE html><html><body>fixture</body></html>` });

const configRoute = (config = CONFIG) => ({ '/docs/search-config.json': json(200, config) });
const algoliaRoute = (results) => ({ '/1/indexes/*/queries': json(200, { results }) });
const queryResult = (nbHits, hits) => ({ nbHits, hits: hits.map((url) => ({ url })) });
// What the Algolia crawler actually records: canonical origin, docs path.
const canonicalHit = (path) => `${SITE_URL}${path}`;
// One well-formed result page: the baseline the response-shape tests mutate.
const goodPage = () => queryResult(MIN_HITS + 2, [canonicalHit('/docs/skills')]);

test('discovers the deployed search config at search-config.json', async () => {
  await withServer(configRoute(), async ({ base }) => {
    const found = await readDeployedConfig(`${base}/docs`);
    assert.equal(found.error, undefined);
    assert.deepEqual(found.config, CONFIG);
    // Monitoring must not leak credential material into logs or annotations.
    assert.ok(found.note.includes('search-config.json'), found.note);
    assert.ok(!found.note.includes(CONFIG.apiKey) && !found.note.includes(CONFIG.appId) && !found.note.includes(CONFIG.indexName), found.note);
  });
});

test('a missing search-config.json fails naming the config URL and the status', async () => {
  await withServer({}, async ({ base }) => {
    const found = await readDeployedConfig(`${base}/docs`);
    assert.match(found.error, /search-config\.json/);
    assert.match(found.error, /404/);
    assert.equal(found.config, undefined);
  });
});

test('a blocked search-config.json fetch names the block, not "bad status"', async () => {
  await withServer(
    { '/docs/search-config.json': { status: 403, headers: { 'x-vercel-mitigated': 'challenge' }, body: '<html>firewall</html>' } },
    async ({ base }) => {
      const found = await readDeployedConfig(`${base}/docs`);
      assert.match(found.error, /Vercel firewall/);
    },
  );
});

test('a malformed search-config.json fails as invalid JSON', async () => {
  await withServer(
    { '/docs/search-config.json': { status: 200, headers: { 'content-type': 'application/json' }, body: 'configure{' } },
    async ({ base }) => {
      const found = await readDeployedConfig(`${base}/docs`);
      assert.match(found.error, /search-config\.json/);
      assert.match(found.error, /not valid JSON/);
    },
  );
});

test('invalid-JSON config failures never echo the raw config, which can contain the key', () => {
  // A half-rendered config whose key literal sits at the syntax error: Node's
  // JSON.parse message quotes the raw text around the error position, so the
  // diagnostic must say only that the config is not valid JSON.
  const raw = `{"appId":"TESTAPPID1","apiKey":${SENTINEL},"indexName":"hermes-agent"}`;
  const { error } = parseSearchConfig(raw);
  assert.match(error, /not valid JSON/);
  assert.ok(!error.includes(SENTINEL), error);
});

test('the search config must be a JSON object with non-empty string appId, apiKey and indexName', () => {
  const cases = [
    ['[]', /object/],
    ['"42"', /object/],
    ['{"appId":"a","apiKey":"b"}', /indexName/],
    ['{"appId":"","apiKey":"b","indexName":"c"}', /appId/],
    ['{"appId":1,"apiKey":"b","indexName":"c"}', /appId/],
  ];
  for (const [raw, expected] of cases) {
    const { error } = parseSearchConfig(raw);
    assert.match(error, expected, `for ${raw}`);
  }
  assert.deepEqual(parseSearchConfig(JSON.stringify(CONFIG)).config, CONFIG);
});

test('Algolia queries go to the app DSN host by default', () => {
  assert.equal(algoliaQueryUrl(CONFIG.appId), `https://${CONFIG.appId}-dsn.algolia.net/1/indexes/*/queries`);
});

test('hitProbeUrl remaps only canonical docs hits, keeping path, query and hash', () => {
  assert.equal(
    hitProbeUrl(`${SITE_URL}/docs/zh-Hans/user-guide/messaging/telegram`, 'http://127.0.0.1:9999/docs').probeUrl,
    'http://127.0.0.1:9999/docs/zh-Hans/user-guide/messaging/telegram',
  );
  assert.equal(
    hitProbeUrl(`${SITE_URL}/docs/skills?source=bundled#install`, 'http://127.0.0.1:9999/docs').probeUrl,
    'http://127.0.0.1:9999/docs/skills?source=bundled#install',
  );
  // Probing the canonical deployment itself: the remap is the identity.
  assert.equal(hitProbeUrl(`${SITE_URL}/docs/cli`, `${SITE_URL}/docs`).probeUrl, `${SITE_URL}/docs/cli`);
  assert.match(hitProbeUrl('https://evil.example.com/docs/cli', 'http://127.0.0.1:9999/docs').rejected, /off-site/);
  assert.match(hitProbeUrl(`${SITE_URL}/blog/post`, 'http://127.0.0.1:9999/docs').rejected, /docs base path/);
});

test('canonical hits are accepted and opened at the probed origin', async () => {
  const hits = {
    [QUERIES[0]]: ['/docs/user-guide/messaging/telegram', '/docs/guides/team-telegram-assistant'],
    [QUERIES[1]]: ['/docs/getting-started/installation'],
    [QUERIES[2]]: ['/docs/skills'],
  };
  await withServer(
    {
      ...configRoute(),
      ...algoliaRoute(QUERIES.map((query) => queryResult(MIN_HITS + 2, hits[query].map(canonicalHit)))),
      ...Object.fromEntries(hits[QUERIES[0]].concat(hits[QUERIES[1]], hits[QUERIES[2]]).map((path) => [path, page()])),
    },
    async ({ base, seen }) => {
      const { failures, notes, hitUrls } = await checkAlgolia(CONFIG, { docsUrl: `${base}/docs`, algoliaBase: base });
      assert.deepEqual(failures, []);
      assert.deepEqual(hitUrls, [
        `${base}/docs/user-guide/messaging/telegram`,
        `${base}/docs/getting-started/installation`,
        `${base}/docs/skills`,
      ]);
      assert.ok(notes.some((note) => note.includes(`"${QUERIES[0]}"`) && note.includes(`${MIN_HITS + 2} hits`)), notes.join('; '));

      const opened = await checkHitPagesOpen(hitUrls);
      assert.deepEqual(opened.failures, []);
      assert.deepEqual(opened.notes, hitUrls.map((url) => `result opens: ${url}`));
      for (const url of hitUrls) assert.ok(seen.includes(`GET ${new URL(url).pathname}`), seen.join(', '));
    },
  );
});

test('off-site hits are rejected, never opened', async () => {
  const offSite = 'https://evil.example.com/docs/telegram';
  await withServer(
    {
      ...configRoute(),
      ...algoliaRoute([
        queryResult(MIN_HITS + 2, [offSite, canonicalHit('/docs/user-guide/messaging/telegram')]),
        queryResult(MIN_HITS + 2, [canonicalHit('/docs/getting-started/installation')]),
        queryResult(MIN_HITS + 2, [canonicalHit('/docs/skills')]),
      ]),
      '/docs/getting-started/installation': page(),
      '/docs/skills': page(),
    },
    async ({ base }) => {
      const { failures, hitUrls } = await checkAlgolia(CONFIG, { docsUrl: `${base}/docs`, algoliaBase: base });
      assert.equal(failures.length, 1);
      assert.match(failures[0], /off-site/);
      assert.match(failures[0], /evil\.example\.com/);
      assert.match(failures[0], new RegExp(QUERIES[0]));
      // The rejected query contributes no URL to open; the others still do.
      assert.deepEqual(hitUrls, [`${base}/docs/getting-started/installation`, `${base}/docs/skills`]);
    },
  );
});

test('hits outside the docs base path are rejected even on the site origin', async () => {
  await withServer(
    {
      ...configRoute(),
      ...algoliaRoute([
        queryResult(MIN_HITS + 2, [canonicalHit('/docs/user-guide/messaging/telegram')]),
        queryResult(MIN_HITS + 2, [canonicalHit('/blog/post')]),
        queryResult(MIN_HITS + 2, [canonicalHit('/docs/skills')]),
      ]),
    },
    async ({ base }) => {
      const { failures, hitUrls } = await checkAlgolia(CONFIG, { docsUrl: `${base}/docs`, algoliaBase: base });
      assert.equal(failures.length, 1);
      assert.match(failures[0], /docs base path/);
      assert.match(failures[0], /\/blog\/post/);
      assert.ok(hitUrls.every((url) => url.startsWith(`${base}/docs/`)), hitUrls.join(', '));
    },
  );
});

test('queries with too few hits are flagged as an empty index', async () => {
  await withServer(
    {
      ...configRoute(),
      ...algoliaRoute([
        queryResult(MIN_HITS - 3, [canonicalHit('/docs/user-guide/messaging/telegram')]),
        queryResult(MIN_HITS + 2, [canonicalHit('/docs/getting-started/installation')]),
        queryResult(MIN_HITS + 2, [canonicalHit('/docs/skills')]),
      ]),
    },
    async ({ base }) => {
      const { failures } = await checkAlgolia(CONFIG, { docsUrl: `${base}/docs`, algoliaBase: base });
      assert.equal(failures.length, 1);
      assert.match(failures[0], new RegExp(`returned ${MIN_HITS - 3} hits`));
      assert.match(failures[0], /index looks empty/);
    },
  );
});

test('an empty Algolia answer fails the probe instead of passing', async () => {
  await withServer(
    {
      ...configRoute(),
      ...algoliaRoute([]),
    },
    async ({ base }) => {
      const { failures, notes, hitUrls } = await checkAlgolia(CONFIG, { docsUrl: `${base}/docs`, algoliaBase: base });
      assert.equal(failures.length, 1);
      assert.match(failures[0], /0 result page/);
      assert.match(failures[0], new RegExp(`${QUERIES.length} queries`));
      assert.deepEqual(notes, []);
      assert.deepEqual(hitUrls, []);
    },
  );
});

test('a truncated or padded Algolia answer fails instead of skipping or inventing queries', async () => {
  const pages = QUERIES.map(() => goodPage());
  const cases = [
    ['truncated', pages.slice(0, 1)],
    ['padded', pages.concat(goodPage())],
  ];
  for (const [name, results] of cases) {
    await withServer(
      {
        ...configRoute(),
        ...algoliaRoute(results),
      },
      async ({ base }) => {
        const { failures, notes, hitUrls } = await checkAlgolia(CONFIG, { docsUrl: `${base}/docs`, algoliaBase: base });
        assert.equal(failures.length, 1, name);
        assert.match(failures[0], /result page/, name);
        assert.match(failures[0], new RegExp(`${QUERIES.length} queries`), name);
        assert.deepEqual(notes, [], name);
        assert.deepEqual(hitUrls, [], name);
      },
    );
  }
});

test('an Algolia answer with no usable results container fails with a diagnostic instead of crashing', async () => {
  const cases = [
    ['results key missing', json(200, { message: 'unexpected gateway shape' })],
    ['results is not an array', json(200, { results: goodPage() })],
    ['body is not JSON', { status: 200, headers: { 'content-type': 'application/json' }, body: '<html>gateway error page</html>' }],
  ];
  for (const [name, route] of cases) {
    await withServer(
      {
        ...configRoute(),
        '/1/indexes/*/queries': route,
      },
      async ({ base }) => {
        const { failures, hitUrls } = await checkAlgolia(CONFIG, { docsUrl: `${base}/docs`, algoliaBase: base });
        assert.equal(failures.length, 1, name);
        assert.match(failures[0], /results array|not valid JSON/, name);
        assert.deepEqual(hitUrls, [], name);
      },
    );
  }
});

test('a result page with an invalid hit count fails as malformed, not as a pass', async () => {
  const cases = [
    ['hit count is a string', { nbHits: `${MIN_HITS + 2}`, hits: [{ url: canonicalHit('/docs/skills') }] }],
    ['hit count is missing', { hits: [{ url: canonicalHit('/docs/skills') }] }],
    ['hit count is negative', { nbHits: -1, hits: [{ url: canonicalHit('/docs/skills') }] }],
  ];
  for (const [name, badPage] of cases) {
    await withServer(
      {
        ...configRoute(),
        ...algoliaRoute([badPage, goodPage(), goodPage()]),
      },
      async ({ base }) => {
        const { failures, hitUrls } = await checkAlgolia(CONFIG, { docsUrl: `${base}/docs`, algoliaBase: base });
        assert.equal(failures.length, 1, name);
        assert.match(failures[0], /no valid hit count/, name);
        assert.match(failures[0], new RegExp(QUERIES[0]), name);
        assert.deepEqual(hitUrls, [`${base}/docs/skills`, `${base}/docs/skills`], name);
      },
    );
  }
});

test('a result page without a usable hits array fails instead of passing', async () => {
  const cases = [
    ['hits array is empty', { nbHits: MIN_HITS + 2, hits: [] }],
    ['hits array is missing', { nbHits: MIN_HITS + 2 }],
    ['hit has no url', { nbHits: MIN_HITS + 2, hits: [{}] }],
  ];
  for (const [name, badPage] of cases) {
    await withServer(
      {
        ...configRoute(),
        ...algoliaRoute([badPage, goodPage(), goodPage()]),
      },
      async ({ base }) => {
        const { failures } = await checkAlgolia(CONFIG, { docsUrl: `${base}/docs`, algoliaBase: base });
        assert.equal(failures.length, 1, name);
        assert.match(failures[0], /no usable hits/, name);
        assert.match(failures[0], new RegExp(QUERIES[0]), name);
      },
    );
  }
});

test('an Algolia API rejection is reported with the status', async () => {
  await withServer(
    {
      ...configRoute(),
      '/1/indexes/*/queries': { status: 403, headers: { 'content-type': 'application/json' }, body: '{"message":"Invalid Application-ID or API key"}' },
    },
    async ({ base }) => {
      const { failures, hitUrls } = await checkAlgolia(CONFIG, { docsUrl: `${base}/docs`, algoliaBase: base });
      assert.equal(failures.length, 1);
      assert.match(failures[0], /Algolia rejected the deployed search key: HTTP 403/);
      assert.deepEqual(hitUrls, []);
    },
  );
});

test('an Algolia rejection never echoes the response body, which can contain the request key', async () => {
  await withServer(
    {
      ...configRoute(),
      '/1/indexes/*/queries': {
        status: 403,
        headers: { 'content-type': 'application/json' },
        // A proxy or provider can hand the request's own API key back in the
        // error body; the probe must never repeat it.
        body: `{"message":"Invalid Application-ID or API key","echoed_request_key":"${CONFIG.apiKey}"}`,
      },
    },
    async ({ base }) => {
      const { failures, hitUrls } = await checkAlgolia(CONFIG, { docsUrl: `${base}/docs`, algoliaBase: base });
      assert.equal(failures.length, 1);
      assert.match(failures[0], /Algolia rejected the deployed search key: HTTP 403/);
      assert.ok(!failures[0].includes(CONFIG.apiKey), failures[0]);
      assert.deepEqual(hitUrls, []);
    },
  );
});

test('a result page that does not open fails the probe', async () => {
  await withServer(
    {
      '/docs/user-guide/messaging/telegram': { status: 404, body: 'gone' },
      '/docs/getting-started/installation': page(500),
      '/docs/skills': page(),
    },
    async ({ base }) => {
      const { failures, notes } = await checkHitPagesOpen([
        `${base}/docs/user-guide/messaging/telegram`,
        `${base}/docs/getting-started/installation`,
        `${base}/docs/skills`,
      ]);
      assert.equal(failures.length, 2);
      assert.match(failures[0], /does not open/);
      assert.match(failures[0], /HTTP 404/);
      assert.match(failures[1], /HTTP 500/);
      assert.deepEqual(notes, [`result opens: ${base}/docs/skills`]);
    },
  );
});
