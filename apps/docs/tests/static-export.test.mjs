import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { runInNewContext } from 'node:vm';

const postbuild = fileURLToPath(new URL('../scripts/postbuild.mjs', import.meta.url));

const englishPages = ['/', '/getting-started/quickstart', '/getting-started/installation', '/guides/automation-blueprints', '/developer-guide/plugins'];
const bilingualPages = [...englishPages, ...englishPages.map((route) => '/zh-Hans' + (route === '/' ? '' : route))];

function exportFixture(t, routes = bilingualPages) {
  const appDir = mkdtempSync(path.join(tmpdir(), 'hermes-docs-export-'));
  t.after(() => rmSync(appDir, { recursive: true, force: true }));
  mkdirSync(path.join(appDir, '.next'));
  for (const route of routes) {
    const file = path.join(appDir, 'out', route, 'index.html');
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, '<!doctype html><html><head><title>Fixture</title></head><body><h1>Fixture</h1></body></html>');
  }
  writeFileSync(path.join(appDir, '.next/prerender-manifest.json'), JSON.stringify({
    version: 4,
    routes: Object.fromEntries([...routes, '/_not-found'].map((route) => [route, {}])),
    dynamicRoutes: {},
    notFoundRoutes: [],
  }));
  return appDir;
}

function runPostbuild(appDir) {
  return spawnSync(process.execPath, [postbuild, '--app-dir', appDir], { encoding: 'utf8' });
}

test('postbuild publishes the exported route set as a public bilingual sitemap', (t) => {
  const appDir = exportFixture(t);
  const result = runPostbuild(appDir);
  assert.equal(result.status, 0, result.stderr);
  const xml = readFileSync(path.join(appDir, 'out/sitemap.xml'), 'utf8');
  const locations = [...xml.matchAll(/<loc>(.*?)<\/loc>/g)].map((match) => match[1]);
  assert.deepEqual(locations.sort(), bilingualPages.map((route) => 'https://hermes-agent.nousresearch.com/docs' + route).sort());
  assert.ok(!xml.includes('_not-found'));
  assert.ok(xml.includes('hreflang="zh-Hans" href="https://hermes-agent.nousresearch.com/docs/zh-Hans/getting-started/quickstart"'));
});

test('legacy aliases preserve language, preview origin, query and fragment', (t) => {
  const appDir = exportFixture(t);
  const result = runPostbuild(appDir);
  assert.equal(result.status, 0, result.stderr);
  const aliases = {
    '/guides/automation-templates': '/guides/automation-blueprints',
    '/guides/build-a-hermes-plugin': '/developer-guide/plugins',
    '/quickstart': '/getting-started/quickstart',
    '/installation': '/getting-started/installation',
  };
  for (const prefix of ['', '/zh-Hans']) {
    for (const [from, to] of Object.entries(aliases)) {
      const target = '/docs' + prefix + to;
      const aliasFile = path.join(appDir, 'out', prefix + from, 'index.html');
      assert.ok(existsSync(aliasFile), `Missing legacy alias: ${prefix + from}`);
      const html = readFileSync(aliasFile, 'utf8');
      assert.ok(html.includes(`href="https://hermes-agent.nousresearch.com${target}"`));
      assert.ok(html.includes(`content="0;url=${target}"`));
      assert.ok(html.includes(`data-redirect="${target}"`));
      const asset = html.match(/<script[^>]+src="([^"]+)"/)[1];
      assert.ok(asset.startsWith('/docs/_next/static/'));
      const script = readFileSync(path.join(appDir, 'out', asset.slice('/docs/'.length)), 'utf8');
      const origin = 'http://127.0.0.1:43123';
      let destination;
      runInNewContext(script, {
        URL,
        document: { currentScript: { dataset: { redirect: target } } },
        window: { location: { origin, search: '?mode=pick&profile=test', hash: '#install', replace: (url) => { destination = url; } } },
      });
      assert.equal(destination, origin + target + '?mode=pick&profile=test#install');
    }
  }
});

test('postbuild refuses empty exported pages', (t) => {
  const appDir = exportFixture(t);
  writeFileSync(path.join(appDir, 'out/getting-started/quickstart/index.html'), '');
  const result = runPostbuild(appDir);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /empty|nonempty/i);
  assert.match(result.stderr, /getting-started\/quickstart/);
});

test('postbuild does not overwrite an existing page with an alias', (t) => {
  const appDir = exportFixture(t);
  const file = path.join(appDir, 'out/quickstart/index.html');
  mkdirSync(path.dirname(file), { recursive: true });
  const original = '<!doctype html><title>Existing page</title>';
  writeFileSync(file, original);
  const result = runPostbuild(appDir);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Refusing to replace existing export: \/quickstart/);
  assert.equal(readFileSync(file, 'utf8'), original);
});

test('repeated postbuild preserves identical sitemap and redirect bytes', (t) => {
  const appDir = exportFixture(t);
  assert.equal(runPostbuild(appDir).status, 0);
  const paths = ['sitemap.xml', 'quickstart/index.html', 'zh-Hans/installation/index.html'];
  const before = paths.map((file) => readFileSync(path.join(appDir, 'out', file), 'utf8'));
  const result = runPostbuild(appDir);
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(paths.map((file) => readFileSync(path.join(appDir, 'out', file), 'utf8')), before);
});
