import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdirSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { pluginPageMetadata, authorPageMetadata } from '../src/lib/catalog-routes.ts';

test('Next route metadata reads params without fetching a README and preserves locale URLs', async (t) => {
  const cwd = process.cwd();
  const fixture = mkdtempSync(path.join(tmpdir(), 'hermes-catalog-meta-'));
  mkdirSync(path.join(fixture, 'public/api'), { recursive: true });
  const plugin = {
    name: 'example', description: 'Example plugin', repo: 'example/plugin',
    sha: 'a'.repeat(40), shaShort: 'aaaaaaa', tier: 'community', category: 'other',
    maintainer: 'Example author', maintainerSlug: 'example-author',
    installCommand: 'hermes plugins install example', readme: true,
    readmeUrl: 'https://raw.githubusercontent.com/example/plugin/' + 'a'.repeat(40) + '/README.md',
  };
  writeFileSync(path.join(fixture, 'public/api/plugins.json'), JSON.stringify([plugin]));
  writeFileSync(path.join(fixture, 'public/api/plugins-meta.json'), '{}');
  process.chdir(fixture);
  t.after(() => { process.chdir(cwd); rmSync(fixture, { recursive: true, force: true }); });
  const fetch = t.mock.method(globalThis, 'fetch', () => { throw new Error('Metadata must not fetch README content'); });
  const metadata = await pluginPageMetadata({ params: Promise.resolve({ name: 'example' }) }, 'zh-Hans');
  assert.equal(metadata.title, 'example · Plugin Catalog');
  assert.equal(metadata.alternates.canonical, 'https://hermes-agent.nousresearch.com/docs/zh-Hans/plugins/example');
  assert.equal(metadata.alternates.languages.en, 'https://hermes-agent.nousresearch.com/docs/plugins/example');
  const author = await authorPageMetadata({ params: Promise.resolve({ author: 'example-author' }) }, 'en');
  assert.equal(author.title, 'Plugins by Example author · Plugin Catalog');
  assert.equal(author.alternates.canonical, 'https://hermes-agent.nousresearch.com/docs/plugins/by/example-author');
  assert.equal(fetch.mock.callCount(), 0);
});
