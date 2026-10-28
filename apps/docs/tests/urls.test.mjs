import assert from 'node:assert/strict';
import test from 'node:test';
import { localizedHref, publicHref, canonicalUrl, splitLocalePath } from '../src/lib/urls.ts';

test('framework links have one locale and no deployment prefix', () => {
  assert.equal(localizedHref('/getting-started/quickstart', 'en'), '/getting-started/quickstart');
  assert.equal(localizedHref('/getting-started/quickstart', 'zh-Hans'), '/zh-Hans/getting-started/quickstart');
  assert.equal(localizedHref('/docs/zh-Hans/getting-started/quickstart', 'en'), '/getting-started/quickstart');
  assert.equal(localizedHref('/docs/skills?source=bundled#install', 'zh-Hans'), '/zh-Hans/skills?source=bundled#install');
});

test('public files and canonical URLs carry exactly one docs mount', () => {
  assert.equal(publicHref('/api/model-catalog.json'), '/docs/api/model-catalog.json');
  assert.equal(publicHref('/docs/api/model-catalog.json'), '/docs/api/model-catalog.json');
  assert.equal(publicHref('/'), '/docs/');
  assert.equal(canonicalUrl('/zh-Hans/user-guide/cli'), 'https://hermes-agent.nousresearch.com/docs/zh-Hans/user-guide/cli');
});

test('outside URLs and document-local fragments are not localized', () => {
  for (const href of ['https://example.com/path', 'mailto:hello@example.com', 'hermes://plugin/install?name=sample', '#setup']) {
    assert.equal(localizedHref(href, 'zh-Hans'), href);
    assert.equal(publicHref(href), href);
  }
});

test('locale parsing respects path segments, queries and the base path', () => {
  assert.deepEqual(splitLocalePath('/docs/zh-Hans/user-guide/cli?x=1#setup'), { locale: 'zh-Hans', path: '/user-guide/cli?x=1#setup' });
  assert.deepEqual(splitLocalePath('/docs/zh-Hans-extra'), { locale: 'en', path: '/zh-Hans-extra' });
  assert.deepEqual(splitLocalePath('/docs'), { locale: 'en', path: '/' });
});
