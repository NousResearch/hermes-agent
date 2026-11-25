import assert from 'node:assert/strict';
import test from 'node:test';
import { catalogAlternates } from '../src/lib/catalog-routes.ts';

test('catalog pages (search) carry slashless canonical and hreflang URLs for both locales', () => {
  const zh = catalogAlternates('/search', 'zh-Hans');
  assert.equal(zh.canonical, 'https://hermes-agent.nousresearch.com/docs/zh-Hans/search');
  assert.equal(zh.languages.en, 'https://hermes-agent.nousresearch.com/docs/search');
  assert.equal(zh.languages['zh-Hans'], zh.canonical);
  assert.equal(zh.languages['x-default'], zh.languages.en);
  assert.equal(catalogAlternates('/search', 'en').canonical, zh.languages.en);
});
