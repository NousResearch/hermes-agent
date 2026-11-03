import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parse } from 'yaml';
import { docs } from '../source.config.ts';
import { documentPageMetadata } from '../src/lib/document-metadata.ts';

function pageMetadata(page) {
  return documentPageMetadata(page, { en: page, 'zh-Hans': page });
}

test('the official home-page SEO title survives the content schema and overrides the site template', () => {
  const home = new URL('../content/en/index.mdx', import.meta.url);
  const frontmatter = parse(readFileSync(home, 'utf8').match(/^---\n([\s\S]*?)\n---/)[1]);
  const data = docs.docs.schema({ path: fileURLToPath(home) }).parse(frontmatter);
  const metadata = pageMetadata({ url: '/', data });
  const officialTitle = 'Hermes Agent Docs (Official) | Nous Research';
  assert.deepEqual(metadata.title, { absolute: officialTitle });
  assert.equal(metadata.openGraph.title, officialTitle);
  assert.equal(data.title, 'Hermes Agent Documentation', 'SEO must not change the visible/sidebar title');
  assert.equal(metadata.alternates.canonical, 'https://hermes-agent.nousresearch.com/docs/');
  assert.equal(metadata.alternates.languages['x-default'], 'https://hermes-agent.nousresearch.com/docs/');
});

test('ordinary documents retain their templated title and locale URL', () => {
  const metadata = pageMetadata({ url: '/zh-Hans/user-guide/cli', data: { title: 'CLI', description: 'CLI reference' } });
  assert.equal(metadata.title, 'CLI');
  assert.equal(metadata.openGraph.title, 'CLI');
  assert.equal(metadata.description, 'CLI reference');
  assert.equal(metadata.alternates.canonical, 'https://hermes-agent.nousresearch.com/docs/zh-Hans/user-guide/cli/');
  assert.equal(metadata.alternates.languages['x-default'], metadata.alternates.languages.en);
});

test('the default-language alternate follows the English page without inventing a missing translation', () => {
  const english = { url: '/guide', data: { title: 'Guide' } };
  const chinese = { url: '/zh-Hans/guide', data: { title: '指南' } };
  for (const page of [english, chinese]) {
    const metadata = documentPageMetadata(page, { en: english, 'zh-Hans': chinese });
    assert.equal(metadata.alternates.languages['x-default'], 'https://hermes-agent.nousresearch.com/docs/guide/');
    assert.equal(metadata.alternates.languages['zh-Hans'], 'https://hermes-agent.nousresearch.com/docs/zh-Hans/guide/');
  }
  const localeOnly = documentPageMetadata(chinese, { 'zh-Hans': chinese });
  assert.equal(localeOnly.alternates.languages['x-default'], localeOnly.alternates.canonical);
});
