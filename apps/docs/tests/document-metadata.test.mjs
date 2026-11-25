import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'yaml';
import { docs } from '../source.config.ts';
import { documentPageMetadata } from '../src/lib/document-metadata.ts';
import { siteMetadata } from '../src/lib/metadata-defaults.ts';
import { publicHref } from '../src/lib/urls.ts';

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
  assert.equal(metadata.alternates.canonical, 'https://hermes-agent.nousresearch.com/docs');
  assert.equal(metadata.alternates.languages['x-default'], 'https://hermes-agent.nousresearch.com/docs');
});

test('description fallback reaches the actual content schema and page metadata', (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'docs-description-schema-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const file = join(directory, 'guide.md');
  writeFileSync(file, '# Guide\n\nFirst **useful** paragraph.\n\nSecond paragraph.\n');
  const schema = docs.docs.schema({ path: file });
  const inferred = schema.parse({});
  assert.equal(inferred.description, 'First useful paragraph.');
  assert.equal(pageMetadata({ url: '/guide', data: inferred }).description, 'First useful paragraph.');
  assert.equal(schema.parse({ description: 'Authored summary.' }).description, 'Authored summary.');
  assert.equal(schema.parse({ description: '' }).description, 'First useful paragraph.');
  assert.equal(schema.parse({ description: '   ' }).description, 'First useful paragraph.');
  writeFileSync(file, '# Guide\n\n```js\nconst onlyCode = true;\n```\n');
  assert.equal(docs.docs.schema({ path: file }).parse({}).description, undefined);
});

test('ordinary documents retain their templated title and locale URL', () => {
  const metadata = pageMetadata({ url: '/zh-Hans/user-guide/cli', data: { title: 'CLI', description: 'CLI reference' } });
  assert.equal(metadata.title, 'CLI');
  assert.equal(metadata.openGraph.title, 'CLI');
  assert.equal(metadata.description, 'CLI reference');
  assert.equal(metadata.alternates.canonical, 'https://hermes-agent.nousresearch.com/docs/zh-Hans/user-guide/cli');
  assert.equal(metadata.alternates.languages['x-default'], metadata.alternates.languages.en);
});

test('the default-language alternate follows the English page without inventing a missing translation', () => {
  const english = { url: '/guide', data: { title: 'Guide' } };
  const chinese = { url: '/zh-Hans/guide', data: { title: '指南' } };
  for (const page of [english, chinese]) {
    const metadata = documentPageMetadata(page, { en: english, 'zh-Hans': chinese });
    assert.equal(metadata.alternates.languages['x-default'], 'https://hermes-agent.nousresearch.com/docs/guide');
    assert.equal(metadata.alternates.languages['zh-Hans'], 'https://hermes-agent.nousresearch.com/docs/zh-Hans/guide');
  }
  const localeOnly = documentPageMetadata(chinese, { 'zh-Hans': chinese });
  assert.equal(localeOnly.alternates.languages['x-default'], localeOnly.alternates.canonical);
});

test('page-level OpenGraph keeps the site banner and name the root layout set', () => {
  // Next merges metadata shallowly per segment (mergeMetadata in
  // metadata-resolution-primitives.js), so this page-level object fully
  // replaces the root's openGraph instead of extending it: it must carry the
  // site identity itself or the page loses og:image and og:site_name.
  const root = siteMetadata('en');
  const metadata = pageMetadata({ url: '/guide', data: { title: 'Guide' } });
  assert.equal(metadata.openGraph.siteName, root.openGraph.siteName);
  assert.deepEqual(metadata.openGraph.images, root.openGraph.images);
  assert.deepEqual(metadata.openGraph.images, [publicHref('/img/hermes-agent-banner.png')]);
  assert.equal(metadata.openGraph.title, 'Guide');
  assert.equal(metadata.openGraph.url, 'https://hermes-agent.nousresearch.com/docs/guide');
});

test('the site root declares the large Twitter card and banner the pages inherit', () => {
  // Next emits no twitter tags unless some segment sets `twitter`
  // (resolveTwitter returns null), and Twitter falls back to og:image when
  // twitter:image is absent — so the root card setting keeps the large-card
  // presentation on every page while a page's own og:image still wins.
  const root = siteMetadata('en');
  assert.equal(root.twitter.card, 'summary_large_image');
  assert.equal(root.openGraph.siteName, 'Hermes Agent');
  assert.deepEqual(root.openGraph.images, [publicHref('/img/hermes-agent-banner.png')]);
  assert.equal(root.other['docsearch:language'], 'en');
});

test("the installation page's SEO title comes from seo_title, not a raw head block", () => {
  const file = new URL('../content/en/getting-started/installation.md', import.meta.url);
  const frontmatter = parse(readFileSync(file, 'utf8').match(/^---\n([\s\S]*?)\n---/)[1]);
  const data = docs.docs.schema({ path: fileURLToPath(file) }).parse(frontmatter);
  const metadata = pageMetadata({ url: '/getting-started/installation', data });
  const officialTitle = 'Install Hermes Agent: Official Guide | Nous Research';
  assert.deepEqual(metadata.title, { absolute: officialTitle });
  assert.equal(metadata.openGraph.title, officialTitle);
  assert.equal(data.title, 'Installation', 'the visible H1/sidebar title stays authored');
  assert.equal(data.hasH1, true);
  assert.equal(metadata.alternates.canonical, 'https://hermes-agent.nousresearch.com/docs/getting-started/installation');
});
