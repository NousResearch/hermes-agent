import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { documentMetadata, documentSlugs, resolveDocumentLink } from '../src/lib/content.mjs';

test('metadata separates the visible H1 from an explicit SEO title', () => {
  const meta = documentMetadata('---\ntitle: Search title\nhide_table_of_contents: true\n---\n\n# Visible title\n', 'index.mdx');
  assert.equal(meta.title, 'Search title');
  assert.equal(meta.hasH1, true);
  assert.equal(meta.hide_table_of_contents, true);
  assert.equal(documentMetadata('```md\n# Example\n```\n\n# Real title\n', 'guide.md').title, 'Real title');
});

test('index filenames and explicit slugs retain their existing document URLs', () => {
  assert.deepEqual(documentSlugs('user-guide/secrets/index.md'), ['user-guide', 'secrets']);
  assert.deepEqual(documentSlugs('index.mdx', '/'), []);
  assert.deepEqual(documentSlugs('guides/example.md', '/guides/renamed'), ['guides', 'renamed']);
});

test('Markdown links resolve relative paths, bare names, overrides and locale fallback', (t) => {
  const root = mkdtempSync(join(tmpdir(), 'hermes-doc-links-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  for (const lang of ['en', 'zh-Hans']) mkdirSync(join(root, lang, 'guides'), { recursive: true });
  writeFileSync(join(root, 'en/guides/target.md'), '---\nslug: /guides/destination\n---\n# Target');
  const source = join(root, 'zh-Hans/guides/start.md');
  writeFileSync(source, '# Start');
  assert.equal(resolveDocumentLink(source, 'target.md?mode=1#anchor', root), '/guides/destination?mode=1#anchor');
  assert.equal(resolveDocumentLink(source, './target.md#anchor', root), '/guides/destination#anchor');
  assert.equal(resolveDocumentLink(source, 'https://example.com/target.md', root), 'https://example.com/target.md');
  assert.equal(resolveDocumentLink(source, '#anchor', root), '#anchor');
  assert.throws(() => resolveDocumentLink(source, './missing.md', root), /missing\.md/);
});

test('extensionless relative links resolve pages, fall back across locales and reject escapes', (t) => {
  const root = mkdtempSync(join(tmpdir(), 'hermes-doc-links-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const en = join(root, 'en');
  const zh = join(root, 'zh-Hans');
  mkdirSync(join(en, 'guides', 'bundle'), { recursive: true });
  mkdirSync(join(zh, 'guides'), { recursive: true });
  writeFileSync(join(en, 'guides', 'target.md'), '---\nslug: /guides/destination\n---\n# Target');
  writeFileSync(join(en, 'guides', 'section.md'), '# Section');
  writeFileSync(join(en, 'guides', 'index.md'), '# Guides');
  writeFileSync(join(en, 'guides', 'bundle', 'index.mdx'), '# Bundle');
  writeFileSync(join(en, 'guides', 'image.png'), 'PNG');
  // Locale fallback order: the zh-Hans page wins over an en namesake that
  // would rename the route, so ./local must not pick up the en slug.
  writeFileSync(join(en, 'guides', 'local.md'), '---\nslug: /guides/en-only-renamed\n---\n# Local EN');
  writeFileSync(join(zh, 'guides', 'local.md'), '# Local ZH');
  const source = join(zh, 'guides', 'start.md');
  writeFileSync(source, '# Start');
  assert.equal(resolveDocumentLink(source, './target', root), '/guides/destination');
  assert.equal(resolveDocumentLink(source, 'target#anchor', root), '/guides/destination#anchor');
  assert.equal(resolveDocumentLink(source, './target?mode=1#anchor', root), '/guides/destination?mode=1#anchor');
  assert.equal(resolveDocumentLink(source, '../guides/section.md', root), '/guides/section');
  assert.equal(resolveDocumentLink(source, '../guides', root), '/guides');
  assert.equal(resolveDocumentLink(source, './bundle', root), '/guides/bundle');
  assert.equal(resolveDocumentLink(source, './local', root), '/guides/local');
  // Non-document assets and absolute logical routes keep their current handling.
  assert.equal(resolveDocumentLink(source, './image.png', root), './image.png');
  assert.equal(resolveDocumentLink(source, '/guides/section', root), '/guides/section');
  assert.throws(() => resolveDocumentLink(source, './missing', root), /missing/);
  assert.throws(() => resolveDocumentLink(source, '../../outside', root), /escapes/);
  // A directory named like a document is not a link target.
  mkdirSync(join(en, 'guides', 'fake.md'));
  assert.throws(() => resolveDocumentLink(source, './fake.md', root), /fake\.md/);
});

test('absolute Markdown links resolve in-locale and reject escapes to existing outside files', (t) => {
  const root = mkdtempSync(join(tmpdir(), 'hermes-doc-links-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  for (const lang of ['en', 'zh-Hans']) mkdirSync(join(root, lang, 'guides'), { recursive: true });
  writeFileSync(join(root, 'en', 'guides', 'target.md'), '---\nslug: /guides/destination\n---\n# Target');
  writeFileSync(join(root, 'outside.md'), '# Outside the locales');
  const source = join(root, 'zh-Hans', 'guides', 'start.md');
  writeFileSync(source, '# Start');
  assert.equal(resolveDocumentLink(source, '/docs/guides/target.md', root), '/guides/destination');
  assert.equal(resolveDocumentLink(source, '/docs/guides/targ%65t.md', root), '/guides/destination');
  assert.throws(() => resolveDocumentLink(source, '/docs/guides/../../outside.md', root), /escapes/);
});

test('a bare .. target cannot reach the content root above the locale', (t) => {
  const root = mkdtempSync(join(tmpdir(), 'hermes-doc-links-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  for (const lang of ['en', 'zh-Hans']) mkdirSync(join(root, lang), { recursive: true });
  writeFileSync(join(root, 'index.md'), '# Content root above the locales');
  const source = join(root, 'zh-Hans', 'index.md');
  writeFileSync(source, '# Locale root');
  assert.throws(() => resolveDocumentLink(source, '..', root), /escapes/);
});

test('link paths with a null byte or malformed percent-encoding fail with clear errors', (t) => {
  const root = mkdtempSync(join(tmpdir(), 'hermes-doc-links-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, 'zh-Hans', 'guides'), { recursive: true });
  const source = join(root, 'zh-Hans', 'guides', 'start.md');
  writeFileSync(source, '# Start');
  assert.throws(() => resolveDocumentLink(source, './evil%00.md', root), /evil%00\.md.*null byte/);
  assert.throws(() => resolveDocumentLink(source, './bad%zz.md', root), /bad%zz\.md.*invalid percent-encoding/);
});
