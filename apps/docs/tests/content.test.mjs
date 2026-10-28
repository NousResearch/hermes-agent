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
