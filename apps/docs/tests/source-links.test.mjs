// Source-level relative-link resolution through the real checker and CLI.
// Run from apps/docs: node --test tests/source-links.test.mjs

import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import { checkSourceLinks } from '../scripts/check-source-links.mjs';
import { remarkDocumentLinks } from '../src/remark/docs.mjs';

const checkerCli = fileURLToPath(new URL('../scripts/check-source-links.mjs', import.meta.url));

/** A scratch app whose content tree mirrors the shapes the corpus uses. */
function buildApp(t, { plant } = {}) {
  const root = mkdtempSync(path.join(tmpdir(), 'hermes-source-links-'));
  const appDir = path.join(root, 'apps', 'docs');
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const en = (...segments) => {
    const file = path.join(appDir, 'content', 'en', ...segments);
    mkdirSync(path.dirname(file), { recursive: true });
    return file;
  };
  writeFileSync(en('index.mdx'), '# Home\n');
  writeFileSync(en('user-guide', 'configuration.md'), '# Configuration\n');
  writeFileSync(en('user-guide', 'features', 'goals.md'), [
    '# Goals\n',
    '[Kanban](./kanban) and [overview](../configuration#prompt-caching)\n',
    '[Secrets](./secrets) and [profile](../../profiles.md)\n',
    '[Shot](./screenshot.png), [external](https://example.com/x), [fragment](#goals), [route](/user-guide/configuration)\n',
    plant?.missingPage ? '[Missing](./s3-missing-extensionless)\n' : '',
  ].join(''));
  writeFileSync(en('user-guide', 'features', 'kanban.md'), '# Kanban\n');
  writeFileSync(en('user-guide', 'features', 'secrets', 'index.md'), '# Secrets\n');
  writeFileSync(en('profiles.md'), '# Profiles\n');
  writeFileSync(en('user-guide', 'features', 'screenshot.png'), 'PNG');
  return appDir;
}

test('extensionless page links resolve; assets, fragments and routes stay out of scope', async (t) => {
  const appDir = buildApp(t);
  const result = await checkSourceLinks(appDir);
  assert.deepEqual(result.errors, []);
  assert.ok(result.links >= 4, `expected the fixture links, got ${result.links}`);
});

test('a missing extensionless page link fails the source check', async (t) => {
  const appDir = buildApp(t, { plant: { missingPage: true } });
  const result = await checkSourceLinks(appDir);
  assert.equal(result.errors.length, 1);
  assert.match(result.errors[0], /s3-missing-extensionless/);
  assert.match(result.errors[0], /goals\.md/);
});

test('the CLI reports the failure and exits nonzero', (t) => {
  const appDir = buildApp(t, { plant: { missingPage: true } });
  const run = spawnSync(process.execPath, [checkerCli, '--app-dir', appDir], { encoding: 'utf8' });
  assert.notEqual(run.status, 0);
  assert.match(run.stderr, /s3-missing-extensionless/);
  const report = JSON.parse(run.stdout.trim().split('\n').pop());
  assert.equal(report.errors, 1);
});

test('the CLI exits zero for a clean tree', (t) => {
  const appDir = buildApp(t);
  const run = spawnSync(process.execPath, [checkerCli, '--app-dir', appDir], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
  const report = JSON.parse(run.stdout.trim().split('\n').pop());
  assert.equal(report.errors, 0);
  assert.ok(report.documents >= 5, `expected the fixture documents, got ${report.documents}`);
});

test('remarkDocumentLinks rewrites extensionless links in the rendered AST', (t) => {
  const appDir = buildApp(t);
  const source = path.join(appDir, 'content', 'en', 'user-guide', 'features', 'goals.md');
  const tree = unified().use(remarkParse).use(remarkGfm)
    .parse('# Goals\n\n[Kanban](./kanban)\n\n[definition]: /user-guide/configuration\n');
  remarkDocumentLinks({ contentDir: path.join(appDir, 'content') })(tree, { path: source });
  const urls = [];
  const walk = (node) => {
    if (node.type === 'link' || node.type === 'definition') urls.push([node.type, node.url]);
    for (const child of node.children ?? []) walk(child);
  };
  walk(tree);
  assert.deepEqual(urls, [
    ['link', '/user-guide/features/kanban'],
    ['definition', '/user-guide/configuration'],
  ]);
});
