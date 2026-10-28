import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { parse } from 'yaml';
import { docs } from '../source.config.ts';
import { canonicalUrl } from '../src/lib/urls.ts';

// Exercise the real metadata export without loading the TSX renderer's
// generated content module or client components into the Node test runner.
const file = new URL('../src/components/document-page.tsx', import.meta.url);
const syntax = ts.createSourceFile(fileURLToPath(file), readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const declaration = syntax.statements.find((node) => ts.isFunctionDeclaration(node) && node.name?.text === 'documentMetadata');
assert.ok(declaration, 'documentMetadata must remain an exported route adapter');
const code = ts.transpileModule(ts.createPrinter().printNode(ts.EmitHint.Unspecified, declaration, syntax), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const loadMetadata = new Function('source', 'canonicalUrl', 'notFound', `const exports = {};\n${code}\nreturn exports.documentMetadata;`);

function pageMetadata(page) {
  const source = { getPage: () => page };
  return loadMetadata(source, canonicalUrl, () => assert.fail('fixture page must exist'))([], 'en');
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
});

test('ordinary documents retain their templated title and locale URL', () => {
  const metadata = pageMetadata({ url: '/zh-Hans/user-guide/cli', data: { title: 'CLI', description: 'CLI reference' } });
  assert.equal(metadata.title, 'CLI');
  assert.equal(metadata.openGraph.title, 'CLI');
  assert.equal(metadata.description, 'CLI reference');
  assert.equal(metadata.alternates.canonical, 'https://hermes-agent.nousresearch.com/docs/zh-Hans/user-guide/cli');
});
