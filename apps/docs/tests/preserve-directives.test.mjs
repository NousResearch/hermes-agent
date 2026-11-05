// Regression coverage for colon-joined prose and anchors. Exercise the real
// configured compiler so plugin ordering cannot silently swallow directives.

import assert from 'node:assert/strict';
import test from 'node:test';
import { createProcessor, evaluate } from '@mdx-js/mdx';
import { applyMdxPreset } from 'fumadocs-mdx/config';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import * as runtime from 'react/jsx-runtime';
import config from '../source.config.ts';
import { calloutLabel } from '../src/remark/admonition-labels.mjs';

const docsDir = fileURLToPath(new URL('..', import.meta.url));
const contentDir = path.join(docsDir, 'content');

const options = await applyMdxPreset(config.mdxOptions)('runtime');
const components = {
  CalloutContainer: ({ type, children }) => createElement('aside', { 'data-type': type }, children),
  CalloutTitle: ({ children }) => createElement('strong', null, children),
  CalloutDescription: ({ children }) => createElement('section', null, children),
  Mermaid: ({ chart }) => createElement('div', { 'data-mermaid': chart }),
  DefaultCalloutLabel: ({ type }) => calloutLabel(type, 'en'),
};
async function render(value) {
  const module = await evaluate(value, { ...options, ...runtime, format: 'md' });
  return renderToStaticMarkup(createElement(module.default, { components }));
}

const DIRECTIVE_TYPES = ['textDirective', 'leafDirective', 'containerDirective'];

function walk(node, visit) {
  visit(node);
  if (node.children) for (const child of node.children) walk(child, visit);
}

function listCorpus(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) listCorpus(full, out);
    else if (/\.(md|mdx)$/.test(entry)) out.push(full);
  }
  return out;
}

test('the real corpus lines with colon text keep both halves when rendered', async () => {
  // en/guides/cron-troubleshooting.md — `9:00 AM every day` lost its `:00`.
  const cronRows = readFileSync(path.join(contentDir, 'en/guides/cron-troubleshooting.md'), 'utf8')
    .split('\n').filter((line) => line.includes('9:00 AM'));
  assert.ok(cronRows.length > 0, 'cron-troubleshooting cron table rows with 9:00 AM moved?');
  for (const row of cronRows) {
    assert.match(await render(row), /9:00 AM/, `table row lost colon text: ${row}`);
  }

  // en/user-guide/messaging/line.md — `1:1 chat` lost its `:1` inside bold.
  const lineRow = readFileSync(path.join(contentDir, 'en/user-guide/messaging/line.md'), 'utf8')
    .split('\n').find((line) => line.includes('**1:1 chat**'));
  assert.ok(lineRow, 'line.md 1:1 chat table row moved?');
  assert.match(await render(lineRow), /<strong>1:1 chat<\/strong>/);
});

test('the iron-proxy heading keeps its port and the anchor deep links use', async () => {
  const heading = readFileSync(path.join(contentDir, 'en/user-guide/egress/iron-proxy.md'), 'utf8')
    .split('\n').find((line) => line.includes('bind-host\\>:9090'));
  assert.ok(heading, 'iron-proxy troubleshooting heading moved?');
  const html = await render(heading);
  assert.match(html, /bind-host&gt;:9090 within 5s/);
  // The pre-migration Docusaurus export linked to ...-bind-host9090-within-5s.
  assert.match(html, /id="iron-proxy-did-not-bind-bind-host9090-within-5s"/);
  assert.doesNotMatch(html, /id="iron-proxy-did-not-bind-bind-host-within-5s"/);
});

test('unknown inline directives keep their label, markup and attributes as text', async () => {
  const html = await render(':abbr[Some **bold** text]{key=val flag}');
  assert.match(html, /:abbr\[Some <strong>bold<\/strong> text\]\{key=val flag\}/);
  assert.doesNotMatch(html, /<div><\/div>/);
});

test('unknown leaf directives render their source text, not an empty box', async () => {
  const html = await render('::youtube\n');
  assert.match(html, /::youtube/);
  assert.doesNotMatch(html, /<div><\/div>/);
});

test('unknown leaf directives with bracket content keep marker and content', async () => {
  const html = await render('::video[Some **bold** title]\n');
  assert.match(html, /::video\[Some <strong>bold<\/strong> title\]/);
});

test('unknown container directives without a label render one marker and one body', async () => {
  const html = await render(':::weird\n\nSome **bold** body.\n\n:::');
  assert.match(html, /<p>:::weird<\/p>/);
  assert.doesNotMatch(html, /:::weird\[/);
  assert.equal((html.match(/<p>Some <strong>bold<\/strong> body\.<\/p>/g) ?? []).length, 1, `body duplicated: ${html}`);
  assert.doesNotMatch(html, /<p><\/p>/);
  assert.match(html, /<p>:::<\/p>/);
  assert.doesNotMatch(html, /data-type/);
});

test('unknown container directives render their markers and keep their body markup', async () => {
  const html = await render(':::weird[Label with [brackets]]\n\nSome **bold** body.\n\n:::');
  assert.match(html, /:::weird\[Label with \[brackets\]\]/);
  assert.match(html, /<strong>bold<\/strong>/);
  assert.match(html, /<p>:::<\/p>/);
  assert.doesNotMatch(html, /data-type/);
});

test('known admonitions and Mermaid keep rendering as components, not source text', async () => {
  const html = await render(':::note\n\nKnown note.\n\n:::\n\n```mermaid\ngraph TD;\n  A-->B;\n```');
  assert.match(html, /data-type="info"/);
  assert.doesNotMatch(html, /:::note/);
  assert.match(html, /data-mermaid/);
});

test('inline code, port autolinks and plain prose were never directives and stay intact', async () => {
  const html = await render('Set `host:9090`, see <https://example.com:8080/health>, or Note: like this.');
  assert.match(html, /<code>host:9090<\/code>/);
  assert.match(html, /href="https:\/\/example.com:8080\/health"/);
  assert.match(html, /Note: like this/);
});

test('no directive node survives the configured remark stage anywhere in the corpus', async () => {
  const files = listCorpus(contentDir);
  for (const locale of ['en', 'zh-Hans']) {
    assert.ok(files.some((file) => file.startsWith(path.join(contentDir, locale) + path.sep)), `no ${locale} content discovered`);
  }

  // The real configured remark chain, observed by a recorder appended after
  // every production plugin, driven by MDX's own parser per file.
  let sites = [];
  const recorder = () => (tree) => walk(tree, (node) => {
    if (DIRECTIVE_TYPES.includes(node.type)) sites.push(`${node.type}(${node.name})`);
  });
  const processors = {
    md: createProcessor({ ...options, remarkPlugins: [...options.remarkPlugins, recorder], rehypePlugins: [], format: 'md' }),
    mdx: createProcessor({ ...options, remarkPlugins: [...options.remarkPlugins, recorder], rehypePlugins: [], format: 'mdx' }),
  };

  const survivors = [];
  for (const file of files) {
    const format = file.endsWith('.mdx') ? 'mdx' : 'md';
    const descriptor = { path: file, value: readFileSync(file, 'utf8') };
    sites = [];
    await processors[format].run(processors[format].parse(descriptor), descriptor);
    for (const site of sites) survivors.push(`${path.relative(contentDir, file)}: ${site}`);
  }
  assert.deepEqual(survivors, [], `${survivors.length} unconsumed directive nodes survive the remark stage:\n${survivors.join('\n')}`);
});
