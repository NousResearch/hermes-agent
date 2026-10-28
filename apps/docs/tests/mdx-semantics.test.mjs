import assert from 'node:assert/strict';
import test from 'node:test';
import { evaluate } from '@mdx-js/mdx';
import { applyMdxPreset } from 'fumadocs-mdx/config';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import * as runtime from 'react/jsx-runtime';
import config from '../source.config.ts';

const options = await applyMdxPreset(config.mdxOptions)('runtime');
const components = {
  CalloutContainer: ({ type, children }) => createElement('aside', { 'data-type': type }, children),
  CalloutTitle: ({ children }) => createElement('strong', null, children),
  CalloutDescription: ({ children }) => createElement('section', null, children),
  Mermaid: ({ chart }) => createElement('div', { 'data-mermaid': chart }),
};
async function render(value) {
  const module = await evaluate(value, { ...options, ...runtime, format: 'md' });
  return renderToStaticMarkup(createElement(module.default, { components }));
}

test('the configured compiler preserves native Markdown disclosures', async () => {
  const html = await render('<details>\n<summary>Show details</summary>\n\nBody.\n\n</details>');
  assert.match(html, /<details>/);
  assert.match(html, /<summary>Show details<\/summary>/);
  assert.match(html, /Body\./);
});

test('titled warning variants retain their meaning and nested brackets', async () => {
  for (const kind of ['caution', 'important']) {
    assert.match(await render(`:::${kind}[The [SILENT] Trick]\n\nBody.\n\n:::`), /data-type="warning"/);
  }
  assert.match(await render(':::tip[The [SILENT] Trick]\n\nBody.\n\n:::'), /The \[SILENT\] Trick/);
});

test('custom anchors keep their identifier and hide their authoring suffix', async () => {
  const html = await render('## Configuration [#stable-id]');
  assert.match(html, /id="stable-id"/);
  assert.doesNotMatch(html, /\[#stable-id\]/);
});

test('unrecognized code languages stay readable rather than breaking the page', async () => {
  const html = await render('```hermes-example-language\nkeep every byte\n```');
  assert.match(html, /keep every byte/);
});
