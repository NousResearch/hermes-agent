import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { evaluate } from '@mdx-js/mdx';
import { applyMdxPreset } from 'fumadocs-mdx/config';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import * as runtime from 'react/jsx-runtime';
import config from '../source.config.ts';
import { calloutLabel } from '../src/remark/admonition-labels.mjs';

const options = await applyMdxPreset(config.mdxOptions)('runtime');
const components = {
  CalloutContainer: ({ type, children }) => createElement('aside', { 'data-type': type }, children),
  CalloutTitle: ({ children }) => createElement('strong', null, children),
  CalloutDescription: ({ children }) => createElement('section', null, children),
  Mermaid: ({ chart }) => createElement('div', { 'data-mermaid': chart }),
  DefaultCalloutLabel: ({ type }) => calloutLabel(type, 'en'),
};
/** The page binds callout labels to the locale it is rendered in, not to the source file. */
const zhComponents = { ...components, DefaultCalloutLabel: ({ type }) => calloutLabel(type, 'zh-Hans') };
async function render(value, filePath, viewing = components) {
  const module = await evaluate(filePath ? { path: filePath, value } : value, { ...options, ...runtime, format: 'md' });
  return renderToStaticMarkup(createElement(module.default, { components: viewing }));
}
async function renderMdx(value) {
  const module = await evaluate(value, { ...options, ...runtime, format: 'mdx' });
  return renderToStaticMarkup(createElement(module.default, { components }));
}
/** The exact bytes of one fenced block from a real corpus page. */
function fenceFrom(page, marker) {
  const start = page.indexOf(marker);
  assert.ok(start !== -1, `fence marker not found: ${marker}`);
  const end = page.indexOf('\n```', start + marker.length);
  assert.ok(end !== -1, `closing fence not found after: ${marker}`);
  return page.slice(start, end + 4);
}

test('the configured compiler preserves native Markdown disclosures', async () => {
  const html = await render('<details>\n<summary>Show details</summary>\n\nBody.\n\n</details>');
  assert.match(html, /<details>/);
  assert.match(html, /<summary>Show details<\/summary>/);
  assert.match(html, /Body\./);
});

test('titled admonitions retain nested brackets and their semantic type', async () => {
  for (const [kind, type] of Object.entries({ caution: 'warning', important: 'info' })) {
    assert.match(await render(`:::${kind}[The [SILENT] Trick]\n\nBody.\n\n:::`), new RegExp(`data-type="${type}"`));
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

test('colon-joined words keep both halves instead of an empty directive', async () => {
  const html = await render('Cron runs at 9:00 AM every day. Use a 1:1 ratio; the tunnel listens on host:9090.');
  assert.match(html, /9:00 AM every day/);
  assert.match(html, /1:1 ratio/);
  assert.match(html, /host:9090/);
  assert.doesNotMatch(html, /<div><\/div>/);
});

test('a port inside a heading keeps its text and its deep-link anchor', async () => {
  const html = await render('### "iron-proxy did not bind \\<bind-host\\>:9090 within 5s"');
  assert.match(html, /bind-host&gt;:9090 within 5s/);
  assert.match(html, /id="iron-proxy-did-not-bind-bind-host9090-within-5s"/);
});

test('the real docker.md fences keep their filename labels', async () => {
  const page = readFileSync(new URL('../content/en/user-guide/docker.md', import.meta.url), 'utf8');
  const conf = await render(fenceFrom(page, '```conf title="asound.conf"'));
  assert.match(conf, /<pre[^>]*title="asound\.conf"/);
  const dockerfile = await render(fenceFrom(page, '```dockerfile title="Dockerfile.audio"'));
  assert.match(dockerfile, /<pre[^>]*title="Dockerfile\.audio"/);
});

test('fence titles with spaces and quoted paths survive the raw re-parse', async () => {
  const spaced = await render('```python title="watch site.py"\nprint("hi")\n```');
  assert.match(spaced, /<pre[^>]*title="watch site\.py"/);
  const home = await render('```yaml title="~/.hermes/config.yaml"\nai: sound\n```');
  assert.match(home, /<pre[^>]*title="~\/\.hermes\/config\.yaml"/);
});

test('MDX-formatted documents keep titled fence labels too', async () => {
  const html = await renderMdx('```yaml title="~/.hermes/config.yaml"\nai: sound\n```');
  assert.match(html, /<pre[^>]*title="~\/\.hermes\/config\.yaml"/);
});

test('raw HTML, highlighting and fence titles keep coexisting', async () => {
  const html = await render(
    '<details>\n<summary>Files</summary>\n\n```yaml title="compose.yaml"\nservices:\n  audio:\n    image: alpine\n```\n\n</details>\n\n```js\nconst x = 1;\n```',
  );
  assert.match(html, /<details>/);
  assert.match(html, /title="compose\.yaml"/);
  assert.match(html, /<span style="[^"]*--shiki-light:#D73A49/);
  assert.doesNotMatch(html, /metastring/);
});

test('admonition aliases preserve informational and success semantics', async () => {
  for (const [name, type] of Object.entries({ tip: 'success', success: 'success', important: 'info', caution: 'warning', danger: 'error' })) {
    const html = await render(`:::${name}[Authored title]\n\nBody.\n\n:::`);
    assert.match(html, new RegExp(`<aside data-type="${type}">`), name);
  }
});

test('unlabeled admonitions show each type\u2019s distinct default label', async () => {
  const labels = {
    note: 'note',
    tip: 'tip',
    info: 'info',
    warning: 'warning',
    caution: 'caution',
    important: 'important',
    danger: 'danger',
    success: 'success',
  };
  for (const [name, label] of Object.entries(labels)) {
    const html = await render(`:::${name}\n\nBody text.\n\n:::`);
    assert.match(html, new RegExp(`<strong>${label}</strong>`), name);
  }
});

test('zh-Hans unlabeled admonitions show translated default labels', async () => {
  const labels = {
    note: '备注',
    tip: '提示',
    info: '信息',
    warning: '注意',
    caution: '警告',
    danger: '危险',
  };
  for (const [name, label] of Object.entries(labels)) {
    const html = await render(`:::${name}\n\n内容。\n\n:::`, 'content/zh-Hans/user-guide/features/cron.md', zhComponents);
    assert.match(html, new RegExp(`<strong>${label}</strong>`), name);
  }
  // Docusaurus kept the legacy aliases untranslated in every locale.
  const important = await render(':::important\n\n内容。\n\n:::', 'content/zh-Hans/user-guide/features/cron.md', zhComponents);
  assert.match(important, /<strong>important<\/strong>/);
});

test('a Chinese route with no zh-Hans source shows the translated label for its English callouts', async () => {
  // Fallback pages compile once from content/en and render on both routes.
  const english = 'content/en/developer-guide/browser-provider-plugin.md';
  assert.match(await render(':::tip\n\nBody.\n\n:::', english, zhComponents), /<strong>提示<\/strong>/);
  assert.match(await render(':::tip\n\nBody.\n\n:::', english), /<strong>tip<\/strong>/);
});

test('titled admonitions keep their authored title and gain no default label', async () => {
  const html = await render(':::note[Custom Title]\n\nBody text.\n\n:::');
  assert.match(html, /<strong>Custom Title<\/strong>/);
  assert.doesNotMatch(html, /<strong>note<\/strong>/);
});

test('unknown prototype-named directives remain literal source', async () => {
  const html = await render(':::constructor\n\nUnrecognized content.\n\n:::');
  assert.match(html, /:::constructor/);
  assert.match(html, /Unrecognized content\./);
});

test('colon-joined text inside admonition titles keeps both halves', async () => {
  const html = await render(':::note[1:1 DMs]\n\nBody text.\n\n:::');
  assert.match(html, /<strong>1:1 DMs<\/strong>/);
  const zh = await render(':::note[1:1 私信（DM）]\n\n内容。\n\n:::', 'content/zh-Hans/user-guide/features/cron.md');
  assert.match(zh, /<strong>1:1 私信（DM）<\/strong>/);
});

test('arXiv identifiers keep their colon inside admonition bodies', async () => {
  const html = await render(':::note\n\nLoRA paper: arXiv:2106.09685\n\n:::');
  assert.match(html, /arXiv:2106\.09685/);
});
