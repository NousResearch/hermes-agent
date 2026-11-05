import assert from 'node:assert/strict';
import test from 'node:test';
import { documentDescription } from '../src/lib/document-description.mjs';

test('falls back to the first prose paragraph after frontmatter, imports, headings, fences and raw JSX', () => {
  const doc = [
    '---',
    'title: Setup',
    '---',
    '',
    "import Chart from './chart.mdx'",
    '',
    'export const version = 2',
    '',
    '<head>',
    '  <title>Install Hermes Agent: Official Guide | Nous Research</title>',
    '  <meta property="og:title" content="Install Hermes Agent: Official Guide | Nous Research" />',
    '</head>',
    '',
    '# Setup',
    '',
    '```bash',
    'hermes setup --quick',
    '```',
    '',
    'Install Hermes with **one command** and a [config file](./config.md) in `~/.hermes`.',
    '',
    'This second paragraph must not become the description.',
  ].join('\n');
  assert.equal(documentDescription(doc), 'Install Hermes with one command and a config file in ~/.hermes.');
});

test('tables and struck text are parsed as Markdown, not description markup', () => {
  const doc = '# Guide\n\n| Name | Value |\n| --- | --- |\n| Only | Table |\n\nThe ~~old~~ new guide.\n';
  assert.equal(documentDescription(doc), 'The old new guide.');
});

test('ordinary colon text remains intact in description prose', () => {
  assert.equal(documentDescription('# Guide\n\nRun at 9:00 for 1:1 DMs; read arXiv:2106.09685.\n'),
    'Run at 9:00 for 1:1 DMs; read arXiv:2106.09685.');
});

test('an authored frontmatter description is honored instead of the fallback', () => {
  const doc = [
    '---',
    'title: Setup',
    'description: Authored description wins.',
    '---',
    '',
    '# Setup',
    '',
    'First prose paragraph is ignored when the frontmatter carries a description.',
  ].join('\n');
  assert.equal(documentDescription(doc), 'Authored description wins.');
});

test('admonitions are not lead prose, CJK soft breaks stay seamless, and prose-free bodies yield undefined', () => {
  const zh = [
    '---',
    'title: 安装',
    '---',
    '',
    '# 安装',
    '',
    ':::tip[提示]',
    '这段提示内容不是页面摘要。',
    ':::',
    '',
    '使用一行安装命令，',
    '两分钟内即可启动并运行 Hermes Agent。',
  ].join('\n');
  assert.equal(documentDescription(zh), '使用一行安装命令，两分钟内即可启动并运行 Hermes Agent。');
  assert.equal(documentDescription('---\ntitle: Empty\n---\n\n# Empty\n\n```js\nonly code\n```\n'), undefined);
});
