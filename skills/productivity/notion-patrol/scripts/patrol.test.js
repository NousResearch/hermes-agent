const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const patrol = require('./patrol.js');

test('parseArgs defaults and repeatable roots', () => {
  assert.deepEqual(patrol.parseArgs([]).roots, patrol.DEFAULT_ROOTS);
  const opts = patrol.parseArgs(['--root', 'https://www.notion.so/Foo-106173b008788028ac4efd380a88308c?v=abc', '--root', '8db170cd-c5ba-4c48-8ef9-302e4e58cede', '--output-dir', '/tmp/out', '--concurrency', '3', '--timeout-ms', '1234', '--no-check']);
  assert.deepEqual(opts.roots, ['106173b008788028ac4efd380a88308c', '8db170cdc5ba4c488ef9302e4e58cede']);
  assert.equal(opts.outputDir, '/tmp/out');
  assert.equal(opts.concurrency, 3);
  assert.equal(opts.timeoutMs, 1234);
  assert.equal(opts.noCheck, true);
});

test('extractUrlsFromBlock extracts external URLs with context and ignores Notion internal URLs', () => {
  const block = {
    id: 'b1',
    type: 'paragraph',
    paragraph: { rich_text: [
      { plain_text: 'See ', href: null },
      { plain_text: 'example', href: 'https://example.com/a?x=1' },
      { plain_text: ' and https://www.notion.so/internal-abc123', href: null },
      { plain_text: ' plus raw https://docs.example.org/path).', href: null },
    ] },
  };
  const rows = patrol.extractUrlsFromBlock(block, 'Page A');
  assert.deepEqual(rows.map(r => r.url).sort(), ['https://docs.example.org/path', 'https://example.com/a?x=1']);
  assert.equal(rows[0].pageTitle, 'Page A');
  assert.match(rows[0].context, /example|docs/);
});

test('extractUrlsFromBlock supports bookmark/embed/file URLs', () => {
  const blocks = [
    { type: 'bookmark', bookmark: { url: 'https://bookmark.example/' } },
    { type: 'embed', embed: { url: 'https://embed.example/' } },
    { type: 'image', image: { type: 'external', external: { url: 'https://image.example/a.png' } } },
  ];
  const urls = blocks.flatMap(b => patrol.extractUrlsFromBlock(b, 'Assets').map(r => r.url));
  assert.deepEqual(urls, ['https://bookmark.example/', 'https://embed.example/', 'https://image.example/a.png']);
});

test('checkUrl uses HEAD then GET on 405 and classifies 2xx/3xx as OK', async () => {
  const calls = [];
  const fakeFetch = async (url, options) => {
    calls.push(options.method);
    if (options.method === 'HEAD') return { status: 405 };
    return { status: 301 };
  };
  const result = await patrol.checkUrl('https://example.com', { fetchImpl: fakeFetch, timeoutMs: 1000 });
  assert.deepEqual(calls, ['HEAD', 'GET']);
  assert.equal(result.statusCode, 301);
  assert.equal(result.judgement, 'OK');
});

test('writeCsv writes BOM UTF-8 and Japanese headers', () => {
  const dir = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'patrol-'));
  const file = patrol.writeCsv([{ pageTitle: 'ページ', url: 'https://e.example', statusCode: 404, judgement: 'NG', context: 'ctx, quote " ok' }], dir, new Date('2026-07-06T00:00:00Z'));
  const buf = fs.readFileSync(file);
  assert.equal(buf.subarray(0, 3).toString('hex'), 'efbbbf');
  const text = buf.toString('utf8');
  assert.match(text, /^﻿ページ名,URL,ステータスコード,判定,Context（文脈）/);
  assert.match(file, /link_check_test_20260706\.csv$/);
  assert.match(text, /"ctx, quote "" ok"/);
});
