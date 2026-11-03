import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { openSearchXml, writeOpenSearch } from '../scripts/opensearch.mjs';

test('the English descriptor points at the English search page and itself', () => {
  const xml = openSearchXml('en');
  assert.ok(xml.startsWith('<?xml version="1.0" encoding="UTF-8"?>'));
  assert.ok(xml.includes('xmlns="http://a9.com/-/spec/opensearch/1.1/"'));
  assert.ok(xml.includes('xmlns:moz="http://www.mozilla.org/2006/browser/search/"'));
  assert.ok(xml.includes('<ShortName>Hermes Agent</ShortName>'));
  assert.ok(xml.includes('<InputEncoding>UTF-8</InputEncoding>'));
  assert.ok(xml.includes(
    '<Url type="text/html" method="get" template="https://hermes-agent.nousresearch.com/docs/search/?q={searchTerms}"/>',
  ));
  assert.ok(xml.includes(
    '<Url type="application/opensearchdescription+xml" rel="self" template="https://hermes-agent.nousresearch.com/docs/opensearch.xml" />',
  ));
  assert.ok(xml.includes(
    '<Image width="16" height="16" type="image/x-icon">https://hermes-agent.nousresearch.com/docs/img/favicon.ico</Image>',
  ));
  assert.ok(xml.includes('<moz:SearchForm>https://hermes-agent.nousresearch.com/docs/</moz:SearchForm>'));
});

test('the Chinese descriptor localizes search while sharing the canonical icon', () => {
  const xml = openSearchXml('zh-Hans');
  assert.ok(xml.includes(
    '<Url type="text/html" method="get" template="https://hermes-agent.nousresearch.com/docs/zh-Hans/search/?q={searchTerms}"/>',
  ));
  assert.ok(xml.includes(
    '<Url type="application/opensearchdescription+xml" rel="self" template="https://hermes-agent.nousresearch.com/docs/zh-Hans/opensearch.xml" />',
  ));
  assert.ok(xml.includes(
    '<Image width="16" height="16" type="image/x-icon">https://hermes-agent.nousresearch.com/docs/img/favicon.ico</Image>',
  ));
  assert.ok(xml.includes('<moz:SearchForm>https://hermes-agent.nousresearch.com/docs/zh-Hans/</moz:SearchForm>'));
});

test('writeOpenSearch publishes both descriptors and reports the files it wrote', (t) => {
  const outputDir = mkdtempSync(join(tmpdir(), 'opensearch-test-'));
  t.after(() => rmSync(outputDir, { recursive: true, force: true }));
  const written = writeOpenSearch(outputDir);
  assert.equal(written.length, 2);
  const root = written.find((file) => file.endsWith(join('opensearch.xml')) && !file.includes('zh-Hans'));
  assert.ok(root, `expected a root descriptor among ${written}`);
  assert.equal(readFileSync(root, 'utf8'), openSearchXml('en'));
  const zh = written.find((file) => file.includes(`zh-Hans${join('')}`) || file.endsWith(join('zh-Hans', 'opensearch.xml')));
  assert.ok(zh, `expected a zh-Hans descriptor among ${written}`);
  assert.equal(readFileSync(zh, 'utf8'), openSearchXml('zh-Hans'));
  assert.equal(
    readFileSync(join(outputDir, 'zh-Hans', 'opensearch.xml'), 'utf8'),
    openSearchXml('zh-Hans'),
  );
});
