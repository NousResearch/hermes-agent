#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');

const DEFAULT_ROOTS = [
  '106173b008788028ac4efd380a88308c',
  '8db170cdc5ba4c488ef9302e4e58cede',
];
const DEFAULT_NOTION_VERSION = '2022-06-28';
const NOTION_BASE = 'https://api.notion.com/v1';

function normalizeNotionId(input) {
  const s = String(input || '').trim();
  const noQuery = s.split(/[?#]/)[0];
  const match = noQuery.match(/([0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})(?:\b|$)/);
  if (!match) return s.replace(/-/g, '');
  return match[1].replace(/-/g, '').toLowerCase();
}

function parseArgs(argv) {
  const opts = { roots: [], outputDir: process.cwd(), concurrency: 10, timeoutMs: 10000, notionVersion: DEFAULT_NOTION_VERSION, noCheck: false, help: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--help' || a === '-h') opts.help = true;
    else if (a === '--root') opts.roots.push(normalizeNotionId(argv[++i]));
    else if (a === '--output-dir') opts.outputDir = argv[++i];
    else if (a === '--concurrency') opts.concurrency = Number(argv[++i]);
    else if (a === '--timeout-ms') opts.timeoutMs = Number(argv[++i]);
    else if (a === '--notion-version') opts.notionVersion = argv[++i];
    else if (a === '--no-check') opts.noCheck = true;
    else throw new Error(`Unknown argument: ${a}`);
  }
  if (opts.roots.length === 0) opts.roots = [...DEFAULT_ROOTS];
  if (!Number.isInteger(opts.concurrency) || opts.concurrency < 1) throw new Error('--concurrency must be a positive integer');
  if (!Number.isInteger(opts.timeoutMs) || opts.timeoutMs < 1) throw new Error('--timeout-ms must be a positive integer');
  return opts;
}

function loadEnvFile(file) {
  if (!fs.existsSync(file)) return;
  for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
    if (!m || process.env[m[1]]) continue;
    process.env[m[1]] = m[2].replace(/^['"]|['"]$/g, '');
  }
}

function getToken() {
  loadEnvFile(path.join(process.cwd(), '.env'));
  const hermesHome = process.env.HERMES_HOME || (process.env.HOME ? path.join(process.env.HOME, '.hermes') : '');
  if (hermesHome) loadEnvFile(path.join(hermesHome, '.env'));
  loadEnvFile(path.join(__dirname, '..', '.env'));
  return process.env.NOTION_API_KEY || process.env.NOTION_TOKEN || process.env.NOTION_API_TOKEN || '';
}

function isExternalUrl(u) {
  try {
    const url = new URL(u);
    return /^https?:$/.test(url.protocol) && !/(^|\.)notion\.so$/.test(url.hostname) && url.hostname !== 'app.notion.com';
  } catch { return false; }
}

function cleanRawUrl(u) { return String(u).replace(/[\s\])}>,。、「」]+$/u, ''); }
function richTextPlain(arr = []) { return arr.map(t => t.plain_text || '').join(''); }
function addUrl(rows, url, pageTitle, context) { const cleaned = cleanRawUrl(url); if (isExternalUrl(cleaned)) rows.push({ pageTitle, url: cleaned, context: context || cleaned }); }

function extractUrlsFromBlock(block, pageTitle = '') {
  const rows = [];
  const type = block.type;
  const value = block[type] || {};
  const texts = value.rich_text || value.caption || [];
  const context = richTextPlain(texts) || value.url || '';
  for (const t of texts) {
    if (t.href) addUrl(rows, t.href, pageTitle, context);
    const raw = t.plain_text || '';
    for (const m of raw.matchAll(/https?:\/\/[^\s<>()"'「」]+/g)) addUrl(rows, m[0], pageTitle, context);
  }
  for (const key of ['bookmark', 'embed', 'link_preview']) if (type === key && value.url) addUrl(rows, value.url, pageTitle, value.url);
  if (['image', 'video', 'file', 'pdf'].includes(type)) {
    const u = value.type === 'external' && value.external ? value.external.url : value.url;
    if (u) addUrl(rows, u, pageTitle, u);
  }
  return rows;
}

async function checkUrl(url, { fetchImpl = fetch, timeoutMs = 10000 } = {}) {
  async function request(method) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try { return await fetchImpl(url, { method, redirect: 'follow', signal: controller.signal }); }
    finally { clearTimeout(timer); }
  }
  try {
    let res = await request('HEAD');
    if (res.status === 405) res = await request('GET');
    return { statusCode: res.status, judgement: res.status >= 200 && res.status < 400 ? 'OK' : 'NG' };
  } catch (e) {
    return { statusCode: e && e.name === 'AbortError' ? 'TIMEOUT' : 'ERROR', judgement: 'NG' };
  }
}

async function mapLimit(items, limit, fn) {
  const out = new Array(items.length); let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) { const i = next++; out[i] = await fn(items[i], i); }
  }));
  return out;
}

function csvEscape(v) { const s = String(v ?? ''); return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; }
function ymd(d = new Date()) { return `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}`; }
function writeCsv(rows, outputDir, date = new Date()) {
  fs.mkdirSync(outputDir, { recursive: true });
  const file = path.join(outputDir, `link_check_test_${ymd(date)}.csv`);
  const header = ['ページ名','URL','ステータスコード','判定','Context（文脈）'];
  const lines = [header, ...rows.map(r => [r.pageTitle, r.url, r.statusCode, r.judgement, r.context])].map(r => r.map(csvEscape).join(','));
  fs.writeFileSync(file, '\ufeff' + lines.join('\n'), 'utf8');
  return file;
}

class NotionClient {
  constructor({ token, notionVersion = DEFAULT_NOTION_VERSION, fetchImpl = fetch }) { this.token = token; this.notionVersion = notionVersion; this.fetchImpl = fetchImpl; }
  async request(method, endpoint, body) {
    const res = await this.fetchImpl(`${NOTION_BASE}${endpoint}`, { method, headers: { Authorization: `Bearer ${this.token}`, 'Notion-Version': this.notionVersion, 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
    if (!res.ok) throw new Error(`Notion API ${res.status} ${endpoint}`);
    return res.json();
  }
  async listChildren(id) { let results = [], cursor; do { const q = cursor ? `?start_cursor=${encodeURIComponent(cursor)}` : ''; const j = await this.request('GET', `/blocks/${id}/children${q}`); results = results.concat(j.results || []); cursor = j.has_more ? j.next_cursor : null; } while (cursor); return results; }
  async queryDatabase(id) { let results = [], cursor; do { const j = await this.request('POST', `/databases/${id}/query`, cursor ? { start_cursor: cursor } : {}); results = results.concat(j.results || []); cursor = j.has_more ? j.next_cursor : null; } while (cursor); return results; }
  async getPage(id) { return this.request('GET', `/pages/${id}`); }
  async getDatabase(id) { return this.request('GET', `/databases/${id}`); }
}

function titleFromObject(obj, fallback) {
  const props = obj.properties || {}; for (const p of Object.values(props)) if (p.type === 'title') return richTextPlain(p.title) || fallback;
  if (obj.title) return richTextPlain(obj.title) || fallback;
  return fallback;
}

async function traverseNotion(rootIds, client) {
  const visited = new Set(); const rows = [];
  async function visit(id, titleHint) {
    id = normalizeNotionId(id); if (visited.has(id)) return; visited.add(id);
    let title = titleHint || id, blocks = [];
    try { title = titleFromObject(await client.getPage(id), title); } catch {}
    try { blocks = await client.listChildren(id); } catch (e) {
      try { title = titleFromObject(await client.getDatabase(id), title); const pages = await client.queryDatabase(id); for (const p of pages) await visit(p.id, titleFromObject(p, p.id)); } catch { throw e; }
      return;
    }
    for (const b of blocks) {
      rows.push(...extractUrlsFromBlock(b, title));
      if (b.type === 'child_page') await visit(b.id, b.child_page && b.child_page.title);
      else if (b.type === 'child_database') { try { for (const p of await client.queryDatabase(b.id)) await visit(p.id, titleFromObject(p, p.id)); } catch {} }
      else if (b.has_children) await visit(b.id, title);
    }
  }
  for (const id of rootIds) await visit(id);
  return rows;
}

function help() { return `Usage: node scripts/patrol.js [--root <page-id-or-url>]... [--output-dir <dir>] [--concurrency <n>] [--timeout-ms <ms>] [--notion-version <version>] [--no-check]\nEnv: NOTION_API_KEY or NOTION_TOKEN or NOTION_API_TOKEN (also .env). Read-only: no Notion writes are performed.`; }

async function main(argv = process.argv.slice(2)) {
  const opts = parseArgs(argv); if (opts.help) { console.log(help()); return 0; }
  const token = getToken(); if (!token) throw new Error('Notion token not found. Set NOTION_API_KEY / NOTION_TOKEN / NOTION_API_TOKEN.');
  const client = new NotionClient({ token, notionVersion: opts.notionVersion });
  const extracted = await traverseNotion(opts.roots, client);
  const checked = opts.noCheck ? extracted.map(r => ({ ...r, statusCode: 'SKIPPED', judgement: 'OK' })) : await mapLimit(extracted, opts.concurrency, async r => ({ ...r, ...(await checkUrl(r.url, { timeoutMs: opts.timeoutMs })) }));
  const file = writeCsv(checked, opts.outputDir);
  const ng = checked.filter(r => r.judgement === 'NG');
  console.log(`Notion Patrol: ${checked.length} URLs checked/extracted, OK=${checked.length - ng.length}, NG=${ng.length}`);
  for (const r of ng) console.log(`NG ${r.statusCode} ${r.url} (${r.pageTitle}) ${r.context}`);
  console.log(`CSV: ${file}`);
  return 0;
}

module.exports = { DEFAULT_ROOTS, normalizeNotionId, parseArgs, isExternalUrl, extractUrlsFromBlock, checkUrl, writeCsv, mapLimit, NotionClient, traverseNotion, main, help };
if (require.main === module) main().catch(e => { console.error(e.message); process.exit(1); });
