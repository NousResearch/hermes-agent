#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

const EXPECTED_HEADERS = [
  '序号',
  '关键帧',
  '时间',
  '镜头',
  '运镜',
  '转场',
  '动作',
  '情绪/细节',
  '台词',
  '旁白',
  '状态/音效',
  '英文',
];

function usage() {
  console.error('Usage: node convert-keyframe-table.mjs <input.md> [--format csv|json] [--out output]');
  process.exit(2);
}

function parseArgs(argv) {
  const args = { input: null, format: 'csv', out: null };
  for (let i = 2; i < argv.length; i += 1) {
    const value = argv[i];
    if (!args.input && !value.startsWith('--')) {
      args.input = value;
    } else if (value === '--format') {
      args.format = argv[++i];
    } else if (value === '--out') {
      args.out = argv[++i];
    } else {
      usage();
    }
  }
  if (!args.input || !['csv', 'json'].includes(args.format)) usage();
  return args;
}

function splitMarkdownRow(line) {
  const trimmed = line.trim();
  if (!trimmed.startsWith('|') || !trimmed.endsWith('|')) return null;
  const cells = [];
  let current = '';
  let escaped = false;

  for (const char of trimmed.slice(1, -1)) {
    if (escaped) {
      current += char === '|' ? '|' : `\\${char}`;
      escaped = false;
    } else if (char === '\\') {
      escaped = true;
    } else if (char === '|') {
      cells.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }

  if (escaped) current += '\\';
  cells.push(current.trim());
  return cells;
}

function isSeparatorRow(cells) {
  return cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function sameHeaders(cells) {
  return cells.length === EXPECTED_HEADERS.length && cells.every((cell, index) => cell === EXPECTED_HEADERS[index]);
}

function extractTables(markdown) {
  const rows = [];
  const lines = markdown.split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const header = splitMarkdownRow(lines[i]);
    if (!header || !sameHeaders(header)) continue;

    const separator = splitMarkdownRow(lines[i + 1] ?? '');
    if (!separator || !isSeparatorRow(separator)) continue;

    i += 2;
    while (i < lines.length) {
      const cells = splitMarkdownRow(lines[i]);
      if (!cells || cells.length !== EXPECTED_HEADERS.length) {
        i -= 1;
        break;
      }
      const row = Object.fromEntries(EXPECTED_HEADERS.map((name, index) => [name, cells[index]]));
      rows.push(row);
      i += 1;
    }
  }
  return rows;
}

function csvEscape(value) {
  const text = String(value ?? '');
  if (/[",\r\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function toCsv(rows) {
  const lines = [EXPECTED_HEADERS.map(csvEscape).join(',')];
  for (const row of rows) {
    lines.push(EXPECTED_HEADERS.map((header) => csvEscape(row[header])).join(','));
  }
  return `${lines.join('\n')}\n`;
}

const args = parseArgs(process.argv);
const inputPath = path.resolve(args.input);
const markdown = fs.readFileSync(inputPath, 'utf8').replace(/^\uFEFF/, '');
const rows = extractTables(markdown);

if (rows.length === 0) {
  console.error(`Warning: no 12-column keyframe table found in "${args.input}". Check that the header matches exactly:\n  | ${EXPECTED_HEADERS.join(' | ')} |`);
  process.exit(1);
}

const output = args.format === 'json'
  ? `${JSON.stringify(rows, null, 2)}\n`
  : toCsv(rows);

if (args.out) {
  fs.writeFileSync(path.resolve(args.out), output, 'utf8');
} else {
  process.stdout.write(output);
}
