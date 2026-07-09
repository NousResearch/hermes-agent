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
  console.error('Usage: node validate-keyframe-table.mjs <input.md>');
  process.exit(2);
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

function parseTimeRange(value) {
  const match = String(value ?? '').trim().match(/^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)s?$/);
  if (!match) return null;
  return { start: Number(match[1]), end: Number(match[2]) };
}

function extractTables(markdown) {
  const lines = markdown.split(/\r?\n/);
  const tables = [];
  const tableErrors = [];

  for (let i = 0; i < lines.length; i += 1) {
    const header = splitMarkdownRow(lines[i]);
    if (!header || !sameHeaders(header)) continue;

    const separator = splitMarkdownRow(lines[i + 1] ?? '');
    if (!separator || !isSeparatorRow(separator)) {
      tableErrors.push(`Line ${i + 2}: 12-column keyframe table header must be followed by a Markdown separator row.`);
      continue;
    }

    const rows = [];
    i += 2;
    while (i < lines.length) {
      const cells = splitMarkdownRow(lines[i]);
      if (!cells) {
        i -= 1;
        break;
      }
      if (isSeparatorRow(cells)) {
        i += 1;
        continue;
      }
      if (cells.length !== EXPECTED_HEADERS.length) {
        tableErrors.push(`Line ${i + 1}: expected ${EXPECTED_HEADERS.length} cells, got ${cells.length}. Escape literal pipes as \\| inside cells.`);
        i -= 1;
        break;
      }
      rows.push({
        line: i + 1,
        data: Object.fromEntries(EXPECTED_HEADERS.map((name, index) => [name, cells[index]])),
      });
      i += 1;
    }

    tables.push({ headerLine: i + 1, rows });
  }

  return { tables, tableErrors };
}

function splitGroups(rows) {
  const groups = [];
  let group = [];

  for (const row of rows) {
    if (row.data['序号'] === '01' && group.length > 0) {
      groups.push(group);
      group = [];
    }
    group.push(row);
  }

  if (group.length > 0) groups.push(group);
  return groups;
}

function validateRows(tables, tableErrors) {
  const errors = [...tableErrors];
  const warnings = [];
  let rowCount = 0;

  for (const table of tables) {
    rowCount += table.rows.length;

    for (const row of table.rows) {
      const data = row.data;
      const time = parseTimeRange(data['时间']);
      if (!time) {
        errors.push(`Line ${row.line}: invalid 时间 "${data['时间']}". Use a range like 9.5-11.2s.`);
      } else if (time.start >= time.end) {
        errors.push(`Line ${row.line}: 时间 start must be earlier than end.`);
      }

      if (/^(旁白|状态|音效|SFX|Ambience)[:：]/i.test(data['台词'])) {
        errors.push(`Line ${row.line}: 台词 column contains non-dialogue prefix. Move narration or sound into 旁白 or 状态/音效.`);
      }
      if (/^(台词|状态|音效|SFX|Ambience)[:：]/i.test(data['旁白'])) {
        errors.push(`Line ${row.line}: 旁白 column contains dialogue or sound prefix. Keep channels separated.`);
      }
      if (/^(台词|旁白)[:：]/.test(data['状态/音效'])) {
        errors.push(`Line ${row.line}: 状态/音效 column contains speech prefix. Move speech into 台词 or 旁白.`);
      }
      if (/(台词|旁白|状态|音效)[:：]/.test(data['动作'])) {
        warnings.push(`Line ${row.line}: 动作 contains audio/text labels; keep 动作 purely visual when possible.`);
      }

      if ((data['台词'] || data['旁白']) && !data['英文']) {
        warnings.push(`Line ${row.line}: 台词 or 旁白 exists but 英文 is empty.`);
      }
      if (data['状态/音效'] && !data['英文']) {
        warnings.push(`Line ${row.line}: 状态/音效 exists but 英文 is empty. Use [SFX: ...], [Ambience: ...], or [N/A].`);
      }
    }

    for (const group of splitGroups(table.rows)) {
      if (group.length !== 9) {
        warnings.push(`Line ${group[0].line}: this group has ${group.length} rows; a full 15s 3x3 group normally has 9 rows.`);
      }

      const starred = group.filter((row) => row.data['关键帧'].includes('⭐')).length;
      if (group.length === 9 && (starred < 1 || starred > 4)) {
        warnings.push(`Line ${group[0].line}: this 9-row group has ${starred} starred keyframes; 2-3 is usually best.`);
      }

      let previousEnd = null;
      for (const row of group) {
        const time = parseTimeRange(row.data['时间']);
        if (!time) continue;
        if (previousEnd !== null && time.start < previousEnd - 0.01) {
          warnings.push(`Line ${row.line}: 时间 overlaps or moves backward inside the 3x3 group.`);
        }
        previousEnd = time.end;
      }
    }
  }

  return { errors, warnings, rowCount };
}

const input = process.argv[2];
if (!input) usage();

const inputPath = path.resolve(input);
const markdown = fs.readFileSync(inputPath, 'utf8').replace(/^\uFEFF/, '');
const { tables, tableErrors } = extractTables(markdown);

if (tables.length === 0) {
  console.error(`Error: no 12-column keyframe table found in "${input}".`);
  console.error(`Expected header:\n| ${EXPECTED_HEADERS.join(' | ')} |`);
  process.exit(1);
}

const { errors, warnings, rowCount } = validateRows(tables, tableErrors);
for (const warning of warnings) console.error(`Warning: ${warning}`);
for (const error of errors) console.error(`Error: ${error}`);

console.error(`Validated ${tables.length} table(s), ${rowCount} row(s): ${errors.length} error(s), ${warnings.length} warning(s).`);
process.exit(errors.length > 0 ? 1 : 0);
