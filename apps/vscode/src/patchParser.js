const path = require('path');

function stripPrefix(filePath) {
  if (!filePath) return filePath;
  let value = filePath.trim();
  if (value === '/dev/null') return value;
  value = value.split('\t')[0].split(' ')[0];
  if (value.startsWith('a/') || value.startsWith('b/')) value = value.slice(2);
  return value;
}

function extractUnifiedDiff(text) {
  if (!text) return '';
  const fenced = [...text.matchAll(/```(?:diff|patch)?\s*\n([\s\S]*?)```/gi)]
    .map((m) => m[1])
    .find((body) => /^diff --git |^---\s+/m.test(body));
  if (fenced) return fenced.trimEnd() + '\n';
  const start = text.search(/^diff --git |^---\s+/m);
  return start >= 0 ? text.slice(start).trimEnd() + '\n' : '';
}

function parsePatch(diffText) {
  const lines = diffText.replace(/\r\n/g, '\n').split('\n');
  const files = [];
  let current = null;
  let hunk = null;

  for (const line of lines) {
    if (line.startsWith('diff --git ')) {
      if (current) files.push(current);
      const parts = line.split(/\s+/);
      current = { oldPath: stripPrefix(parts[2]), newPath: stripPrefix(parts[3]), hunks: [], rawHeader: [line] };
      hunk = null;
      continue;
    }
    if (line.startsWith('--- ')) {
      if (!current) current = { oldPath: stripPrefix(line.slice(4)), newPath: undefined, hunks: [], rawHeader: [] };
      current.oldPath = stripPrefix(line.slice(4));
      current.rawHeader.push(line);
      hunk = null;
      continue;
    }
    if (line.startsWith('+++ ')) {
      if (!current) current = { oldPath: undefined, newPath: stripPrefix(line.slice(4)), hunks: [], rawHeader: [] };
      current.newPath = stripPrefix(line.slice(4));
      current.rawHeader.push(line);
      hunk = null;
      continue;
    }
    const match = line.match(/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/);
    if (match) {
      if (!current) continue;
      hunk = {
        oldStart: Number(match[1]),
        oldCount: match[2] ? Number(match[2]) : 1,
        newStart: Number(match[3]),
        newCount: match[4] ? Number(match[4]) : 1,
        lines: [line]
      };
      current.hunks.push(hunk);
      continue;
    }
    if (hunk) hunk.lines.push(line);
    else if (current) current.rawHeader.push(line);
  }
  if (current) files.push(current);
  return files.filter((file) => file.newPath && file.newPath !== '/dev/null' && file.hunks.length > 0);
}

function applyPatchToContent(original, filePatch) {
  const source = original.replace(/\r\n/g, '\n').split('\n');
  if (source.length && source[source.length - 1] === '') source.pop();
  const output = [];
  let cursor = 0;

  for (const hunk of filePatch.hunks) {
    const targetIndex = Math.max(0, hunk.oldStart - 1);
    while (cursor < targetIndex && cursor < source.length) output.push(source[cursor++]);

    for (const line of hunk.lines.slice(1)) {
      if (line.startsWith('\\ No newline at end of file')) continue;
      const marker = line[0];
      const value = line.slice(1);
      if (marker === ' ') {
        output.push(source[cursor] !== undefined ? source[cursor] : value);
        cursor += 1;
      } else if (marker === '-') {
        cursor += 1;
      } else if (marker === '+') {
        output.push(value);
      }
    }
  }
  while (cursor < source.length) output.push(source[cursor++]);
  return output.join('\n') + (original.endsWith('\n') ? '\n' : '');
}

function safeJoin(root, relativePath) {
  const resolved = path.resolve(root, relativePath);
  const rootResolved = path.resolve(root);
  if (resolved !== rootResolved && !resolved.startsWith(rootResolved + path.sep)) {
    throw new Error(`Patch path escapes workspace: ${relativePath}`);
  }
  return resolved;
}

module.exports = { extractUnifiedDiff, parsePatch, applyPatchToContent, safeJoin };
