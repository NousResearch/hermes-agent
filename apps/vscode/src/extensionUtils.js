const fsp = require('fs/promises');
const path = require('path');

function summarizePatch(diffText) {
  const files = [];
  let current;
  const finish = () => {
    if (current) files.push(current);
    current = undefined;
  };
  for (const line of String(diffText || '').replace(/\r\n/g, '\n').split('\n')) {
    if (line.startsWith('diff --git ')) {
      finish();
      const parts = line.split(/\s+/);
      current = { path: stripDiffPath(parts[3] || parts[2]), additions: 0, deletions: 0, hunks: 0 };
      continue;
    }
    if (line.startsWith('--- ') && !current) {
      current = { path: stripDiffPath(line.slice(4)), additions: 0, deletions: 0, hunks: 0 };
      continue;
    }
    if (line.startsWith('+++ ') && current) {
      const newPath = stripDiffPath(line.slice(4));
      if (newPath !== '/dev/null') current.path = newPath;
      continue;
    }
    if (!current) continue;
    if (line.startsWith('@@ ')) current.hunks += 1;
    else if (line.startsWith('+') && !line.startsWith('+++ ')) current.additions += 1;
    else if (line.startsWith('-') && !line.startsWith('--- ')) current.deletions += 1;
  }
  finish();
  const cleaned = files.filter((file) => file.path && (file.additions || file.deletions || file.hunks));
  return {
    files: cleaned,
    additions: cleaned.reduce((sum, file) => sum + file.additions, 0),
    deletions: cleaned.reduce((sum, file) => sum + file.deletions, 0),
    hunks: cleaned.reduce((sum, file) => sum + file.hunks, 0)
  };
}

function stripDiffPath(value) {
  if (!value) return value;
  const cleaned = value.trim().split('\t')[0].split(' ')[0];
  if (cleaned === '/dev/null') return cleaned;
  return cleaned.startsWith('a/') || cleaned.startsWith('b/') ? cleaned.slice(2) : cleaned;
}

async function detectTestCommand(cwd, fsApi = fsp) {
  const exists = async (name) => {
    try { await fsApi.stat(path.join(cwd, name)); return true; } catch { return false; }
  };
  if (await exists('package.json')) return 'npm test';
  if (await exists('pyproject.toml') || await exists('pytest.ini')) return 'python -m pytest';
  if (await exists('Package.swift')) return 'swift test';
  if (await exists('go.mod')) return 'go test ./...';
  if (await exists('Cargo.toml')) return 'cargo test';
  if (await exists('Makefile')) return 'make test';
  return 'echo "No standard test command detected. Edit this command and press Enter."';
}

module.exports = { summarizePatch, stripDiffPath, detectTestCommand };
