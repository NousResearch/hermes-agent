const assert = require('assert');
const { summarizePatch, stripDiffPath, detectTestCommand } = require('../src/extensionUtils');

const deletionOnly = `diff --git a/old.txt b/old.txt\ndeleted file mode 100644\n--- a/old.txt\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-one\n-two\n`;
const deletionSummary = summarizePatch(deletionOnly);
assert.strictEqual(deletionSummary.files.length, 1);
assert.strictEqual(deletionSummary.files[0].path, 'old.txt');
assert.strictEqual(deletionSummary.deletions, 2);
assert.strictEqual(deletionSummary.additions, 0);
assert.strictEqual(deletionSummary.hunks, 1);

const mixed = `diff --git a/src/a.js b/src/a.js\n--- a/src/a.js\n+++ b/src/a.js\n@@ -1 +1,2 @@\n-old\n+new\n+line\ndiff --git a/src/b.js b/src/b.js\n--- a/src/b.js\n+++ b/src/b.js\n@@ -1 +1 @@\n keep\n`;
const mixedSummary = summarizePatch(mixed);
assert.strictEqual(mixedSummary.files.length, 2);
assert.strictEqual(mixedSummary.additions, 2);
assert.strictEqual(mixedSummary.deletions, 1);
assert.strictEqual(mixedSummary.hunks, 2);
assert.strictEqual(stripDiffPath('b/foo/bar.js\t2026'), 'foo/bar.js');

async function main() {
  const fakeFs = (names) => ({
    async stat(file) {
      const base = file.split(/[\\/]/).pop();
      if (!names.has(base)) throw Object.assign(new Error('missing'), { code: 'ENOENT' });
      return { isFile: () => true };
    }
  });
  assert.strictEqual(await detectTestCommand('/repo', fakeFs(new Set(['package.json']))), 'npm test');
  assert.strictEqual(await detectTestCommand('/repo', fakeFs(new Set(['pyproject.toml']))), 'python -m pytest');
  assert.strictEqual(await detectTestCommand('/repo', fakeFs(new Set(['Package.swift']))), 'swift test');
  assert.strictEqual(await detectTestCommand('/repo', fakeFs(new Set(['go.mod']))), 'go test ./...');
  assert.strictEqual(await detectTestCommand('/repo', fakeFs(new Set(['Cargo.toml']))), 'cargo test');
  assert.strictEqual(await detectTestCommand('/repo', fakeFs(new Set(['Makefile']))), 'make test');
  assert.match(await detectTestCommand('/repo', fakeFs(new Set())), /No standard test command detected/);
  console.log('extensionUtils tests ok');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
