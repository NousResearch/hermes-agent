const assert = require('assert');
const {
  summarizePatch,
  stripDiffPath,
  detectTestCommand,
  pickPackageScript,
  packageScriptCommand,
  buildHermesPrompt,
  buildTerminalDebugPrompt,
  buildTestAnalysisPrompt,
  modeInstructions,
  truncateText
} = require('../src/extensionUtils');

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

assert.strictEqual(pickPackageScript({ test: 'vitest', check: 'tsc' }, 'example'), 'test');
assert.strictEqual(pickPackageScript({ check: 'tsc', lint: 'eslint .' }, 'example'), 'check');
assert.strictEqual(pickPackageScript({ 'package:check': 'node scripts/package-check.js' }, 'hermes-vscode'), 'package:check');
assert.strictEqual(packageScriptCommand('npm', 'test'), 'npm test');
assert.strictEqual(packageScriptCommand('npm', 'check'), 'npm run check');
assert.strictEqual(packageScriptCommand('pnpm', 'check'), 'pnpm run check');
assert.strictEqual(packageScriptCommand('yarn', 'lint'), 'yarn lint');
assert.strictEqual(packageScriptCommand('bun', 'test'), 'bun test');

const prompt = buildHermesPrompt('fix it', 'Active file: app.js', { mode: 'debug', wantsPatch: true });
assert.match(prompt, /VS Code\/Cursor extension/);
assert.match(prompt, /Mode: debug/);
assert.match(prompt, /unified diff/);
assert.match(prompt, /Active file: app\.js/);
assert.match(prompt, /## User request\n\nfix it/);
assert.match(modeInstructions('security'), /trust boundaries/);
assert.match(modeInstructions('unknown'), /general coding assistant/);

const truncated = truncateText('abcdefghij', 4);
assert.strictEqual(truncated, 'abcd\n\n...[truncated 6 chars]');
assert.strictEqual(truncateText('abc', 10), 'abc');
const debugPrompt = buildTerminalDebugPrompt('npm test', 'x'.repeat(20), 'failed with exit 1', 8);
assert.match(debugPrompt, /Command:/);
assert.match(debugPrompt, /npm test/);
assert.match(debugPrompt, /truncated 12 chars/);
const testPrompt = buildTestAnalysisPrompt('npm test', 'pass', 'passed', 100);
assert.match(testPrompt, /detected test command `npm test` passed/);
assert.match(testPrompt, /```text\npass/);

async function main() {
  const fakeFs = (files) => ({
    async stat(file) {
      if (!Object.prototype.hasOwnProperty.call(files, normalize(file))) {
        const base = file.split(/[\\/]/).pop();
        if (!Object.prototype.hasOwnProperty.call(files, base)) {
          throw Object.assign(new Error('missing'), { code: 'ENOENT' });
        }
      }
      return { isFile: () => true };
    },
    async readFile(file) {
      const key = normalize(file);
      const base = file.split(/[\\/]/).pop();
      if (Object.prototype.hasOwnProperty.call(files, key)) return files[key];
      if (Object.prototype.hasOwnProperty.call(files, base)) return files[base];
      throw Object.assign(new Error('missing'), { code: 'ENOENT' });
    }
  });

  assert.strictEqual(
    await detectTestCommand('/repo', fakeFs({ 'package.json': JSON.stringify({ scripts: { test: 'vitest' } }) })),
    'npm test'
  );
  assert.strictEqual(
    await detectTestCommand('/repo', fakeFs({ 'package.json': JSON.stringify({ scripts: { check: 'tsc --noEmit' } }), 'pnpm-lock.yaml': '' })),
    'pnpm run check'
  );
  assert.strictEqual(
    await detectTestCommand('/repo', fakeFs({ 'package.json': JSON.stringify({ scripts: { lint: 'eslint .' } }), 'yarn.lock': '' })),
    'yarn lint'
  );
  assert.strictEqual(
    await detectTestCommand('/repo', fakeFs({ 'package.json': JSON.stringify({ scripts: {} }), 'bun.lockb': '' })),
    'bun test'
  );
  assert.strictEqual(
    await detectTestCommand('/repo/apps/vscode', fakeFs({ 'package.json': JSON.stringify({ name: 'hermes-vscode', scripts: { test: 'node test.js', 'package:check': 'npm run test && node scripts/package-check.js' } }) })),
    'npm run package:check'
  );
  assert.strictEqual(
    await detectTestCommand('/repo', fakeFs({ 'apps/vscode/package.json': JSON.stringify({ scripts: { 'package:check': 'npm run test && node scripts/package-check.js' } }) })),
    'npm run package:check --workspace apps/vscode'
  );
  assert.strictEqual(await detectTestCommand('/repo', fakeFs({ 'pyproject.toml': '' })), 'python -m pytest');
  assert.strictEqual(await detectTestCommand('/repo', fakeFs({ 'pyproject.toml': '', 'uv.lock': '' })), 'uv run pytest');
  assert.strictEqual(await detectTestCommand('/repo', fakeFs({ 'pytest.ini': '' })), 'python -m pytest');
  assert.strictEqual(await detectTestCommand('/repo', fakeFs({ 'Package.swift': '' })), 'swift test');
  assert.strictEqual(await detectTestCommand('/repo', fakeFs({ 'go.mod': '' })), 'go test ./...');
  assert.strictEqual(await detectTestCommand('/repo', fakeFs({ 'Cargo.toml': '' })), 'cargo test');
  assert.strictEqual(await detectTestCommand('/repo', fakeFs({ 'Makefile': '' })), 'make test');
  assert.match(await detectTestCommand('/repo', fakeFs({})), /No standard test command detected/);
  console.log('extensionUtils tests ok');
}

function normalize(file) {
  return file.replace(/\\/g, '/').replace(/^\/repo\//, '').replace(/^\/repo$/, '');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
