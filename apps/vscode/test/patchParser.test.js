const assert = require('assert');
const { extractUnifiedDiff, parsePatch, applyPatchToContent, safeJoin } = require('../src/patchParser');

const response = `Here is the patch:\n\n\`\`\`diff\ndiff --git a/foo.txt b/foo.txt\n--- a/foo.txt\n+++ b/foo.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n\`\`\`\n`;
const diff = extractUnifiedDiff(response);
assert(diff.includes('diff --git a/foo.txt b/foo.txt'));
const files = parsePatch(diff);
assert.strictEqual(files.length, 1);
assert.strictEqual(files[0].newPath, 'foo.txt');
assert.strictEqual(applyPatchToContent('one\ntwo\nthree\n', files[0]), 'one\nTWO\nthree\n');
assert.throws(() => safeJoin('/tmp/root', '../escape'));
assert.strictEqual(safeJoin('/tmp/root', 'a/b').startsWith('/tmp/root/'), true);
console.log('patchParser tests ok');
