import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const bridgeSource = readFileSync(path.join(__dirname, 'bridge.js'), 'utf8');

test('ffmpeg audio conversion uses argument-array execution', () => {
  assert.match(bridgeSource, /import\s+\{\s*execFileSync\s*\}\s+from\s+['"]child_process['"]/);
  assert.doesNotMatch(bridgeSource, /import\s+\{\s*execSync\s*\}\s+from\s+['"]child_process['"]/);
  assert.doesNotMatch(bridgeSource, /execSync\(\s*`ffmpeg/);
  assert.match(bridgeSource, /execFileSync\(\s*['"]ffmpeg['"],\s*\[/);
});
