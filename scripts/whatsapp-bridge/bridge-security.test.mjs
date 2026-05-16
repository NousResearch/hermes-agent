import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const bridgeSource = readFileSync(new URL('./bridge.js', import.meta.url), 'utf8');

test('ffmpeg conversion avoids shell command execution', () => {
  assert.match(bridgeSource, /import \{ execFileSync \} from 'child_process';/);
  assert.doesNotMatch(bridgeSource, /\bexecSync\b/);
  assert.match(
    bridgeSource,
    /execFileSync\(\s*'ffmpeg',\s*\[\s*'-y',\s*'-i',\s*filePath,\s*'-ar',\s*'48000',\s*'-ac',\s*'1',\s*'-c:a',\s*'libopus',\s*tmpPath\s*\]/s
  );
});
