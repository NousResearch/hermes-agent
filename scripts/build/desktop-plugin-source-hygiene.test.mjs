import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync, spawnSync } from 'node:child_process';
import { readdirSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));
const plugins = 'apps/desktop/src/plugins';
const git = (...args) => execFileSync('git', ['-C', root, ...args], { encoding: 'utf8' });
const ignored = path => spawnSync('git', ['-C', root, 'check-ignore', '--no-index', '-q', path]).status;

test('generated Desktop entries stay ignored while source-only entries remain trackable', () => {
  const tracked = new Set(git('ls-files', `${plugins}/*/plugin.js`).trim().split('\n').filter(Boolean));
  const generated = new Set(readdirSync(resolve(root, plugins), { withFileTypes: true })
    .filter(dir => dir.isDirectory() && existsSync(resolve(root, plugins, dir.name, 'plugin.tsx')))
    .map(dir => `${plugins}/${dir.name}/plugin.js`));
  for (const path of generated) {
    assert.equal(tracked.has(path), false, `${path} must not shadow its TSX source`);
    assert.equal(ignored(path), 0, `${path} must be ignored`);
  }
  for (const path of tracked) assert.equal(ignored(path), 1, `${path} is an authored source`);
});
