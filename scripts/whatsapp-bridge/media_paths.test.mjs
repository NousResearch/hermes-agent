import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { resolveAllowedMediaPath } from './media_paths.js';

function makeCache(t) {
  const tempDir = fs.mkdtempSync(path.join(tmpdir(), 'hermes-wa-media-paths-'));
  t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }));
  const cache = path.join(tempDir, 'cache');
  fs.mkdirSync(cache);
  return { tempDir, cache };
}

test('rejects a cache symlink escaping to a sibling before statting its target', t => {
  const { tempDir, cache } = makeCache(t);
  const outside = path.join(tempDir, 'cache-outside');
  fs.mkdirSync(outside);
  const outsideFile = path.join(outside, 'private.txt');
  fs.writeFileSync(outsideFile, 'outside');
  const link = path.join(cache, 'escape');
  // Junctions exercise real Windows path resolution without symlink privileges.
  fs.symlinkSync(outside, link, process.platform === 'win32' ? 'junction' : 'dir');
  const candidate = path.join(link, 'private.txt');
  assert.equal(fs.realpathSync(candidate), fs.realpathSync(outsideFile));

  const stat = t.mock.method(fs, 'statSync');
  assert.equal(resolveAllowedMediaPath(candidate, [cache], 100), null);
  // Returning null alone also passes if the forbidden stat happened first.
  assert.equal(stat.mock.callCount(), 0);
});

test('accepts a regular cache file while preserving file and size restrictions', t => {
  const { cache } = makeCache(t);
  const media = path.join(cache, 'image.png');
  const content = Buffer.from('cached media');
  fs.writeFileSync(media, content);

  const stat = t.mock.method(fs, 'statSync');
  assert.equal(resolveAllowedMediaPath(media, [cache], content.length), fs.realpathSync(media));
  assert.equal(stat.mock.callCount(), 1);
  assert.equal(stat.mock.calls[0].arguments[0], fs.realpathSync(media));
  assert.equal(resolveAllowedMediaPath(media, [cache], content.length - 1), null);
  const directory = path.join(cache, 'nested');
  fs.mkdirSync(directory);
  assert.equal(resolveAllowedMediaPath(directory, [cache], content.length), null);
  assert.equal(resolveAllowedMediaPath(cache, [cache], content.length), null);
  assert.equal(resolveAllowedMediaPath(path.join(cache, 'missing.png'), [cache], content.length), null);
  assert.equal(resolveAllowedMediaPath('image.png', [cache], content.length), null);
});
