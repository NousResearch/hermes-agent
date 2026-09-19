/**
 * Unit tests for sessionIdentity(): the /health `session` field the
 * WhatsApp adapter compares against its own session path to tell its own
 * bridge from a foreign profile's bridge on the same port.
 *
 * bridge.js itself is NOT imported here (it starts a server at module
 * load); the helper must stay pure — see bridge.native.test.mjs.
 */

import { strict as assert } from 'node:assert';
import { mkdirSync, mkdtempSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { sessionIdentity } from './bridge_helpers.js';

// -- relative input becomes absolute ---------------------------------------
{
  const resolved = sessionIdentity('relative/session');
  assert.ok(path.isAbsolute(resolved), 'sessionIdentity must return an absolute path');
}

// -- `..` segments are normalized ------------------------------------------
{
  const resolved = sessionIdentity('a/../b');
  assert.strictEqual(resolved, path.resolve('b'));
}

// -- symlinks are NOT resolved (path.resolve, not realpath) -----------------
{
  const root = mkdtempSync(path.join(tmpdir(), 'wa-session-identity-'));
  const realDir = path.join(root, 'real');
  const linkDir = path.join(root, 'link');
  mkdirSync(realDir);
  symlinkSync(realDir, linkDir, 'dir');
  const resolved = sessionIdentity(linkDir);
  assert.ok(
    resolved.includes('link') && !resolved.includes('real'),
    `symlink must not be resolved, got: ${resolved}`
  );
}
