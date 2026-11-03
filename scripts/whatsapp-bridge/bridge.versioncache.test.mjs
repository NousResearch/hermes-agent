/**
 * Unit tests for the on-disk WhatsApp version cache.
 *
 * Regression tests for the 2026-08-17 GitHub outage trap: with
 * raw.githubusercontent.com unreachable, fetchLatestBaileysVersion() times
 * out, the resolver hands back the Baileys library default, and WhatsApp
 * rejects the handshake with stream error 405 in an endless reconnect loop.
 * The in-memory fallback cannot help a bridge that restarted during the
 * outage — it never had a successful fetch in this process. Persisting the
 * last successfully fetched version gives a cold start a known-good tier.
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const roots = [];
function tempDir() {
  const dir = mkdtempSync(path.join(tmpdir(), 'wa-version-cache-'));
  roots.push(dir);
  return dir;
}

import { createVersionResolver } from './bridge_helpers.js';

// A successful fetch writes the version through to disk, so the NEXT process
// to start has something better than the library default to fall back on.
{
  const cacheFile = path.join(tempDir(), 'wa-version-cache.json');
  const resolveVersion = createVersionResolver(
    async () => ({ version: [2, 3000, 1023223821] }),
    { log: () => {}, cacheFile },
  );

  assert.deepEqual(await resolveVersion(), [2, 3000, 1023223821]);

  const persisted = JSON.parse(readFileSync(cacheFile, 'utf8'));
  assert.deepEqual(persisted.version, [2, 3000, 1023223821]);
  assert.ok(!Number.isNaN(Date.parse(persisted.fetched_at)), 'fetched_at is an ISO timestamp');
}

// The outage scenario end to end: a fresh resolver (no in-memory version,
// exactly like a bridge restarted mid-outage) reads the last known-good
// version off disk instead of falling through to the library default.
{
  const cacheFile = path.join(tempDir(), 'wa-version-cache.json');
  writeFileSync(cacheFile, JSON.stringify({
    version: [2, 3000, 1023223821],
    fetched_at: '2026-08-16T12:00:00.000Z',
  }));

  const logs = [];
  const resolveVersion = createVersionResolver(
    async () => { throw new Error('network down'); },
    { log: line => logs.push(line), cacheFile },
  );

  assert.deepEqual(await resolveVersion(), [2, 3000, 1023223821]);
  assert.equal(logs.length, 1);
  assert.match(logs[0], /network down/);
  assert.match(logs[0], /from disk/);
  assert.match(logs[0], /2\.3000\.1023223821/);
}

// Once the disk tier has been used it is held in memory, so a second failing
// call does not re-read the file — and a later success replaces both tiers.
{
  const cacheFile = path.join(tempDir(), 'wa-version-cache.json');
  writeFileSync(cacheFile, JSON.stringify({ version: [2, 3000, 1], fetched_at: 'x' }));

  let reads = 0;
  const resolveVersion = createVersionResolver(
    async () => { reads += 1; throw new Error('still down'); },
    { log: () => {}, cacheFile },
  );

  assert.deepEqual(await resolveVersion(), [2, 3000, 1]);
  // Overwrite the file: an in-memory hit must win, so this is never observed.
  writeFileSync(cacheFile, JSON.stringify({ version: [9, 9, 9], fetched_at: 'x' }));
  assert.deepEqual(await resolveVersion(), [2, 3000, 1]);
  assert.equal(reads, 2);
}

// A fresh fetch outranks the disk cache and rewrites it.
{
  const cacheFile = path.join(tempDir(), 'wa-version-cache.json');
  writeFileSync(cacheFile, JSON.stringify({ version: [2, 3000, 1], fetched_at: 'x' }));

  const resolveVersion = createVersionResolver(
    async () => ({ version: [2, 3000, 2] }),
    { log: () => {}, cacheFile },
  );

  assert.deepEqual(await resolveVersion(), [2, 3000, 2]);
  assert.deepEqual(JSON.parse(readFileSync(cacheFile, 'utf8')).version, [2, 3000, 2]);
}

// Garbage on disk is treated as absent, not trusted: unparseable JSON, a
// missing/ill-typed version field, and out-of-range tuple lengths all fall
// through to the library default rather than feeding junk to makeWASocket().
{
  const badPayloads = [
    'not json at all',
    '{}',
    JSON.stringify({ version: '2.3000.1' }),
    JSON.stringify({ version: [] }),
    JSON.stringify({ version: [2] }),
    JSON.stringify({ version: [2, 3000, 1, 4, 5] }),
    JSON.stringify({ version: [2, '3000', 1] }),
    JSON.stringify({ version: [2, 3000.5, 1] }),
    JSON.stringify({ version: [2, null, 1] }),
  ];

  for (const payload of badPayloads) {
    const cacheFile = path.join(tempDir(), 'wa-version-cache.json');
    writeFileSync(cacheFile, payload);

    const logs = [];
    const resolveVersion = createVersionResolver(
      async () => { throw new Error('network down'); },
      { log: line => logs.push(line), cacheFile },
    );

    assert.equal(await resolveVersion(), null, `payload should be rejected: ${payload}`);
    assert.match(logs[0], /library default/);
  }
}

// A cacheFile that can never be created (its parent is a regular file) must
// not break the connection path: the resolver still returns the fetched
// version, and it complains at most once no matter how often it reconnects.
{
  const blocker = path.join(tempDir(), 'not-a-directory');
  writeFileSync(blocker, 'i am a file');
  const cacheFile = path.join(blocker, 'wa-version-cache.json');

  const logs = [];
  const resolveVersion = createVersionResolver(
    async () => ({ version: [2, 3000, 7] }),
    { log: line => logs.push(line), cacheFile },
  );

  assert.deepEqual(await resolveVersion(), [2, 3000, 7]);
  assert.deepEqual(await resolveVersion(), [2, 3000, 7]);
  assert.equal(logs.length, 1, 'the write failure is reported once, not per reconnect');
  assert.match(logs[0], /persist/);
}

// An unreadable cache path on the failure side is equally inert.
{
  const blocker = path.join(tempDir(), 'not-a-directory');
  writeFileSync(blocker, 'i am a file');
  const cacheFile = path.join(blocker, 'wa-version-cache.json');

  const logs = [];
  const resolveVersion = createVersionResolver(
    async () => { throw new Error('network down'); },
    { log: line => logs.push(line), cacheFile },
  );

  assert.equal(await resolveVersion(), null);
  assert.match(logs[0], /library default/);
}

// A missing cache file is the ordinary first-run case, not an error.
{
  const cacheFile = path.join(tempDir(), 'nested', 'wa-version-cache.json');
  const logs = [];
  const resolveVersion = createVersionResolver(
    async () => { throw new Error('network down'); },
    { log: line => logs.push(line), cacheFile },
  );

  assert.equal(await resolveVersion(), null);
  assert.match(logs[0], /library default/);
}

// Without a cacheFile the resolver behaves exactly as it did before: memory
// only, no disk access, library default before the first success.
{
  const logs = [];
  let calls = 0;
  const resolveVersion = createVersionResolver(
    async () => {
      calls += 1;
      if (calls === 1) throw new Error('network down');
      return { version: [2, 3000, 3] };
    },
    { log: line => logs.push(line) },
  );

  assert.equal(await resolveVersion(), null);
  assert.match(logs[0], /library default/);
  assert.deepEqual(await resolveVersion(), [2, 3000, 3]);
}

// The parent directory of the cache file is created on demand, so a first
// successful fetch on a brand-new session dir still persists.
{
  const cacheFile = path.join(tempDir(), 'session', 'nested', 'wa-version-cache.json');
  const resolveVersion = createVersionResolver(
    async () => ({ version: [2, 3000, 11] }),
    { log: () => {}, cacheFile },
  );

  assert.deepEqual(await resolveVersion(), [2, 3000, 11]);
  assert.deepEqual(JSON.parse(readFileSync(cacheFile, 'utf8')).version, [2, 3000, 11]);
}

// The timeout tier persists too: a hung fetch that loses the race still
// leaves the previously persisted version in place for the next process.
{
  const dir = tempDir();
  const cacheFile = path.join(dir, 'wa-version-cache.json');
  mkdirSync(dir, { recursive: true });
  writeFileSync(cacheFile, JSON.stringify({ version: [2, 3000, 5], fetched_at: 'x' }));

  const logs = [];
  const resolveVersion = createVersionResolver(
    () => new Promise(() => {}),
    { timeoutMs: 20, log: line => logs.push(line), cacheFile },
  );

  assert.deepEqual(await resolveVersion(), [2, 3000, 5]);
  assert.match(logs[0], /timed out/);
  assert.match(logs[0], /from disk/);
  assert.deepEqual(JSON.parse(readFileSync(cacheFile, 'utf8')).version, [2, 3000, 5]);
}

for (const dir of roots) {
  try { rmSync(dir, { recursive: true, force: true }); } catch {}
}

console.log('bridge.versioncache.test.mjs: all assertions passed');
