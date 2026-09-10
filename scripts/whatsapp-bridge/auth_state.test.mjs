/**
 * Auth-state durability tests.
 *
 * These do NOT mock the filesystem. A mocked ENOSPC proves only that the mock
 * was configured; the outage we are preventing was a real kernel write
 * failure, so the tests induce a real one with `ulimit -f` (RLIMIT_FSIZE) and
 * let the kernel fail the write mid-flight. The stock Baileys writer loses the
 * file under that condition; the atomic writer does not.
 *
 * Run: node --test auth_state.test.mjs
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, statSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { writeFileAtomic, useAtomicMultiFileAuthState } from './auth_state.js';

const HERE = import.meta.dirname;
// RLIMIT_FSIZE via `ulimit -f` is POSIX-only. The bridge itself is
// cross-platform, so skip (don't fail) the harness on Windows.
const POSIX = process.platform !== 'win32';

function freshDir() {
  return mkdtempSync(join(tmpdir(), 'authstate-'));
}

/**
 * Run `script` in a child node process under RLIMIT_FSIZE, so a large write
 * genuinely fails partway through (EFBIG) exactly as ENOSPC did in production.
 * XFSZ is trapped so the child survives to report what happened on disk.
 */
function runUnderFileSizeLimit(script, blocks = 1) {
  const file = join(freshDir(), 'probe.mjs');
  writeFileSync(file, script);
  const out = execFileSync(
    'bash',
    ['-c', `trap '' XFSZ; ulimit -f ${blocks}; node ${file}`],
    { cwd: HERE, encoding: 'utf8' },
  );
  return JSON.parse(out.trim().split('\n').pop());
}

// ── The bug, reproduced against the stock implementation ─────────────────

test('BASELINE: a plain writeFile destroys existing creds when the write fails', { skip: !POSIX }, () => {
  const dir = freshDir();
  const target = join(dir, 'creds.json');
  const result = runUnderFileSizeLimit(`
    import { writeFile } from 'fs/promises';
    import { writeFileSync, statSync, readFileSync } from 'fs';
    const p = ${JSON.stringify(target)};
    writeFileSync(p, JSON.stringify({ me: 'real-identity' }));
    const before = readFileSync(p, 'utf8');
    let code = null;
    try { await writeFile(p, JSON.stringify({ pad: 'y'.repeat(8000) })); }
    catch (e) { code = e.code; }
    let parsed = true;
    try { JSON.parse(readFileSync(p, 'utf8')); } catch { parsed = false; }
    console.log(JSON.stringify({ code, before, stillValid: parsed, size: statSync(p).size }));
  `);

  assert.ok(result.code, 'expected the kernel to fail the write');
  // This is the outage: the pre-existing, perfectly good credentials are gone.
  assert.equal(result.stillValid, false,
    'baseline was expected to corrupt the file — if this fails, the harness no longer reproduces the bug');
  rmSync(dir, { recursive: true, force: true });
});

// ── The guarantee ────────────────────────────────────────────────────────

test('atomic write leaves existing creds intact when the write fails', { skip: !POSIX }, () => {
  const dir = freshDir();
  const target = join(dir, 'creds.json');
  const result = runUnderFileSizeLimit(`
    import { writeFileAtomic } from ${JSON.stringify(join(HERE, 'auth_state.js'))};
    import { writeFileSync, statSync, readFileSync, readdirSync } from 'fs';
    const p = ${JSON.stringify(target)};
    const original = JSON.stringify({ me: 'real-identity' });
    writeFileSync(p, original);
    let code = null;
    try { await writeFileAtomic(p, JSON.stringify({ pad: 'y'.repeat(8000) })); }
    catch (e) { code = e.code || e.message; }
    console.log(JSON.stringify({
      code,
      content: readFileSync(p, 'utf8'),
      intact: readFileSync(p, 'utf8') === original,
      leftovers: readdirSync(${JSON.stringify(dir)}).filter(f => f.includes('.tmp-')),
    }));
  `);

  assert.ok(result.code, 'the write must still fail — we are not hiding the error');
  assert.equal(result.intact, true, 'existing credentials must survive a failed write');
  assert.deepEqual(result.leftovers, [], 'no temp files may be left behind');
  rmSync(dir, { recursive: true, force: true });
});

test('atomic write is a real replacement on the happy path', async () => {
  const dir = freshDir();
  const p = join(dir, 'file.json');
  await writeFileAtomic(p, '{"a":1}');
  assert.equal(readFileSync(p, 'utf8'), '{"a":1}');
  await writeFileAtomic(p, '{"a":2}');
  assert.equal(readFileSync(p, 'utf8'), '{"a":2}');
  assert.deepEqual(readdirSync(dir), ['file.json']);
  rmSync(dir, { recursive: true, force: true });
});

// ── Corruption must be loud, not silently re-paired ──────────────────────

test('a 0-byte creds.json throws instead of minting a new identity', async () => {
  const dir = freshDir();
  writeFileSync(join(dir, 'creds.json'), '');   // exactly what the outage left
  await assert.rejects(
    () => useAtomicMultiFileAuthState(dir),
    /empty or corrupt/,
    'an emptied creds file must fail loudly, not silently re-pair',
  );
  rmSync(dir, { recursive: true, force: true });
});

test('a truncated/garbage creds.json also throws', async () => {
  const dir = freshDir();
  writeFileSync(join(dir, 'creds.json'), '{"noiseKey":{"private":{"typ');
  await assert.rejects(() => useAtomicMultiFileAuthState(dir), /empty or corrupt/);
  rmSync(dir, { recursive: true, force: true });
});

test('an ABSENT creds.json is a normal first run and mints fresh creds', async () => {
  const dir = freshDir();
  const { state, saveCreds } = await useAtomicMultiFileAuthState(dir);
  assert.ok(state.creds, 'first run must produce usable creds');
  await saveCreds();
  assert.ok(statSync(join(dir, 'creds.json')).size > 0);
  rmSync(dir, { recursive: true, force: true });
});

// ── Compatibility with sessions written by stock Baileys ─────────────────

test('reads a session directory written by the stock implementation', async () => {
  const dir = freshDir();
  const { useMultiFileAuthState } = await import('@whiskeysockets/baileys');
  const stock = await useMultiFileAuthState(dir);
  await stock.saveCreds();
  await stock.state.keys.set({ 'pre-key': { '1': { public: Buffer.from([1, 2, 3]) } } });

  const ours = await useAtomicMultiFileAuthState(dir);
  assert.equal(ours.state.creds.registrationId, stock.state.creds.registrationId,
    'must load the identity the stock writer persisted');
  const keys = await ours.state.keys.get('pre-key', ['1']);
  assert.ok(keys['1'], 'must read keys written by the stock implementation');
  rmSync(dir, { recursive: true, force: true });
});

test('key round-trip survives a reopen (Buffer encoding preserved)', async () => {
  const dir = freshDir();
  const first = await useAtomicMultiFileAuthState(dir);
  await first.saveCreds();
  await first.state.keys.set({ 'pre-key': { '7': { public: Buffer.from([9, 8, 7]) } } });

  const second = await useAtomicMultiFileAuthState(dir);
  const got = await second.state.keys.get('pre-key', ['7']);
  assert.ok(Buffer.isBuffer(got['7'].public), 'BufferJSON round-trip must yield a Buffer');
  assert.deepEqual([...got['7'].public], [9, 8, 7]);
  rmSync(dir, { recursive: true, force: true });
});

test('setting a null key deletes its file', async () => {
  const dir = freshDir();
  const auth = await useAtomicMultiFileAuthState(dir);
  await auth.state.keys.set({ 'pre-key': { '3': { public: Buffer.from([1]) } } });
  assert.ok(readdirSync(dir).includes('pre-key-3.json'));
  await auth.state.keys.set({ 'pre-key': { '3': null } });
  assert.ok(!readdirSync(dir).includes('pre-key-3.json'));
  rmSync(dir, { recursive: true, force: true });
});

test('stale temp files from a previous crash are swept on open', async () => {
  const dir = freshDir();
  writeFileSync(join(dir, 'creds.json.tmp-999-deadbeef'), 'junk');
  await useAtomicMultiFileAuthState(dir);
  assert.deepEqual(readdirSync(dir).filter((f) => f.includes('.tmp-')), []);
  rmSync(dir, { recursive: true, force: true });
});

test('concurrent saves on one file serialize without interleaving', async () => {
  const dir = freshDir();
  const auth = await useAtomicMultiFileAuthState(dir);
  await Promise.all(Array.from({ length: 25 }, () => auth.saveCreds()));
  const parsed = JSON.parse(readFileSync(join(dir, 'creds.json'), 'utf8'));
  assert.ok(parsed.registrationId !== undefined, 'file must remain valid JSON under concurrency');
  assert.deepEqual(readdirSync(dir).filter((f) => f.includes('.tmp-')), []);
  rmSync(dir, { recursive: true, force: true });
});

test('sweep removes only real temp files, not keys whose id contains .tmp-', async () => {
  const dir = freshDir();
  // A sender-key id can legitimately contain this substring; a naive
  // `name.includes('.tmp-')` sweep would delete a live key at startup.
  const lookalike = join(dir, 'sender-key-1234.tmp-abc@s.whatsapp.net.json');
  writeFileSync(lookalike, JSON.stringify({ keep: true }));
  writeFileSync(join(dir, 'creds.json.tmp-999-deadbeef'), 'junk');

  await useAtomicMultiFileAuthState(dir);

  assert.ok(readdirSync(dir).includes('sender-key-1234.tmp-abc@s.whatsapp.net.json'),
    'a real key file must survive the sweep');
  assert.deepEqual(readdirSync(dir).filter((f) => /\.tmp-\d+-[0-9a-f]+$/.test(f)), [],
    'genuine temp files must still be swept');
  rmSync(dir, { recursive: true, force: true });
});

test('a failed saveCreds rejects (caller must catch) and leaves creds intact', async () => {
  const dir = freshDir();
  const { saveCreds } = await useAtomicMultiFileAuthState(dir);
  await saveCreds();
  const original = readFileSync(join(dir, 'creds.json'), 'utf8');

  // Make the directory read-only so the temp-file create fails.
  const { chmodSync } = await import('node:fs');
  chmodSync(dir, 0o500);
  try {
    await assert.rejects(() => saveCreds(), 'a failed write must reject, not resolve silently');
  } finally {
    chmodSync(dir, 0o700);
  }
  assert.equal(readFileSync(join(dir, 'creds.json'), 'utf8'), original,
    'the previous credentials must survive a failed save');
  rmSync(dir, { recursive: true, force: true });
});
