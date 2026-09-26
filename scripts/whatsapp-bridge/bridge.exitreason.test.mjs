/**
 * Unit tests for the terminal-disconnect classification and the bridge ->
 * gateway exit contract.
 *
 * Regression tests for #80088. The bridge exits 1 when WhatsApp ends the
 * session, and the gateway sees only that exit code, where a logged-out
 * session is indistinguishable from a crash. The gateway therefore classified
 * every exit as retryable and re-spawned the bridge forever against
 * credentials that cannot work. The bridge now records why it is going in
 * `bridge-exit.json` beside the session.
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and a Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { TERMINAL_DISCONNECT_REASONS, terminalDisconnectReason, writeBridgeExit } from './bridge_helpers.js';

// -- terminalDisconnectReason ---------------------------------------------

// 401 covers both an explicit logout and the linked device being removed from
// the phone: WhatsApp answers the same code for both, and neither can be
// revived by another reconnect.
assert.equal(terminalDisconnectReason(401), 'logged_out');
assert.equal(TERMINAL_DISCONNECT_REASONS[401], 'logged_out');

// Every other close stays retryable. Disabling the reconnect path for a
// dropped connection (408), a server-side close (428), a replaced session
// (440), an expired session (500), an unavailable service (503), a forbidden
// account (403), or the post-pairing restart request (515) would be a worse
// bug than the one this contract fixes, so none of them may be classified as
// terminal. An unknown or absent code must also stay retryable: refusing to
// retry an exit nobody explained would strand a recoverable bridge.
for (const code of [408, 428, 440, 500, 503, 515, 403, 200, 0, -1, undefined, null, 'nonsense']) {
  assert.equal(terminalDisconnectReason(code), null, `${code} must stay retryable`);
}

// -- writeBridgeExit ------------------------------------------------------

// The record lands beside the session with the reason the adapter keys on.
{
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-bridge-exit-'));
  try {
    assert.equal(writeBridgeExit(dir, { reason: 'logged_out', statusCode: 401 }), true);
    const record = JSON.parse(readFileSync(path.join(dir, 'bridge-exit.json'), 'utf8'));
    assert.equal(record.reason, 'logged_out');
    assert.equal(record.statusCode, 401);
    assert.ok(record.at, 'the record carries a timestamp');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// A session directory that does not exist yet is created, so the record is
// never lost just because this is the first bridge start.
{
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-bridge-exit-'));
  const nested = path.join(dir, 'session');
  try {
    assert.equal(writeBridgeExit(nested, { reason: 'logged_out', statusCode: 401 }), true);
    assert.ok(existsSync(path.join(nested, 'bridge-exit.json')));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// A write that cannot happen is reported and swallowed: the bridge must still
// be free to exit, and a session directory blocked by a regular file is the
// realistic shape of that failure.
{
  const dir = mkdtempSync(path.join(tmpdir(), 'hermes-bridge-exit-'));
  const logs = [];
  try {
    const blocker = path.join(dir, 'not-a-directory');
    writeFileSync(blocker, 'x');
    assert.equal(
      writeBridgeExit(path.join(blocker, 'session'), { reason: 'logged_out', statusCode: 401 }, { log: line => logs.push(line) }),
      false,
    );
    assert.equal(logs.length, 1);
    assert.match(logs[0], /Could not record the bridge exit reason/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

console.log('bridge.exitreason.test.mjs: all assertions passed');
