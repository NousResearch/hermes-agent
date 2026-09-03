/**
 * Unit + end-to-end tests for the last-resort process guards (#97108).
 *
 * Regression family: saveMedia()'s try/catch (the #59261 guard) contains the
 * awaited-rejection half of inbound media download failures, but a Baileys
 * media stream under undici/HTTP2 can emit 'error' with no listener outside
 * the awaited promise chain. Without an uncaughtException handler that throw
 * kills the bridge process and loses the in-memory inbound queue — WhatsApp
 * does not redeliver. These tests pin both halves: the guard's logging
 * behavior, and — via a real child process reproducing the issue's
 * deterministic simulation — that a listener-less stream 'error' no longer
 * kills the process once the guards are installed.
 *
 * These tests avoid importing bridge.js (it starts an HTTP server and a
 * Baileys socket at module load, same reason as bridge.reconnect.test.mjs);
 * the guards live in bridge_helpers.js for exactly that reason.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { installProcessGuards } from './bridge_helpers.js';

const here = path.dirname(fileURLToPath(import.meta.url));

test('installProcessGuards logs uncaught exceptions and keeps the target alive', () => {
  const target = new EventEmitter();
  const lines = [];
  installProcessGuards({ log: (l) => lines.push(l), target });

  const boom = new Error('HTTP/2 stream reset');
  target.emit('uncaughtException', boom);

  assert.equal(lines.length, 1);
  assert.match(lines[0], /uncaught exception contained — bridge kept alive/);
  assert.match(lines[0], /HTTP\/2 stream reset/);
  assert.ok(lines[0].includes(boom.stack), 'full stack lands in the log');
});

test('installProcessGuards logs unhandled rejections too', () => {
  const target = new EventEmitter();
  const lines = [];
  installProcessGuards({ log: (l) => lines.push(l), target });

  target.emit('unhandledRejection', new Error('socket hang up'));

  assert.equal(lines.length, 1);
  assert.match(lines[0], /unhandled rejection contained — bridge kept alive/);
});

test('non-Error rejection reasons are stringified, not dereferenced', () => {
  const target = new EventEmitter();
  const lines = [];
  installProcessGuards({ log: (l) => lines.push(l), target });

  target.emit('unhandledRejection', 'plain string reason');

  assert.equal(lines.length, 1);
  assert.match(lines[0], /plain string reason/);
});

test('a broken log sink never turns the guard into the killer', () => {
  const target = new EventEmitter();
  installProcessGuards({
    log: () => {
      throw new Error('log pipeline is broken');
    },
    target,
  });

  // The guard must swallow its own logging failure instead of rethrowing.
  target.emit('uncaughtException', new Error('original failure'));
});

// Real-process end-to-end anchors. The child script reproduces the issue's
// deterministic simulation: resolve the awaited download, then on a later
// tick emit 'error' on a listener-less Readable — the exact escape hatch the
// saveMedia() try/catch cannot reach.

const guardedChild = `
import { Readable } from 'node:stream';
import { installProcessGuards } from './bridge_helpers.js';
const lines = [];
installProcessGuards({ log: (l) => lines.push(l), target: process });
process.on('exit', (code) => {
  if (code !== 0) return;
  if (lines.length < 1 || !lines[0].includes('kept alive')) process.exitCode = 2;
});
process.nextTick(() => {
  // Listener-less 'error' outside any awaited promise chain (#97108 repro).
  new Readable({ read() {} }).emit('error', new Error('HTTP/2 stream reset'));
});
setTimeout(() => {
  // Reaching this timer at all proves the process survived the throw.
  console.log('ALIVE_AFTER_UNHANDLED_STREAM_ERROR');
}, 200);
`;

const bareChild = `
import { Readable } from 'node:stream';
process.nextTick(() => {
  new Readable({ read() {} }).emit('error', new Error('HTTP/2 stream reset'));
});
setTimeout(() => {
  console.log('SHOULD_NOT_GET_HERE');
}, 200);
`;

test('end-to-end: guards keep the bridge process alive through a listener-less stream error', () => {
  const out = execFileSync(
    process.execPath,
    ['--input-type=module', '-e', guardedChild],
    { cwd: here, encoding: 'utf8', timeout: 15000 },
  );
  assert.match(out, /ALIVE_AFTER_UNHANDLED_STREAM_ERROR/);
});

test('negative control: without the guards the same error kills the process', () => {
  let sawDeath = false;
  try {
    execFileSync(
      process.execPath,
      ['--input-type=module', '-e', bareChild],
      { cwd: here, encoding: 'utf8', timeout: 15000 },
    );
  } catch (err) {
    // Node exits non-zero on the unhandled 'error' event — that is the bug.
    sawDeath = err.status !== 0;
    assert.ok(sawDeath, 'child should die non-zero without the guards');
    assert.doesNotMatch(String(err.stdout || ''), /SHOULD_NOT_GET_HERE/);
    return;
  }
  assert.fail('child unexpectedly survived without the guards installed');
});
