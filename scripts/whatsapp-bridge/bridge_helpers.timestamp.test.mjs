/**
 * Unit tests for the bridge's log-line stamping: formatLogStamp,
 * installConsoleStamps and writeJsonLine.
 *
 * Regression for issue #97021: startup/connection lifecycle console.log/
 * console.warn lines (bridge listening, connected, logged out, reconnect,
 * "[bridge] ..." warnings) carried no timestamp, and the platform adapter
 * captures the bridge's stdout/stderr verbatim into bridge.log, making
 * bridge.log impossible to sequence on its own during incident forensics.
 * Machine-read JSON event lines (pair events parsed line-by-line by the
 * dashboard pairing watcher, `ignored` events) must stay byte-identical.
 */

import { strict as assert } from 'node:assert';

import { formatLogStamp, installConsoleStamps, writeJsonLine } from './bridge_helpers.js';

const LOCAL_STAMP_PREFIX_RE = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} /;

function captureWrites(stream, fn) {
  const chunks = [];
  const originalWrite = stream.write;
  stream.write = (chunk, ...rest) => {
    chunks.push(String(chunk));
    const callback = rest.find(arg => typeof arg === 'function');
    if (callback) callback();
    return true;
  };
  try {
    fn();
  } finally {
    stream.write = originalWrite;
  }
  return chunks;
}

function withStampedConsole(fn) {
  const saved = { log: console.log, warn: console.warn, error: console.error };
  installConsoleStamps();
  try {
    fn();
  } finally {
    Object.assign(console, saved);
  }
}

// The stamp is local wall-clock time in Python's asctime shape.
{
  assert.equal(formatLogStamp(new Date(2026, 8, 28, 3, 4, 5, 62)), '2026-09-28 03:04:05,062');
}

// A console.log line is stamped with the current local time, and the message
// text survives byte-for-byte after the stamp.
withStampedConsole(() => {
  const before = Date.now();
  const out = captureWrites(process.stdout, () => {
    console.log('❌ Logged out. Delete session and restart to re-authenticate.');
  });

  assert.equal(out.length, 1);
  assert.match(out[0], LOCAL_STAMP_PREFIX_RE);
  assert.ok(out[0].endsWith(' ❌ Logged out. Delete session and restart to re-authenticate.\n'));
  const [day, time] = out[0].split(' ');
  const stampedMs = new Date(`${day}T${time.replace(',', '.')}`).getTime();
  assert.ok(Math.abs(stampedMs - before) < 5000, `stamp ${day} ${time} is not local now`);
});

// console.warn keeps its multi-argument formatting and still goes to stderr.
withStampedConsole(() => {
  const err = captureWrites(process.stderr, () => {
    console.warn('[bridge] failed to send read receipt:', 'boom');
  });

  assert.equal(err.length, 1);
  assert.match(err[0], LOCAL_STAMP_PREFIX_RE);
  assert.ok(err[0].endsWith(' [bridge] failed to send read receipt: boom\n'));
});

// A JSON event line is not stamped, even with the console wrapped, so the
// pairing watcher's per-line json.loads still parses it.
withStampedConsole(() => {
  const out = captureWrites(process.stdout, () => {
    writeJsonLine({ ts: 1790000000000, event: 'qr', qr: 'abc' });
  });

  assert.deepEqual(out, ['{"ts":1790000000000,"event":"qr","qr":"abc"}\n']);
});

// An empty console.log() stays a bare newline rather than a lone stamp.
withStampedConsole(() => {
  assert.deepEqual(captureWrites(process.stdout, () => console.log()), ['\n']);
});

console.log('bridge_helpers.timestamp.test.mjs: all assertions passed');
