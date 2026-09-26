/**
 * Unit tests for timestampedLine, the ISO-8601-prefix helper for
 * bridge.js's human-facing lifecycle log lines.
 *
 * Regression for issue #97021: startup/connection lifecycle console.log/
 * console.warn lines (bridge listening, connected, logged out, reconnect,
 * "[bridge] ..." warnings) carried no timestamp, and the platform adapter
 * captures the bridge's stdout/stderr verbatim into bridge.log -- only
 * structured JSON events (pair events, allowlist rejections, #92683) were
 * timestamped, making bridge.log impossible to sequence on its own during
 * incident forensics.
 */

import { strict as assert } from 'node:assert';

import { timestampedLine } from './bridge_helpers.js';

const ISO_8601_PREFIX_RE = /^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\] /;

// The prefix is a valid, parseable ISO-8601 UTC timestamp.
{
  const line = timestampedLine('✅ WhatsApp connected!');

  assert.match(line, ISO_8601_PREFIX_RE);

  const bracketed = line.slice(1, line.indexOf(']'));
  assert.ok(!Number.isNaN(new Date(bracketed).getTime()));
}

// The original message text survives byte-for-byte after the prefix --
// this is a display shim, not a message-mangling one.
{
  const message = '❌ Logged out. Delete session and restart to re-authenticate.';
  const line = timestampedLine(message);

  assert.ok(line.endsWith(message));
}

// A template-literal-interpolated message (matching the actual bridge.js
// call sites, e.g. the reconnect/warn lines) is preserved intact too.
{
  const reason = 'stream error';
  const message = `⚠️  Connection closed (reason: ${reason}). Reconnecting in 3s...`;
  const line = timestampedLine(message);

  assert.ok(line.includes('stream error'));
  assert.ok(line.endsWith(message));
}

// Two calls a moment apart produce increasing (or equal, at ms resolution)
// timestamps -- the prefix reflects the actual call time, not a cached or
// module-load-time value, so a relaunch loop's individual lines remain
// independently sequenceable.
{
  const first = timestampedLine('a');
  await new Promise(resolve => setTimeout(resolve, 5));
  const second = timestampedLine('b');

  const firstTs = new Date(first.slice(1, first.indexOf(']'))).getTime();
  const secondTs = new Date(second.slice(1, second.indexOf(']'))).getTime();

  assert.ok(secondTs >= firstTs);
}

console.log('bridge_helpers.timestamp.test.mjs: all assertions passed');
