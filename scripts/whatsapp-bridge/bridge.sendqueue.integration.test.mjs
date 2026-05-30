/**
 * Integration test: verify bridge.js has the send queue wired in.
 * This reads the source file and asserts the required patterns exist.
 */
import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(resolve(__dirname, 'bridge.js'), 'utf-8');

// 1. Queue primitives exist
assert.ok(src.includes('let _sendQueue = Promise.resolve()'),
  'bridge.js must define _sendQueue');
assert.ok(src.includes('function enqueueSend(fn)'),
  'bridge.js must define enqueueSend');

// 2. sendWithTimeout is wrapped
assert.ok(src.includes('return enqueueSend(() =>'),
  'sendWithTimeout must call enqueueSend');

// 3. The queue is before sendWithTimeout in the file
const queueIdx = src.indexOf('let _sendQueue');
const sendTimeoutIdx = src.indexOf('function sendWithTimeout');
assert.ok(queueIdx < sendTimeoutIdx,
  '_sendQueue must be defined before sendWithTimeout');

console.log('✅ bridge.js send queue integration check passed.');
