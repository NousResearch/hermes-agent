/**
 * Regression tests for the server-ack watch (camelludo/hermes-agent, Go2Stone/freightman#2463):
 * a reconnected socket swallowed every send for 36h while /health said "connected".
 */
import { strict as assert } from 'node:assert';
import { EventEmitter } from 'node:events';
import { createAckWatch } from './ack_watch.js';

function fakeSocket() {
  return { ws: new EventEmitter(), ev: new EventEmitter() };
}

let clock = 0;
const now = () => clock;

// A send the server acks never turns the bridge unhealthy.
{
  const watch = createAckWatch({ staleMs: 120_000, now });
  const sock = fakeSocket();
  watch.attach(sock);
  watch.sent('A');
  sock.ws.emit('CB:ack,class:message', { attrs: { id: 'A' } });
  clock += 600_000;
  assert.deepEqual(watch.health(), {
    stale: false, unackedSends: 0, oldestUnackedSec: 0, sendsTracked: 1, sendsAcked: 1,
  });
}

// A send with no ack goes stale only after staleMs.
{
  clock = 0;
  const watch = createAckWatch({ staleMs: 120_000, now });
  watch.attach(fakeSocket());
  watch.sent('B');
  clock = 119_000;
  assert.equal(watch.health().stale, false);
  clock = 121_000;
  assert.equal(watch.health().stale, true);
  assert.equal(watch.health().unackedSends, 1);
}

// A delivery receipt counts as an ack; receipts for other people's messages do not.
{
  clock = 0;
  const watch = createAckWatch({ staleMs: 1, now });
  const sock = fakeSocket();
  watch.attach(sock);
  watch.sent('C');
  sock.ev.emit('messages.update', [{ key: { id: 'C', fromMe: false }, update: { status: 3 } }]);
  assert.equal(watch.health().unackedSends, 1);
  sock.ev.emit('messages.update', [{ key: { id: 'C', fromMe: true }, update: { status: 3 } }]);
  assert.equal(watch.health().unackedSends, 0);
}

// A new socket forgets the old one's pending sends (they can never be acked).
{
  clock = 0;
  const watch = createAckWatch({ staleMs: 1, now });
  watch.attach(fakeSocket());
  watch.sent('D');
  clock = 10;
  assert.equal(watch.health().stale, true);
  watch.attach(fakeSocket());
  assert.equal(watch.health().stale, false);
}

console.log('ack_watch: all tests passed');
