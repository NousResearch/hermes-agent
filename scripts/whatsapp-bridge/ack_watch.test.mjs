/**
 * Regression tests for the server-ack watch (camelludo/hermes-agent, Go2Stone/freightman#2463):
 * a reconnected socket swallowed every send for 36h while /health said "connected".
 */
import { strict as assert } from 'node:assert';
import { EventEmitter } from 'node:events';
import { bridgeStatus, createAckWatch, normalizeJid } from './ack_watch.js';

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

// Per chat: only sends to that chat since the moment count, and only acked ones are acked.
{
  clock = 1_000;
  const watch = createAckWatch({ now });
  const sock = fakeSocket();
  watch.attach(sock);
  watch.sent('OLD', '971500000001@s.whatsapp.net');
  sock.ws.emit('CB:ack,class:message', { attrs: { id: 'OLD' } });
  clock = 5_000;
  watch.sent('E1', '971500000001@s.whatsapp.net');
  watch.sent('E2', '971500000001:7@s.whatsapp.net');
  watch.sent('OTHER', '34600000000@s.whatsapp.net');
  sock.ws.emit('CB:ack,class:message', { attrs: { id: 'E1' } });
  assert.deepEqual(watch.sends('971500000001', 5_000), { sent: 2, acked: 1 });
  sock.ev.emit('messages.update', [{ key: { id: 'E2', fromMe: true }, update: { status: 2 } }]);
  assert.deepEqual(watch.sends('971500000001@s.whatsapp.net', 5_000), { sent: 2, acked: 2 });
  assert.deepEqual(watch.sends('971500000001@s.whatsapp.net', 9_000), { sent: 0, acked: 0 });
  assert.deepEqual(watch.sends('123@lid', 0), { sent: 0, acked: 0 });
}

// Replay of 2026-10-17 → 10-19: after a reconnect, sendMessage keeps resolving but WhatsApp
// never acks. /health must stop saying "connected" within the window, and the per-chat answer
// must never call the reply acked — not even after the bridge moves to a new socket.
{
  clock = 0;
  const watch = createAckWatch({ staleMs: 120_000, now });
  const sock = fakeSocket();
  watch.attach(sock);
  watch.sent('MOHAMMED-1', '971556003130@s.whatsapp.net');
  assert.equal(bridgeStatus('connected', watch.health()), 'connected');
  clock = 121_000;
  assert.equal(bridgeStatus('connected', watch.health()), 'sends_unacked');
  assert.equal(watch.health().unackedSends, 1);
  clock = 600_000;
  assert.equal(bridgeStatus('connected', watch.health()), 'sends_unacked');
  assert.deepEqual(watch.sends('971556003130', 0), { sent: 1, acked: 0 });
  watch.attach(fakeSocket());
  assert.deepEqual(watch.sends('971556003130', 0), { sent: 1, acked: 0 });
  assert.equal(bridgeStatus('reconnecting', watch.health()), 'reconnecting');
}

// The log stays bounded.
{
  const watch = createAckWatch({ now, keep: 3 });
  for (const id of ['a', 'b', 'c', 'd']) watch.sent(id, '1');
  assert.deepEqual(watch.sends('1', 0), { sent: 3, acked: 0 });
}

assert.equal(normalizeJid('34600000000'), '34600000000@s.whatsapp.net');
assert.equal(normalizeJid('34600000000:12@s.whatsapp.net'), '34600000000@s.whatsapp.net');
assert.equal(normalizeJid('9876@lid'), '9876@lid');
assert.equal(normalizeJid(''), '');

console.log('ack_watch: all tests passed');
