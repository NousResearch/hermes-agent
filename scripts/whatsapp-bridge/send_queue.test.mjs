import assert from 'node:assert/strict';
import test from 'node:test';

import { createSendQueue } from './send_queue.js';

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// A fake socket that records when each send starts and ends, and fails the test
// if two sends are ever in flight at once.
function recordingSocket(durations) {
  const events = [];
  let inFlight = 0;
  return {
    events,
    sendMessage(chatId, payload) {
      inFlight += 1;
      assert.equal(inFlight, 1, `overlapping send on the socket: ${payload.label}`);
      events.push(`start ${payload.label}`);
      const ms = durations[payload.label] ?? 0;
      return delay(ms).then(() => {
        inFlight -= 1;
        events.push(`end ${payload.label}`);
        return { key: { id: payload.label } };
      });
    },
  };
}

test('a best-effort send that misses its deadline keeps the socket until it settles', async () => {
  const sock = recordingSocket({ reaction: 120, reply: 0 });
  const { sendWithTimeout } = createSendQueue({ getSocket: () => sock, defaultTimeoutMs: 1000 });

  const reaction = sendWithTimeout('chat', { label: 'reaction' }, {}, 30, { skipIfExpired: true });
  reaction.catch(() => sock.events.push('caller deadline'));
  const reply = sendWithTimeout('chat', { label: 'reply' });

  await assert.rejects(reaction, /timed out after 0.03s/);
  await reply;
  // The caller gave up while the reaction was still on the socket, and the reply waited for it.
  assert.deepEqual(sock.events, ['start reaction', 'caller deadline', 'end reaction', 'start reply', 'end reply']);
});

test('a best-effort send whose deadline passed while queued never starts', async () => {
  const sock = recordingSocket({ slow: 80, reaction: 0 });
  const { sendWithTimeout } = createSendQueue({ getSocket: () => sock, defaultTimeoutMs: 1000 });

  const slow = sendWithTimeout('chat', { label: 'slow' });
  const reaction = sendWithTimeout('chat', { label: 'reaction' }, {}, 20, { skipIfExpired: true });

  await assert.rejects(reaction, /timed out/);
  await slow;
  await delay(10);
  assert.deepEqual(sock.events, ['start slow', 'end slow']);
});

test('a hung best-effort send frees the queue at the default timeout', async () => {
  const hung = { sendMessage: (_c, payload) => (payload.label === 'hung' ? new Promise(() => {}) : Promise.resolve({ key: { id: payload.label } })) };
  const { sendWithTimeout } = createSendQueue({ getSocket: () => hung, defaultTimeoutMs: 60 });

  const started = Date.now();
  const reaction = sendWithTimeout('chat', { label: 'hung' }, {}, 20, { skipIfExpired: true });
  const reply = sendWithTimeout('chat', { label: 'reply' }, {}, 500);

  await assert.rejects(reaction, /timed out after 0.02s/);
  const sent = await reply;
  assert.equal(sent.key.id, 'reply');
  assert.ok(Date.now() - started >= 55, 'the reply waited for the ceiling, not just the short deadline');
});

test('a regular send keeps the existing deadline behavior', async () => {
  const sock = recordingSocket({ reply: 0 });
  const { sendWithTimeout } = createSendQueue({ getSocket: () => sock, defaultTimeoutMs: 1000 });
  const sent = await sendWithTimeout('chat', { label: 'reply' }, { quoted: null });
  assert.equal(sent.key.id, 'reply');
});
