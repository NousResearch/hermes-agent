import test from 'node:test';
import assert from 'node:assert/strict';

import { classifyInboundMessageGate } from './inbound_message_gate.js';

function makeAllowlist(allowedIds) {
  const set = new Set(allowedIds);
  return (id) => set.has(id);
}

test('fromMe messages pass through inbound gate', () => {
  assert.deepEqual(classifyInboundMessageGate({ fromMe: true }), { action: 'pass' });
});

test('self-chat mode rejects non-self messages by default', () => {
  assert.deepEqual(
    classifyInboundMessageGate({
      fromMe: false,
      mode: 'self-chat',
      allowNonSelf: false,
    }),
    { action: 'drop', reason: 'self_chat_mode_rejects_non_self' },
  );
});

test('human-in-loop opt-in lets customer DMs reach Python pre-dispatch in self-chat mode', () => {
  assert.deepEqual(
    classifyInboundMessageGate({
      fromMe: false,
      mode: 'self-chat',
      allowNonSelf: true,
    }),
    { action: 'pass', reason: 'human_in_loop_allows_non_self' },
  );
});

test('bot mode keeps allowlist protection without human-in-loop opt-in', () => {
  assert.deepEqual(
    classifyInboundMessageGate({
      fromMe: false,
      mode: 'bot',
      allowNonSelf: false,
      allowlistMatches: makeAllowlist(['owner@s.whatsapp.net']),
      senderId: 'customer@s.whatsapp.net',
    }),
    { action: 'drop', reason: 'allowlist_mismatch' },
  );
});

test('human-in-loop opt-in bypasses sender allowlist in bot mode', () => {
  assert.deepEqual(
    classifyInboundMessageGate({
      fromMe: false,
      mode: 'bot',
      allowNonSelf: true,
      allowlistMatches: makeAllowlist(['owner@s.whatsapp.net']),
      senderId: 'customer@s.whatsapp.net',
    }),
    { action: 'pass', reason: 'human_in_loop_bypasses_allowlist' },
  );
});
