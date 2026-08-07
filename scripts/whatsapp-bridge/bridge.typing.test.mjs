/**
 * Unit tests for WhatsApp typing presence forwarding.
 */

import { strict as assert } from 'node:assert';

import { sendTypingPresence } from './bridge_helpers.js';

{
  const calls = [];
  const sock = {
    sendPresenceUpdate: async (...args) => calls.push(args),
  };

  const presence = await sendTypingPresence(
    sock,
    '15551234567@s.whatsapp.net',
    'paused',
  );

  assert.equal(presence, 'paused');
  assert.deepEqual(calls, [['paused', '15551234567@s.whatsapp.net']]);
}

{
  const calls = [];
  const sock = {
    sendPresenceUpdate: async (...args) => calls.push(args),
  };

  const presence = await sendTypingPresence(
    sock,
    '15551234567@s.whatsapp.net',
    'unexpected',
  );

  assert.equal(presence, 'composing');
  assert.deepEqual(calls, [['composing', '15551234567@s.whatsapp.net']]);
}

console.log('bridge.typing.test.mjs: all assertions passed');
