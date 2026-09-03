import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveWhatsAppNumber } from '../phone-resolver.js';

test('resolves a national-format input to WhatsApp canonical JIDs', async () => {
  const calls = [];
  const sock = {
    onWhatsApp: async (number) => {
      calls.push(number);
      return [
        { jid: '557199258283@s.whatsapp.net', exists: true },
        { jid: 'ignored@s.whatsapp.net', exists: false },
      ];
    },
  };

  const result = await resolveWhatsAppNumber(sock, '+55 (71) 99925-8283');

  assert.deepEqual(calls, ['5571999258283']);
  assert.deepEqual(result, {
    number: '5571999258283',
    exists: true,
    matches: [{ jid: '557199258283@s.whatsapp.net', exists: true }],
  });
});

test('rejects inputs without digits before calling WhatsApp', async () => {
  let called = false;
  const sock = { onWhatsApp: async () => { called = true; } };

  await assert.rejects(
    () => resolveWhatsAppNumber(sock, 'not a number'),
    { message: 'number must contain digits' },
  );
  assert.equal(called, false);
});
