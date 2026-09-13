import { strict as assert } from 'node:assert';

import { buildTextSendPayload } from './bridge_helpers.js';


{
  const result = buildTextSendPayload('hello group', {
    mentions: ['+15551234567', '149606612619433@lid', '15557654321:4@s.whatsapp.net'],
  });

  assert.deepEqual(result, {
    content: {
      text: 'hello group',
      contextInfo: {
        mentionedJid: [
          '15551234567@s.whatsapp.net',
          '149606612619433@lid',
          '15557654321@s.whatsapp.net',
        ],
      },
    },
    options: {},
  });
}

{
  assert.deepEqual(buildTextSendPayload('plain text'), {
    content: { text: 'plain text' },
    options: {},
  });
}

assert.throws(
  () => buildTextSendPayload('bad', { mentions: ['alice'] }),
  /invalid whatsapp mention/i,
);

console.log('  ✓ outbound mentions become Baileys contextInfo.mentionedJid');
