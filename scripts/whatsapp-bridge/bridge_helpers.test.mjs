import test from 'node:test';
import assert from 'node:assert/strict';

import { extractBridgeEvent } from './bridge_helpers.js';

// #106066: a quoted message can arrive inside the same wrappers as a top-level
// message (ephemeralMessage for disappearing-message replies). The quote ID and
// flag survived, but the extractor returned empty text for wrapped payloads.
async function quotedEvent(quotedMessage) {
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'reply-id', remoteJid: 'example@g.us' },
      message: {
        extendedTextMessage: {
          text: 'confirmed',
          contextInfo: {
            stanzaId: 'original-id',
            participant: 'example@s.whatsapp.net',
            quotedMessage,
          },
        },
      },
    },
    chatId: 'example@g.us',
    senderId: 'sender@s.whatsapp.net',
    senderNumber: 'sender',
    botIds: [],
    isGroup: true,
  });
  assert.equal(event.quotedMessageId, 'original-id');
  assert.equal(event.hasQuotedMessage, true);
  return event;
}

test('plain quoted conversation keeps its text', async () => {
  const event = await quotedEvent({ conversation: 'Example appointment at 11:40' });
  assert.equal(event.quotedText, 'Example appointment at 11:40');
});

test('ephemeral-wrapped quoted conversation keeps its text (#106066)', async () => {
  const event = await quotedEvent({
    ephemeralMessage: { message: { conversation: 'Example appointment at 11:40' } },
  });
  assert.equal(event.quotedText, 'Example appointment at 11:40');
});

test('view-once wrapped quoted conversation keeps its text', async () => {
  const event = await quotedEvent({
    viewOnceMessage: { message: { conversation: 'once upon a quote' } },
  });
  assert.equal(event.quotedText, 'once upon a quote');
});

test('wrapped quoted media caption still resolves', async () => {
  const event = await quotedEvent({
    ephemeralMessage: { message: { imageMessage: { caption: 'pic caption' } } },
  });
  assert.equal(event.quotedText, 'pic caption');
});
