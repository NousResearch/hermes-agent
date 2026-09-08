// Regression tests for quoted-text extraction through Baileys envelope
// wrappers (disappearing messages, view-once media, document-with-caption).
//
// A reply that quotes such a message nests the quoted payload under
// `contextInfo.quotedMessage.<wrapper>.message`, and the extractor used to
// read `quotedMessage.conversation` directly — so the quote text was lost
// while the quoted-message id survived (issue #106066).
//
// Imports only the pure helper module, so this runs without the Baileys
// runtime dependency: `node scripts/whatsapp-bridge/quoted_wrappers.test.mjs`.

import test from 'node:test';
import assert from 'node:assert/strict';

import { extractBridgeEvent } from './bridge_helpers.js';

const QUOTE = 'Example appointment at 11:40';

function replyQuoting(quotedMessage) {
  return extractBridgeEvent({
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
}

test('an unwrapped conversation quote keeps its text', async () => {
  const event = await replyQuoting({ conversation: QUOTE });
  assert.equal(event.quotedMessageId, 'original-id');
  assert.equal(event.hasQuotedMessage, true);
  assert.equal(event.quotedText, QUOTE);
});

test('an ephemeralMessage-wrapped quote keeps its text', async () => {
  const event = await replyQuoting({
    ephemeralMessage: { message: { conversation: QUOTE } },
  });
  assert.equal(event.quotedMessageId, 'original-id');
  assert.equal(event.hasQuotedMessage, true);
  assert.equal(event.quotedText, QUOTE);
});

test('a viewOnce-wrapped media caption is preserved', async () => {
  const event = await replyQuoting({
    viewOnceMessage: { message: { imageMessage: { caption: QUOTE } } },
  });
  assert.equal(event.quotedText, QUOTE);
});

test('nested envelopes (ephemeral wrapping view-once) are peeled', async () => {
  const event = await replyQuoting({
    ephemeralMessage: {
      message: {
        viewOnceMessageV2: { message: { extendedTextMessage: { text: QUOTE } } },
      },
    },
  });
  assert.equal(event.quotedText, QUOTE);
});

test('a documentWithCaption-wrapped quote keeps its caption', async () => {
  const event = await replyQuoting({
    documentWithCaptionMessage: {
      message: { documentMessage: { caption: QUOTE } },
    },
  });
  assert.equal(event.quotedText, QUOTE);
});

test('a missing quoted message still yields empty text without throwing', async () => {
  const event = await replyQuoting(undefined);
  assert.equal(event.quotedText, '');
  assert.equal(event.hasQuotedMessage, false);
});
