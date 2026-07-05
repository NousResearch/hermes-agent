/**
 * Unit tests for WhatsApp-native bridge payload helpers.
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';
import { createHash } from 'node:crypto';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { getAggregateVotesInPollMessage } from '@whiskeysockets/baileys';

import {
  buildPollPayload,
  buildTextSendPayload,
  createBoundedMessageStore,
  extractBridgeEvent,
  mediaPayloadForFile,
  pollCreationMessageFromPayload,
  pollUpdateForAggregation,
} from './bridge_helpers.js';

// -- quoted outbound text -------------------------------------------------
{
  const store = createBoundedMessageStore(2);
  store.remember({
    key: {
      id: 'inbound-1',
      remoteJid: '15551234567@s.whatsapp.net',
      participant: '15550001111@s.whatsapp.net',
      fromMe: false,
    },
    message: { conversation: 'original text' },
  });

  const { content, options } = buildTextSendPayload('reply text', {
    chatId: '15551234567@s.whatsapp.net',
    replyTo: 'inbound-1',
    messageStore: store,
  });

  assert.deepEqual(content, { text: 'reply text' });
  assert.equal(options.quoted.key.id, 'inbound-1');
  assert.equal(options.quoted.message.conversation, 'original text');
  console.log('  ✓ text replies include Baileys quoted message when resolvable');
}

{
  const store = createBoundedMessageStore(2);
  const { content, options } = buildTextSendPayload('plain text', {
    chatId: '15551234567@s.whatsapp.net',
    replyTo: 'missing-id',
    messageStore: store,
  });

  assert.deepEqual(content, { text: 'plain text' });
  assert.deepEqual(options, {});
  console.log('  ✓ unresolved replyTo falls back to plain text');
}

// -- inbound quote/media/native metadata --------------------------------
{
  const event = await extractBridgeEvent({
    msg: {
      key: {
        id: 'incoming-1',
        remoteJid: '15551234567@s.whatsapp.net',
        participant: '15550001111@s.whatsapp.net',
        fromMe: false,
      },
      pushName: 'Tester',
      messageTimestamp: 123,
      message: {
        extendedTextMessage: {
          text: 'approved',
          contextInfo: {
            stanzaId: 'outbound-1',
            participant: '15559998888@s.whatsapp.net',
            remoteJid: '15551234567@s.whatsapp.net',
            quotedMessage: { conversation: 'approve deploy?' },
          },
        },
      },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
    botIds: ['15559998888@s.whatsapp.net'],
    downloadMedia: async () => Buffer.from(''),
  });

  assert.equal(event.quotedMessageId, 'outbound-1');
  assert.equal(event.quotedParticipant, '15559998888@s.whatsapp.net');
  assert.equal(event.quotedRemoteJid, '15551234567@s.whatsapp.net');
  assert.equal(event.quotedText, 'approve deploy?');
  assert.equal(event.hasQuotedMessage, true);
  assert.equal(event.body, 'approved');
  console.log('  ✓ inbound quoted metadata includes quoted text');
}

{
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'doc-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: {
        documentMessage: {
          caption: 'see attached',
          fileName: 'report.pdf',
          mimetype: 'application/pdf',
        },
      },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
    downloadMedia: async () => Buffer.from('pdf'),
    writeMediaFile: async () => '/tmp/report.pdf',
  });

  assert.equal(event.hasMedia, true);
  assert.equal(event.mediaType, 'document');
  assert.equal(event.mime, 'application/pdf');
  assert.equal(event.fileName, 'report.pdf');
  assert.equal(event.nativeType, 'documentMessage');
  assert.deepEqual(event.mediaUrls, ['/tmp/report.pdf']);
  console.log('  ✓ inbound document metadata preserves MIME and filename');
}

{
  const cacheDir = mkdtempSync(path.join(tmpdir(), 'hermes-wa-doc-'));
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'doc-2', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: {
        documentMessage: {
          caption: 'see attached',
          fileName: 'report',
          mimetype: 'application/pdf',
        },
      },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
    downloadMedia: async () => Buffer.from('pdf'),
    cacheDirs: { document: cacheDir },
  });

  assert.equal(event.mediaUrls.length, 1);
  assert.ok(event.mediaUrls[0].endsWith('_report.pdf'), event.mediaUrls[0]);
  console.log('  ✓ MIME extension is preserved when document filename has none');
}

{
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'loc-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: {
        locationMessage: {
          name: 'HQ',
          degreesLatitude: 41.015,
          degreesLongitude: 28.979,
        },
      },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
  });

  assert.equal(event.mediaType, 'location');
  assert.equal(event.body, '[Location: HQ 41.015,28.979]');
  assert.deepEqual(event.nativeMetadata.location, {
    name: 'HQ',
    address: '',
    latitude: 41.015,
    longitude: 28.979,
    isLive: false,
  });
  console.log('  ✓ native location messages get text fallback and metadata');
}

{
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'poll-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: {
        pollCreationMessage: {
          name: 'Approve deploy?',
          options: [{ optionName: 'Approve' }, { optionName: 'Deny' }],
          selectableOptionsCount: 1,
        },
      },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15550001111@s.whatsapp.net',
    senderNumber: '15550001111',
  });

  assert.equal(event.mediaType, 'poll');
  assert.equal(event.body, '[Poll: Approve deploy? Options: Approve, Deny]');
  assert.deepEqual(event.nativeMetadata.poll.options, ['Approve', 'Deny']);
  console.log('  ✓ poll creation messages get text fallback and metadata');
}

// -- outbound media/poll helpers -----------------------------------------
{
  const payload = mediaPayloadForFile({
    buffer: Buffer.from('gif89a'),
    filePath: '/tmp/loop.gif',
    mediaType: 'image',
    caption: 'loop',
  });

  assert.ok(payload.image, 'pure helper fallback keeps raw GIF as image bytes');
  assert.equal(payload.gifPlayback, undefined);
  assert.equal(payload.mimetype, 'image/gif');
  assert.equal(payload.caption, 'loop');
  console.log('  ✓ local GIF helper fallback stays truthful; live bridge converts to gifPlayback when possible');
}

{
  const payload = buildPollPayload({
    question: 'Proceed?',
    options: ['Approve', 'Deny'],
    selectableCount: 1,
  });

  assert.equal(payload.poll.name, 'Proceed?');
  assert.deepEqual(payload.poll.values, ['Approve', 'Deny']);
  assert.equal(payload.poll.selectableCount, 1);
  assert.equal(Buffer.isBuffer(payload.poll.messageSecret), true);
  assert.equal(payload.poll.messageSecret.length, 32);
  assert.deepEqual(pollCreationMessageFromPayload(payload), {
    messageContextInfo: {
      messageSecret: payload.poll.messageSecret,
    },
    pollCreationMessageV3: {
      name: 'Proceed?',
      options: [{ optionName: 'Approve' }, { optionName: 'Deny' }],
      selectableOptionsCount: 1,
    },
  });
  console.log('  ✓ poll payload primitive carries a cacheable vote secret');
}

{
  const pollCreation = {
    key: {
      id: 'poll-creation',
      remoteJid: '15551234567@s.whatsapp.net',
      fromMe: true,
    },
    message: {
      messageContextInfo: {
        messageSecret: Buffer.from('0123456789abcdef0123456789abcdef'),
      },
      pollCreationMessageV3: {
        name: 'Proceed?',
        options: [{ optionName: 'Approve' }, { optionName: 'Deny' }],
        selectableOptionsCount: 1,
      },
    },
  };
  const voteKey = {
    id: 'vote-message',
    remoteJid: '15551234567@s.whatsapp.net',
    participant: '15550001111@s.whatsapp.net',
    fromMe: false,
  };
  const encryptedVote = {
    encPayload: Buffer.from('payload'),
    encIv: Buffer.from('iv'),
  };

  const attempts = [];
  const pollUpdate = pollUpdateForAggregation({
    pollUpdateMessage: {
      pollCreationMessageKey: pollCreation.key,
      vote: encryptedVote,
      senderTimestampMs: 123,
    },
    pollUpdateMessageKey: voteKey,
    pollCreation,
    decryptPollVote: (vote, ctx) => {
      attempts.push({ pollCreatorJid: ctx.pollCreatorJid, voterJid: ctx.voterJid });
      assert.equal(vote, encryptedVote);
      assert.equal(ctx.pollMsgId, 'poll-creation');
      assert.equal(ctx.pollEncKey, pollCreation.message.messageContextInfo.messageSecret);
      if (ctx.pollCreatorJid !== 'creator-lid@lid') {
        throw new Error('wrong creator jid');
      }
      assert.equal(ctx.voterJid, '15550001111@s.whatsapp.net');
      return {
        selectedOptions: [createHash('sha256').update(Buffer.from('Approve')).digest()],
      };
    },
    getKeyAuthor: (key, meId = 'me') => (key?.fromMe ? meId : key?.participant || key?.remoteJid || ''),
    meId: 'classic-me@s.whatsapp.net',
    pollCreatorJids: ['classic-me@s.whatsapp.net', 'creator-lid@lid'],
  });

  assert.deepEqual(attempts.map(item => item.pollCreatorJid), ['classic-me@s.whatsapp.net', 'creator-lid@lid']);

  assert.equal(pollUpdate.pollUpdateMessageKey.id, 'vote-message');
  assert.equal(pollUpdate.senderTimestampMs, 123);
  const aggregation = getAggregateVotesInPollMessage({
    message: pollCreation.message,
    pollUpdates: [pollUpdate],
  });
  assert.deepEqual(
    aggregation.map(option => ({ name: option.name, voters: option.voters })),
    [
      { name: 'Approve', voters: ['15550001111@s.whatsapp.net'] },
      { name: 'Deny', voters: [] },
    ],
  );
  console.log('  ✓ encrypted poll upserts are wrapped into Baileys aggregation shape');
}

console.log('\n✅ All WhatsApp native bridge helper tests passed.');
