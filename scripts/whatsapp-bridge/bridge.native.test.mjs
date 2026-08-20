/**
 * Unit tests for WhatsApp-native bridge payload helpers.
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';
import { createHash } from 'node:crypto';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { getAggregateVotesInPollMessage } from '@whiskeysockets/baileys';

import {
  buildAlbumPlan,
  buildPollPayload,
  buildTextSendPayload,
  createBoundedMessageStore,
  appendMediaFailureNote,
  extractBridgeEvent,
  inboundReadReceiptKeys,
  mediaPayloadForFile,
  pollCreationMessageFromPayload,
  pollUpdateForAggregation,
  prepareAlbumItems,
  sendAlbumSequence,
} from './bridge_helpers.js';

// -- inbound read receipts ------------------------------------------------
{
  const groupKey = {
    id: 'incoming-group-1',
    remoteJid: '120363001234567890@g.us',
    participant: '15550001111@s.whatsapp.net',
    fromMe: false,
  };

  assert.deepEqual(inboundReadReceiptKeys({ key: groupKey, enabled: false }), []);
  assert.deepEqual(
    inboundReadReceiptKeys({ key: { ...groupKey, fromMe: true }, enabled: true }),
    [],
  );
  const receiptKeys = inboundReadReceiptKeys({ key: groupKey, enabled: true });
  assert.equal(receiptKeys.length, 1);
  assert.equal(receiptKeys[0], groupKey);
  assert.equal(receiptKeys[0].participant, groupKey.participant);
  console.log('  ✓ inbound read receipts preserve the original group message key');
}

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
  assert.deepEqual(event.readReceiptKey, {
    id: 'incoming-1',
    remoteJid: '15551234567@s.whatsapp.net',
    participant: '15550001111@s.whatsapp.net',
    fromMe: false,
  });
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
  const calls = [];
  const parentKey = {
    id: 'album-parent',
    remoteJid: '15551234567@s.whatsapp.net',
    fromMe: true,
  };
  const send = async (chatId, payload) => {
    calls.push({ chatId, payload });
    if (calls.length === 1) {
      return { key: parentKey };
    }
    return { key: { id: `album-child-${calls.length - 1}`, remoteJid: chatId, fromMe: true } };
  };
  const items = [
    { type: 'image', filePath: '/tmp/one.jpg', payload: { image: Buffer.from('one'), mimetype: 'image/jpeg' } },
    { type: 'image', filePath: '/tmp/two.jpg', payload: { image: Buffer.from('two'), mimetype: 'image/jpeg' } },
    { type: 'video', filePath: '/tmp/three.mp4', payload: { video: Buffer.from('three'), mimetype: 'video/mp4' } },
  ];

  const plan = buildAlbumPlan(items);
  assert.deepEqual(plan.parentPayload, {
    album: { expectedImageCount: 2, expectedVideoCount: 1 },
  });

  const result = await sendAlbumSequence({
    chatId: '15551234567@s.whatsapp.net',
    items,
    send,
  });

  assert.equal(calls.length, 4);
  assert.deepEqual(calls[0].payload, plan.parentPayload);
  for (const childCall of calls.slice(1)) {
    assert.deepEqual(childCall.payload.albumParentKey, parentKey);
  }
  assert.equal(result.success, true);
  assert.equal(result.parentMessageId, 'album-parent');
  assert.deepEqual(result.childMessageIds, ['album-child-1', 'album-child-2', 'album-child-3']);
  assert.equal(result.items[0].filePath, '/tmp/one.jpg');
  console.log('  ✓ album sequence sends one parent then associated media children');
}

{
  const albumDir = mkdtempSync(path.join(tmpdir(), 'hermes-wa-album-'));
  const first = path.join(albumDir, 'first.jpg');
  const second = path.join(albumDir, 'second.mp4');
  writeFileSync(first, Buffer.from('first'));
  writeFileSync(second, Buffer.from('second'));

  const prepared = prepareAlbumItems([
    { filePath: first, mediaType: 'image' },
    { filePath: second, mediaType: 'video', caption: 'clip' },
  ]);

  assert.equal(prepared.length, 2);
  assert.equal(prepared[0].type, 'image');
  assert.equal(prepared[0].filePath, first);
  assert.equal(prepared[0].createPayload().mimetype, 'image/jpeg');
  assert.equal(prepared[1].type, 'video');
  assert.equal(prepared[1].createPayload().caption, 'clip');
  console.log('  ✓ album files are fully validated and prepared before parent send');
}

{
  const calls = [];
  const send = async (chatId, payload) => {
    calls.push({ chatId, payload });
    if (calls.length === 1) return { key: { id: 'parent', remoteJid: chatId } };
    if (calls.length === 3) throw new Error('child upload failed');
    return { key: { id: `child-${calls.length - 1}`, remoteJid: chatId } };
  };

  const result = await sendAlbumSequence({
    chatId: '15551234567@s.whatsapp.net',
    items: [
      { type: 'image', filePath: '/tmp/one.jpg', payload: { image: Buffer.from('one') } },
      { type: 'image', filePath: '/tmp/two.jpg', payload: { image: Buffer.from('two') } },
      { type: 'image', filePath: '/tmp/three.jpg', payload: { image: Buffer.from('three') } },
    ],
    send,
  });

  assert.equal(calls.length, 4, 'later children still send after one child fails');
  assert.equal(result.success, false);
  assert.equal(result.status, 'partial_failure');
  assert.deepEqual(result.childMessageIds, ['child-1', 'child-3']);
  assert.deepEqual(result.items[1], {
    index: 1,
    filePath: '/tmp/two.jpg',
    success: false,
    error: 'child upload failed',
  });
  console.log('  ✓ partial album failures are explicit and do not duplicate successful children');
}

{
  const result = await sendAlbumSequence({
    chatId: '15551234567@s.whatsapp.net',
    items: [
      { type: 'image', payload: { image: Buffer.from('one') } },
      { type: 'image', payload: { image: Buffer.from('two') } },
    ],
    send: async () => { throw new Error('parent send timed out'); },
  });

  assert.deepEqual(result, {
    success: false,
    attempted: true,
    status: 'parent_failure',
    parentMessageId: null,
    childMessageIds: [],
    items: [],
    error: 'parent send timed out',
  });
  console.log('  ✓ parent failures are distinguished from preflight validation failures');
}

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

// -- media download failure containment (port of nanoclaw#2895) -----------
{
  assert.equal(appendMediaFailureNote('hello', []), 'hello');
  assert.equal(
    appendMediaFailureNote('check this out', ['image']),
    'check this out\n[image could not be downloaded]',
  );
  // Regression guard: an uncaptioned failed image must still produce a
  // non-empty body, or the empty-message guard drops the whole message.
  assert.equal(appendMediaFailureNote('', ['image']), '[image could not be downloaded]');
  assert.equal(
    appendMediaFailureNote('', ['image', 'document']),
    '[image could not be downloaded] [document could not be downloaded]',
  );
  console.log('  ✓ appendMediaFailureNote formats failure notes');
}

{
  // A throwing downloadMedia (expired CDN URL) must not reject out of
  // extractBridgeEvent — before this guard the whole upsert batch died and
  // the message was silently dropped.
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'img-fail-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: { imageMessage: { caption: '', mimetype: 'image/jpeg' } },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15551234567@s.whatsapp.net',
    senderNumber: '15551234567',
    downloadMedia: async () => { throw new Error('Failed to fetch stream from https://mmg.whatsapp.net/x'); },
    cacheDirs: { image: mkdtempSync(path.join(tmpdir(), 'wa-media-')) },
  });
  assert.equal(event.hasMedia, true);
  assert.equal(event.mediaUrls.length, 0);
  assert.equal(event.body, '[image could not be downloaded]');
  console.log('  ✓ failed media download is contained and surfaced in body');
}

{
  // Captioned message keeps the caption and appends the failure note.
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'doc-fail-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: false },
      messageTimestamp: 123,
      message: { documentMessage: { caption: 'see attached', fileName: 'q.pdf', mimetype: 'application/pdf' } },
    },
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15551234567@s.whatsapp.net',
    senderNumber: '15551234567',
    downloadMedia: async () => { throw new Error('boom'); },
    cacheDirs: { document: mkdtempSync(path.join(tmpdir(), 'wa-media-')) },
  });
  assert.equal(event.body, 'see attached\n[document could not be downloaded]');
  assert.equal(event.mediaUrls.length, 0);
  console.log('  ✓ captioned failed download keeps caption and appends note');
}

console.log('\n✅ All WhatsApp native bridge helper tests passed.');
