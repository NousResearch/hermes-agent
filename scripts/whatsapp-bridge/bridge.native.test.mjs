/**
 * Unit tests for WhatsApp-native bridge payload helpers.
 *
 * These tests avoid importing bridge.js because that file starts an HTTP
 * server and Baileys socket at module load. Keep the helper module pure.
 */

import { strict as assert } from 'node:assert';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { getAggregateVotesInPollMessage } from '@whiskeysockets/baileys';

import {
  buildPollPayload,
  buildTextSendPayload,
  acknowledgeDeliveryReceipts,
  createDeliveryReceiptQueue,
  createOwnerMessageQueue,
  ownerMessageTokenMatches,
  ownerMessageDeliveryMode,
  ownerSendFenceStatus,
  providerSendErrorResponse,
  providerSendIntentStatus,
  createBoundedMessageStore,
  deliveryReceiptFromMessageUpdate,
  deliveryReceiptFromUserReceiptUpdate,
  appendMediaFailureNote,
  extractBridgeEvent,
  inboundReadReceiptKeys,
  mediaPayloadForFile,
  pollCreationMessageFromPayload,
  pollUpdateForAggregation,
} from './bridge_helpers.js';

// -- provider-send intent policy ------------------------------------------
{
  const base = { expectedSecret: 'secret-value', authorizationHeader: undefined };
  assert.equal(providerSendIntentStatus({ ...base, externalConsumer: false }), 'standard');
  assert.equal(providerSendIntentStatus({
    ...base, externalConsumer: true, sendIntent: 'automatic', expectedOwnerFenceSequence: 0,
  }), 'automatic');
  assert.equal(providerSendIntentStatus({
    ...base, externalConsumer: true, sendIntent: 'automatic', expectedOwnerFenceSequence: undefined,
  }), 'invalid');
  assert.equal(providerSendIntentStatus({
    ...base, externalConsumer: true, sendIntent: 'human', authorizationHeader: 'Bearer secret-value',
  }), 'human');
  assert.equal(providerSendIntentStatus({
    ...base, externalConsumer: true, sendIntent: 'human', authorizationHeader: 'Bearer wrong',
  }), 'unauthorized');
  assert.equal(providerSendIntentStatus({ ...base, externalConsumer: true }), 'invalid');
  assert.deepEqual(providerSendErrorResponse(
    Object.assign(new Error('Owner intervention fenced this send'), { code: 'OWNER_FENCED' }),
    ['sent-1'],
  ), {
    statusCode: 409,
    body: {
      error: 'Owner intervention fenced this send',
      code: 'OWNER_FENCED',
      retryable: false,
      partial: true,
      messageId: 'sent-1',
      messageIds: ['sent-1'],
    },
  });
  assert.deepEqual(providerSendErrorResponse(
    Object.assign(new Error('queue expired'), { code: 'SEND_QUEUE_EXPIRED' }),
  ), {
    statusCode: 503,
    body: {
      error: 'queue expired',
      code: 'SEND_QUEUE_EXPIRED',
      retryable: true,
      partial: false,
      messageId: undefined,
      messageIds: [],
    },
  });
  console.log('  ✓ expired bridge queue waits return a retryable provider-safe response');
  console.log('  ✓ partial owner fences are terminal conflicts with sent IDs');
  console.log('  ✓ send intent fails closed only for external consumers and authenticates humans');
}

// -- owner-device messages -------------------------------------------------
{
  assert.equal(ownerMessageDeliveryMode(true, false), 'inbound');
  assert.equal(ownerMessageDeliveryMode(true, true), 'external');
  assert.equal(ownerMessageDeliveryMode(false, true), 'inbound');
  console.log('  ✓ generic owner forwarding and external durable consumption stay distinct');
}

{
  assert.equal(ownerMessageTokenMatches('secret-value', 'Bearer secret-value'), true);
  assert.equal(ownerMessageTokenMatches('secret-value', 'Bearer secret-valuE'), false);
  assert.equal(ownerMessageTokenMatches('secret-value', ''), false);
  assert.equal(ownerMessageTokenMatches('', 'Bearer secret-value'), false);
  console.log('  ✓ owner-device API bearer tokens use exact authenticated matching');
}

{
  const queueDir = mkdtempSync(path.join(tmpdir(), 'hermes-owner-queue-'));
  const queue = createOwnerMessageQueue({ directory: queueDir, pageSize: 1 });
  const first = {
    messageId: 'owner-1',
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15551234567@s.whatsapp.net',
    fromOwner: true,
    type: 'text',
    body: 'Assumi o atendimento.',
    timestamp: 1_723_636_800,
  };
  assert.equal(queue.add(first), true);
  assert.equal(queue.add({ ...first }), false);
  const firstQueued = { ...first, ownerFenceSequence: 1 };
  assert.deepEqual(queue.snapshot(), [firstQueued]);
  const reopened = createOwnerMessageQueue({ directory: queueDir, pageSize: 1 });
  assert.deepEqual(reopened.snapshot(), [firstQueued]);
  assert.equal(reopened.lastSequence(first.chatId), 1);
  assert.match(readFileSync(path.join(queueDir, 'owner-messages.json'), 'utf8'), /owner-1/);
  assert.equal(reopened.acknowledge([{ messageId: 'owner-1' }]), 1);
  const afterAck = createOwnerMessageQueue({ directory: queueDir });
  assert.deepEqual(afterAck.snapshot(), []);
  assert.equal(afterAck.lastSequence(first.chatId), 1);
  console.log('  ✓ owner-device messages persist across restart until exact acknowledgement');
}

{
  const malformedJsonDir = mkdtempSync(path.join(tmpdir(), 'hermes-owner-malformed-json-'));
  writeFileSync(path.join(malformedJsonDir, 'owner-messages.json'), '{not-json');
  assert.throws(
    () => createOwnerMessageQueue({ directory: malformedJsonDir }),
    SyntaxError,
  );

  const invalidSchemaDir = mkdtempSync(path.join(tmpdir(), 'hermes-owner-invalid-schema-'));
  writeFileSync(path.join(invalidSchemaDir, 'owner-messages.json'), JSON.stringify({
    nextSequence: 2,
    entries: [{ sequence: 1, event: { messageId: 'owner-lost' } }],
    lastSequenceByChat: {},
    unresolvedLidFences: {},
    tombstones: {},
  }));
  assert.throws(
    () => createOwnerMessageQueue({ directory: invalidSchemaDir }),
    /Invalid owner message queue state/,
  );

  const invalidNestedMapDir = mkdtempSync(path.join(tmpdir(), 'hermes-owner-invalid-map-'));
  writeFileSync(path.join(invalidNestedMapDir, 'owner-messages.json'), JSON.stringify({
    nextSequence: 1,
    entries: [],
    lastSequenceByChat: { '15551234567@s.whatsapp.net': '1' },
    unresolvedLidFences: {},
    tombstones: {},
  }));
  assert.throws(
    () => createOwnerMessageQueue({ directory: invalidNestedMapDir }),
    /Invalid owner message queue state/,
  );
  console.log('  ✓ readable corrupt owner queue state fails closed instead of resetting');
}

{
  const queueDir = mkdtempSync(path.join(tmpdir(), 'hermes-owner-tombstone-'));
  let currentTime = 1_723_636_800_000;
  const retentionMs = 30 * 24 * 60 * 60 * 1000;
  const options = { directory: queueDir, now: () => currentTime, tombstoneRetentionMs: retentionMs };
  const event = {
    messageId: 'owner-replayed-after-ack',
    chatId: '15551234567@s.whatsapp.net',
    senderId: '15551234567@s.whatsapp.net',
    fromOwner: true,
    type: 'text',
    body: 'Handled already.',
    timestamp: 1_723_636_800,
  };
  const queue = createOwnerMessageQueue(options);
  assert.equal(queue.add(event), true);
  assert.equal(queue.acknowledge([{ messageId: event.messageId }]), 1);
  const persisted = JSON.parse(readFileSync(path.join(queueDir, 'owner-messages.json'), 'utf8'));
  assert.equal(persisted.tombstones[event.messageId].sequence, 1);

  const reopened = createOwnerMessageQueue(options);
  assert.equal(reopened.add(event), false);
  assert.equal(reopened.lastSequence(event.chatId), 1);
  assert.deepEqual(reopened.snapshot(), []);

  currentTime += retentionMs + 1;
  const afterReplayWindow = createOwnerMessageQueue(options);
  assert.equal(afterReplayWindow.add(event), true);
  assert.equal(afterReplayWindow.snapshot()[0].ownerFenceSequence, 2);
  console.log('  ✓ ACK tombstones survive restart, preserve sequence, and expire only by replay age');
}

{
  const queue = createOwnerMessageQueue({
    directory: mkdtempSync(path.join(tmpdir(), 'hermes-owner-fence-')),
  });
  const chatId = '15551234567@s.whatsapp.net';
  assert.equal(ownerSendFenceStatus(queue, chatId, 0), 'allowed');
  assert.equal(queue.add({
    messageId: 'owner-fence-1', chatId, senderId: chatId, fromOwner: true,
    type: 'text', body: 'Assumi.', timestamp: 1_723_636_800,
  }), true);
  assert.equal(ownerSendFenceStatus(queue, chatId, 0), 'fenced');
  assert.equal(ownerSendFenceStatus(queue, chatId, 1), 'allowed');
  assert.equal(ownerSendFenceStatus(queue, chatId, 2), 'fenced');
  assert.equal(ownerSendFenceStatus(queue, chatId, undefined), 'invalid');
  console.log('  ✓ owner sequence fences stale automatic sends before provider dispatch');
}

{
  const queueDir = mkdtempSync(path.join(tmpdir(), 'hermes-owner-alias-fence-'));
  const queue = createOwnerMessageQueue({ directory: queueDir });
  const lidChatId = '267383306489914@lid';
  const canonicalChatId = '19175395595@s.whatsapp.net';
  const base = {
    senderId: lidChatId, fromOwner: true, type: 'text', timestamp: 1_723_636_800,
  };

  assert.equal(queue.add({
    ...base, messageId: 'owner-alias-lid', chatId: lidChatId, body: 'LID first',
  }), true);
  assert.equal(ownerSendFenceStatus(queue, canonicalChatId, 0), 'fenced');
  assert.equal(ownerSendFenceStatus(queue, '15550001111@s.whatsapp.net', 0), 'fenced');
  assert.equal(ownerSendFenceStatus(
    createOwnerMessageQueue({ directory: queueDir }), canonicalChatId, 0,
  ), 'fenced');
  assert.equal(queue.acknowledge([{ messageId: 'owner-alias-lid' }]), 1);
  assert.equal(ownerSendFenceStatus(
    createOwnerMessageQueue({ directory: queueDir }), '15550001111@s.whatsapp.net', 0,
  ), 'fenced');

  // Baileys can persist this exact mapping only after the owner event arrives.
  writeFileSync(
    path.join(queueDir, 'lid-mapping-19175395595.json'),
    JSON.stringify('267383306489914'),
  );
  writeFileSync(
    path.join(queueDir, 'lid-mapping-267383306489914_reverse.json'),
    JSON.stringify('19175395595'),
  );

  assert.equal(ownerSendFenceStatus(queue, canonicalChatId, 0), 'fenced');
  assert.equal(ownerSendFenceStatus(queue, canonicalChatId, 1), 'allowed');
  assert.equal(ownerSendFenceStatus(queue, canonicalChatId, 2), 'fenced');
  assert.equal(ownerSendFenceStatus(queue, '15550001111@s.whatsapp.net', 0), 'allowed');
  assert.equal(queue.add({
    ...base,
    messageId: 'owner-alias-canonical',
    chatId: canonicalChatId,
    senderId: canonicalChatId,
    body: 'canonical second',
  }), true);
  assert.equal(ownerSendFenceStatus(queue, lidChatId, 1), 'fenced');
  assert.equal(ownerSendFenceStatus(queue, lidChatId, 2), 'allowed');
  assert.equal(ownerSendFenceStatus(queue, lidChatId, 3), 'fenced');

  assert.equal(queue.acknowledge([
    { messageId: 'owner-alias-lid' },
    { messageId: 'owner-alias-canonical' },
  ]), 1);
  const reopened = createOwnerMessageQueue({ directory: queueDir });
  assert.equal(ownerSendFenceStatus(reopened, canonicalChatId, 2), 'allowed');
  assert.equal(ownerSendFenceStatus(reopened, lidChatId, 2), 'allowed');
  console.log('  ✓ owner fences use exact persisted LID/canonical aliases across mapping, ACK, and restart');
}

{
  const queue = createOwnerMessageQueue({
    directory: mkdtempSync(path.join(tmpdir(), 'hermes-owner-invalid-')),
    pageSize: 2,
  });
  assert.equal(queue.add({ messageId: 'bad', fromOwner: false }), false);
  assert.equal(queue.add({
    messageId: 'owner-group', chatId: '123@g.us', senderId: '123@g.us', fromOwner: true,
    type: 'text', body: 'no', timestamp: 1,
  }), false);
  assert.equal(queue.add({
    messageId: 'owner-image', chatId: '15551234567@s.whatsapp.net',
    senderId: '15551234567@s.whatsapp.net', fromOwner: true,
    hasMedia: true, mediaType: 'image', body: '', timestamp: 1,
  }), true);
  assert.deepEqual(queue.snapshot(), [{
    messageId: 'owner-image', chatId: '15551234567@s.whatsapp.net',
    senderId: '15551234567@s.whatsapp.net', fromOwner: true,
    type: 'image', body: '', timestamp: 1,
    ownerFenceSequence: 1,
    disposition: 'unsupported_media', requiresIntervention: true,
  }]);
  assert.equal(queue.size(), 1);
  console.log('  ✓ owner-device queue rejects non-owner/non-personal and quarantines media');
}

{
  const queue = createOwnerMessageQueue({
    directory: mkdtempSync(path.join(tmpdir(), 'hermes-owner-pages-')),
    pageSize: 1,
  });
  const base = {
    chatId: '15551234567@s.whatsapp.net', senderId: '15551234567@s.whatsapp.net',
    fromOwner: true, type: 'text', timestamp: 1,
  };
  assert.equal(queue.add({ ...base, messageId: 'page-1', body: 'first' }), true);
  assert.equal(queue.add({ ...base, messageId: 'page-2', body: 'second' }), true);
  assert.equal(queue.add({ ...base, messageId: 'page-3', body: 'x'.repeat(25_000) }), true);
  const firstPage = queue.page();
  assert.equal(firstPage.messages[0].messageId, 'page-1');
  assert.equal(firstPage.hasMore, true);
  const secondPage = queue.page({ cursor: firstPage.nextCursor });
  assert.equal(secondPage.messages[0].messageId, 'page-2');
  const thirdPage = queue.page({ cursor: secondPage.nextCursor });
  assert.equal(thirdPage.messages[0].messageId, 'page-3');
  assert.equal(thirdPage.messages[0].body.length, 25_000);
  assert.equal(thirdPage.hasMore, false);
  console.log('  ✓ owner-device cursor reaches later and oversized text without ACKing earlier pages');
}

// -- outbound delivery/read receipts --------------------------------------
{
  const key = { id: 'outbound-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: true };
  const now = () => '2026-08-14T12:00:00.000Z';

  assert.deepEqual(deliveryReceiptFromMessageUpdate({ key, update: { status: 2 }, now }), {
    messageId: 'outbound-1',
    status: 'sent',
    occurredAt: '2026-08-14T12:00:00.000Z',
  });
  assert.equal(
    deliveryReceiptFromMessageUpdate({ key, update: { status: 3 }, now }).status,
    'delivered',
  );
  assert.equal(
    deliveryReceiptFromMessageUpdate({ key, update: { status: 4 }, now }).status,
    'read',
  );
  assert.equal(
    deliveryReceiptFromMessageUpdate({ key, update: { status: 5 }, now }).status,
    'read',
  );
  assert.equal(
    deliveryReceiptFromMessageUpdate({ key: { ...key, fromMe: false }, update: { status: 4 }, now }),
    null,
  );
  assert.equal(
    deliveryReceiptFromMessageUpdate({
      key: { ...key, remoteJid: '120363001234567890@g.us' },
      update: { status: 4 },
      now,
    }),
    null,
  );
  assert.equal(
    deliveryReceiptFromMessageUpdate({
      key: { ...key, remoteJid: 'status@broadcast' },
      update: { status: 4 },
      now,
    }),
    null,
  );
  assert.equal(
    deliveryReceiptFromUserReceiptUpdate({
      key: { ...key, remoteJid: 'status@broadcast' },
      receipt: { readTimestamp: 1_723_636_900 },
      now,
    }),
    null,
  );
  assert.equal(
    deliveryReceiptFromMessageUpdate({ key: { ...key, id: '' }, update: { status: 4 }, now }),
    null,
  );
  assert.equal(deliveryReceiptFromMessageUpdate({ key, update: { status: 1 }, now }), null);

  assert.deepEqual(
    deliveryReceiptFromUserReceiptUpdate({
      key,
      receipt: { receiptTimestamp: 1_723_636_800 },
      now,
    }),
    {
      messageId: 'outbound-1',
      status: 'delivered',
      occurredAt: '2024-08-14T12:00:00.000Z',
    },
  );
  assert.equal(
    deliveryReceiptFromUserReceiptUpdate({
      key,
      receipt: { receiptTimestamp: 1, readTimestamp: 1_723_636_900 },
      now,
    }).status,
    'read',
  );
  assert.equal(
    deliveryReceiptFromUserReceiptUpdate({ key, receipt: {}, now }),
    null,
  );
  assert.equal(
    deliveryReceiptFromUserReceiptUpdate({
      key: { ...key, remoteJid: '120363001234567890@g.us' },
      receipt: { readTimestamp: 1_723_636_900 },
      now,
    }),
    null,
  );
  console.log('  ✓ outbound delivery receipts are normalized without recipient identifiers');
}

{
  const queue = [
    { messageId: 'outbound-1', status: 'delivered', occurredAt: '2026-08-14T12:00:00Z' },
    { messageId: 'outbound-1', status: 'read', occurredAt: '2026-08-14T12:01:00Z' },
  ];
  const acknowledged = acknowledgeDeliveryReceipts(queue, [
    { messageId: 'outbound-1', status: 'delivered' },
  ]);
  assert.equal(acknowledged, 1);
  assert.deepEqual(queue, [
    { messageId: 'outbound-1', status: 'read', occurredAt: '2026-08-14T12:01:00Z' },
  ]);
  console.log('  ✓ receipts remain queued until their exact status is acknowledged');
}

{
  let currentTime = 1_723_636_800_000;
  const retentionMs = 30 * 24 * 60 * 60 * 1000;
  const receipts = createDeliveryReceiptQueue({
    capacity: 2,
    pageSize: 1,
    tombstoneRetentionMs: retentionMs,
    now: () => currentTime,
  });
  const delivered = {
    messageId: 'outbound-queue-1', status: 'delivered', occurredAt: '2026-08-14T12:00:00Z',
  };
  const read = {
    messageId: 'outbound-queue-1', status: 'read', occurredAt: '2026-08-14T12:01:00Z',
  };
  const other = {
    messageId: 'outbound-queue-2', status: 'sent', occurredAt: '2026-08-14T12:02:00Z',
  };
  assert.equal(receipts.add(delivered), true);
  assert.equal(receipts.add(delivered), false);
  assert.equal(receipts.add(read), true);
  assert.equal(receipts.add(other), true);
  assert.equal(receipts.size(), 2);
  assert.deepEqual(receipts.snapshot(), [read]);
  assert.equal(receipts.acknowledge([{ messageId: read.messageId, status: read.status }]), 1);
  assert.deepEqual(receipts.snapshot(), [other]);
  assert.equal(receipts.add(read), false);
  assert.equal(receipts.add(delivered), false);

  const monotonic = {
    messageId: 'outbound-monotonic', status: 'delivered', occurredAt: '2026-08-14T12:02:30Z',
  };
  assert.equal(receipts.add(monotonic), true);
  assert.equal(receipts.acknowledge([
    { messageId: monotonic.messageId, status: monotonic.status },
  ]), 1);
  assert.equal(receipts.add({
    ...monotonic, status: 'read', occurredAt: '2026-08-14T12:02:45Z',
  }), true);

  const unmatched = {
    messageId: 'outbound-unmatched-ack', status: 'sent', occurredAt: '2026-08-14T12:03:00Z',
  };
  assert.equal(receipts.acknowledge([
    { messageId: unmatched.messageId, status: unmatched.status },
  ]), 0);
  assert.equal(receipts.add(unmatched), true);

  const outOfOrder = createDeliveryReceiptQueue({ capacity: 4, pageSize: 4 });
  const lower = {
    messageId: 'outbound-out-of-order', status: 'sent', occurredAt: '2026-08-14T12:03:10Z',
  };
  const higher = {
    messageId: lower.messageId, status: 'read', occurredAt: '2026-08-14T12:03:20Z',
  };
  assert.equal(outOfOrder.add(lower), true);
  assert.equal(outOfOrder.add(higher), true);
  assert.equal(outOfOrder.acknowledge([
    { messageId: higher.messageId, status: higher.status },
  ]), 2);
  assert.deepEqual(outOfOrder.snapshot(), []);

  currentTime += retentionMs + 1;
  assert.equal(receipts.add(read), true);
  assert.equal(receipts.size(), 2);
  console.log('  ✓ receipt ACK tombstones suppress replay/regression, expire by age, and ignore unmatched ACKs');
}

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
