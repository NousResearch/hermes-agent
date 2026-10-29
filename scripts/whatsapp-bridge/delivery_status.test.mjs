/**
 * Delivery/read status events for messages Hermes sent (logged to gateway.log by the adapter).
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { proto } from '@whiskeysockets/baileys';

import { deliveryStatusEvent } from './bridge_helpers.js';

const Status = proto.WebMessageInfo.Status;
const own = new Set(['sent-1']);
const isOwnSend = id => own.has(id);
const dmKey = { id: 'sent-1', remoteJid: '15551234567@s.whatsapp.net', fromMe: true };

test('each 1:1 status change of our own send becomes one event', () => {
  const names = [];
  for (const status of [Status.SERVER_ACK, Status.DELIVERY_ACK, Status.READ, Status.PLAYED]) {
    const event = deliveryStatusEvent({ key: dmKey, update: { status, messageTimestamp: 1760000000 }, isOwnSend });
    assert.equal(event.event, 'delivery_status');
    assert.equal(event.messageId, 'sent-1');
    assert.equal(event.chatId, '15551234567@s.whatsapp.net');
    assert.equal(event.timestamp, 1760000000);
    names.push(event.status);
  }
  assert.deepEqual(names, ['server_ack', 'delivered', 'read', 'played']);
});

test('a failed send carries the error code', () => {
  const event = deliveryStatusEvent({
    key: dmKey, update: { status: Status.ERROR, messageStubParameters: ['463'] }, isOwnSend,
  });
  assert.equal(event.status, 'error');
  assert.equal(event.error, '463');
});

test('group receipts name the recipient and the furthest state', () => {
  const key = { id: 'sent-1', remoteJid: '120363001234567890@g.us', fromMe: true };
  const read = deliveryStatusEvent({
    key, receipt: { userJid: '15550001111@s.whatsapp.net', readTimestamp: 1760000100 }, isOwnSend,
  });
  assert.deepEqual(
    { status: read.status, recipientId: read.recipientId, timestamp: read.timestamp },
    { status: 'read', recipientId: '15550001111@s.whatsapp.net', timestamp: 1760000100 },
  );
  const delivered = deliveryStatusEvent({
    key, receipt: { userJid: '15550001111@s.whatsapp.net', receiptTimestamp: 1760000050 }, isOwnSend,
  });
  assert.equal(delivered.status, 'delivered');
});

test('nothing is reported for messages Hermes did not send', () => {
  const update = { status: Status.READ };
  // Typed by a human on the linked phone: fromMe but not one of our sends.
  assert.equal(deliveryStatusEvent({ key: { ...dmKey, id: 'human-typed' }, update, isOwnSend }), null);
  // Inbound message (another device of ours read it).
  assert.equal(deliveryStatusEvent({ key: { ...dmKey, fromMe: false }, update, isOwnSend }), null);
  // Poll vote / edit / stub updates carry no status.
  assert.equal(deliveryStatusEvent({ key: dmKey, update: { pollUpdates: [] }, isOwnSend }), null);
  assert.equal(deliveryStatusEvent({ key: dmKey, update: { status: Status.PENDING }, isOwnSend }), null);
  assert.equal(deliveryStatusEvent({ key: dmKey, receipt: { userJid: 'x@s.whatsapp.net' }, isOwnSend }), null);
});
