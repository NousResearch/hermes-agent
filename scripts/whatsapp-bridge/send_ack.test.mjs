// Real /send handler + real ack gate; only the WhatsApp network is fake.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';
import { EventEmitter } from 'node:events';
import { proto } from '@whiskeysockets/baileys';
const Status = proto.WebMessageInfo.Status;
const source = fs.readFileSync(new URL('./bridge.js', import.meta.url), 'utf8');
const gateModule = new URL('./send_ack.js', import.meta.url);
const createAckSender = fs.existsSync(gateModule) ? (await import(gateModule)).createAckSender : null;

async function runHandler(behavior = () => {}, { group = true, audit = null } = {}) {
  let handler, response, httpStatus = 200, calls = 0, refreshes = 0;
  const receipts = [], order = [];
  const chatId = group ? '120363406889865453@g.us' : '100@s.whatsapp.net';
  const sock = {
    user: { id: '100@s.whatsapp.net' }, ev: new EventEmitter(), ws: new EventEmitter(),
    groupMetadata: async () => { refreshes++; order.push('refresh'); return {}; },
  };
  const sendWithTimeout = async (chat, payload, options = {}) => {
    calls++; order.push('send');
    const key = { id: options.messageId || 'OFFLINE-PENDING', remoteJid: chat, fromMe: true };
    const override = await behavior({ sock, key, calls, Status });
    return override === undefined ? { key, status: Status.PENDING } : override;
  };
  const sender = createAckSender?.({
    getSocket: () => sock, send: sendWithTimeout,
    record: audit || (line => receipts.push(line)),
    metadataTimeoutMs: 20, sendTimeoutMs: 20, ackTimeoutMs: 20,
  });
  const context = {
    app: { post: (route, fn) => { assert.equal(route, '/send'); handler = fn; } },
    sock, connectionState: 'connected',
    splitLongMessage: text => [text], formatOutgoingMessage: text => text,
    buildTextSendPayload: text => ({ content: { text }, options: {} }),
    sendWithTimeout, sendWithServerAck: sender?.send,
    trackSentMessageId: () => {}, messageStore: { remember: () => {} },
    observeSendAck: sender?.observe, proto,
  };
  // Exercise the actual messages.update listener too (non-poll branch).
  const listenerStart = source.indexOf("  sock.ev.on('messages.update',");
  const listenerEnd = source.indexOf("  sock.ev.on('messages.upsert',", listenerStart);
  const rawListenerStart = source.indexOf("  sock.ws.on('CB:ack,class:message',");
  if (rawListenerStart !== -1) vm.runInNewContext(source.slice(rawListenerStart, listenerStart), context);
  vm.runInNewContext(source.slice(listenerStart, listenerEnd), context);
  const start = source.indexOf("app.post('/send',");
  vm.runInNewContext(source.slice(start, source.indexOf('// Edit a previously sent message', start)), context);
  const res = { status: value => { httpStatus = value; return res; }, json: value => { response = value; } };
  await handler({ body: { chatId, message: 'OFFLINE ONLY' } }, res);
  return { httpStatus, response, calls, refreshes, receipts, order };
}

test('real /send rejects PENDING-only local keys without any server ack', async () => {
  const result = await runHandler();
  assert.notEqual(result.response?.success, true, 'Live handler falsely reports success for PENDING without any server acknowledgement');
  assert.notEqual(result.httpStatus, 200);
  assert.equal(result.response.status, 'PENDING');
  assert.equal(result.calls, 2, 'one retry only');
  assert.deepEqual(result.order, ['refresh', 'send', 'refresh', 'send']);
  assert.equal(result.receipts.length, 2);
  assert.ok(result.receipts.every(line => line.error === 'ACK_TIMEOUT'));
  assert.equal(new Set(result.receipts.map(line => line.message_id)).size, 1, 'retry preserves protocol id');
});

function emit(sock, key, status) { sock.ev.emit('messages.update', [{ key, update: { status } }]); }

test('real server message ACK stanza becomes SERVER_ACK (Baileys rc13 group gap)', async () => {
  const result = await runHandler(({ sock, key }) => {
    sock.ws.emit('CB:ack,class:message', { attrs: { class: 'message', id: key.id, from: key.remoteJid } });
  });
  assert.equal(result.httpStatus, 200);
  assert.equal(result.response.status, 'SERVER_ACK');
  assert.equal(result.calls, 1);
});

test('server ACK stanza with an error must not become success', async () => {
  const result = await runHandler(({ sock, key }) => {
    sock.ws.emit('CB:ack,class:message', { attrs: { class: 'message', id: key.id, from: key.remoteJid, error: '479' } });
  });
  assert.equal(result.httpStatus, 502);
  assert.notEqual(result.response.status, 'SERVER_ACK');
});

test('ack before local return is retained by the real messages.update listener', async () => {
  const result = await runHandler(({ sock, key, Status }) => { emit(sock, key, Status.SERVER_ACK); });
  assert.equal(result.httpStatus, 200);
  assert.equal(result.response.success, true);
  assert.equal(result.response.status, 'SERVER_ACK');
  assert.equal(result.calls, 1);
  assert.equal(result.refreshes, 1);
  assert.equal(result.receipts[0].message_id, result.response.messageId);
  assert.equal(result.receipts[0].status, 'SERVER_ACK');
  assert.equal(result.receipts[0].error, null);
});

test('delayed delivery ack is awaited, not just the local key', async () => {
  let ackEmitted = false;
  const result = await runHandler(({ sock, key, Status }) => {
    setTimeout(() => { ackEmitted = true; emit(sock, key, Status.DELIVERY_ACK); }, 5);
  });
  assert.equal(ackEmitted, true);
  assert.equal(result.response.status, 'DELIVERY_ACK');
  assert.equal(result.calls, 1);
});

test('a local SERVER_ACK status without an update is not evidence', async () => {
  const result = await runHandler(({ key, Status }) => ({ key, status: Status.SERVER_ACK }));
  assert.equal(result.httpStatus, 502);
});

test('ERROR is not discarded as a falsy enum value', async () => {
  const result = await runHandler(({ sock, key, Status }) => { emit(sock, key, Status.ERROR); });
  assert.equal(result.httpStatus, 502);
  assert.equal(result.response.status, 'ERROR');
  assert.equal(result.calls, 2);
  assert.ok(result.receipts.every(line => line.error === 'SERVER_ERROR'));
});

test('retry after metadata refresh may server-ack with the same logical id', async () => {
  const result = await runHandler(({ sock, key, calls, Status }) => {
    if (calls === 2) emit(sock, key, Status.SERVER_ACK);
  });
  assert.equal(result.httpStatus, 200);
  assert.equal(result.calls, 2);
  assert.equal(result.refreshes, 2);
  assert.deepEqual(result.receipts.map(line => line.status), ['PENDING', 'SERVER_ACK']);
  assert.equal(new Set(result.receipts.map(line => line.message_id)).size, 1);
});

test('wrong chat, wrong id and incoming status cannot satisfy the gate', async () => {
  const result = await runHandler(({ sock, key, Status }) => {
    emit(sock, { ...key, remoteJid: 'other@g.us' }, Status.READ);
    emit(sock, { ...key, id: 'other' }, Status.SERVER_ACK);
    emit(sock, { ...key, fromMe: false }, Status.DELIVERY_ACK);
  });
  assert.equal(result.httpStatus, 502);
  assert.equal(result.response.status, 'PENDING');
});

test('missing local id fails closed even if an ack exists', async () => {
  const result = await runHandler(({ sock, key, Status }) => {
    emit(sock, key, Status.SERVER_ACK);
    return {};
  });
  assert.equal(result.httpStatus, 502);
  assert.equal(result.response.error, 'INVALID_MESSAGE_KEY');
});

test('non-group failure has no refresh and no retry', async () => {
  const result = await runHandler(() => {}, { group: false });
  assert.equal(result.httpStatus, 502);
  assert.equal(result.calls, 1);
  assert.equal(result.refreshes, 0);
});

test('audit failure is fail-closed without retrying an accepted send', async () => {
  const result = await runHandler(({ sock, key, Status }) => { emit(sock, key, Status.SERVER_ACK); }, {
    audit: () => { throw new Error('disk unavailable'); },
  });
  assert.equal(result.httpStatus, 502);
  assert.equal(result.response.error, 'ACK_AUDIT_FAILED');
  assert.equal(result.calls, 1);
});

test('durable receipt has only allowlisted metadata, not body or arbitrary error', async () => {
  const result = await runHandler(() => { throw new Error('secret body token'); });
  assert.equal(result.httpStatus, 502);
  assert.ok(result.receipts.every(line => line.error === 'SEND_FAILED'));
  assert.doesNotMatch(JSON.stringify(result.receipts), /secret|token|OFFLINE ONLY/);
  const { appendAckReceipt } = await import('./send_ack.js');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const dir = fs.mkdtempSync(join(tmpdir(), 'wa-ack-test-'));
  try {
    const file = join(dir, 'receipt.jsonl');
    appendAckReceipt(file, result.receipts[0]);
    assert.deepEqual(JSON.parse(fs.readFileSync(file, 'utf8')), result.receipts[0]);
    assert.equal(fs.statSync(file).mode & 0o777, 0o600);
  } finally { fs.rmSync(dir, { recursive: true }); }
});
