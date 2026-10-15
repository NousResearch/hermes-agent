import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import { proto } from '@whiskeysockets/baileys';
import { createBoundedMessageStore } from './bridge_helpers.js';
import { handleReactionRequest } from './reaction.js';

const store = createBoundedMessageStore();
const groupKey = { id: 'group-in', remoteJid: '123@g.us', participant: '555@lid', fromMe: false };
const ownKey = { id: 'own-dm', remoteJid: '111@s.whatsapp.net', fromMe: true };
const lidKey = { id: 'lid-dm', remoteJid: '999@lid', fromMe: false };
for (const key of [groupKey, ownKey, lidKey]) store.remember({ key, message: { conversation: 'hello' } });
store.remember({ key: { id: 'own-group', remoteJid: '123@g.us', fromMe: true },
  participant: '777@lid', message: { conversation: 'sent' } });
store.remember({ key: { id: 'unknown-group-author', remoteJid: '123@g.us', fromMe: false },
  message: { conversation: 'received' } });

async function request(body, { connected = true, aliases = new Set(), send = async () => {} } = {}) {
  const response = { statusCode: 200, status(code) { this.statusCode = code; return this; },
    json(value) { this.body = value; return this; } };
  const calls = [];
  await handleReactionRequest({ body }, response, {
    connected: () => connected, messageStore: store, expandAliases: () => aliases,
    send: async (...args) => { calls.push(args); return send(...args); },
  });
  return { response, calls };
}

{
  const { response, calls } = await request({ chatId: '123@g.us', messageId: 'group-in', emoji: '👍' });
  assert.equal(response.body.success, true);
  assert.deepEqual(calls, [['123@g.us', { react: { text: '👍', key: groupKey } }]]);
}
{
  const { response, calls } = await request({ chatId: '111@s.whatsapp.net', messageId: 'own-dm', emoji: '' });
  assert.equal(response.body.success, true);
  assert.deepEqual(calls, [['111@s.whatsapp.net', { react: { text: '', key: ownKey } }]]);
}
{
  const { response, calls } = await request({ chatId: '111@s.whatsapp.net', messageId: 'lid-dm', emoji: '❤️' });
  assert.equal(response.statusCode, 404);
  assert.equal(calls.length, 0);
}
{
  const { response, calls } = await request({ chatId: '111@s.whatsapp.net', messageId: 'missing', emoji: '❤️' });
  assert.equal(response.statusCode, 404);
  assert.equal(calls.length, 0);
}
{
  const { response, calls } = await request({ chatId: '111@s.whatsapp.net', messageId: 'lid-dm', emoji: '❤️' }, { aliases: new Set(['111', '999']) });
  assert.equal(response.body.success, true);
  assert.deepEqual(calls, [['999@lid', { react: { text: '❤️', key: lidKey } }]]);
}
{
  const { response, calls } = await request({ chatId: '123@g.us', messageId: 'own-group', emoji: '👍' });
  assert.equal(response.body.success, true);
  assert.equal(calls[0][1].react.key.participant, '777@lid');
  assert.equal(calls[0][1].react.key.fromMe, true);
  const nativeMessage = proto.Message.fromObject({ reactionMessage: calls[0][1].react });
  const decoded = proto.Message.decode(proto.Message.encode(nativeMessage).finish());
  assert.equal(decoded.reactionMessage.key.id, 'own-group');
  assert.equal(decoded.reactionMessage.key.remoteJid, '123@g.us');
  assert.equal(decoded.reactionMessage.key.participant, '777@lid');
  assert.equal(decoded.reactionMessage.key.fromMe, true);
  assert.equal(decoded.reactionMessage.text, '👍');
}
{
  const { response, calls } = await request({ chatId: '123@g.us', messageId: 'unknown-group-author', emoji: '👍' });
  assert.equal(response.statusCode, 404);
  assert.equal(calls.length, 0);
}
{
  const sameDigits = { id: 'same-digits', remoteJid: '111@lid', fromMe: false };
  store.remember({ key: sameDigits, message: { conversation: 'test' } });
  const { response, calls } = await request({ chatId: '111@s.whatsapp.net', messageId: 'same-digits', emoji: '👍' },
    { aliases: new Set(['111']) });
  assert.equal(response.statusCode, 404);
  assert.equal(calls.length, 0);
}
{
  const { response, calls } = await request({ chatId: '123@g.us', messageId: 'group-in', emoji: '👍' }, { connected: false });
  assert.equal(response.statusCode, 503);
  assert.equal(calls.length, 0);
}
{
  const tracked = [];
  const response = { statusCode: 200, status(code) { this.statusCode = code; return this; },
    json(value) { this.body = value; return this; } };
  await handleReactionRequest({ body: { chatId: '123@g.us', messageId: 'group-in', emoji: '👍' } }, response, {
    connected: () => true, messageStore: store, expandAliases: () => new Set(),
    send: async () => { throw new Error('offline'); }, trackSent: (sent) => tracked.push(sent),
  });
  assert.equal(response.statusCode, 500);
  assert.deepEqual(tracked, []);
}

// Exercise the same route handler over real loopback HTTP without a Baileys
// socket or account. The fake sender receives the exact native payload.
{
  const calls = [];
  const sentIds = [];
  const server = createServer(async (req, raw) => {
    if (req.method !== 'POST' || req.url !== '/reaction') { raw.writeHead(404).end(); return; }
    let body = '';
    for await (const chunk of req) body += chunk;
    const res = {
      statusCode: 200,
      status(code) { this.statusCode = code; return this; },
      json(value) { raw.writeHead(this.statusCode, { 'content-type': 'application/json' }).end(JSON.stringify(value)); },
    };
    await handleReactionRequest({ body: JSON.parse(body) }, res, {
      connected: () => true, messageStore: store, expandAliases: () => new Set(),
      send: async (...args) => { calls.push(args); return { key: { id: 'reaction-out' } }; },
      trackSent: (sent) => sentIds.push(sent.key.id),
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const url = `http://127.0.0.1:${server.address().port}/reaction`;
    const post = (messageId) => fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ chatId: '123@g.us', messageId, emoji: '😂' }) });
    assert.equal((await post('group-in')).status, 200);
    assert.equal((await post('own-dm')).status, 404);
    assert.deepEqual(calls, [['123@g.us', { react: { text: '😂', key: groupKey } }]]);
    assert.deepEqual(sentIds, ['reaction-out']);
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
}
