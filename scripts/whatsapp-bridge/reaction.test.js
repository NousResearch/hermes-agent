import assert from 'node:assert/strict';
import test from 'node:test';

import { extractBridgeEvent } from './bridge_helpers.js';
import { buildReactionPayload, registerReactionRoute } from './reaction.js';

function responseRecorder() {
  return {
    statusCode: 200,
    payload: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      this.payload = payload;
      return this;
    },
  };
}

test('buildReactionPayload targets a direct chat message', () => {
  const payload = buildReactionPayload({
    chatId: '123@s.whatsapp.net',
    messageId: 'msg-1',
    emoji: '👀',
    fromMe: false,
  });

  assert.deepEqual(payload, {
    react: {
      text: '👀',
      key: {
        remoteJid: '123@s.whatsapp.net',
        id: 'msg-1',
        fromMe: false,
      },
    },
  });
});

test('buildReactionPayload includes group participant when provided', () => {
  const payload = buildReactionPayload({
    chatId: '123@g.us',
    messageId: 'msg-2',
    emoji: '✅',
    senderId: '456@s.whatsapp.net',
    fromMe: false,
  });

  assert.equal(payload.react.key.participant, '456@s.whatsapp.net');
});

test('buildReactionPayload preserves fromMe for self-chat messages', () => {
  const payload = buildReactionPayload({
    chatId: '123@s.whatsapp.net',
    messageId: 'msg-3',
    emoji: '❌',
    fromMe: true,
  });

  assert.equal(payload.react.key.fromMe, true);
});

test('fromMe survives bridge event extraction into the reaction key', async () => {
  const event = await extractBridgeEvent({
    msg: {
      key: {
        id: 'owner-msg-1',
        remoteJid: '123@s.whatsapp.net',
        fromMe: true,
      },
      pushName: 'Owner',
      messageTimestamp: 123,
      message: { conversation: 'owner forwarded this' },
    },
    chatId: '123@s.whatsapp.net',
    senderId: '123@s.whatsapp.net',
    senderNumber: '123',
  });

  const payload = buildReactionPayload({
    chatId: event.chatId,
    messageId: event.messageId,
    emoji: '👀',
    senderId: event.senderId,
    fromMe: event.fromMe,
  });

  assert.equal(event.fromMe, true);
  assert.equal(payload.react.key.fromMe, true);
});

test('registered /react route sends through the injected serialized send path', async () => {
  let routePath;
  let routeHandler;
  const app = {
    post(path, handler) {
      routePath = path;
      routeHandler = handler;
    },
  };
  const sendCalls = [];
  registerReactionRoute(app, {
    getSocket: () => ({}),
    getConnectionState: () => 'connected',
    sendWithTimeout: async (...args) => {
      sendCalls.push(args);
    },
  });

  assert.equal(routePath, '/react');
  const res = responseRecorder();
  await routeHandler({
    body: {
      chatId: '123@g.us',
      messageId: 'msg-4',
      emoji: '✅',
      senderId: '456@s.whatsapp.net',
      fromMe: true,
    },
  }, res);

  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.payload, { success: true });
  assert.equal(sendCalls.length, 1);
  assert.equal(sendCalls[0][0], '123@g.us');
  assert.equal(sendCalls[0][1].react.key.fromMe, true);
  assert.equal(sendCalls[0][1].react.key.participant, '456@s.whatsapp.net');
});
