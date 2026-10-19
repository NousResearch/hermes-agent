import assert from 'node:assert/strict';
import test from 'node:test';

import { extractBridgeEvent } from './bridge_helpers.js';
import { createOutboundIdTracker, isOwnSendEcho } from './outbound_ids.js';
import { classifyOwnerMessageGate } from './owner_message_gate.js';
import { REACTION_SEND_TIMEOUT_MS, buildReactionPayload, registerReactionRoute } from './reaction.js';

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

test('buildReactionPayload uses empty text to remove a reaction', () => {
  const payload = buildReactionPayload({
    chatId: '123@s.whatsapp.net',
    messageId: 'msg-4',
    emoji: '',
    fromMe: false,
  });

  assert.equal(payload.react.text, '');
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
    trackSentMessageId: () => {},
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

test('registered /react route accepts empty emoji to remove a reaction', async () => {
  let routeHandler;
  const app = {
    post(_path, handler) {
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
    trackSentMessageId: () => {},
  });

  const res = responseRecorder();
  await routeHandler({
    body: {
      chatId: '123@s.whatsapp.net',
      messageId: 'msg-6',
      emoji: '',
      fromMe: false,
    },
  }, res);

  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.payload, { success: true });
  assert.equal(sendCalls.length, 1);
  assert.equal(sendCalls[0][1].react.text, '');
});

test('registerReactionRoute refuses to register without the sent-id tracker', () => {
  assert.throws(() => registerReactionRoute({ post() {} }, {
    getSocket: () => ({}),
    getConnectionState: () => 'connected',
    sendWithTimeout: async () => ({}),
  }), TypeError);
});

test('a sent reaction is recorded so its fromMe echo is dropped at intake', async () => {
  let routeHandler;
  const app = {
    post(_path, handler) {
      routeHandler = handler;
    },
  };
  const recentlySent = createOutboundIdTracker(16);
  registerReactionRoute(app, {
    getSocket: () => ({}),
    getConnectionState: () => 'connected',
    // Baileys resolves sendMessage with the WAMessage it sent; its key id is the
    // id the echo upsert carries.
    sendWithTimeout: async (chatId, payload) => ({
      key: { id: 'reaction-echo-1', remoteJid: chatId, fromMe: true },
      message: { reactionMessage: payload.react },
    }),
    trackSentMessageId: (sent) => recentlySent.remember(sent?.key?.id),
  });

  const res = responseRecorder();
  await routeHandler({
    body: { chatId: '123@s.whatsapp.net', messageId: 'msg-8', emoji: '👀', fromMe: true },
  }, res);
  assert.equal(res.statusCode, 200);

  const echo = await extractBridgeEvent({
    msg: {
      key: { id: 'reaction-echo-1', remoteJid: '123@s.whatsapp.net', fromMe: true },
      messageTimestamp: 124,
      message: {
        reactionMessage: { text: '👀', key: { id: 'msg-8', remoteJid: '123@s.whatsapp.net', fromMe: true } },
      },
    },
    chatId: '123@s.whatsapp.net',
    senderId: '123@s.whatsapp.net',
    senderNumber: '123',
  });
  assert.equal(echo.body, '[Reaction: 👀 to msg-8]');

  // Self-chat intake: the reply prefix never matches a reaction, so only the
  // recorded id stops the echo from becoming a new turn.
  assert.equal(isOwnSendEcho({
    fromMe: true,
    messageId: echo.messageId,
    body: echo.body,
    replyPrefix: '☤ *Hermes Agent*\n────────────\n',
    recentlySent,
  }), true);
  // Bot mode with WHATSAPP_FORWARD_OWNER_MESSAGES: the owner gate drops it as an echo.
  assert.deepEqual(classifyOwnerMessageGate({
    fromMe: true,
    fromOwnerEnabled: true,
    recentlySent,
    allowlistMatches: () => true,
    messageId: echo.messageId,
    chatId: echo.chatId,
  }), { action: 'drop_echo' });
});

test('the /react route sends with a short deadline that is dropped once expired', async () => {
  let routeHandler;
  const app = {
    post(_path, handler) {
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
    trackSentMessageId: () => {},
  });

  await routeHandler({
    body: { chatId: '123@s.whatsapp.net', messageId: 'msg-9', emoji: '👀' },
  }, responseRecorder());

  // A stalled reaction may hold the shared send queue for this long at most,
  // not for the 60 s default that real replies use.
  assert.equal(sendCalls.length, 1);
  assert.equal(sendCalls[0][3], REACTION_SEND_TIMEOUT_MS);
  assert.ok(REACTION_SEND_TIMEOUT_MS < 5000);
  assert.deepEqual(sendCalls[0][4], { skipIfExpired: true });
});
