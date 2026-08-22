import test from 'node:test';
import assert from 'node:assert/strict';

import { outboundUnreadTarget } from './bridge_helpers.js';

function makeSentMessage(id = 'sent-1') {
  return { key: { id, remoteJid: '15551234567@s.whatsapp.net', fromMe: true } };
}

test('marks unread only in self-chat mode when enabled', () => {
  const sentMessage = makeSentMessage();
  assert.equal(
    outboundUnreadTarget({ mode: 'self-chat', enabled: true, sentMessage }),
    sentMessage,
  );
});

test('bot mode never marks unread, even when enabled', () => {
  const sentMessage = makeSentMessage();
  assert.equal(
    outboundUnreadTarget({ mode: 'bot', enabled: true, sentMessage }),
    null,
  );
});

test('disabled by default preserves current behavior', () => {
  const sentMessage = makeSentMessage();
  assert.equal(
    outboundUnreadTarget({ mode: 'self-chat', enabled: false, sentMessage }),
    null,
  );
});

test('no-op when there is no sent message to target', () => {
  assert.equal(
    outboundUnreadTarget({ mode: 'self-chat', enabled: true, sentMessage: undefined }),
    null,
  );
  assert.equal(
    outboundUnreadTarget({ mode: 'self-chat', enabled: true, sentMessage: {} }),
    null,
  );
});
