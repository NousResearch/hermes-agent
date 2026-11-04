import test from 'node:test';
import assert from 'node:assert/strict';

import { formatOutgoingMessage, isSelfChatId } from './self_chat_prefix.js';

const USER = { id: '447700900001:12@s.whatsapp.net', lid: '123456789012345:12@lid' };
const PREFIX = '☤ *Hermes Agent*\n────────────\n';

function format(chatId, { mode = 'self-chat', replyPrefix = PREFIX, user = USER } = {}) {
  return formatOutgoingMessage('hello', { mode, replyPrefix, chatId, user });
}

test('self-chat addressed by phone JID is recognised', () => {
  assert.equal(isSelfChatId('447700900001@s.whatsapp.net', USER), true);
});

test('self-chat addressed by LID is recognised', () => {
  assert.equal(isSelfChatId('123456789012345@lid', USER), true);
});

test('another contact is not the self-chat', () => {
  assert.equal(isSelfChatId('447700900999@s.whatsapp.net', USER), false);
  assert.equal(isSelfChatId('120363000000000000@g.us', USER), false);
});

test('missing chat or account identity is never the self-chat', () => {
  assert.equal(isSelfChatId('', USER), false);
  assert.equal(isSelfChatId('447700900001@s.whatsapp.net', undefined), false);
  assert.equal(isSelfChatId('@s.whatsapp.net', { id: '', lid: '' }), false);
});

test('self-chat mode prefixes replies in the self-chat', () => {
  assert.equal(format('447700900001@s.whatsapp.net'), `${PREFIX}hello`);
  assert.equal(format('123456789012345@lid'), `${PREFIX}hello`);
});

test('self-chat mode does not prefix sends to other contacts or groups', () => {
  assert.equal(format('447700900999@s.whatsapp.net'), 'hello');
  assert.equal(format('120363000000000000@g.us'), 'hello');
});

test('bot mode never prefixes', () => {
  assert.equal(format('447700900001@s.whatsapp.net', { mode: 'bot' }), 'hello');
});

test('an empty prefix disables prefixing', () => {
  assert.equal(format('447700900001@s.whatsapp.net', { replyPrefix: '' }), 'hello');
});
