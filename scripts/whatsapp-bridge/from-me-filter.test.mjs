import test from 'node:test';
import assert from 'node:assert/strict';

import { shouldFilterFromMeMessage } from './from-me-filter.js';

const SELF_NUMBER = '34652029134';
const SELF_LID = '67427329167522';
const SELF_CHAT_DM = `${SELF_NUMBER}@s.whatsapp.net`;
const SELF_CHAT_LID_DM = `${SELF_LID}@lid`;
const FRIEND_DM = '15551234567@s.whatsapp.net';
const GROUP_CHAT = '120363001234567890@g.us';
const STATUS_BROADCAST = 'status@broadcast';

// --- Status broadcasts always dropped --------------------------------------

test('status broadcast fromMe is always dropped (self-chat mode)', () => {
  assert.equal(
    shouldFilterFromMeMessage({
      chatId: STATUS_BROADCAST,
      isGroup: false,
      mode: 'self-chat',
      myNumber: SELF_NUMBER,
      myLid: SELF_LID,
    }),
    true,
  );
});

test('status broadcast fromMe is always dropped (bot mode)', () => {
  assert.equal(
    shouldFilterFromMeMessage({
      chatId: STATUS_BROADCAST,
      isGroup: false,
      mode: 'bot',
    }),
    true,
  );
});

// --- Bot mode drops everything --------------------------------------------

test('bot mode drops all fromMe messages including groups', () => {
  assert.equal(
    shouldFilterFromMeMessage({ chatId: GROUP_CHAT, isGroup: true, mode: 'bot' }),
    true,
  );
  assert.equal(
    shouldFilterFromMeMessage({ chatId: FRIEND_DM, isGroup: false, mode: 'bot' }),
    true,
  );
  assert.equal(
    shouldFilterFromMeMessage({ chatId: SELF_CHAT_DM, isGroup: false, mode: 'bot' }),
    true,
  );
});

// --- Self-chat mode: groups must pass through (#20143) --------------------

test('self-chat mode forwards fromMe group messages to gateway (#20143)', () => {
  assert.equal(
    shouldFilterFromMeMessage({
      chatId: GROUP_CHAT,
      isGroup: true,
      mode: 'self-chat',
      myNumber: SELF_NUMBER,
      myLid: SELF_LID,
    }),
    false,
    'group fromMe must NOT be filtered — gateway group_policy handles routing',
  );
});

test('self-chat mode forwards fromMe group messages even without sock.user identity', () => {
  // Even if sock.user is missing (e.g. early connection), groups still pass.
  assert.equal(
    shouldFilterFromMeMessage({
      chatId: GROUP_CHAT,
      isGroup: true,
      mode: 'self-chat',
    }),
    false,
  );
});

// --- Self-chat mode: DM filtering preserved ------------------------------

test('self-chat mode forwards user own self-chat DM (classic format)', () => {
  assert.equal(
    shouldFilterFromMeMessage({
      chatId: SELF_CHAT_DM,
      isGroup: false,
      mode: 'self-chat',
      myNumber: SELF_NUMBER,
      myLid: SELF_LID,
    }),
    false,
  );
});

test('self-chat mode forwards user own self-chat DM (LID format)', () => {
  assert.equal(
    shouldFilterFromMeMessage({
      chatId: SELF_CHAT_LID_DM,
      isGroup: false,
      mode: 'self-chat',
      myNumber: SELF_NUMBER,
      myLid: SELF_LID,
    }),
    false,
  );
});

test('self-chat mode drops fromMe DMs to other contacts (echo prevention)', () => {
  assert.equal(
    shouldFilterFromMeMessage({
      chatId: FRIEND_DM,
      isGroup: false,
      mode: 'self-chat',
      myNumber: SELF_NUMBER,
      myLid: SELF_LID,
    }),
    true,
  );
});

test('self-chat mode drops fromMe DMs when sock.user identity is missing', () => {
  // Without identity, can't recognise self-chat — fail closed for DMs.
  assert.equal(
    shouldFilterFromMeMessage({
      chatId: SELF_CHAT_DM,
      isGroup: false,
      mode: 'self-chat',
      myNumber: '',
      myLid: '',
    }),
    true,
  );
});
