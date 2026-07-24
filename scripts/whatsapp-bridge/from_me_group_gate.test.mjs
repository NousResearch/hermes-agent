import test from 'node:test';
import assert from 'node:assert/strict';

import { classifyFromMeGroupGate } from './from_me_group_gate.js';

test('non-fromMe is not applicable', () => {
  assert.equal(
    classifyFromMeGroupGate({ isGroup: true, fromMe: false }).action,
    'not_applicable',
  );
});

test('status broadcasts always drop', () => {
  assert.equal(
    classifyFromMeGroupGate({
      isGroup: false,
      fromMe: true,
      isStatus: true,
    }).action,
    'drop_status',
  );
});

test('fromMe DM is not applicable (handled by owner_message_gate)', () => {
  assert.equal(
    classifyFromMeGroupGate({
      isGroup: false,
      fromMe: true,
      mode: 'bot',
      forwardOwnerMessages: true,
    }).action,
    'not_applicable',
  );
});

test('bot mode drops group fromMe by default', () => {
  assert.equal(
    classifyFromMeGroupGate({
      isGroup: true,
      fromMe: true,
      mode: 'bot',
      forwardOwnerMessages: false,
    }).action,
    'drop_from_me_group',
  );
});

test('bot mode forwards group fromMe when FORWARD_OWNER_MESSAGES is on', () => {
  assert.equal(
    classifyFromMeGroupGate({
      isGroup: true,
      fromMe: true,
      mode: 'bot',
      forwardOwnerMessages: true,
    }).action,
    'forward_owner',
  );
});

test('self-chat mode always forwards group fromMe', () => {
  assert.equal(
    classifyFromMeGroupGate({
      isGroup: true,
      fromMe: true,
      mode: 'self-chat',
      forwardOwnerMessages: false,
    }).action,
    'forward_owner',
  );
});

test('echo of our own /send is dropped even when forward is enabled', () => {
  assert.equal(
    classifyFromMeGroupGate({
      isGroup: true,
      fromMe: true,
      mode: 'bot',
      forwardOwnerMessages: true,
      recentlySent: true,
    }).action,
    'drop_echo',
  );
});
