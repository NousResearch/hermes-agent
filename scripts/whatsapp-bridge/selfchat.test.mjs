import test from 'node:test';
import assert from 'node:assert/strict';

import { deriveSelfIdentifiers, isOwnSelfChat } from './selfchat.js';

test('deriveSelfIdentifiers strips device suffix and host from id and lid', () => {
  const self = deriveSelfIdentifiers({
    id: '923344176856:58@s.whatsapp.net',
    lid: '158939576553626:58@lid',
  });
  assert.deepEqual(self, { myNumber: '923344176856', myLid: '158939576553626' });
});

test('deriveSelfIdentifiers tolerates missing fields', () => {
  assert.deepEqual(deriveSelfIdentifiers(null), { myNumber: '', myLid: '' });
  assert.deepEqual(deriveSelfIdentifiers({}), { myNumber: '', myLid: '' });
});

test('isOwnSelfChat accepts the user\'s own number JID', () => {
  const self = { myNumber: '923344176856', myLid: '158939576553626' };
  assert.equal(isOwnSelfChat('923344176856@s.whatsapp.net', self), true);
});

test('isOwnSelfChat accepts the user\'s own LID JID (regression: fromMe=false self-chat)', () => {
  // WhatsApp delivers the user's own self-chat addressed by LID with
  // fromMe=false; recognising it by LID is the core of the bug fix.
  const self = { myNumber: '923344176856', myLid: '158939576553626' };
  assert.equal(isOwnSelfChat('158939576553626@lid', self), true);
});

test('isOwnSelfChat rejects a stranger DM', () => {
  const self = { myNumber: '923344176856', myLid: '158939576553626' };
  assert.equal(isOwnSelfChat('19175395595@s.whatsapp.net', self), false);
});

test('isOwnSelfChat rejects groups and status broadcasts', () => {
  const self = { myNumber: '923344176856', myLid: '158939576553626' };
  assert.equal(isOwnSelfChat('120363024639251251@g.us', self), false);
  assert.equal(isOwnSelfChat('status@broadcast', self), false);
});

test('isOwnSelfChat handles empty inputs safely', () => {
  assert.equal(isOwnSelfChat('', { myNumber: '923344176856' }), false);
  assert.equal(isOwnSelfChat('923344176856@s.whatsapp.net', {}), false);
});
