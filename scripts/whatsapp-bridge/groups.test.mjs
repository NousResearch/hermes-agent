import test from 'node:test';
import assert from 'node:assert/strict';

import { summarizeParticipatingGroups, inboundPolicyRejection, borderoWriteRejection } from './bridge_helpers.js';

test('summarizes only valid group JIDs and sorts without exposing participants', () => {
  const result = summarizeParticipatingGroups({
    '120363000000000002@g.us': { subject: 'Borderô Saldanha', participants: [{ id: 'secret' }] },
    '120363000000000001@g.us': { subject: 'Borderô UBBO', participants: [{ id: 'secret' }] },
    '123@s.whatsapp.net': { subject: 'DM', participants: [] },
    'bad': { subject: 'invalid', participants: [] },
  });

  assert.deepEqual(result, [
    { group_jid: '120363000000000001@g.us', name: 'Borderô UBBO' },
    { group_jid: '120363000000000002@g.us', name: 'Borderô Saldanha' },
  ]);
});

test('returns an empty list for malformed group metadata', () => {
  assert.deepEqual(summarizeParticipatingGroups(null), []);
  assert.deepEqual(summarizeParticipatingGroups({ nope: null }), []);
});

test('Borderô ingress is exact-group-only and all bridge writes are blocked', () => {
  const allowed = new Set(['120363000000000001@g.us', '120363000000000002@g.us']);
  assert.equal(inboundPolicyRejection({ borderoReadOnly: true, isGroup: false }), 'bordero_dm_disabled');
  assert.equal(inboundPolicyRejection({ borderoReadOnly: true, isGroup: true, groupJid: '999@g.us', allowedGroupJids: allowed }), 'bordero_group_not_allowlisted');
  assert.equal(inboundPolicyRejection({ borderoReadOnly: true, isGroup: true, groupJid: '120363000000000001@g.us', allowedGroupJids: allowed, mode: 'bot' }), null);
  for (const path of ['/send', '/edit', '/send-media', '/send-poll', '/send-location', '/typing', '/read']) {
    assert.equal(borderoWriteRejection({ enabled: true, method: 'POST', path }), 'bordero_read_only_writes_disabled');
  }
  assert.equal(borderoWriteRejection({ enabled: true, method: 'GET', path: '/messages' }), null);
});
