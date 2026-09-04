import test from 'node:test';
import assert from 'node:assert/strict';

import { summarizeParticipatingGroups } from './bridge_helpers.js';

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
