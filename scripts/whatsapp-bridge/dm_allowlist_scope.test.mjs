import test from 'node:test';
import assert from 'node:assert/strict';

import { shouldEnforceDmSenderAllowlist } from './dm_allowlist_scope.js';

test('DM stranger under allowlist policy is enforced', () => {
  assert.equal(
    shouldEnforceDmSenderAllowlist({
      isGroup: false,
      fromMe: false,
      dmPolicy: 'allowlist',
    }),
    true,
  );
});

test('group messages never run the DM sender allowlist', () => {
  assert.equal(
    shouldEnforceDmSenderAllowlist({
      isGroup: true,
      fromMe: false,
      dmPolicy: 'allowlist',
    }),
    false,
  );
});

test('pairing policy skips bridge allowlist (Python pairing handles it)', () => {
  assert.equal(
    shouldEnforceDmSenderAllowlist({
      isGroup: false,
      fromMe: false,
      dmPolicy: 'pairing',
    }),
    false,
  );
});

test('fromMe never runs the stranger DM allowlist', () => {
  assert.equal(
    shouldEnforceDmSenderAllowlist({
      isGroup: false,
      fromMe: true,
      dmPolicy: 'allowlist',
    }),
    false,
  );
});
