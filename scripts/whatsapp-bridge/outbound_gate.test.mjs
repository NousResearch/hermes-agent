/**
 * Unit tests for the outbound recipient gate (outbound_gate.js).
 *
 * Same convention as bridge.native.test.mjs: pure Node test runner, no
 * server/Baileys boot — the gate factory takes env, sessionDir and an
 * account getter so everything is faked here.
 */

import { strict as assert } from 'node:assert';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import { createOutboundGate, expandOutboundAliases } from './outbound_gate.js';

const ACCOUNT = { id: '33600000000:13@s.whatsapp.net', lid: '40000000000000:13@lid' };
const SELF_PHONE = '33600000000';
const SELF_LID = '40000000000000';
const CONTACT = '33699999999'; // some contact of the linked account
const CONTACT_LID = '44999999999999';

function makeSessionDir() {
  const dir = mkdtempSync(path.join(tmpdir(), 'wa-gate-'));
  // phone -> lid and lid -> phone twins, in the Baileys multi-file shape
  writeFileSync(path.join(dir, `lid-mapping-${CONTACT}.json`), JSON.stringify(CONTACT_LID));
  writeFileSync(path.join(dir, `lid-mapping-${CONTACT_LID}_reverse.json`), JSON.stringify(CONTACT));
  writeFileSync(path.join(dir, `lid-mapping-${SELF_LID}_reverse.json`), JSON.stringify(SELF_PHONE));
  return dir;
}

function gate(overrides = {}, account = ACCOUNT) {
  const env = {
    WHATSAPP_ALLOWED_USERS: `+${SELF_PHONE}`,
    WHATSAPP_HOME_CHANNEL: `${SELF_LID}@lid`,
    ...overrides,
  };
  return createOutboundGate({ env, sessionDir: makeSessionDir(), getAccount: () => account });
}

test('linked account passes as phone, LID, or with device suffix', () => {
  const isAllowed = gate();
  assert.equal(isAllowed(`${SELF_PHONE}@s.whatsapp.net`), true);
  assert.equal(isAllowed(`${SELF_PHONE}:13@s.whatsapp.net`), true);
  assert.equal(isAllowed(`${SELF_LID}@lid`), true);
  assert.equal(isAllowed(`+${SELF_PHONE}`), true);
});

test('home channel LID passes even when numeric allowlist only carries the phone', () => {
  assert.equal(gate()(`${SELF_LID}@lid`), true);
});

test('arbitrary contact is blocked in every identity format', () => {
  const isAllowed = gate();
  assert.equal(isAllowed(`${CONTACT}@s.whatsapp.net`), false);
  assert.equal(isAllowed(`${CONTACT_LID}@lid`), false);
  assert.equal(isAllowed(`+${CONTACT}`), false);
});

test('group and broadcast JIDs are blocked by default', () => {
  const isAllowed = gate();
  assert.equal(isAllowed('120363001122334455@g.us'), false);
  assert.equal(isAllowed('status@broadcast'), false);
});

test('explicit WHATSAPP_OUTBOUND_ALLOWED opens single chats', () => {
  const isAllowed = gate({ WHATSAPP_OUTBOUND_ALLOWED: `${CONTACT}@s.whatsapp.net` });
  assert.equal(isAllowed(`${CONTACT}@s.whatsapp.net`), true);
  assert.equal(isAllowed(`${CONTACT_LID}@lid`), true, 'LID twin resolves through the session mapping');
  assert.equal(isAllowed('33611111111@s.whatsapp.net'), false);
});

test('outbound groups need an explicit allow entry', () => {
  const isAllowed = gate({ WHATSAPP_OUTBOUND_ALLOWED: '120363001122334455@g.us' });
  assert.equal(isAllowed('120363001122334455@g.us'), true);
  assert.equal(isAllowed('120363009999999999@g.us'), false);
});

test('groups already inbound-allowlisted may receive replies', () => {
  const isAllowed = gate({ WHATSAPP_GROUP_ALLOWED_USERS: '120363001122334455@g.us' });
  assert.equal(isAllowed('120363001122334455@g.us'), true);
  assert.equal(isAllowed('120363009999999999@g.us'), false);
});

test('wildcard and escape hatch', () => {
  assert.equal(gate({ WHATSAPP_OUTBOUND_ALLOWED: '*' })(`${CONTACT}@s.whatsapp.net`), true);
  assert.equal(gate({ WHATSAPP_OUTBOUND_ALLOW_ALL: '1' })(`${CONTACT}@s.whatsapp.net`), true);
});

test('empty/garbage chatId is denied', () => {
  const isAllowed = gate();
  assert.equal(isAllowed(undefined), false);
  assert.equal(isAllowed(''), false);
  assert.equal(isAllowed('status@broadcast'), false);
});

test('gate fails closed when no allowlist config exists at all', () => {
  const isAllowed = createOutboundGate({
    env: {},
    sessionDir: makeSessionDir(),
    getAccount: () => ACCOUNT,
  });
  assert.equal(isAllowed(`${SELF_PHONE}@s.whatsapp.net`), true, 'linked account itself still allowed');
  assert.equal(isAllowed(`${CONTACT}@s.whatsapp.net`), false);
});

test('account unavailable before connect only allows explicit list entries', () => {
  const isAllowed = gate({}, null);
  assert.equal(isAllowed(`${SELF_PHONE}@s.whatsapp.net`), true, 'via WHATSAPP_ALLOWED_USERS');
  assert.equal(isAllowed(`${CONTACT}@s.whatsapp.net`), false);
});

test('expandOutboundAliases resolves phone<->lid twins both ways', () => {
  const dir = makeSessionDir();
  assert.ok(expandOutboundAliases(`${CONTACT}@s.whatsapp.net`, dir).has(CONTACT_LID));
  assert.ok(expandOutboundAliases(`${CONTACT_LID}@lid`, dir).has(CONTACT));
});
