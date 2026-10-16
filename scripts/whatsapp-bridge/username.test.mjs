/**
 * Unit tests for WhatsApp @username targets and group member identity.
 *
 * Pure helpers only (bridge.js starts a server at import). The query builder
 * runs against Baileys' real USyncQuery/USyncUser, so these tests check the
 * stanza WhatsApp actually receives.
 *
 * Run: node scripts/whatsapp-bridge/username.test.mjs
 */

import { strict as assert } from 'node:assert';
import { USyncQuery, USyncUser } from '@whiskeysockets/baileys';

import {
  buildUsernameQuery,
  groupMembersFromMetadata,
  lidFromUsernameResult,
  parseWhatsAppUsername,
  resolveWhatsAppUsername,
} from './bridge_helpers.js';

const usync = { USyncQuery, USyncUser };

// -- parseWhatsAppUsername ------------------------------------------------
{
  assert.equal(parseWhatsAppUsername('@bykahwai'), 'bykahwai');
  assert.equal(parseWhatsAppUsername('  @lss.me  '), 'lss.me');
  assert.equal(parseWhatsAppUsername('@Some_User.9'), 'Some_User.9');
  console.log('  ✓ @username targets parse to the bare username');
}
{
  // JIDs, numbers and malformed handles are not usernames: they pass through.
  for (const value of [
    '278992619872335@lid', '60123456789@s.whatsapp.net', '120363001234567890@g.us',
    '+60123456789', 'bykahwai', '@ab', `@${'a'.repeat(36)}`, '@has space', '@a@b',
    '', null, undefined,
  ]) {
    assert.equal(parseWhatsAppUsername(value), null, String(value));
  }
  console.log('  ✓ JIDs, phone numbers and invalid handles are not usernames');
}

// -- buildUsernameQuery ---------------------------------------------------
{
  const query = buildUsernameQuery(usync, 'bykahwai');
  assert.deepEqual(query.protocols.map((p) => p.name), ['contact', 'lid']);
  assert.equal(query.users.length, 1);
  assert.equal(query.users[0].username, 'bykahwai');
  assert.equal(query.users[0].usernameKey, undefined);
  const contact = query.protocols[0].getUserElement(query.users[0]);
  assert.deepEqual(contact, { tag: 'contact', attrs: { username: 'bykahwai' } });
  console.log('  ✓ a username query asks for contact + lid by username');
}
{
  const query = buildUsernameQuery(usync, 'bykahwai', '4821');
  const contact = query.protocols[0].getUserElement(query.users[0]);
  assert.deepEqual(contact.attrs, { username: 'bykahwai', pin: '4821' });
  console.log('  ✓ the owner\'s PIN rides along as the contact pin');
}

// -- lidFromUsernameResult ------------------------------------------------
{
  assert.equal(
    lidFromUsernameResult({ list: [{ contact: true, id: '278992619872335@lid' }] }),
    '278992619872335@lid');
  assert.equal(
    lidFromUsernameResult({ list: [{ contact: true, id: '60123@s.whatsapp.net', lid: '111@lid' }] }),
    '111@lid');
  assert.equal(lidFromUsernameResult({ list: [] }), null);
  assert.equal(lidFromUsernameResult({ list: [{ contact: false, id: '1@lid' }] }), null);
  assert.equal(lidFromUsernameResult(undefined), null);
  console.log('  ✓ only a found contact yields an @lid');
}

// -- resolveWhatsAppUsername ----------------------------------------------
{
  const seen = [];
  const execute = async (query) => {
    seen.push(query);
    return { list: [{ contact: true, id: '19056887390267@lid' }] };
  };
  assert.equal(await resolveWhatsAppUsername(execute, usync, 'adlinakama'), '19056887390267@lid');
  assert.equal(seen.length, 1);
  assert.equal(seen[0].users[0].username, 'adlinakama');

  const nobody = async () => ({ list: [] });
  assert.equal(await resolveWhatsAppUsername(nobody, usync, 'nobody_here'), null);
  console.log('  ✓ resolve sends one query and returns the @lid, or null');
}

// -- groupMembersFromMetadata ---------------------------------------------
{
  const members = groupMembersFromMetadata([
    // LID-addressed group, number shared
    { id: '115191106797782@lid', phoneNumber: '60123456789@s.whatsapp.net', admin: 'admin' },
    // number hidden behind a username
    { id: '278992619872335@lid', username: 'bykahwai', admin: null },
    // phone-addressed group: the id IS the number
    { id: '60198765432@s.whatsapp.net', lid: '222@lid' },
  ]);
  assert.deepEqual(members, [
    { id: '115191106797782@lid', phone: '60123456789@s.whatsapp.net', username: null, admin: 'admin' },
    { id: '278992619872335@lid', phone: null, username: 'bykahwai', admin: null },
    { id: '60198765432@s.whatsapp.net', phone: '60198765432@s.whatsapp.net', username: null, admin: null },
  ]);
  assert.deepEqual(groupMembersFromMetadata(undefined), []);
  console.log('  ✓ group members carry phone (null when hidden), username and admin');
}

console.log('username tests passed');
