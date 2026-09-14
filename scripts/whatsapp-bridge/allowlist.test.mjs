import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';

import {
  expandWhatsAppIdentifiers,
  matchesAllowedMessage,
  matchesAllowedUser,
  normalizeWhatsAppIdentifier,
  parseAllowedUsers,
} from './allowlist.js';

// Synthetic fixtures only. Never put real phone numbers or LIDs in tests.
const TEST_PHONE = '15555550123';
const TEST_LID = '900000000000001';
const OTHER_PHONE = '15555550999';
const OTHER_LID = '900000000000002';

test('normalizeWhatsAppIdentifier strips jid syntax and plus prefix', () => {
  assert.equal(normalizeWhatsAppIdentifier(`+${TEST_PHONE}@s.whatsapp.net`), TEST_PHONE);
  assert.equal(normalizeWhatsAppIdentifier(`${TEST_LID}@lid`), TEST_LID);
  assert.equal(normalizeWhatsAppIdentifier(`${TEST_PHONE}:12@s.whatsapp.net`), TEST_PHONE);
});

test('expandWhatsAppIdentifiers resolves phone and lid aliases from session files', () => {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-allowlist-'));

  try {
    writeFileSync(path.join(sessionDir, `lid-mapping-${TEST_PHONE}.json`), JSON.stringify(TEST_LID));
    writeFileSync(path.join(sessionDir, `lid-mapping-${TEST_LID}_reverse.json`), JSON.stringify(TEST_PHONE));

    const aliases = expandWhatsAppIdentifiers(`${TEST_LID}@lid`, sessionDir);
    assert.deepEqual([...aliases].sort(), [TEST_PHONE, TEST_LID].sort());
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});

test('matchesAllowedUser accepts mapped lid sender when allowlist only contains phone number', () => {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-allowlist-'));

  try {
    writeFileSync(path.join(sessionDir, `lid-mapping-${TEST_PHONE}.json`), JSON.stringify(TEST_LID));
    writeFileSync(path.join(sessionDir, `lid-mapping-${TEST_LID}_reverse.json`), JSON.stringify(TEST_PHONE));

    const allowedUsers = parseAllowedUsers(`+${TEST_PHONE}`);
    assert.equal(matchesAllowedUser(`${TEST_LID}@lid`, allowedUsers, sessionDir), true);
    assert.equal(matchesAllowedUser(`${OTHER_LID}@lid`, allowedUsers, sessionDir), false);
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});

test('matchesAllowedUser treats * as allow-all wildcard', () => {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-allowlist-'));

  try {
    const allowedUsers = parseAllowedUsers('*');
    assert.equal(matchesAllowedUser(`${TEST_PHONE}@s.whatsapp.net`, allowedUsers, sessionDir), true);
    assert.equal(matchesAllowedUser(`${TEST_LID}@lid`, allowedUsers, sessionDir), true);
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});

test('matchesAllowedUser rejects everyone when allowlist is empty (#8389)', () => {
  // Regression guard: empty allowlist used to return true (allow-everyone),
  // which let any stranger DM the bridge and trigger a Python-side
  // pairing-code reply. Secure default is now "reject unless explicitly
  // configured"; operators who want an open bot must set `*`.
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-allowlist-'));

  try {
    const empty = parseAllowedUsers('');
    assert.equal(empty.size, 0);
    assert.equal(matchesAllowedUser(`${TEST_PHONE}@s.whatsapp.net`, empty, sessionDir), false);
    assert.equal(matchesAllowedUser(`${TEST_LID}@lid`, empty, sessionDir), false);

    // Null/undefined allowlist (defensive) also rejects.
    assert.equal(matchesAllowedUser(`${TEST_PHONE}@s.whatsapp.net`, null, sessionDir), false);
    assert.equal(matchesAllowedUser(`${TEST_PHONE}@s.whatsapp.net`, undefined, sessionDir), false);
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});

test('matchesAllowedMessage authorizes a raw LID through its alternate phone JID', () => {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-allowlist-'));
  try {
    const allowedUsers = parseAllowedUsers(`+${TEST_PHONE}`);
    const key = {
      remoteJid: `${TEST_LID}@lid`,
      remoteJidAlt: `${TEST_PHONE}@s.whatsapp.net`,
    };
    assert.equal(matchesAllowedMessage(key.remoteJid, key, allowedUsers, sessionDir), true);
    assert.equal(matchesAllowedMessage(`${TEST_LID}@lid`, {
      participant: `${TEST_LID}@lid`,
      participantAlt: `${TEST_PHONE}@s.whatsapp.net`,
    }, allowedUsers, sessionDir), true);
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});

test('matchesAllowedMessage denies unresolved, unrelated, and empty-allowlist aliases', () => {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-allowlist-'));
  try {
    const allowedUsers = parseAllowedUsers(`+${TEST_PHONE}`);
    const rawLid = `${TEST_LID}@lid`;
    assert.equal(matchesAllowedMessage(rawLid, { remoteJid: rawLid }, allowedUsers, sessionDir), false);
    assert.equal(matchesAllowedMessage(rawLid, {
      remoteJid: rawLid,
      remoteJidAlt: `${OTHER_PHONE}@s.whatsapp.net`,
    }, allowedUsers, sessionDir), false);
    // senderPn is not part of Baileys 7.0.0-rc13's WAMessageKey contract.
    assert.equal(matchesAllowedMessage(rawLid, {
      remoteJid: rawLid,
      senderPn: `${TEST_PHONE}@s.whatsapp.net`,
    }, allowedUsers, sessionDir), false);
    assert.equal(matchesAllowedMessage(rawLid, {
      remoteJid: rawLid,
      participantAlt: `${TEST_PHONE}@s.whatsapp.net`,
    }, parseAllowedUsers(''), sessionDir), false);
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});
