import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';

import {
  expandWhatsAppIdentifiers,
  matchesAllowedSenderWithAlt,
  matchesAllowedUser,
  normalizeWhatsAppIdentifier,
  parseAllowedUsers,
} from './allowlist.js';

test('normalizeWhatsAppIdentifier strips jid syntax and plus prefix', () => {
  assert.equal(normalizeWhatsAppIdentifier('+19175395595@s.whatsapp.net'), '19175395595');
  assert.equal(normalizeWhatsAppIdentifier('267383306489914@lid'), '267383306489914');
  assert.equal(normalizeWhatsAppIdentifier('19175395595:12@s.whatsapp.net'), '19175395595');
});

test('expandWhatsAppIdentifiers resolves phone and lid aliases from session files', () => {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-allowlist-'));

  try {
    writeFileSync(path.join(sessionDir, 'lid-mapping-19175395595.json'), JSON.stringify('267383306489914'));
    writeFileSync(path.join(sessionDir, 'lid-mapping-267383306489914_reverse.json'), JSON.stringify('19175395595'));

    const aliases = expandWhatsAppIdentifiers('267383306489914@lid', sessionDir);
    assert.deepEqual([...aliases].sort(), ['19175395595', '267383306489914']);
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});

test('matchesAllowedUser accepts mapped lid sender when allowlist only contains phone number', () => {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-allowlist-'));

  try {
    writeFileSync(path.join(sessionDir, 'lid-mapping-19175395595.json'), JSON.stringify('267383306489914'));
    writeFileSync(path.join(sessionDir, 'lid-mapping-267383306489914_reverse.json'), JSON.stringify('19175395595'));

    const allowedUsers = parseAllowedUsers('+19175395595');
    assert.equal(matchesAllowedUser('267383306489914@lid', allowedUsers, sessionDir), true);
    assert.equal(matchesAllowedUser('188012763865257@lid', allowedUsers, sessionDir), false);
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});

test('matchesAllowedUser treats * as allow-all wildcard', () => {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-allowlist-'));

  try {
    const allowedUsers = parseAllowedUsers('*');
    assert.equal(matchesAllowedUser('19175395595@s.whatsapp.net', allowedUsers, sessionDir), true);
    assert.equal(matchesAllowedUser('267383306489914@lid', allowedUsers, sessionDir), true);
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
    assert.equal(matchesAllowedUser('19175395595@s.whatsapp.net', empty, sessionDir), false);
    assert.equal(matchesAllowedUser('267383306489914@lid', empty, sessionDir), false);

    // Null/undefined allowlist (defensive) also rejects.
    assert.equal(matchesAllowedUser('19175395595@s.whatsapp.net', null, sessionDir), false);
    assert.equal(matchesAllowedUser('19175395595@s.whatsapp.net', undefined, sessionDir), false);
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});

test('matchesAllowedSenderWithAlt falls back to participantAlt when no lid mapping exists (#72529)', () => {
  // Baileys v7 group messages arrive with key.participant in LID form and
  // the phone number only in key.participantAlt. A fresh bot session has no
  // lid-mapping-*.json files, so the LID alone can never match a phone
  // allowlist — the alt identifier must be consulted as a fallback.
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-allowlist-'));

  try {
    const allowedUsers = parseAllowedUsers('+19175395595');

    // No mapping file: LID by itself is rejected today (the #72529 drop)…
    assert.equal(matchesAllowedUser('267383306489914@lid', allowedUsers, sessionDir), false);
    // …but the alt phone identifier rescues the match.
    assert.equal(
      matchesAllowedSenderWithAlt('267383306489914@lid', '19175395595@s.whatsapp.net', allowedUsers, sessionDir),
      true
    );

    // A stranger's LID with a non-allowlisted alt stays rejected.
    assert.equal(
      matchesAllowedSenderWithAlt('188012763865257@lid', '15550001111@s.whatsapp.net', allowedUsers, sessionDir),
      false
    );

    // Missing / empty alt keeps the existing behaviour.
    assert.equal(matchesAllowedSenderWithAlt('267383306489914@lid', '', allowedUsers, sessionDir), false);
    assert.equal(matchesAllowedSenderWithAlt('267383306489914@lid', undefined, allowedUsers, sessionDir), false);
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});

test('matchesAllowedSenderWithAlt still resolves via lid-mapping files without an alt', () => {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-allowlist-'));

  try {
    writeFileSync(path.join(sessionDir, 'lid-mapping-19175395595.json'), JSON.stringify('267383306489914'));
    writeFileSync(path.join(sessionDir, 'lid-mapping-267383306489914_reverse.json'), JSON.stringify('19175395595'));

    const allowedUsers = parseAllowedUsers('+19175395595');
    assert.equal(matchesAllowedSenderWithAlt('267383306489914@lid', '', allowedUsers, sessionDir), true);
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});
