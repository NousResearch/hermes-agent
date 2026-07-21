import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';

import {
  expandWhatsAppIdentifiers,
  isGroupChatAllowed,
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

test('allowlist group policy rejects a group missing from the group allowlist', () => {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-group-allowlist-'));
  try {
    const allowedGroups = parseAllowedUsers('120363001234567890@g.us');
    assert.equal(
      isGroupChatAllowed(
        '120363009999999999@g.us',
        'allowlist',
        allowedGroups,
        sessionDir,
      ),
      false,
    );
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});

test('allowlist group policy allows a listed group', () => {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-group-allowlist-'));
  try {
    const allowedGroups = parseAllowedUsers('120363001234567890@g.us');
    assert.equal(
      isGroupChatAllowed(
        '120363001234567890@g.us',
        'allowlist',
        allowedGroups,
        sessionDir,
      ),
      true,
    );
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
});

test('open group policy allows a group without an allowlist', () => {
  assert.equal(
    isGroupChatAllowed(
      '120363009999999999@g.us',
      'open',
      new Set(),
      '',
    ),
    true,
  );
});

test('non-open group policies reject groups unless explicitly allowlisted', () => {
  for (const policy of ['disabled', 'pairing', '']) {
    assert.equal(
      isGroupChatAllowed(
        '120363001234567890@g.us',
        policy,
        parseAllowedUsers('*'),
        '',
      ),
      false,
      `policy ${policy || '<empty>'}`,
    );
  }
});

test('bridge enforces group policy before media extraction', () => {
  const source = readFileSync(new URL('./bridge.js', import.meta.url), 'utf8');
  const gateComment = source.indexOf('// Apply group policy in the bridge before');
  const extraction = source.indexOf('const event = await extractBridgeEvent({');

  assert.notEqual(gateComment, -1, 'group policy gate must exist');
  assert.notEqual(extraction, -1, 'media-capable extraction must exist');
  assert.ok(gateComment < extraction, 'group policy gate must precede extraction');

  const preExtraction = source.slice(gateComment, extraction);
  assert.match(preExtraction, /isGroupChatAllowed\(/);
  assert.match(preExtraction, /group_policy_rejected_before_media/);
  assert.match(preExtraction, /continue;/);
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
