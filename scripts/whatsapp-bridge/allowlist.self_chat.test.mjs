/**
 * Behaviour tests for isSelfChatIdentity — which identifiers count as the
 * account's OWN in self-chat mode.
 *
 * Two properties pull in opposite directions and both are pinned here:
 *
 *   * after a WhatsApp number change the "Message yourself" self-chat can stay
 *     bound to the PREVIOUS number's LID while sock.user carries the new one,
 *     so the account's own messages must still be recognized when the operator
 *     declares that number; and
 *   * a stranger's DM must never be mistaken for the account's own chat — not
 *     even when the session already holds that contact's lid-mapping file
 *     (#8389: accepting it fires a pairing-code reply at an arbitrary sender).
 *
 * The session's lid-mapping-*.json files are contact maps, so expanding a
 * DECLARED identity through them is safe, while matching the incoming
 * identifier against them is not. These tests keep that distinction.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';

import { isSelfChatIdentity, parseAllowedUsers } from './allowlist.js';

// Synthetic fixtures only: the 555 exchange is reserved for fiction, and the
// LIDs are arbitrary digit strings. Nothing here is a real subscriber number.
const ACCOUNT_PHONE = '15551234567';
const ACCOUNT_LID = '267383306489914';
const PREVIOUS_PHONE = '15551234568';
const PREVIOUS_LID = '100000000000001';
const STRANGER_PHONE = '15551234569';
const STRANGER_LID = '299900001111222';

const ACCOUNT_ID = `${ACCOUNT_PHONE}:10@s.whatsapp.net`;
const ACCOUNT_LID_JID = `${ACCOUNT_LID}@lid`;

function withSessionDir(fn) {
  const sessionDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-self-chat-'));
  try {
    return fn(sessionDir);
  } finally {
    rmSync(sessionDir, { recursive: true, force: true });
  }
}

function writeLidMapping(sessionDir, phone, lid) {
  writeFileSync(path.join(sessionDir, `lid-mapping-${phone}.json`), JSON.stringify(lid));
  writeFileSync(path.join(sessionDir, `lid-mapping-${lid}_reverse.json`), JSON.stringify(phone));
}

test('current sock.user number and lid are self', () => {
  withSessionDir((sessionDir) => {
    const account = { accountId: ACCOUNT_ID, accountLid: ACCOUNT_LID_JID, sessionDir };

    assert.equal(isSelfChatIdentity(`${ACCOUNT_PHONE}@s.whatsapp.net`, account), true);
    assert.equal(isSelfChatIdentity(ACCOUNT_LID_JID, account), true);
  });
});

test("a self-chat stays bound to the previous number's LID, so a declared previous number is self", () => {
  withSessionDir((sessionDir) => {
    writeLidMapping(sessionDir, ACCOUNT_PHONE, ACCOUNT_LID);
    writeLidMapping(sessionDir, PREVIOUS_PHONE, PREVIOUS_LID);

    const account = {
      accountId: ACCOUNT_ID,
      accountLid: ACCOUNT_LID_JID,
      selfIdentifiers: parseAllowedUsers(`+${ACCOUNT_PHONE},+${PREVIOUS_PHONE}`),
      sessionDir,
    };

    assert.equal(isSelfChatIdentity(`${PREVIOUS_LID}@lid`, account), true);
    assert.equal(isSelfChatIdentity(`${PREVIOUS_PHONE}@s.whatsapp.net`, account), true);
  });
});

test("a known contact's identities are not self, even with lid-mapping files on disk (#8389)", () => {
  withSessionDir((sessionDir) => {
    writeLidMapping(sessionDir, ACCOUNT_PHONE, ACCOUNT_LID);
    writeLidMapping(sessionDir, STRANGER_PHONE, STRANGER_LID);

    const account = {
      accountId: ACCOUNT_ID,
      accountLid: ACCOUNT_LID_JID,
      selfIdentifiers: parseAllowedUsers(ACCOUNT_PHONE),
      sessionDir,
    };

    assert.equal(isSelfChatIdentity(`${STRANGER_LID}@lid`, account), false);
    assert.equal(isSelfChatIdentity(`${STRANGER_PHONE}@s.whatsapp.net`, account), false);
  });
});

test('an empty allowlist declares no self identity beyond the current account', () => {
  withSessionDir((sessionDir) => {
    const account = {
      accountId: ACCOUNT_ID,
      accountLid: ACCOUNT_LID_JID,
      selfIdentifiers: parseAllowedUsers(''),
      sessionDir,
    };

    assert.equal(isSelfChatIdentity(`${PREVIOUS_LID}@lid`, account), false);
    assert.equal(isSelfChatIdentity(ACCOUNT_LID_JID, account), true);
  });
});

test('the "*" allow-all wildcard never means "this is me"', () => {
  withSessionDir((sessionDir) => {
    const account = {
      accountId: ACCOUNT_ID,
      accountLid: ACCOUNT_LID_JID,
      selfIdentifiers: parseAllowedUsers('*'),
      sessionDir,
    };

    assert.equal(isSelfChatIdentity(`${STRANGER_LID}@lid`, account), false);
    assert.equal(isSelfChatIdentity(`${STRANGER_PHONE}@s.whatsapp.net`, account), false);
  });
});

test('missing or blank identifiers are not self', () => {
  assert.equal(isSelfChatIdentity('', { accountId: ACCOUNT_ID, accountLid: ACCOUNT_LID_JID }), false);
  assert.equal(isSelfChatIdentity(undefined, {}), false);
  assert.equal(isSelfChatIdentity(`${ACCOUNT_LID}@lid`, {}), false);
});
