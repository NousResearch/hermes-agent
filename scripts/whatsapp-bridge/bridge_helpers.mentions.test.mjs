/**
 * Mention name resolution, bridge side: the label behind a mentioned id must come
 * from data Baileys actually delivers, and reach the bridge event as
 * `mentionedNames` (rendered by gateway/platforms/whatsapp_common.py).
 *
 * Group participants are built with Baileys' own extractGroupMetadata from a
 * synthetic <group> node, so the test breaks if the participant shape drifts
 * (it carries no display name: only id / phoneNumber / lid / username).
 * All ids are synthetic.
 */

import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { extractGroupMetadata } from '@whiskeysockets/baileys/lib/Socket/groups.js';

import { createMentionNameResolver, extractBridgeEvent } from './bridge_helpers.js';

const GROUP = '120363000000000000@g.us';
const ALICE_LID = '100000000000002';
const ALICE_PN = '15551234567';
const CAROL_LID = '100000000000003';
const CAROL_PN = '15550001111';
const BOB_PN = '15550002222';
const BOT_LID = '100000000000001';

function groupMetadata() {
  return extractGroupMetadata({
    tag: 'iq',
    attrs: {},
    content: [{
      tag: 'group',
      attrs: { id: GROUP.split('@')[0], subject: 'Synthetic group', creation: '1' },
      content: [
        // LID-addressed member with a phone twin, no username.
        { tag: 'participant', attrs: { jid: `${ALICE_LID}@lid`, phone_number: `${ALICE_PN}@s.whatsapp.net` } },
        // PN-addressed member with a LID twin and a WhatsApp username.
        { tag: 'participant', attrs: { jid: `${CAROL_PN}@s.whatsapp.net`, lid: `${CAROL_LID}@lid`, username: 'carol' } },
      ],
    }],
  });
}

function inbound({ id, participant, participantAlt, pushName, text = 'hi', mentionedJid, fromMe = false }) {
  return {
    key: { id, remoteJid: GROUP, participant, participantAlt, fromMe },
    pushName,
    messageTimestamp: 1,
    message: { extendedTextMessage: { text, contextInfo: mentionedJid ? { mentionedJid } : undefined } },
  };
}

async function eventFor(resolver, msg) {
  resolver.learnMessage(msg);
  return extractBridgeEvent({
    msg,
    chatId: GROUP,
    senderId: msg.key.participant,
    senderNumber: msg.key.participant.split('@')[0],
    isGroup: true,
    resolveMentionNames: resolver.resolve,
  });
}

test('real participant shape carries no display name', () => {
  for (const participant of groupMetadata().participants) {
    assert.equal(participant.name, undefined);
    assert.equal(participant.notify, undefined);
  }
});

test('a pushName learned under one id form resolves a mention under its twin', async () => {
  let fetches = 0;
  const resolver = createMentionNameResolver({ fetchGroupMetadata: async () => { fetches += 1; return groupMetadata(); } });
  // Alice spoke earlier; this key carries only her phone form.
  resolver.learnMessage(inbound({ id: 'm1', participant: `${ALICE_PN}@s.whatsapp.net`, pushName: 'Alice' }));

  // Bob mentions her by LID: only the metadata PN<->LID twin connects the two.
  const event = await eventFor(resolver, inbound({
    id: 'm2',
    participant: `${BOB_PN}@s.whatsapp.net`,
    pushName: 'Bob',
    text: `@${ALICE_LID} did you see this?`,
    mentionedJid: [`${ALICE_LID}@lid`],
  }));

  assert.deepEqual(event.mentionedIds, [`${ALICE_LID}@lid`]);
  assert.deepEqual(event.mentionedNames, { [ALICE_LID]: 'Alice' });
  assert.equal(fetches, 1);
});

test('the twin pair on a message key needs no metadata round-trip', async () => {
  const resolver = createMentionNameResolver({ fetchGroupMetadata: async () => assert.fail('metadata not needed') });
  resolver.learnMessage(inbound({
    id: 'm1', participant: `${ALICE_LID}@lid`, participantAlt: `${ALICE_PN}@s.whatsapp.net`, pushName: 'Alice',
  }));

  const event = await eventFor(resolver, inbound({
    id: 'm2', participant: `${BOB_PN}@s.whatsapp.net`, pushName: 'Bob', mentionedJid: [`${ALICE_PN}@s.whatsapp.net`],
  }));

  assert.deepEqual(event.mentionedNames, { [ALICE_PN]: 'Alice' });
});

test('username is the fallback label and unknown members are never named', async () => {
  let fetches = 0;
  const resolver = createMentionNameResolver({ fetchGroupMetadata: async () => { fetches += 1; return groupMetadata(); } });
  const mention = (id, mentionedJid) => inbound({ id, participant: `${BOB_PN}@s.whatsapp.net`, pushName: 'Bob', mentionedJid });

  const event = await eventFor(resolver, mention('m1', [`${CAROL_LID}@lid`, `${ALICE_LID}@lid`]));
  // Carol never spoke but has a username; Alice has neither a pushName nor a username.
  assert.deepEqual(event.mentionedNames, { [CAROL_LID]: 'carol' });

  // Still-unresolved mentions do not re-query metadata inside the TTL.
  await eventFor(resolver, mention('m2', [`${ALICE_LID}@lid`]));
  assert.equal(fetches, 1);
});

test('own messages and failed metadata never invent or drop names', async () => {
  const resolver = createMentionNameResolver({ fetchGroupMetadata: async () => { throw new Error('offline'); } });
  // The bot's own pushName is not learned (Python labels the bot itself).
  resolver.learnMessage(inbound({ id: 'm0', participant: `${BOT_LID}@lid`, pushName: 'Bot', fromMe: true }));
  resolver.learnMessage(inbound({ id: 'm1', participant: `${CAROL_PN}@s.whatsapp.net`, pushName: 'Carol' }));

  const event = await eventFor(resolver, inbound({
    id: 'm2',
    participant: `${BOB_PN}@s.whatsapp.net`,
    pushName: 'Bob',
    mentionedJid: [`${BOT_LID}@lid`, `${ALICE_LID}@lid`, `${CAROL_PN}@s.whatsapp.net`],
  }));

  // Pushname-learned names survive the metadata failure; nothing else is filled in.
  assert.deepEqual(event.mentionedNames, { [CAROL_PN]: 'Carol' });
  assert.equal(event.body, 'hi');
});

test('learned names are bounded', async () => {
  const resolver = createMentionNameResolver({ limit: 2 });
  for (const [n, pn] of [['Alice', ALICE_PN], ['Carol', CAROL_PN], ['Bob', BOB_PN]]) {
    resolver.learnMessage(inbound({ id: n, participant: `${pn}@s.whatsapp.net`, pushName: n }));
  }

  const names = await resolver.resolve([ALICE_PN, CAROL_PN, BOB_PN].map((pn) => `${pn}@s.whatsapp.net`), GROUP);

  assert.deepEqual(names, { [CAROL_PN]: 'Carol', [BOB_PN]: 'Bob' });
});
