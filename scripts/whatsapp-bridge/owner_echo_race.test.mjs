import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import { proto } from '@whiskeysockets/baileys';
import { createAckSender } from './send_ack.js';
import { createOutboundIdTracker } from './outbound_ids.js';
import { classifyOwnerMessageGate } from './owner_message_gate.js';

const source = readFileSync(new URL('./bridge.js', import.meta.url), 'utf8');

test('bridge drops its fromMe echo before the local send returns or server ACK arrives', async () => {
  const recentlySentIds = createOutboundIdTracker();
  const chatId = '49731443138613@lid';
  const receipts = [];
  let echoAssertion;
  let observedId;
  let sender;
  const context = vm.createContext({
    createAckSender,
    sock: { user: { id: '100@s.whatsapp.net' } },
    recentlySentIds,
    appendAckReceipt: (_file, receipt) => receipts.push(receipt),
    sendWithTimeout: async (chat, _payload, options) => {
      observedId = options.messageId;
      assert.ok(observedId, 'ACK sender must pre-generate the protocol ID');
      // Simulate messages.upsert synchronously inside the network send,
      // before either local completion or the later server acknowledgement.
      const { action } = classifyOwnerMessageGate({
        fromMe: true, fromOwnerEnabled: true, recentlySent: recentlySentIds,
        allowlistMatches: id => id === chatId, messageId: observedId, chatId: chat,
      });
      // Preserve the assertion: createAckSender deliberately sanitizes send errors.
      try { assert.equal(action, 'drop_echo'); } catch (error) { echoAssertion = error; }
      const key = { id: observedId, remoteJid: chat, fromMe: true };
      setImmediate(() => sender.observeSendAck(key, { status: proto.WebMessageInfo.Status.SERVER_ACK }));
      return { key };
    },
  });
  // Evaluate the real production wiring, not a copied version of the fix.
  const start = source.indexOf('const { observe: observeSendAck, send: sendWithServerAck } = createAckSender({');
  const end = source.indexOf('\n});', start);
  assert.ok(start >= 0 && end > start, 'bridge ACK sender wiring must exist');
  const remember = source.match(/function rememberSentId\(id\) \{[\s\S]*?\n\}/);
  assert.ok(remember, 'bridge tracker helper must exist');
  sender = vm.runInContext(`${source.slice(start, end + 4)}\n${remember[0]}\n({ observeSendAck, sendWithServerAck })`, context);
  const sent = await sender.sendWithServerAck(chatId, { text: 'offline echo race test' });
  if (echoAssertion) throw echoAssertion;
  assert.equal(sent.key.id, observedId);
  assert.equal(sent.ackStatus, 'SERVER_ACK');
  assert.equal(receipts.length, 1);
  // Owner forwarding remains enabled for genuinely owner-typed messages.
  assert.equal(classifyOwnerMessageGate({
    fromMe: true, fromOwnerEnabled: true, recentlySent: recentlySentIds,
    allowlistMatches: id => id === chatId, messageId: 'OWNER-TYPED', chatId,
  }).action, 'forward_owner');
});
