/**
 * Passive-capture regression tests.  bridge.js is not imported (it starts a
 * socket and HTTP server at load), so its wiring is checked statically.
 */

import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { appendPassiveRecord, buildPassiveRecord, isOwnSelfChat, isPassiveTerminal } from './passive_capture.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const bridgeSrc = readFileSync(path.join(here, 'bridge.js'), 'utf8');

const dmEvent = {
  chatId: '420111222333@s.whatsapp.net',
  senderId: '420111222333@s.whatsapp.net',
  senderName: 'Kontakt',
  isGroup: false,
  messageId: 'ABC1',
  timestamp: 1794200000,
  body: 'ahoj',
  hasMedia: false,
};

test('only inbound messages are terminal in passive mode', () => {
  assert.equal(isPassiveTerminal({ enabled: true, fromMe: false }), true);
  assert.equal(isPassiveTerminal({ enabled: true, fromMe: true }), false);
  assert.equal(isPassiveTerminal({ enabled: false, fromMe: false }), false);
  assert.equal(isPassiveTerminal({ enabled: '', fromMe: false }), false);
});

test('records keep DM and group identity and drop empty stubs', () => {
  const dm = buildPassiveRecord(dmEvent, new Date('2026-11-10T10:00:00Z'));
  assert.equal(dm.direction, 'inbound');
  assert.equal(dm.isGroup, false);
  assert.equal(dm.messageId, 'ABC1');
  assert.equal(dm.capturedAt, '2026-11-10T10:00:00.000Z');
  const group = buildPassiveRecord({ ...dmEvent, chatId: '1203630000@g.us', isGroup: true });
  assert.equal(group.isGroup, true);
  assert.equal(group.chatId, '1203630000@g.us');
  assert.equal(buildPassiveRecord({ ...dmEvent, body: '   ' }), null);
  assert.ok(buildPassiveRecord({ ...dmEvent, body: '', hasMedia: true, mediaType: 'image' }));
});

test('append writes private NDJSON and repairs a torn last line', () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'wa-passive-'));
  const file = path.join(dir, 'sub', 'passive-capture.jsonl');
  appendPassiveRecord(file, buildPassiveRecord(dmEvent));
  assert.equal(statSync(file).mode & 0o777, 0o600);
  assert.equal(statSync(path.dirname(file)).mode & 0o777, 0o700);

  writeFileSync(file, readFileSync(file, 'utf8') + '{"torn":', { flag: 'w' });
  appendPassiveRecord(file, buildPassiveRecord({ ...dmEvent, messageId: 'ABC2' }));
  const lines = readFileSync(file, 'utf8').split('\n');
  assert.equal(lines.at(-1), '');
  assert.equal(JSON.parse(lines[0]).messageId, 'ABC1');
  assert.equal(lines[1], '{"torn":');
  assert.equal(JSON.parse(lines[2]).messageId, 'ABC2');
  assert.equal(statSync(file).mode & 0o777, 0o600);
});

test('bridge.js: passive branch precedes every agent-queue push in messages.upsert', () => {
  const start = bridgeSrc.indexOf("sock.ev.on('messages.upsert'");
  assert.ok(start > 0);
  const handler = bridgeSrc.slice(start);
  const passive = handler.indexOf('isPassiveTerminal({ enabled: PASSIVE_CAPTURE_ENABLED');
  const push = handler.indexOf('messageQueue.push(event)');
  assert.ok(passive > 0 && push > passive, 'passive check must come before messageQueue.push');
  const branch = handler.slice(passive, handler.indexOf('\n      }\n', passive));
  assert.match(branch, /capturePassiveEvent\(event\);\s*continue;/);
});

test('bridge.js: poll votes are not enqueued in passive mode', () => {
  const fn = bridgeSrc.slice(bridgeSrc.indexOf('function enqueuePollUpdateEvent('));
  const body = fn.slice(0, fn.indexOf('messageQueue.push(event)'));
  assert.match(body, /if \(PASSIVE_CAPTURE_ENABLED\) return;/);
});

test('bridge.js: /health exposes passiveCapture', () => {
  const health = bridgeSrc.slice(bridgeSrc.indexOf("app.get('/health'"));
  assert.match(health.slice(0, 600), /passiveCapture: PASSIVE_CAPTURE_ENABLED/);
});

test('isOwnSelfChat matches only the account\'s own chat (phone or LID)', () => {
  const user = { id: '420603715271:46@s.whatsapp.net', lid: '162444404109355:46@lid' };
  assert.equal(isOwnSelfChat('420603715271@s.whatsapp.net', user), true);
  assert.equal(isOwnSelfChat('162444404109355@lid', user), true);
  assert.equal(isOwnSelfChat('420111222333@s.whatsapp.net', user), false);
  assert.equal(isOwnSelfChat('25838674292800@lid', user), false);
  assert.equal(isOwnSelfChat('420603715271@g.us', user), false);
  assert.equal(isOwnSelfChat('', user), false);
  assert.equal(isOwnSelfChat('420603715271@s.whatsapp.net', undefined), false);
});

test('bridge.js: passive self-chat bypasses only the owner gate, echo filter still applies', () => {
  const start = bridgeSrc.indexOf("sock.ev.on('messages.upsert'");
  const handler = bridgeSrc.slice(start);
  const selfChat = handler.indexOf('PASSIVE_CAPTURE_ENABLED && isOwnSelfChat(chatId, sock.user)');
  const gate = handler.indexOf('classifyOwnerMessageGate({');
  const echo = handler.indexOf('recentlySentIds.has(msg.key.id)');
  const push = handler.indexOf('messageQueue.push(event)');
  assert.ok(selfChat > 0 && gate > selfChat, 'self-chat branch must wrap the owner gate');
  assert.ok(echo > selfChat && push > echo, 'echo filter must run before the agent queue');
  const between = handler.slice(selfChat, gate);
  assert.doesNotMatch(between, /fromOwner = true/);
});
