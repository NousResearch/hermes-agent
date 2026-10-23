/**
 * Contract for the inbound observation log (WHATSAPP_OBSERVER).
 *
 * The bridge keeps no message archive and messageStore is a bounded
 * quotes/polls cache, so the observation line is the only reconstruction of
 * what was said. Only the pure helpers are covered here: importing bridge.js
 * would start an HTTP server and a Baileys socket at module load.
 */

import { strict as assert } from 'node:assert';
import test from 'node:test';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { appendObservedLine, observedContent, observedLine, observerEnabled } from './bridge_helpers.js';

test('observerEnabled is opt-in: only an affirmative value enables it', () => {
  assert.equal(observerEnabled(undefined), false);  // default off: this writes message content to disk
  assert.equal(observerEnabled(''), false);
  assert.equal(observerEnabled('off'), false);
  assert.equal(observerEnabled('no'), false);
  assert.equal(observerEnabled('on'), true);
  assert.equal(observerEnabled('TRUE'), true);
  assert.equal(observerEnabled(' 1 '), true);
});

test('observedLine writes one JSON object per line', () => {
  const line = observedLine({
    direction: 'in',
    chatId: '120363001234567890@g.us',
    senderId: '51940397467',
    messageId: 'ABC123',
    text: 'hello group',
    hasMedia: false,
  }, new Date('2026-10-13T20:00:00Z'));

  assert.ok(line.endsWith('\n'), 'must end in a newline');
  assert.equal(line.split('\n').length, 2, 'a single line');
  const obj = JSON.parse(line);
  assert.equal(obj.ts, '2026-10-13T20:00:00.000Z');
  assert.equal(obj.direction, 'in');
  assert.equal(obj.chatId, '120363001234567890@g.us');
  assert.equal(obj.sender, '51940397467');
  assert.equal(obj.messageId, 'ABC123');
  assert.equal(obj.text, 'hello group');
});

test('observedLine defaults absent fields to empty, never undefined', () => {
  const obj = JSON.parse(observedLine({ chatId: 'g@g.us' }));
  for (const key of ['sender', 'messageId', 'text', 'mediaType']) {
    assert.equal(obj[key], '', `${key} must be ''`);
  }
  assert.equal(obj.hasMedia, false);
  assert.equal(obj.direction, 'in');
});

test('appendObservedLine creates the directory and appends without truncating', () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-observer-'));
  const file = path.join(dir, 'nested', 'observed.jsonl');
  try {
    assert.equal(appendObservedLine(file, observedLine({ chatId: 'a@g.us', text: 'one' })), true);
    assert.equal(appendObservedLine(file, observedLine({ chatId: 'a@g.us', text: 'two' })), true);
    const lines = readFileSync(file, 'utf8').trim().split('\n');
    assert.equal(lines.length, 2, 'appends, does not truncate');
    assert.equal(JSON.parse(lines[1]).text, 'two');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('appendObservedLine returns false instead of throwing on an unwritable path', () => {
  // A logging failure must not take the intake path down with it. The parent is
  // a regular file, so creating its "directory" fails on every platform.
  const dir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-observer-'));
  const blocker = path.join(dir, 'not-a-dir');
  try {
    writeFileSync(blocker, 'x');
    assert.equal(appendObservedLine(path.join(blocker, 'observed.jsonl'), 'x\n'), false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('observedContent reads text and media off the raw payload', () => {
  assert.deepEqual(
    observedContent({ conversation: 'plain text' }),
    { text: 'plain text', hasMedia: false, mediaType: '' },
  );
  assert.deepEqual(
    observedContent({ extendedTextMessage: { text: 'extended' } }),
    { text: 'extended', hasMedia: false, mediaType: '' },
  );
  assert.deepEqual(
    observedContent({ imageMessage: { caption: 'a photo' } }),
    { text: 'a photo', hasMedia: true, mediaType: 'image' },
  );
  // An uncaptioned attachment still reports its type; the text stays empty.
  assert.deepEqual(
    observedContent({ videoMessage: {} }),
    { text: '', hasMedia: true, mediaType: 'video' },
  );
  assert.deepEqual(
    observedContent({ videoMessage: { gifPlayback: true } }).mediaType,
    'gif',
  );
  assert.deepEqual(
    observedContent({ pttMessage: {} }).mediaType,
    'audio',
  );
  assert.deepEqual(
    observedContent({ documentMessage: { caption: 'report', fileName: 'r.pdf' } }),
    { text: 'report', hasMedia: true, mediaType: 'document' },
  );
  assert.deepEqual(
    observedContent({ stickerMessage: {} }).mediaType,
    'sticker',
  );
  assert.deepEqual(observedContent(null), { text: '', hasMedia: false, mediaType: '' });
});

test('observedContent is shallow: it never downloads media or decodes envelopes', () => {
  // This runs before the intake gates, on messages the gates are about to drop,
  // so it must not carry extractBridgeEvent's cost (a CDN fetch per attachment).
  // An envelope-wrapped payload is reported empty instead of being peeled: the
  // observer records that a message arrived, not a best-effort decode of it.
  const msg = { ephemeralMessage: { message: { imageMessage: { caption: 'hidden' } } } };
  assert.deepEqual(observedContent(msg), { text: '', hasMedia: false, mediaType: '' });
});

test('the observation log is one parseable object per line', () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-observer-'));
  const file = path.join(dir, 'observed.jsonl');
  try {
    for (const text of ['one', 'two', 'three']) {
      appendObservedLine(file, observedLine({ chatId: 'g@g.us', text }));
    }
    const raw = readFileSync(file, 'utf8');
    const parsed = raw.split('\n').filter(Boolean).map((l) => JSON.parse(l));
    assert.equal(parsed.length, 3);
    assert.deepEqual(parsed.map((o) => o.text), ['one', 'two', 'three']);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('a pre-existing log is preserved rather than rewritten', () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'hermes-wa-observer-'));
  const file = path.join(dir, 'observed.jsonl');
  try {
    writeFileSync(file, '{"ts":"2026-01-01T00:00:00.000Z","text":"earlier"}\n');
    appendObservedLine(file, observedLine({ chatId: 'g@g.us', text: 'later' }));
    const lines = readFileSync(file, 'utf8').trim().split('\n');
    assert.equal(lines.length, 2);
    assert.equal(JSON.parse(lines[0]).text, 'earlier');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
