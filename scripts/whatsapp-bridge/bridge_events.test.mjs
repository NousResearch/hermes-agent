import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { mkdtempSync, rmSync, readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';

import { recordBridgeEvent, resolveEventsFile } from './bridge_events.js';

function tmp() {
  return mkdtempSync(path.join(os.tmpdir(), 'bridge-events-test-'));
}

test('resolveEventsFile defaults to a peer of the session dir and honors env override', () => {
  const dir = tmp();
  try {
    const sessionDir = path.join(dir, 'whatsapp', 'session');
    assert.equal(resolveEventsFile(sessionDir), path.join(sessionDir, '..', 'bridge-events.jsonl'));
    process.env.WHATSAPP_BRIDGE_EVENTS_FILE = path.join(dir, 'custom-events.jsonl');
    assert.equal(resolveEventsFile(sessionDir), path.join(dir, 'custom-events.jsonl'));
  } finally {
    delete process.env.WHATSAPP_BRIDGE_EVENTS_FILE;
    rmSync(dir, { recursive: true, force: true });
  }
});

test('recordBridgeEvent appends timestamped JSONL entries', () => {
  const dir = tmp();
  try {
    const sessionDir = path.join(dir, 'whatsapp', 'session');
    mkdirSync(sessionDir, { recursive: true });
    recordBridgeEvent(sessionDir, 'bridge_started', { mode: 'bot', port: 3000 });
    recordBridgeEvent(sessionDir, 'connected', {});
    recordBridgeEvent(sessionDir, 'disconnected', { reason: 428 });
    const lines = readFileSync(resolveEventsFile(sessionDir), 'utf8').trim().split('\n');
    assert.equal(lines.length, 3);
    const first = JSON.parse(lines[0]);
    assert.equal(first.event, 'bridge_started');
    assert.equal(first.port, 3000);
    assert.ok(!Number.isNaN(Date.parse(first.ts)));
    assert.equal(JSON.parse(lines[2]).reason, 428);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('recordBridgeEvent never throws on an unwritable destination', () => {
  const dir = tmp();
  try {
    const blocker = path.join(dir, 'blocker');
    writeFileSync(blocker, 'x');
    process.env.WHATSAPP_BRIDGE_EVENTS_FILE = path.join(blocker, 'nested', 'events.jsonl');
    assert.doesNotThrow(() => recordBridgeEvent(path.join(dir, 's'), 'connected', {}));
  } finally {
    delete process.env.WHATSAPP_BRIDGE_EVENTS_FILE;
    rmSync(dir, { recursive: true, force: true });
  }
});

test('recordBridgeEvent rotates the file past the size cap', () => {
  const dir = tmp();
  try {
    const sessionDir = path.join(dir, 'whatsapp', 'session');
    mkdirSync(sessionDir, { recursive: true });
    const file = resolveEventsFile(sessionDir);
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, Buffer.alloc(5 * 1024 * 1024 + 1, 0x61));
    recordBridgeEvent(sessionDir, 'connected', {});
    assert.ok(existsSync(`${file}.1`), 'rotated file exists');
    const lines = readFileSync(file, 'utf8').trim().split('\n');
    assert.equal(lines.length, 1);
    assert.equal(JSON.parse(lines[0]).event, 'connected');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
