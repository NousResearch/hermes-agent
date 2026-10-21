/**
 * A logged-out Baileys session must be deleted, not renamed. A leftover
 * creds.json is what the dashboard treated as "Linked".
 */

import { strict as assert } from 'node:assert';
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { prepareLoggedOutSession } from './bridge_helpers.js';

const parent = mkdtempSync(path.join(tmpdir(), 'wa-logout-'));
const sessionDir = path.join(parent, 'session');
mkdirSync(sessionDir);
writeFileSync(path.join(sessionDir, 'creds.json'), '{"me":{"id":"15551234567"}}');
writeFileSync(path.join(sessionDir, 'app-state-sync-key-1.json'), '{}');

const plan = prepareLoggedOutSession(sessionDir, rmSync);

assert.equal(plan.status, 'needs_qr');
assert.equal(plan.restartDelayMs, 1000);
assert.equal(existsSync(sessionDir), false);
assert.equal(existsSync(path.join(sessionDir, 'creds.json')), false);

const leftovers = readdirSync(parent);
assert.deepEqual(leftovers, [], 'logged-out session is deleted, not backed up');

rmSync(parent, { recursive: true, force: true });
console.log('bridge.logged-out.test.mjs: all assertions passed');
