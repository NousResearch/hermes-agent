#!/usr/bin/env node
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const router = path.join(root, 'scripts', 'dry-run-route.sh');
const mapping = path.join(root, 'references', 'repo_mappings.json');

function route(mission) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lah-write-v4-'));
  const file = path.join(dir, 'mission.txt');
  fs.writeFileSync(file, `${mission}\n`);
  const result = spawnSync('bash', [router, mapping, file], { encoding: 'utf8' });
  fs.rmSync(dir, { recursive: true, force: true });
  assert.equal(result.status, 0, result.stderr);
  const first = result.stdout.indexOf('{');
  return JSON.parse(result.stdout.slice(first, result.stdout.lastIndexOf('}') + 1));
}

const crossRole = route('Implement runtime telemetry and update operational memory.');
assert.deepEqual([...crossRole.write_allowed_repos].sort(), ['cartelogic-v2', 'openclaw-runtime']);
assert.equal(crossRole.write_intents.length, 2);

const context = route('Review discovery platform context for a runtime design.');
assert.deepEqual(context.write_allowed_repos, []);

const archived = route('Update the archived OpenClaw agent-memory files.');
assert.equal(archived.decision, 'BLOCKED');
assert.deepEqual(archived.write_allowed_repos, []);

const conflict = route('In lah-core, update the gateway runtime.');
assert.equal(conflict.decision, 'BLOCKED');
assert.deepEqual(conflict.write_allowed_repos, []);
assert.ok(conflict.conflicts.length > 0);

const ambiguous = route('Improve campaign intelligence.');
assert.equal(ambiguous.decision, 'AMBIGUOUS');
assert.deepEqual(ambiguous.write_allowed_repos, []);

console.log('v4 write-policy tests: 5 passed');
