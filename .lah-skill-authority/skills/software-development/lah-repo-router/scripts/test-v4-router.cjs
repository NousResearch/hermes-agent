#!/usr/bin/env node
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const router = path.join(root, 'scripts', 'dry-run-route.sh');
const mapping = path.join(root, 'references', 'repo_mappings.json');
const fixtures = JSON.parse(fs.readFileSync(path.join(root, 'references', 'v4-adversarial-fixtures.json'), 'utf8'));

function route(mission) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lah-router-v4-'));
  const missionFile = path.join(dir, 'mission.txt');
  fs.writeFileSync(missionFile, `${mission}\n`);
  const result = spawnSync('bash', [router, mapping, missionFile], { encoding: 'utf8' });
  fs.rmSync(dir, { recursive: true, force: true });
  assert.equal(result.status, 0, result.stderr || result.stdout);
  const first = result.stdout.indexOf('{');
  const last = result.stdout.lastIndexOf('}');
  assert.ok(first >= 0 && last > first, `No receipt for: ${mission}`);
  return JSON.parse(result.stdout.slice(first, last + 1));
}

for (const fixture of fixtures) {
  const receipt = route(fixture.mission);
  assert.equal(receipt.schema_version, '4', fixture.id);
  assert.equal(receipt.decision, fixture.expected_decision, fixture.id);
  for (const key of [
    'expected_implementation_repo',
    'expected_execution_repo',
    'expected_primary_role',
    'expected_memory_repo',
    'expected_skill_knowledge_repo',
    'expected_business_asset_repo'
  ]) {
    if (fixture[key] !== undefined) assert.equal(receipt[key.replace('expected_', '')], fixture[key], fixture.id);
  }
  if (fixture.expected_write_allowed_repos) {
    assert.deepEqual([...receipt.write_allowed_repos].sort(), [...fixture.expected_write_allowed_repos].sort(), fixture.id);
  }
  for (const forbidden of fixture.expected_not_route_to || []) {
    assert.notEqual(receipt.repository_authority, forbidden, `${fixture.id}: routed to forbidden repo`);
    assert.notEqual(receipt.implementation_repo, forbidden, `${fixture.id}: implementation forbidden repo`);
  }
  if (fixture.id === 'codegraph-cannot-resolve-conflict') {
    assert.equal(receipt.codegraph_used, false, fixture.id);
    assert.equal(receipt.decision, 'AMBIGUOUS', fixture.id);
  }
  assert.ok(receipt.role_evidence && typeof receipt.role_evidence === 'object', fixture.id);
  assert.ok(Array.isArray(receipt.write_intents), fixture.id);
  assert.ok(Array.isArray(receipt.conflicts), fixture.id);
}

const forward = route(fixtures[1].mission);
const reverse = route(fixtures[2].mission);
for (const key of ['decision', 'implementation_repo', 'execution_repo', 'memory_repo', 'write_allowed_repos']) {
  assert.deepEqual(forward[key], reverse[key], `order independence: ${key}`);
}

const repeatA = route(fixtures[1].mission);
const repeatB = route(fixtures[1].mission);
for (const key of Object.keys(repeatA)) {
  if (!['routing_ms', 'total_ms'].includes(key)) assert.deepEqual(repeatA[key], repeatB[key], `repeatability: ${key}`);
}

console.log(`v4 adversarial tests: ${fixtures.length + 2} passed`);
