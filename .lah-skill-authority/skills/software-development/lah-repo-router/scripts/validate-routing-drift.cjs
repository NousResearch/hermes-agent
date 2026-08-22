#!/usr/bin/env node
/** Validate one canonical v4 ontology and documentation handoff. */
const fs = require('node:fs');
const crypto = require('node:crypto');
const path = require('node:path');

const skillRoot = path.resolve(__dirname, '..');
const mappingPath = path.join(skillRoot, 'references', 'repo_mappings.json');
const mapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));
const codexPath = '/home/deploy/.codex/skills/lah-repo-router/SKILL.md';
const hermesPath = path.join(skillRoot, 'SKILL.md');
const docs = [codexPath, hermesPath].map((file) => ({ file, text: fs.readFileSync(file, 'utf8') }));
const failures = [];

if (mapping.schema_version !== '4') failures.push('canonical mapping schema_version is not 4');
if (!mapping.repositories.some((repo) => repo.repository_id === 'lah-stack-skills' && repo.roles.includes('SKILL_KNOWLEDGE'))) {
  failures.push('lah-stack-skills SKILL_KNOWLEDGE authority missing');
}
for (const role of ['IMPLEMENTATION', 'EXECUTION_RUNTIME', 'GOVERNANCE', 'MEMORY', 'CONTEXT', 'SKILL_KNOWLEDGE', 'BUSINESS_ASSET']) {
  if (!mapping.repositories.some((repo) => repo.roles.includes(role))) failures.push(`no repository declares role ${role}`);
}
for (const doc of docs) {
  if (!doc.text.includes('repo_mappings.json')) failures.push(`${doc.file} does not reference canonical mapping`);
  if (!doc.text.includes('schema v4')) failures.push(`${doc.file} does not declare schema v4`);
  if (doc.text.includes('mission prefix LAH_ resolves to lah-core')) failures.push(`${doc.file} retains prefix authority doctrine`);
}

const fingerprint = crypto.createHash('sha256').update(fs.readFileSync(mappingPath)).digest('hex');
console.log(JSON.stringify({
  schema_version: mapping.schema_version,
  canonical_mapping: mappingPath,
  ontology_fingerprint: fingerprint,
  docs_checked: docs.map((doc) => doc.file),
  failures,
  verdict: failures.length ? 'DRIFT_DETECTED' : 'DRIFT_CHECK_PASS',
}, null, 2));
process.exit(failures.length ? 1 : 0);
