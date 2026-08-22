#!/usr/bin/env node
/** Three-way source/install hash and semantic-contract validator. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const cp = require('node:child_process');
const sourceRoot = path.resolve(__dirname, '..');
const expectedHermes = '/home/deploy/.hermes/skills/software-development/lah-repo-router';
const expectedCodex = '/home/deploy/.codex/skills/lah-repo-router';
const HERMES = [
  'SKILL.md', 'scripts/dry-run-route.sh', 'scripts/dry-run-route.cjs', 'scripts/test-v4-router.cjs', 'scripts/test-v4-write-policy.cjs', 'scripts/run-escalation-tests.cjs', 'scripts/validate-routing-drift.cjs', 'scripts/validate-installation-drift.cjs', 'scripts/validate-installation-drift.sh', 'references/repo_mappings.json', 'references/receipt-schema-v4.json', 'references/source-contract.json', 'references/v4-adversarial-fixtures.json', 'references/escalation-tests.txt', 'references/test-missions.txt', 'references/router-output-format.md', 'references/dry-run-test-results.md', 'references/refresh-limitations.md'
];
const CODEX = [{ source: 'templates/codex-SKILL.md', target: 'SKILL.md' }];
function hash(file) { return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'); }
function parse(argv) {
  const out = { source: sourceRoot, hermes: expectedHermes, codex: expectedCodex, json: false };
  for (let i = 0; i < argv.length; i += 1) { const a = argv[i]; if (a === '--source') out.source = path.resolve(argv[++i]); else if (a === '--hermes') out.hermes = path.resolve(argv[++i]); else if (a === '--codex') out.codex = path.resolve(argv[++i]); else if (a === '--json') out.json = true; else { console.error(`unknown argument ${a}`); process.exit(2); } }
  return out;
}
function classify(result, category, detail) { result.failures.push({ category, ...detail }); }
function compare(result, source, target, relative) {
  if (!fs.existsSync(source)) { classify(result, 'SOURCE_DRIFT', { relative, source, reason: 'source artifact missing' }); return; }
  if (!fs.existsSync(target)) { classify(result, 'MISSING_ARTIFACT', { relative, target }); return; }
  if (hash(source) !== hash(target)) classify(result, 'INSTALLATION_DRIFT', { relative, source_sha256: hash(source), target_sha256: hash(target), target });
}
function main() {
  const args = parse(process.argv.slice(2));
  const result = { source: args.source, hermes: args.hermes, codex: args.codex, mapping_fingerprint: null, source_to_hermes: 'PASS', source_to_codex: 'PASS', hermes_to_codex: 'PASS', semantic_equivalence: 'PASS', failures: [] };
  const contractPath = path.join(args.source, 'references/source-contract.json');
  const mappingPath = path.join(args.source, 'references/repo_mappings.json');
  if (!fs.existsSync(contractPath) || !fs.existsSync(mappingPath)) classify(result, 'SOURCE_DRIFT', { reason: 'canonical contract or mapping missing' });
  else { const contract = JSON.parse(fs.readFileSync(contractPath, 'utf8')); result.mapping_fingerprint = hash(mappingPath); if (result.mapping_fingerprint !== contract.mapping_fingerprint) classify(result, 'SOURCE_DRIFT', { reason: 'mapping fingerprint does not match source contract', expected: contract.mapping_fingerprint, actual: result.mapping_fingerprint }); }
  for (const relative of HERMES) compare(result, path.join(args.source, relative), path.join(args.hermes, relative), relative);
  for (const item of CODEX) compare(result, path.join(args.source, item.source), path.join(args.codex, item.target), item.target);
  if (result.failures.some((f) => f.category === 'INSTALLATION_DRIFT' || f.category === 'MISSING_ARTIFACT')) { result.source_to_hermes = result.failures.some((f) => f.target?.startsWith(args.hermes)) ? 'FAIL' : 'PASS'; result.source_to_codex = result.failures.some((f) => f.target?.startsWith(args.codex)) ? 'FAIL' : 'PASS'; }
  const hermesMapping = path.join(args.hermes, 'references/repo_mappings.json');
  const codexSkill = path.join(args.codex, 'SKILL.md');
  if (fs.existsSync(hermesMapping) && hash(hermesMapping) !== result.mapping_fingerprint) result.hermes_to_codex = 'FAIL';
  if (!fs.existsSync(codexSkill)) result.hermes_to_codex = 'FAIL';
  if (result.failures.length) result.semantic_equivalence = 'FAIL';
  result.verdict = result.failures.length ? 'DRIFT_CHECK_FAIL' : 'DRIFT_CHECK_PASS';
  if (args.json) console.log(JSON.stringify(result, null, 2)); else console.log(JSON.stringify(result, null, 2));
  process.exitCode = result.failures.length ? 1 : 0;
}
main();
