#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const EXPECTED_FILES = [
  'seedance-script-designer/README.md',
  'seedance-script-designer/SKILL.md',
  'seedance-script-designer/references/story-structure-method.md',
  'seedance-script-designer/references/dialogue-voice-method.md',
  'seedance-script-designer/references/shot-breakdown-method.md',
  'seedance-script-designer/references/output-schema.md',
  'seedance-script-designer/references/storyboard-annotation-rules.md',
  'seedance-script-designer/references/quick-modes.md',
  'seedance-script-designer/references/release-checklist.md',
  'seedance-script-designer/assets/README.md',
  'seedance-script-designer/assets/asset-manifest-template.md',
  'seedance-script-designer/scripts/README.md',
  'seedance-script-designer/scripts/convert-keyframe-table.mjs',
  'seedance-script-designer/scripts/validate-keyframe-table.mjs',
  'seedance-script-designer/scripts/check-dist-freshness.mjs',
  'seedance-script-designer/scripts/verify-package-contents.mjs',
  'seedance-script-designer/examples/minimal-12col-keyframes.md',
  'seedance-script-designer/examples/asset-manifest-example.md',
  'seedance-script-designer/examples/audio-sync-example.md',
];

function usage() {
  console.error('Usage: node verify-package-contents.mjs <dist-file.skill>');
  process.exit(2);
}

function normalizeTarPath(value) {
  return value.replaceAll('\\', '/').replace(/^\.\//, '').replace(/\/$/, '');
}

const distArg = process.argv[2];
if (!distArg) usage();

const distFile = path.resolve(distArg);
if (!fs.existsSync(distFile) || !fs.statSync(distFile).isFile()) {
  console.error(`Error: package file not found: ${distFile}`);
  process.exit(1);
}

let output;
try {
  output = execFileSync('tar', ['-tf', distFile], { encoding: 'utf8' });
} catch (error) {
  console.error(`Error: failed to list package contents with tar: ${error.message}`);
  process.exit(1);
}

const actual = output
  .split(/\r?\n/)
  .map((line) => normalizeTarPath(line.trim()))
  .filter(Boolean)
  .sort();

const expected = EXPECTED_FILES.map(normalizeTarPath).sort();
const actualSet = new Set(actual);
const expectedSet = new Set(expected);

const missing = expected.filter((file) => !actualSet.has(file));
const extra = actual.filter((file) => !expectedSet.has(file));

for (const file of missing) console.error(`Missing: ${file}`);
for (const file of extra) console.error(`Extra: ${file}`);

console.error(`Package contents: ${actual.length} file(s), expected ${expected.length}.`);

if (missing.length > 0 || extra.length > 0) {
  console.error('Error: package contents do not match the expected skill manifest.');
  process.exit(1);
}

console.error('OK: package contents match the expected skill manifest.');
