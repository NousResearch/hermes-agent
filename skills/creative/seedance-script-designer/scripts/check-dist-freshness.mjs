#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

function usage() {
  console.error('Usage: node check-dist-freshness.mjs <skill-source-dir> <dist-file>');
  process.exit(2);
}

function walkFiles(root) {
  const files = [];
  const skip = new Set(['.git', 'node_modules', 'dist']);

  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (skip.has(entry.name)) continue;
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkFiles(fullPath));
    } else if (entry.isFile()) {
      files.push(fullPath);
    }
  }

  return files;
}

const [sourceArg, distArg] = process.argv.slice(2);
if (!sourceArg || !distArg) usage();

const sourceDir = path.resolve(sourceArg);
const distFile = path.resolve(distArg);

if (!fs.existsSync(sourceDir) || !fs.statSync(sourceDir).isDirectory()) {
  console.error(`Error: source dir not found: ${sourceDir}`);
  process.exit(1);
}

if (!fs.existsSync(distFile) || !fs.statSync(distFile).isFile()) {
  console.error(`Error: dist file not found: ${distFile}`);
  process.exit(1);
}

const sourceFiles = walkFiles(sourceDir);
if (sourceFiles.length === 0) {
  console.error(`Error: no source files found under ${sourceDir}`);
  process.exit(1);
}

let newest = { file: null, mtimeMs: 0 };
for (const file of sourceFiles) {
  const mtimeMs = fs.statSync(file).mtimeMs;
  if (mtimeMs > newest.mtimeMs) newest = { file, mtimeMs };
}

const distMtimeMs = fs.statSync(distFile).mtimeMs;
const newestTime = new Date(newest.mtimeMs).toISOString();
const distTime = new Date(distMtimeMs).toISOString();

console.error(`Newest source: ${newest.file}`);
console.error(`Source mtime: ${newestTime}`);
console.error(`Dist mtime:   ${distTime}`);

if (distMtimeMs < newest.mtimeMs) {
  console.error('Error: dist package is stale. Rebuild it before distributing the skill.');
  process.exit(1);
}

console.error('OK: dist package is fresh.');
