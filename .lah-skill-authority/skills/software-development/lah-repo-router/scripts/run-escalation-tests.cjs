#!/usr/bin/env node
/** LAH Repo Router v4 regression entrypoint. */
const { spawnSync } = require('node:child_process');
const path = require('node:path');

const test = path.join(__dirname, 'test-v4-router.cjs');
const result = spawnSync(process.execPath, [test], { encoding: 'utf8' });
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
process.exit(result.status === 0 ? 0 : 1);
