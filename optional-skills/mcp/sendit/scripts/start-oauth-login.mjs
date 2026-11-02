#!/usr/bin/env node

import { existsSync, mkdirSync, openSync, readFileSync, writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';

const serverName = process.env.SENDIT_HERMES_MCP_SERVER || 'sendit';
const logDir = process.env.SENDIT_HERMES_LOG_DIR || '/tmp/sendit-hermes';
const logPath = `${logDir}/oauth.log`;
const pidPath = `${logDir}/oauth.pid`;
const authUrlPattern = /https:\/\/sendit\.infiniteappsai\.com\/oauth\/authorize[^\s"'<>]+/;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

mkdirSync(logDir, { recursive: true });

const out = openSync(logPath, 'w');
const child = spawn('hermes', ['mcp', 'login', serverName], {
  detached: true,
  stdio: ['ignore', out, out],
});

child.unref();
writeFileSync(pidPath, `${child.pid}\n`, 'utf8');

console.log(`Started Hermes SendIt OAuth login on VPS. PID: ${child.pid}`);
console.log(`MCP server name: ${serverName}`);
console.log(`Log: ${logPath}`);

let authUrl = '';
for (let i = 0; i < 60; i += 1) {
  await sleep(500);
  if (!existsSync(logPath)) continue;

  const log = readFileSync(logPath, 'utf8');
  const match = log.match(authUrlPattern);
  if (match) {
    authUrl = match[0];
    break;
  }
}

if (!authUrl) {
  console.log('');
  console.log('Authorization URL was not found yet. Ask Hermes to inspect:');
  console.log(`  ${logPath}`);
  console.log('If the process exited, rerun this script.');
  process.exit(1);
}

console.log('');
console.log('Send this authorization URL to the user:');
console.log(authUrl);
console.log('');
console.log('After the user signs in, ask them to paste the full localhost callback URL.');
console.log('Then run:');
console.log("  node scripts/complete-oauth-callback.mjs '<PASTED_CALLBACK_URL>'");
