#!/usr/bin/env node
import { spawn } from 'node:child_process';
import process from 'node:process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const isWin = process.platform === 'win32';

const script = isWin ? 'start-samagent.bat' : './start-samagent.sh';
const command = isWin ? 'cmd.exe' : 'bash';
const args = isWin ? ['/c', path.join(rootDir, script)] : [path.join(rootDir, script)];

const child = spawn(command, args, {
  cwd: rootDir,
  stdio: 'inherit',
  env: process.env,
});

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 0);
  }
});
