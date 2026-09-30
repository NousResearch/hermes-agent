#!/usr/bin/env node
import { spawn } from 'node:child_process';
import process from 'node:process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const isWin = process.platform === 'win32';

const venvPytest = isWin
  ? path.join(rootDir, '.samagent-venv', 'Scripts', 'pytest.exe')
  : path.join(rootDir, '.samagent-venv', 'bin', 'pytest');

if (!fs.existsSync(venvPytest)) {
  console.error(`Error: pytest not found at ${venvPytest}`);
  console.error('Please run "npm run dev" or "start-samagent.bat" to set up the environment.');
  process.exit(1);
}

const args = ['tests/samagent/test_samagent_pipeline.py', '-v', ...process.argv.slice(2)];
const env = { ...process.env, PYTHONPATH: rootDir };

const child = spawn(venvPytest, args, {
  cwd: rootDir,
  stdio: 'inherit',
  env,
});

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 0);
  }
});
