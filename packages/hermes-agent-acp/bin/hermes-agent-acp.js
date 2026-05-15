#!/usr/bin/env node
'use strict';

const { spawn, spawnSync } = require('node:child_process');

const HERMES_AGENT_VERSION = '0.13.0';
const HERMES_SPEC = `hermes-agent[acp]==${HERMES_AGENT_VERSION}`;

function commandExists(command) {
  const result = spawnSync(command, ['--version'], { stdio: 'ignore' });
  return !result.error && result.status === 0;
}

function buildCommand(argv, exists = commandExists) {
  if (exists('uvx')) {
    return {
      command: 'uvx',
      args: ['--from', HERMES_SPEC, 'hermes-acp', ...argv],
    };
  }

  if (exists('uv')) {
    return {
      command: 'uv',
      args: ['tool', 'run', '--from', HERMES_SPEC, 'hermes-acp', ...argv],
    };
  }

  return null;
}

function main() {
  const argv = process.argv.slice(2);
  const command = buildCommand(argv);

  if (!command) {
    console.error('Hermes Agent ACP requires uv or uvx to launch the Python package.');
    console.error('Install uv from https://docs.astral.sh/uv/getting-started/installation/');
    console.error('Then retry this agent from Zed.');
    process.exit(127);
  }

  const child = spawn(command.command, command.args, {
    stdio: 'inherit',
    env: process.env,
  });

  child.on('error', (error) => {
    console.error(`Failed to start Hermes Agent ACP: ${error.message}`);
    process.exit(1);
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 0);
  });
}

if (require.main === module) {
  main();
}

module.exports = { buildCommand, HERMES_AGENT_VERSION, HERMES_SPEC };
