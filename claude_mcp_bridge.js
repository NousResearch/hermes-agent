#!/usr/bin/env node
/**
 * Claude Code MCP Bridge - Node.js proxy
 *
 * Wraps `claude mcp serve` with a PTY to keep it alive,
 * then exposes stdin/stdout as a stdio MCP server.
 */

const { spawn } = require('child_process');
const pty = require('ptyw');

// For debugging
const DEBUG = process.env.DEBUG === '1';

let ptyMaster = null;
let claudeProc = null;

function debug(...args) {
  if (DEBUG) console.error('[claude-mcp-bridge]', ...args);
}

async function main() {
  debug('Starting Claude MCP bridge...');

  // Open PTY
  const ptyInfo = pty.open();
  ptyMaster = ptyInfo.fd;

  debug(`PTY opened: master=${ptyMaster}`);

  // Spawn claude mcp serve with the PTY as stdin/stdout
  claudeProc = spawn('/usr/local/bin/claude', ['mcp', 'serve', '--verbose'], {
    stdio: [ptyMaster, ptyMaster, 'inherit'],
    env: {
      ...process.env,
      HOME: process.env.HOME || '/root',
      TERM: process.env.TERM || 'xterm-256color',
      CLAUDE_CODE_SIMPLE: '1',
    },
    detached: false,
    cwd: process.env.HOME || '/root',
  });

  debug(`Claude process spawned: pid=${claudeProc.pid}`);

  claudeProc.on('exit', (code, signal) => {
    debug(`Claude exited: code=${code}, signal=${signal}`);
    process.exit(code || 0);
  });

  claudeProc.on('error', (err) => {
    debug('Claude error:', err.message);
    process.exit(1);
  });

  // Keep this process alive
  // Just wait for the PTY master to close or a signal
  process.on('SIGINT', () => {
    debug('SIGINT received');
    if (claudeProc && claudeProc.exitCode === null) {
      claudeProc.kill('SIGINT');
    }
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    debug('SIGTERM received');
    if (claudeProc && claudeProc.exitCode === null) {
      claudeProc.kill('SIGTERM');
    }
    process.exit(0);
  });

  // Just keep the process alive
  debug('Bridge ready, waiting for input...');
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});