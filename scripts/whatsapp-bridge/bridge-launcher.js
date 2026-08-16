#!/usr/bin/env node

const guidance = [
  'WhatsApp bridge serving is gateway-managed.',
  'Run `hermes gateway` to launch the authenticated private IPC bridge.',
  'Run `hermes whatsapp` for pairing-only mode (no control server).',
  'Native Windows serving is not supported; use WSL2/Linux.',
].join('\n');

console.error(guidance);
process.exitCode = 1;
