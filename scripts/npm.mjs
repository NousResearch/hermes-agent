#!/usr/bin/env node
// npm.mjs — run an npm subcommand with a scrubbed environment.
//
// `npm run` exports the resolved config as npm_config_* env vars. When the
// resolved allow-scripts value (project or user .npmrc) reaches a child npm
// this way, npm >= 11.17 treats it as a CLI/env-layer policy and rejects
// project-scoped install/ci/audit/exec with EALLOWSCRIPTS. Removing it lets
// the child fall back to package.json#allowScripts — the intended policy —
// so no allow-scripts setting is weakened or bypassed by this wrapper.
//
// Usage: node scripts/npm.mjs <npm args...>
//   node scripts/npm.mjs install --workspace apps/desktop

import { spawnSync } from 'node:child_process'

for (const key of Object.keys(process.env)) {
  if (key.toLowerCase().replace(/-/g, '_') === 'npm_config_allow_scripts') {
    delete process.env[key]
  }
}

const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'
const result = spawnSync(npm, process.argv.slice(2), {
  stdio: 'inherit',
  // npm.cmd is a batch file — Windows requires a shell to exec it
  // (spawnSync without shell raises EINVAL on .cmd since Node 20.12.2).
  shell: process.platform === 'win32',
})

if (result.error) {
  console.error(`npm.mjs: failed to spawn ${npm}: ${result.error.message}`)
  process.exit(1)
}
process.exit(result.status ?? 1)
