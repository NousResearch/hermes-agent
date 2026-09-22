#!/usr/bin/env node
// run-electron.mjs — exec the real Electron binary with a scrubbed env.
//
// Host Electron apps (Devin Desktop, other Electron shells) can leak
// ELECTRON_RUN_AS_NODE=1 into shells they spawn. Under it `electron` runs as
// plain Node: the app never opens and `electron --version` prints the
// embedded Node version. Electron checks presence, not value — an empty
// string still triggers it — so the variable must be removed, not blanked.
//
// Usage: node scripts/run-electron.mjs [electron args...]
//   node scripts/run-electron.mjs .

import { spawnSync } from 'node:child_process'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
// The `electron` package's entry point exports the path to the binary.
const electronBin = require('electron')

for (const key of Object.keys(process.env)) {
  if (key.toUpperCase() === 'ELECTRON_RUN_AS_NODE') {
    delete process.env[key]
  }
}

const result = spawnSync(electronBin, process.argv.slice(2), {
  stdio: 'inherit',
  env: process.env,
})

if (result.error) {
  console.error(`run-electron: failed to spawn ${electronBin}: ${result.error.message}`)
  process.exit(1)
}
process.exit(result.status ?? 1)
