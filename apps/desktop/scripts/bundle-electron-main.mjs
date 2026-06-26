#!/usr/bin/env node
// bundle-electron-main.mjs — bundles electron/main.cjs into a single
// self-contained file so the nix build doesn't need to ship node_modules/.
//
// `electron` is provided by the runtime; `node-pty` is staged separately
// via stage-native-deps.cjs.  `preload.cjs` is NOT require()'d by main —
// Electron loads it via path.join(__dirname, 'preload.cjs') — so it stays
// as a separate file and doesn't need bundling.
import { build } from 'esbuild'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { renameSync } from 'node:fs'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const entry = resolve(root, 'electron/main.cjs')
const tmp = resolve(root, 'electron/main.bundled.cjs')

await build({
  entryPoints: [entry],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node20',
  outfile: tmp,
  external: ['electron', 'node-pty'],
  logLevel: 'info'
})

// Overwrite the original with the bundled version.
renameSync(tmp, entry)

console.log(`bundled ${entry}`)
